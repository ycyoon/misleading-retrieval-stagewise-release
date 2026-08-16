"""M2: Candidate-Blind Relational Corroboration Verifier.

Small torch model over 3 per-source nodes (C/R/P). Each node embeds
question [SEP] answer [SEP] evidence-snippet with a FROZEN text encoder
(bge-base-en-v1.5 via sentence_transformers, or a deterministic hash encoder
for tests/debug). Node vectors = MLP(embedding) + learned role embedding
(+ absent-flag embedding for missing candidates). Pairwise relation logits
over {support, contradict, unknown} feed one round of relation-aware attention;
three heads: per-source confidence (aux), PRIMARY (answer_context vs abstain),
RECOVERY (4-way over deduped candidates with merged-candidate masking).

Torch-free callers must check torch_available() before using the model classes.
The abstain threshold is chosen on dev by the same c=2 utility sweep as
models.gate.Gate.select_threshold, then FROZEN.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.src.data.views import InputView
from experiments.src.features.matching import normalize
from experiments.src.features.node_features import ACTIONS, PAIRS, SOURCES

try:  # torch is optional at import time; callers check torch_available().
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover - exercised via monkeypatch in tests
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_OK = False

_ANSWER_ATTR = {
    "answer_context": "context_answer",
    "answer_reference": "reference_answer",
    "answer_parametric": "parametric_answer",
}
_MATCH_KEY = {
    "answer_context": "context",
    "answer_reference": "reference",
    "answer_parametric": "parametric",
}

DEFAULT_THRESHOLD = 0.5
DEFAULT_LOSS_WEIGHTS = {"primary": 1.0, "recovery": 1.0,
                        "source_conf": 0.3, "relation": 0.3,
                        # M3: lambda_cf; 0 keeps exact M2 behavior.
                        "counterfactual": 0.0}
_NEG_INF = -1e9

# ---------------------------------------------------------------------------
# M3 counterfactual-consistency constants.  Condition / dataset labels are
# LOSS-ONLY side information (never model features; the forward pass sees only
# embeddings + absent flags).  CF_PAIRS are (should-be-higher, should-be-lower)
# indices into CF_CONDITIONS for the margin-ranking terms on p(answer_context):
#   max(0, m - p^clean + p^poison)  and  max(0, m - p^mixed + p^noanswer).
CF_CONDITIONS = ("C_suf_clean", "C_poison", "C_mixed", "C_insuf")
CF_PAIRS = ((0, 1), (2, 3))
DEFAULT_CF_MARGIN = 0.2
DEFAULT_DRO_ETA = 0.1


def brier_binary(p, y) -> float:
    """Mean squared error between probabilities and binary labels."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece_binary(p, y, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins over [0, 1]."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return 0.0
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def torch_available() -> bool:
    """True iff torch imported; the relational model degrades to a clear
    fallback path in scripts/train.py when this is False."""
    return bool(_TORCH_OK)


def resolve_device(device: str = "auto") -> str:
    """'auto' -> cuda when available, else cpu; explicit strings pass through."""
    if device == "auto":
        return "cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu"
    return device


# ---------------------------------------------------------------------------
# Frozen text encoders + disk-cached batch encoding
# ---------------------------------------------------------------------------

class HashEncoder:
    """Deterministic torch-free bag-of-tokens hashing encoder (tests/debug).

    Feature-hashes normalized unigram tokens into `dim` signed buckets and
    l2-normalizes. No network, no model download, stable across processes.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)
        self.name = f"hash-{self.dim}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in normalize(text).split():
                digest = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class BgeEncoder:
    """FROZEN sentence_transformers encoder (default BAAI/bge-base-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str = "auto", batch_size: int = 128) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.name = f"bge:{model_name}"
        self._model = None  # lazy

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        emb = model.encode(list(texts), batch_size=self.batch_size,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def make_encoder(cfg: Optional[Dict[str, Any]] = None):
    """Encoder from config: backend 'bge' (default) or 'hash'."""
    cfg = dict(cfg or {})
    backend = cfg.get("backend", "bge")
    if backend == "hash":
        return HashEncoder(dim=int(cfg.get("dim", 64)))
    if backend == "bge":
        return BgeEncoder(model_name=cfg.get("model_name", "BAAI/bge-base-en-v1.5"),
                          device=cfg.get("device", "auto"),
                          batch_size=int(cfg.get("batch_size", 128)))
    raise ValueError(f"unknown encoder backend {backend!r}; want 'bge' or 'hash'")


class EmbeddingCache:
    """Sharded per-text .npy embedding cache keyed by sha256(encoder|text)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def key(encoder_name: str, text: str) -> str:
        return hashlib.sha256(
            (encoder_name + "\x00" + text).encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npy"

    def get(self, key: str) -> Optional[np.ndarray]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    def put(self, key: str, vec: np.ndarray) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(vec, dtype=np.float32))
        tmp.replace(path)


def encode_node_texts(texts: Sequence[Sequence[Optional[str]]], encoder,
                      cache: Optional[EmbeddingCache] = None) -> np.ndarray:
    """(N, 3, dim) float32 node embeddings; absent (None) texts -> zeros.

    Unique texts are batch-encoded once; with a cache, embeddings are read
    from / written to disk so repeated runs cost zero encoder calls.
    """
    unique: Dict[str, Optional[np.ndarray]] = {}
    for row in texts:
        for t in row:
            if t is not None and t not in unique:
                unique[t] = None
    todo: List[str] = []
    for t in unique:
        if cache is not None:
            hit = cache.get(EmbeddingCache.key(encoder.name, t))
            if hit is not None:
                unique[t] = hit.astype(np.float32)
                continue
        todo.append(t)
    if todo:
        encoded = encoder.encode(todo)
        for t, vec in zip(todo, encoded):
            vec = np.asarray(vec, dtype=np.float32)
            unique[t] = vec
            if cache is not None:
                cache.put(EmbeddingCache.key(encoder.name, t), vec)
    dim = int(encoder.dim) if not unique else int(next(iter(unique.values())).shape[0])
    out = np.zeros((len(texts), len(SOURCES), dim), dtype=np.float32)
    for i, row in enumerate(texts):
        for j, t in enumerate(row):
            if t is not None:
                out[i, j] = unique[t]
    return out


# ---------------------------------------------------------------------------
# Dev threshold sweep (identical semantics to gate.Gate.select_threshold)
# ---------------------------------------------------------------------------

def sweep_threshold(p: np.ndarray, proposed: Sequence[str],
                    candidate_matches: Sequence[Dict[str, bool]],
                    c: float = 2.0) -> float:
    """Utility-maximizing abstain threshold (+1 correct, -c wrong, 0 abstain);
    ties prefer the higher (more abstaining) threshold. Mirrors gate.py."""
    p = np.asarray(p, dtype=float)
    correct = np.array([
        bool(m.get(_MATCH_KEY[a], False)) if a != "abstain" else False
        for m, a in zip(candidate_matches, proposed)
    ])
    candidates = np.unique(np.concatenate([p, [0.0, 1.0 + 1e-9]]))
    best_u, best_t = -np.inf, float(candidates[-1])
    for t in np.sort(candidates):  # ascending; >= keeps highest tied t
        answered = p >= t
        n_ans = int(answered.sum())
        n_correct = int(correct[answered].sum())
        u = n_correct - c * (n_ans - n_correct)
        if u >= best_u:
            best_u, best_t = u, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Torch model
# ---------------------------------------------------------------------------

if _TORCH_OK:

    class RelationalVerifier(nn.Module):
        """3-node relational net: node MLP + role/absent embeddings, pairwise
        relation logits, one relation-aware attention round, three heads."""

        def __init__(self, embed_dim: int, hidden_dim: int = 256,
                     dropout: float = 0.1) -> None:
            super().__init__()
            h = int(hidden_dim)
            self.embed_dim = int(embed_dim)
            self.hidden_dim = h
            self.node_mlp = nn.Sequential(
                nn.Linear(embed_dim, h), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(h, h))
            self.role_emb = nn.Embedding(len(SOURCES), h)
            self.absent_emb = nn.Parameter(torch.zeros(h))
            self.ln_node = nn.LayerNorm(h)
            # relation encoder over [h_i, h_j, h_i*h_j, |h_i-h_j|]
            self.rel_mlp = nn.Sequential(
                nn.Linear(4 * h, h), nn.GELU(), nn.Linear(h, 3))
            # relation-aware attention (single round, single head)
            self.q = nn.Linear(h, h, bias=False)
            self.k = nn.Linear(h, h, bias=False)
            self.v = nn.Linear(h, h, bias=False)
            self.edge_score = nn.Linear(3, 1)
            self.edge_val = nn.Linear(3, h)
            self.self_rel = nn.Parameter(torch.zeros(3))
            self.out_proj = nn.Linear(h, h)
            self.ln_attn = nn.LayerNorm(h)
            self.dropout = nn.Dropout(dropout)
            # heads
            self.conf_head = nn.Linear(h, 1)
            self.primary_head = nn.Sequential(
                nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(h, 1))
            self.rec_src_head = nn.Linear(h, 1)
            self.rec_abstain_head = nn.Linear(h, 1)
            self.scale = h ** -0.5

        def forward(self, emb: "torch.Tensor",
                    absent: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            """emb (B,3,D) float, absent (B,3) float {0,1}."""
            n_src = len(SOURCES)
            hnode = (self.node_mlp(emb)
                     + self.role_emb.weight.unsqueeze(0)
                     + absent.unsqueeze(-1) * self.absent_emb)
            hnode = self.ln_node(hnode)  # (B,3,h)
            hi = hnode.unsqueeze(2).expand(-1, -1, n_src, -1)
            hj = hnode.unsqueeze(1).expand(-1, n_src, -1, -1)
            pair = torch.cat([hi, hj, hi * hj, (hi - hj).abs()], dim=-1)
            rel_logits = self.rel_mlp(pair)  # (B,3,3,3) directed [i,j]
            rel_probs = rel_logits.softmax(dim=-1)
            eye = torch.eye(n_src, device=emb.device,
                            dtype=torch.bool).view(1, n_src, n_src, 1)
            self_probs = self.self_rel.softmax(dim=-1).view(1, 1, 1, 3)
            rel_probs = torch.where(eye, self_probs.expand_as(rel_probs), rel_probs)
            scores = torch.matmul(self.q(hnode),
                                  self.k(hnode).transpose(1, 2)) * self.scale
            scores = scores + self.edge_score(rel_probs).squeeze(-1)
            attn = scores.softmax(dim=-1)  # (B,3,3) over sender j
            msg = self.v(hnode).unsqueeze(1) + self.edge_val(rel_probs)
            agg = (attn.unsqueeze(-1) * msg).sum(dim=2)
            hout = self.ln_attn(hnode + self.dropout(self.out_proj(agg)))
            pooled = hout.mean(dim=1)
            primary_logit = self.primary_head(
                torch.cat([hout[:, 0], pooled], dim=-1)).squeeze(-1)
            conf_logits = self.conf_head(hout).squeeze(-1)  # (B,3)
            rec_logits = torch.cat(
                [self.rec_src_head(hout).squeeze(-1),
                 self.rec_abstain_head(pooled)], dim=-1)  # (B,4) ACTIONS order
            rel_sup = torch.stack(
                [rel_logits[:, i, j] for i, j, _tag in PAIRS], dim=1)  # (B,3,3)
            return {"primary_logit": primary_logit, "conf_logits": conf_logits,
                    "rec_logits": rec_logits, "rel_logits": rel_sup}

    def counterfactual_loss(p: "torch.Tensor", cond_id: "torch.Tensor",
                            cf_group: "torch.Tensor",
                            margin: float = DEFAULT_CF_MARGIN) -> "torch.Tensor":
        """M3 L_cf: margin ranking over same-group condition pairs on
        p(answer_context) = sigmoid(primary_logit):

            max(0, m - p^clean + p^poison) + max(0, m - p^mixed + p^noanswer)

        cond_id indexes CF_CONDITIONS (-1 = not a CF condition), cf_group is an
        integer question-group id; both are LOSS-ONLY (never features).  A
        group lacking either condition of a pair contributes no term for that
        pair (the term is DROPPED, documented behavior).  Returns the mean over
        all present terms, or 0 when no pair is complete — batches must
        therefore be group-sampled so pairs co-occur (RelationalGate does)."""
        cond_np = cond_id.detach().cpu().numpy()
        grp_np = cf_group.detach().cpu().numpy()
        pos: Dict[int, Dict[int, int]] = {}
        for c in range(len(CF_CONDITIONS)):
            rows = np.nonzero(cond_np == c)[0]
            pos[c] = {int(g): int(i) for g, i in zip(grp_np[rows], rows)}
        hi_idx: List[int] = []
        lo_idx: List[int] = []
        for hi, lo in CF_PAIRS:
            common = pos[hi].keys() & pos[lo].keys()
            hi_idx.extend(pos[hi][g] for g in common)
            lo_idx.extend(pos[lo][g] for g in common)
        if not hi_idx:
            return p.new_zeros(())
        ihi = torch.as_tensor(hi_idx, device=p.device, dtype=torch.long)
        ilo = torch.as_tensor(lo_idx, device=p.device, dtype=torch.long)
        return F.relu(margin - p[ihi] + p[ilo]).mean()

    def fit_temperature(nll_fn, device: str,
                        max_iter: int = 100) -> Tuple[float, float, float]:
        """Scalar temperature minimizing nll_fn(T) via LBFGS on log T
        (positivity by construction).  Returns (T, nll_before, nll_after);
        keeps T=1 when optimization does not improve the NLL."""
        log_t = torch.zeros((), device=device, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=int(max_iter))

        def closure():
            opt.zero_grad()
            loss = nll_fn(log_t.exp())
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            t = float(log_t.exp().clamp(1e-3, 1e3))
            one = torch.ones((), device=device)
            nll_before = float(nll_fn(one))
            nll_after = float(nll_fn(one * t))
        if not np.isfinite(t) or nll_after > nll_before:
            return 1.0, nll_before, nll_before
        return t, nll_before, nll_after


# ---------------------------------------------------------------------------
# Trainer / predictor wrapper
# ---------------------------------------------------------------------------

def pack_data(emb: np.ndarray, absent: Sequence[Sequence[float]],
              action_mask: Sequence[Sequence[bool]],
              primary_y: Optional[Sequence[int]] = None,
              recovery_y: Optional[Sequence[str]] = None,
              relation_y: Optional[Sequence[Sequence[int]]] = None,
              conf_y: Optional[Sequence[Sequence[float]]] = None,
              cond_id: Optional[Sequence[int]] = None,
              cf_group: Optional[Sequence[int]] = None,
              dro_group: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Numpy training/eval bundle for RelationalGate (labels optional).

    cond_id / cf_group / dro_group are M3 LOSS-ONLY side arrays (counterfactual
    pairing + Group-DRO group ids); they never enter the forward pass."""
    data: Dict[str, np.ndarray] = {
        "emb": np.asarray(emb, dtype=np.float32),
        "absent": np.asarray(absent, dtype=np.float32),
        "mask": np.asarray(action_mask, dtype=bool),
    }
    if primary_y is not None:
        data["primary_y"] = np.asarray(primary_y, dtype=np.float32)
    if recovery_y is not None:
        data["recovery_y"] = np.asarray(
            [ACTIONS.index(a) for a in recovery_y], dtype=np.int64)
    if relation_y is not None:
        data["relation_y"] = np.asarray(relation_y, dtype=np.int64)
    if conf_y is not None:
        data["conf_y"] = np.asarray(conf_y, dtype=np.float32)
    if cond_id is not None:
        data["cond_id"] = np.asarray(cond_id, dtype=np.int64)
    if cf_group is not None:
        data["cf_group"] = np.asarray(cf_group, dtype=np.int64)
    if dro_group is not None:
        data["dro_group"] = np.asarray(dro_group, dtype=np.int64)
    return data


def subset_data(data: Dict[str, np.ndarray], idx) -> Dict[str, np.ndarray]:
    return {k: v[np.asarray(idx)] for k, v in data.items()}


class RelationalGate:
    """M2/M3 wrapper: fit (AdamW + early stop on dev action loss), dev-frozen
    thresholds per head, gate-compatible predict, save/load.

    M3 additions (each independently switchable; defaults preserve exact M2
    behavior):
      * counterfactual consistency loss (loss_weights['counterfactual'] > 0
        + cond_id/cf_group arrays in the train bundle) — batches are then
        group-sampled so every condition of a group lands in the same batch;
      * Group-DRO (dro=True + dro_group array): exponentiated-gradient
        worst-group reweighting of the ACTION loss only (aux losses stay ERM);
        group ids are used in the LOSS only, never as features;
      * post-hoc temperature scaling via calibrate() (dev LBFGS on NLL);
        stored temperatures are applied inside _forward_all, so the dev-frozen
        threshold sweep and predict() see calibrated probabilities."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1,
                 loss_weights: Optional[Dict[str, float]] = None,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, max_epochs: int = 100, patience: int = 10,
                 min_epochs: int = 1, seed: int = 13, device: str = "auto",
                 cf_margin: float = DEFAULT_CF_MARGIN, dro: bool = False,
                 dro_eta: float = DEFAULT_DRO_ETA) -> None:
        if not _TORCH_OK:
            raise RuntimeError(
                "model_type 'relational' needs torch; install torch or use "
                "model_type logreg/gbt/mlp")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.loss_weights = {**DEFAULT_LOSS_WEIGHTS, **(loss_weights or {})}
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.seed = int(seed)
        self.cf_margin = float(cf_margin)
        self.dro = bool(dro)
        self.dro_eta = float(dro_eta)
        self.device = resolve_device(device)
        torch.manual_seed(self.seed)
        self.model = RelationalVerifier(self.embed_dim, self.hidden_dim,
                                        self.dropout).to(self.device)
        self.thresholds: Dict[str, float] = {}
        self.temperatures: Dict[str, float] = {}  # per-head T; missing -> 1.0
        self._dro_q = None  # (G,) exponentiated-gradient group weights
        self.history: List[Dict[str, float]] = []

    def dro_q(self) -> Optional[List[float]]:
        """Final Group-DRO weights (None when DRO was not used)."""
        if self._dro_q is None:
            return None
        return [float(x) for x in self._dro_q.detach().cpu()]

    # -- loss --------------------------------------------------------------
    def _batch_losses(self, out: Dict[str, "torch.Tensor"],
                      batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        w = self.loss_weights
        zero = out["primary_logit"].new_zeros(())
        losses = {"primary": zero, "recovery": zero,
                  "source_conf": zero, "relation": zero, "counterfactual": zero}
        per_primary = per_recovery = None
        if w.get("primary", 0) > 0 and "primary_y" in batch:
            per_primary = F.binary_cross_entropy_with_logits(
                out["primary_logit"], batch["primary_y"], reduction="none")
            losses["primary"] = per_primary.mean()
        if w.get("recovery", 0) > 0 and "recovery_y" in batch:
            masked = out["rec_logits"].masked_fill(~batch["mask"], _NEG_INF)
            per_recovery = F.cross_entropy(masked, batch["recovery_y"],
                                           reduction="none")
            losses["recovery"] = per_recovery.mean()
        if w.get("source_conf", 0) > 0 and "conf_y" in batch:
            present = 1.0 - batch["absent"]
            per = F.binary_cross_entropy_with_logits(
                out["conf_logits"], batch["conf_y"], reduction="none")
            losses["source_conf"] = (per * present).sum() / present.sum().clamp(min=1.0)
        if w.get("relation", 0) > 0 and "relation_y" in batch:
            losses["relation"] = F.cross_entropy(
                out["rel_logits"].reshape(-1, 3), batch["relation_y"].reshape(-1))
        if (w.get("counterfactual", 0) > 0 and "cond_id" in batch
                and "cf_group" in batch):
            losses["counterfactual"] = counterfactual_loss(
                torch.sigmoid(out["primary_logit"]), batch["cond_id"],
                batch["cf_group"], margin=self.cf_margin)
        losses["action"] = (w.get("primary", 0) * losses["primary"]
                            + w.get("recovery", 0) * losses["recovery"])
        # Per-row action loss for Group-DRO reweighting (None when no labels).
        per_action = None
        if per_primary is not None or per_recovery is not None:
            per_action = out["primary_logit"].new_zeros(
                out["primary_logit"].shape)
            if per_primary is not None:
                per_action = per_action + w.get("primary", 0) * per_primary
            if per_recovery is not None:
                per_action = per_action + w.get("recovery", 0) * per_recovery
        losses["_action_per_row"] = per_action
        losses["total"] = (losses["action"]
                           + w.get("source_conf", 0) * losses["source_conf"]
                           + w.get("relation", 0) * losses["relation"]
                           + w.get("counterfactual", 0) * losses["counterfactual"])
        return losses

    def _dro_action(self, per_row: "torch.Tensor",
                    group_ids: "torch.Tensor") -> "torch.Tensor":
        """Group-DRO robust action loss for one batch: per-group mean losses,
        exponentiated-gradient update q_g <- q_g * exp(eta * L_g) (then global
        renormalization), and the q-weighted mean over the groups present in
        the batch (weights renormalized over present groups to keep the loss
        scale comparable to ERM)."""
        uniq = torch.unique(group_ids)
        means = torch.stack([per_row[group_ids == g].mean() for g in uniq])
        with torch.no_grad():
            self._dro_q[uniq] = self._dro_q[uniq] * torch.exp(
                self.dro_eta * means.detach())
            self._dro_q = self._dro_q / self._dro_q.sum()
        qb = self._dro_q[uniq]
        return (qb / qb.sum() * means).sum()

    def _to_device(self, data: Dict[str, np.ndarray],
                   idx=None) -> Dict[str, "torch.Tensor"]:
        sub = data if idx is None else subset_data(data, idx)
        out = {}
        for key, arr in sub.items():
            t = torch.from_numpy(np.ascontiguousarray(arr))
            out[key] = t.to(self.device)
        return out

    def _eval_loss(self, data: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """(action_loss, total_loss) over data, minibatched, no grad."""
        self.model.eval()
        n = len(data["emb"])
        tot_action = tot_total = 0.0
        with torch.no_grad():
            for lo in range(0, n, self.batch_size):
                idx = np.arange(lo, min(lo + self.batch_size, n))
                batch = self._to_device(data, idx)
                out = self.model(batch["emb"], batch["absent"])
                losses = self._batch_losses(out, batch)
                tot_action += float(losses["action"]) * len(idx)
                tot_total += float(losses["total"]) * len(idx)
        return tot_action / max(n, 1), tot_total / max(n, 1)

    # -- training ----------------------------------------------------------
    def _train_batches(self, train: Dict[str, np.ndarray], rng,
                       group_sample: bool = False):
        """Shuffled batch index arrays.  With group_sample=True (M3
        counterfactual loss), whole cf_group blocks are kept together: group
        order is shuffled and groups are appended until the batch reaches
        batch_size rows, so every condition of a group shares a batch."""
        n = len(train["emb"])
        if not group_sample:
            perm = rng.permutation(n)
            for lo in range(0, n, self.batch_size):
                yield perm[lo:lo + self.batch_size]
            return
        groups = np.asarray(train["cf_group"])
        uniq = np.unique(groups)
        rows_of = {g: np.nonzero(groups == g)[0] for g in uniq}
        buf: List[int] = []
        for g in rng.permutation(uniq):
            buf.extend(rows_of[g].tolist())
            if len(buf) >= self.batch_size:
                yield np.asarray(buf)
                buf = []
        if buf:
            yield np.asarray(buf)

    def fit(self, train: Dict[str, np.ndarray],
            dev: Optional[Dict[str, np.ndarray]] = None,
            verbose: bool = False) -> "RelationalGate":
        """AdamW; early stop on dev action loss (falls back to train action
        loss when dev is None); restores the best epoch's weights.

        M3: when the counterfactual weight is active and the bundle carries
        cond_id/cf_group, batches group-sample (all conditions of a group
        together); when dro=True and dro_group is present, the batch ACTION
        loss is Group-DRO-reweighted (the early-stop monitor stays the plain
        ERM dev action loss)."""
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        use_cf = (self.loss_weights.get("counterfactual", 0) > 0
                  and "cond_id" in train and "cf_group" in train)
        use_dro = self.dro and "dro_group" in train
        if use_dro:
            n_groups = int(np.max(train["dro_group"])) + 1
            self._dro_q = torch.full((n_groups,), 1.0 / n_groups,
                                     device=self.device)
        best_loss, best_state, best_epoch, bad = np.inf, None, -1, 0
        monitor = dev if dev is not None and len(dev["emb"]) else train
        for epoch in range(self.max_epochs):
            self.model.train()
            for idx in self._train_batches(train, rng, group_sample=use_cf):
                batch = self._to_device(train, idx)
                out = self.model(batch["emb"], batch["absent"])
                losses = self._batch_losses(out, batch)
                loss = losses["total"]
                if use_dro and losses["_action_per_row"] is not None:
                    loss = (loss - losses["action"]
                            + self._dro_action(losses["_action_per_row"],
                                               batch["dro_group"]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            action_loss, total_loss = self._eval_loss(monitor)
            self.history.append({"epoch": epoch, "monitor_action_loss": action_loss,
                                 "monitor_total_loss": total_loss})
            if verbose:
                print(f"[relational] epoch={epoch} action={action_loss:.4f} "
                      f"total={total_loss:.4f}")
            if action_loss < best_loss - 1e-6:
                best_loss, best_epoch, bad = action_loss, epoch, 0
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            else:
                bad += 1
                if epoch + 1 >= self.min_epochs and bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_epoch = best_epoch
        self.model.to(self.device)
        return self

    # -- scoring -----------------------------------------------------------
    def _forward_all(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Probabilities for all rows; M3 temperatures (default 1.0) divide
        the head logits BEFORE sigmoid/softmax, so calibration flows into the
        threshold sweep and predict() alike."""
        self.model.eval()
        t_primary = float(self.temperatures.get("primary", 1.0))
        t_recovery = float(self.temperatures.get("recovery", 1.0))
        n = len(data["emb"])
        chunks: Dict[str, List[np.ndarray]] = {}
        with torch.no_grad():
            for lo in range(0, n, self.batch_size):
                idx = np.arange(lo, min(lo + self.batch_size, n))
                batch = self._to_device(data, idx)
                out = self.model(batch["emb"], batch["absent"])
                masked = (out["rec_logits"] / t_recovery).masked_fill(
                    ~batch["mask"], _NEG_INF)
                step = {
                    "p_primary": torch.sigmoid(out["primary_logit"] / t_primary),
                    "rec_probs": masked.softmax(dim=-1),
                    "conf": torch.sigmoid(out["conf_logits"]),
                    "rel_probs": out["rel_logits"].softmax(dim=-1),
                }
                for k, v in step.items():
                    chunks.setdefault(k, []).append(v.cpu().numpy())
        return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}

    def _logits_all(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Raw (uncalibrated) head logits for all rows, minibatched, no grad."""
        self.model.eval()
        n = len(data["emb"])
        prim: List[np.ndarray] = []
        rec: List[np.ndarray] = []
        with torch.no_grad():
            for lo in range(0, n, self.batch_size):
                idx = np.arange(lo, min(lo + self.batch_size, n))
                batch = self._to_device(data, idx)
                out = self.model(batch["emb"], batch["absent"])
                prim.append(out["primary_logit"].cpu().numpy())
                rec.append(out["rec_logits"].cpu().numpy())
        return {"primary_logit": np.concatenate(prim, axis=0),
                "rec_logits": np.concatenate(rec, axis=0)}

    # -- calibration (M3; dev-only, before threshold selection) -------------
    def calibrate(self, dev: Dict[str, np.ndarray],
                  heads: Sequence[str] = ("primary", "recovery"),
                  max_iter: int = 100) -> Dict[str, Dict[str, float]]:
        """Post-hoc temperature scaling of the action heads on DEV: LBFGS on
        the dev NLL (BCE for primary, mask-aware CE for recovery).  Stores T
        per head in self.temperatures (applied by _forward_all) and returns a
        report {head: {temperature, nll/ece/brier before-after}}.  Call BEFORE
        select_threshold so the frozen threshold sees calibrated scores."""
        logits = self._logits_all(dev)
        report: Dict[str, Dict[str, float]] = {}
        if "primary" in heads and "primary_y" in dev:
            z = torch.from_numpy(logits["primary_logit"]).to(self.device)
            y = torch.from_numpy(
                np.asarray(dev["primary_y"], dtype=np.float32)).to(self.device)
            t, nll_b, nll_a = fit_temperature(
                lambda tt: F.binary_cross_entropy_with_logits(z / tt, y),
                self.device, max_iter=max_iter)
            self.temperatures["primary"] = t
            yn = np.asarray(dev["primary_y"], dtype=float)
            zn = logits["primary_logit"].astype(np.float64)
            p_b = 1.0 / (1.0 + np.exp(-zn))
            p_a = 1.0 / (1.0 + np.exp(-zn / t))
            report["primary"] = {
                "temperature": t, "nll_before": nll_b, "nll_after": nll_a,
                "ece_before": ece_binary(p_b, yn), "ece_after": ece_binary(p_a, yn),
                "brier_before": brier_binary(p_b, yn),
                "brier_after": brier_binary(p_a, yn)}
        if "recovery" in heads and "recovery_y" in dev:
            z = torch.from_numpy(logits["rec_logits"]).to(self.device)
            mask = torch.from_numpy(
                np.asarray(dev["mask"], dtype=bool)).to(self.device)
            y = torch.from_numpy(
                np.asarray(dev["recovery_y"], dtype=np.int64)).to(self.device)
            t, nll_b, nll_a = fit_temperature(
                lambda tt: F.cross_entropy(
                    (z / tt).masked_fill(~mask, _NEG_INF), y),
                self.device, max_iter=max_iter)
            self.temperatures["recovery"] = t
            y_np = np.asarray(dev["recovery_y"], dtype=int)
            stats: Dict[str, float] = {"temperature": t,
                                       "nll_before": nll_b, "nll_after": nll_a}
            for tag, tt in (("before", 1.0), ("after", t)):
                probs = torch.softmax(
                    (z / tt).masked_fill(~mask, _NEG_INF), dim=-1).cpu().numpy()
                conf = probs.max(axis=1)
                correct = (probs.argmax(axis=1) == y_np).astype(float)
                onehot = np.eye(probs.shape[1])[y_np]
                stats[f"ece_{tag}"] = ece_binary(conf, correct)
                stats[f"brier_{tag}"] = float(
                    np.mean(np.sum((probs - onehot) ** 2, axis=1)))
            report["recovery"] = stats
        return report

    def head_scores(self, head: str,
                    data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Per-row (p_answer, proposed action): sigmoid P for primary; best
        unmasked answer-class probability for recovery (gate semantics)."""
        fwd = self._forward_all(data)
        n = len(data["emb"])
        if head == "primary":
            return fwd["p_primary"], ["answer_context"] * n
        if head != "recovery":
            raise ValueError(f"unknown head {head!r}")
        probs = fwd["rec_probs"]  # (N,4), masked classes ~0
        mask = data["mask"]
        p = np.zeros(n)
        proposed = ["abstain"] * n
        for i in range(n):
            allowed = [j for j in range(3) if mask[i, j]]
            if not allowed:
                continue
            best = max(allowed, key=lambda j: probs[i, j])
            p[i] = float(probs[i, best])
            proposed[i] = ACTIONS[best]
        return p, proposed

    # -- threshold selection (dev; then FROZEN) ----------------------------
    def select_threshold(self, head: str, data: Dict[str, np.ndarray],
                         candidate_matches: Sequence[Dict[str, bool]],
                         c: float = 2.0) -> float:
        p, proposed = self.head_scores(head, data)
        thr = sweep_threshold(p, proposed, candidate_matches, c=c)
        self.thresholds[head] = thr
        return thr

    # -- prediction --------------------------------------------------------
    def predict(self, head: str, data: Dict[str, np.ndarray],
                views: Sequence[InputView]) -> List[Dict[str, Any]]:
        """Gate-compatible per-row decisions {action, answer, p_answer,
        proposed_action}; abstains below the frozen threshold or when the
        proposed candidate answer is missing."""
        p, proposed = self.head_scores(head, data)
        thr = self.thresholds.get(head, DEFAULT_THRESHOLD)
        out: List[Dict[str, Any]] = []
        for view, pi, act in zip(views, p, proposed):
            action, answer = "abstain", None
            if act != "abstain" and pi >= thr:
                candidate = getattr(view, _ANSWER_ATTR[act])
                if isinstance(candidate, str) and normalize(candidate):
                    action, answer = act, candidate
            out.append({"action": action, "answer": answer,
                        "p_answer": float(pi), "proposed_action": act})
        return out

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "hparams": {
                "embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim,
                "dropout": self.dropout, "loss_weights": self.loss_weights,
                "lr": self.lr, "weight_decay": self.weight_decay,
                "batch_size": self.batch_size, "max_epochs": self.max_epochs,
                "patience": self.patience, "min_epochs": self.min_epochs,
                "seed": self.seed, "cf_margin": self.cf_margin,
                "dro": self.dro, "dro_eta": self.dro_eta,
            },
            "thresholds": self.thresholds,
            "temperatures": self.temperatures,
            "dro_q": self.dro_q(),
            "actions": list(ACTIONS),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "RelationalGate":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("actions") != list(ACTIONS):
            raise ValueError(f"{path}: action vocabulary drift")
        gate = cls(device=device, **payload["hparams"])
        gate.model.load_state_dict(payload["state_dict"])
        gate.model.to(gate.device)
        gate.thresholds = dict(payload["thresholds"])
        gate.temperatures = dict(payload.get("temperatures") or {})
        return gate
