"""CGRV: Competence-Gated Relational Verifier (DESIGN_CGRV.md made executable).

Structured-bottleneck model over the three per-source candidates (C/R/P).
Everything the decision layer sees is compressed to interpretable scalars:

  1. NODE SCALARS (source-side observables only, InputView discipline):
     competence gates g_s = sigmoid(MLP over that source's observable block)
       g_R: retrieval-score stats (mean/max/std/margin/count + missing
            indicator) + reference passage count + answer length
       g_P: parametric presence / abstain-text / answer length + optional
            familiarity probe value (nullable -> missing indicator)
       g_C: context passage count/length stats + reader commit + answer length
     commitment c_s = presence indicator (answered & non-abstain).
  2. TYPED RELATIONS: shared 4-way head per unordered pair (CR, CP, RP) over
     {equivalent, compatible, contradictory, unknown} from frozen answer
     embeddings ('question [SEP] answer' only — no evidence); supervised by
     72B pairwise silver labels (outputs/relation_labels; LOSS-ONLY join by
     (dataset, qid, condition); missing/judge-failure pairs masked with -1;
     'unknown' = absent-pair class).
  3. DECISION LAYER: scalar vector z = [g_C,g_R,g_P, c_C,c_R,c_P, 9 relation
     probs, veto=p_contra_CR*g_R, support=p_eq_CR*g_R, defer=(1-g_R)*g_P,
     suspicion=g_R*(1-c_R)]; evidence state sigma = presence pattern (8
     states, hard assignment v1); per-state linear head over z with SIGN
     CONSTRAINTS via softplus (p_answer increasing in support/p_eq_CR,
     decreasing in veto/suspicion/p_contra_CR); p(answer_context) =
     sigmoid(h_sigma(z)). RECOVERY head analogous (per-state linear -> 4
     masked logits over deduped candidates; no sign constraints dictated).
  4. TRAINING: action BCE/CE + 0.3 * relation CE + 0.5 * counterfactual
     pairing loss (reuses relational.counterfactual_loss + group-sampled
     batches); NO Group-DRO. Per-state temperature calibration on dev for
     PRIMARY (global T for recovery), then the dev-frozen c=2 threshold.

EXP7 gate-desaturation switches (each independently togglable, ALL DEFAULT OFF
so every pre-EXP7 number reproduces exactly; none of them adds a parameter, so
the torch RNG stream and the state_dict are unchanged):
  (A) loss_weights['source_conf'] > 0
        supervised per-source confidence head, TIED to the gate: the gate logit
        IS the confidence logit, trained with BCE against the LOSS-ONLY target
        conf_y[i, s] = 1 iff source s's answer alias-matches gold
        (LabelBuilder.candidate_matches). Gold is a TRAINING TARGET, never an
        input. Absent sources are masked out of the term.
  (B) model gate_standardize=True
        z-score each observable with TRAIN-split mean/std frozen at fit time
        (CGRVGate.obs_stats, persisted by save()); constant columns -> std 1.
  (C) loss_weights['gate_l2'] > 0
        mean-square penalty on the gate logits (keeps them near 0 unless the
        evidence pushes), which directly attacks the g -> 1 saturation optimum.

Ablation switches (each a constructor flag, exercised by configs/cgrv.yaml):
  gates=False           g_s = 1 (competence gates off)
  typed_relations=False binary matcher-relation head {support, contradict,
                        unknown}; z gets (p_support, 0, p_contradict)
  state_mixture=False   single decision head for all presence states
  monotone=False        unconstrained decision weights

Torch-free callers must check relational.torch_available() first (same
contract as relational.py; the encoder/cache/threshold machinery is reused
from there).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.src.data.views import InputView
from experiments.src.features import node_features as nf
from experiments.src.features.matching import normalize
from experiments.src.models import relational as rel
from experiments.src.models.relational import (  # noqa: F401 (re-exported)
    DEFAULT_CF_MARGIN,
    EmbeddingCache,
    HashEncoder,
    make_encoder,
    sweep_threshold,
    torch_available,
)

if rel.torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - torch-free import must still succeed
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

ACTIONS = nf.ACTIONS
SOURCES = nf.SOURCES
PAIRS = nf.PAIRS
SEP = nf.SEP
# Public names of the three competence gates, aligned with SOURCES order
# (the column order of the model's `g` tensor).
GATE_KEYS = ("g_C", "g_R", "g_P")

_ANSWER_ATTR = {
    "context": "context_answer",
    "reference": "reference_answer",
    "parametric": "parametric_answer",
}
_MATCH_KEY = {"answer_context": "context", "answer_reference": "reference",
              "answer_parametric": "parametric"}

DEFAULT_THRESHOLD = 0.5
# `source_conf` and `gate_l2` are the EXP7 gate-desaturation switches; both
# default to 0.0 so every result produced before EXP7 reproduces byte-for-byte
# (neither switch adds a parameter, so the torch RNG stream is untouched too).
DEFAULT_LOSS_WEIGHTS = {"primary": 1.0, "recovery": 1.0,
                        "relation": 0.3, "counterfactual": 0.5,
                        "source_conf": 0.0, "gate_l2": 0.0}
# Recommended weights when the corresponding switch is turned ON (configs may
# override; DEFAULT_LOSS_WEIGHTS keeps them at 0 = OFF).
SOURCE_CONF_ON_WEIGHT = 0.3
GATE_L2_ON_WEIGHT = 0.01
_NEG_INF = -1e9
_STD_FLOOR = 1e-6

# ---------------------------------------------------------------------------
# Typed relation vocabulary
# ---------------------------------------------------------------------------

# Model class order for the TYPED head; silver files use the first three
# strings ('null' in a silver file = pair not judged -> masked, see
# relation_targets). 'unknown' is the ABSENT-PAIR class.
REL_CLASSES = ("equivalent", "compatible", "contradictory", "unknown")
REL_UNKNOWN_ID = REL_CLASSES.index("unknown")
PAIR_TAGS = ("CR", "CP", "RP")  # silver-file keys, aligned with nf.PAIRS
N_TYPED = len(REL_CLASSES)
N_BINARY = len(nf.RELATIONS)  # matcher classes (support, contradict, unknown)

# ---------------------------------------------------------------------------
# Observable per-source scalar features (InputView ONLY; no gold anywhere)
# ---------------------------------------------------------------------------

# Feature blocks per gate. Counts/lengths are log1p-compressed so the tiny
# MLPs see bounded inputs. All nullable quantities carry a missing indicator.
OBS_C = ("c_n_passages", "c_total_chars", "c_mean_chars", "c_commit", "c_len")
OBS_R = ("r_mean", "r_max", "r_std", "r_margin", "r_count", "r_missing",
         "r_n_passages", "r_len")
OBS_P = ("p_present", "p_abstain_text", "p_len",
         "p_familiarity", "p_familiarity_missing")
OBS_FEATURES = OBS_C + OBS_R + OBS_P
OBS_SLICES = {
    "context": slice(0, len(OBS_C)),
    "reference": slice(len(OBS_C), len(OBS_C) + len(OBS_R)),
    "parametric": slice(len(OBS_C) + len(OBS_R), len(OBS_FEATURES)),
}

_FORBIDDEN_NAME_SUBSTRINGS = ("gold", "a_false", "condition", "dataset",
                              "covered", "label", "poison")
for _name in OBS_FEATURES:
    for _bad in _FORBIDDEN_NAME_SUBSTRINGS:
        assert _bad not in _name, f"forbidden observable feature name: {_name}"
assert len(set(OBS_FEATURES)) == len(OBS_FEATURES)


def _n_tokens(answer: Optional[str]) -> float:
    if not isinstance(answer, str):
        return 0.0
    return float(len(normalize(answer).split()))


def _is_abstain_text(text: Optional[str]) -> bool:
    if not isinstance(text, str):
        return False
    nt = normalize(text)
    return any(sub in nt for sub in nf.ABSTAIN_SUBSTRINGS)


def observable_features(view: InputView,
                        familiarity: Optional[float] = None) -> List[float]:
    """Flat observable vector in OBS_FEATURES order for one row.

    familiarity is an OPTIONAL cached parametric-familiarity probe value in
    [0, 1] (outputs/familiarity/<ds>.jsonl, keyed by qid at the caller);
    None -> 0.0 + missing indicator 1.0.  Consumes InputView only.
    """
    if not isinstance(view, InputView):
        raise TypeError(
            f"observable_features requires InputView, got "
            f"{type(view).__name__}; raw rows are forbidden (leakage guard)")
    feats: Dict[str, float] = {}
    # --- context block ---
    passages = view.supplied_context or []
    lens = [float(len(str(p.get("text") or ""))) for p in passages]
    feats["c_n_passages"] = math.log1p(float(len(passages)))
    feats["c_total_chars"] = math.log1p(float(sum(lens)))
    feats["c_mean_chars"] = math.log1p(sum(lens) / len(lens) if lens else 0.0)
    feats["c_commit"] = (1.0 if nf.usable_answer(view.context_answer)
                         is not None else 0.0)
    feats["c_len"] = math.log1p(_n_tokens(view.context_answer))
    # --- reference block ---
    scores = view.retrieval_scores
    if scores:
        vals = sorted((float(s) for s in scores), reverse=True)
        mean = sum(vals) / len(vals)
        feats["r_mean"] = mean
        feats["r_max"] = vals[0]
        feats["r_std"] = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        feats["r_margin"] = vals[0] - vals[1] if len(vals) > 1 else 0.0
        feats["r_count"] = math.log1p(float(len(vals)))
        feats["r_missing"] = 0.0
    else:
        for k in ("r_mean", "r_max", "r_std", "r_margin", "r_count"):
            feats[k] = 0.0
        feats["r_missing"] = 1.0
    feats["r_n_passages"] = math.log1p(float(len(view.reference_passages or [])))
    feats["r_len"] = math.log1p(_n_tokens(view.reference_answer))
    # --- parametric block ---
    feats["p_present"] = (1.0 if nf.usable_answer(view.parametric_answer)
                          is not None else 0.0)
    feats["p_abstain_text"] = 1.0 if _is_abstain_text(view.parametric_answer) else 0.0
    feats["p_len"] = math.log1p(_n_tokens(view.parametric_answer))
    if familiarity is None:
        feats["p_familiarity"] = 0.0
        feats["p_familiarity_missing"] = 1.0
    else:
        feats["p_familiarity"] = float(familiarity)
        feats["p_familiarity_missing"] = 0.0
    assert set(feats) == set(OBS_FEATURES)
    return [feats[k] for k in OBS_FEATURES]


def load_familiarity(root: str | Path) -> Dict[str, float]:
    """qid -> familiarity value from outputs/familiarity/*.jsonl (all files).

    Missing directory / files -> empty dict (features fall back to the
    missing indicator; the model must not require this input)."""
    root = Path(root)
    out: Dict[str, float] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("qid") is not None and rec.get("familiarity") is not None:
                    out[str(rec["qid"])] = float(rec["familiarity"])
    return out


# ---------------------------------------------------------------------------
# Answer-only node texts ('question [SEP] answer'; evidence NOT embedded)
# ---------------------------------------------------------------------------

def answer_texts(view: InputView) -> List[Optional[str]]:
    """Per-node embed text in SOURCES order; None for absent candidates."""
    if not isinstance(view, InputView):
        raise TypeError("answer_texts requires InputView (leakage guard)")
    out: List[Optional[str]] = []
    for source in SOURCES:
        answer = nf.usable_answer(getattr(view, _ANSWER_ATTR[source]))
        out.append(None if answer is None else view.question + SEP + answer)
    return out


# ---------------------------------------------------------------------------
# Whole-channel source ablation (which evidence source is worth having at all)
# ---------------------------------------------------------------------------

# Every field that belongs to one evidence channel. Nulling these on the Row
# rather than patching the tensors is what keeps the ablation honest: the
# absent flag, the zeroed node embedding, the observable block's own
# missing-value convention, the silver relation targets, the per-source
# confidence targets and the recovery action mask are all derived from the Row,
# so a removed channel cannot leak back in through any of them.
_SOURCE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "reference": ("reference_answer", "reference_passages", "retrieval_scores",
                  "reference_generation_scores"),
    "parametric": ("parametric_answer", "parametric_generation_scores"),
}


def ablate_sources(rows: Sequence[Any], sources_off: Sequence[str]
                   ) -> List[Any]:
    """Rows with each named evidence channel removed entirely.

    The context channel cannot be removed: the decision is whether to return
    the context answer, so a run without it has no action to take.
    """
    off = tuple(sources_off)
    if not off:
        return list(rows)
    bad = [s for s in off if s not in _SOURCE_FIELDS]
    if bad:
        raise ValueError(
            f"sources_off names must be in {tuple(_SOURCE_FIELDS)}: {bad}")
    import dataclasses

    fields = [f for s in off for f in _SOURCE_FIELDS[s]]
    return [dataclasses.replace(r, **{f: None for f in fields}) for r in rows]


def build_cgrv_inputs(views: Sequence[InputView],
                      familiarity: Optional[Dict[str, float]] = None
                      ) -> Dict[str, list]:
    """Batch inputs: texts (answer-only), absent flags, observable scalars,
    matcher silver relations (binary-ablation supervision), recovery action
    mask + canonical map (dedupe reused from node_features)."""
    fam = familiarity or {}
    out: Dict[str, list] = {"texts": [], "absent": [], "obs": [],
                            "matcher_relation_y": [], "action_mask": [],
                            "canonical": []}
    for view in views:
        texts = answer_texts(view)
        struct = nf.recovery_structure(view)
        out["texts"].append(texts)
        out["absent"].append([1.0 if t is None else 0.0 for t in texts])
        out["obs"].append(observable_features(view, fam.get(view.qid)))
        out["matcher_relation_y"].append(nf.relation_silver(view))
        out["action_mask"].append(struct["action_mask"])
        out["canonical"].append(struct["canonical"])
    return out


# ---------------------------------------------------------------------------
# Typed silver relation labels (LOSS-ONLY; joined by (dataset, qid, condition))
# ---------------------------------------------------------------------------

def load_relation_labels(paths: Sequence[str | Path]
                         ) -> Dict[Tuple[str, str, str], Dict[str, Optional[str]]]:
    """(dataset, qid, condition) -> {'CR'/'CP'/'RP': label-or-None} from the
    outputs/relation_labels jsonl files produced by gen_relation_labels.py."""
    out: Dict[Tuple[str, str, str], Dict[str, Optional[str]]] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[(rec["dataset"], rec["qid"], rec["condition"])] = \
                    rec["relations"]
    return out


def relation_targets(rows: Sequence[Any], absent: Sequence[Sequence[float]],
                     labels: Optional[Dict[Tuple[str, str, str],
                                           Dict[str, Optional[str]]]]
                     ) -> np.ndarray:
    """(N, 3) int64 typed-relation targets per PAIR_TAGS; LOSS-ONLY (reads
    Row.dataset/condition, which are forbidden as features).

      either answer absent          -> REL_UNKNOWN_ID (absent-pair class)
      silver in the 3 typed labels  -> its index
      silver null/'unknown'/missing -> -1 (masked out of the CE)
    """
    n = len(rows)
    y = np.full((n, len(PAIR_TAGS)), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        rec = None
        if labels is not None:
            rec = labels.get((row.dataset, row.qid, row.condition))
        for k, (a, b, _tag) in enumerate(PAIRS):
            if absent[i][a] or absent[i][b]:
                y[i, k] = REL_UNKNOWN_ID
                continue
            lab = None if rec is None else rec.get(PAIR_TAGS[k])
            if lab in REL_CLASSES[:3]:
                y[i, k] = REL_CLASSES.index(lab)
            # else: judged-unknown / null / no record -> stays masked (-1)
    return y


# ---------------------------------------------------------------------------
# Data bundle
# ---------------------------------------------------------------------------

def pack_data(emb: np.ndarray, absent: Sequence[Sequence[float]],
              obs: Sequence[Sequence[float]],
              action_mask: Sequence[Sequence[bool]],
              primary_y: Optional[Sequence[int]] = None,
              recovery_y: Optional[Sequence[str]] = None,
              relation_y: Optional[np.ndarray] = None,
              matcher_relation_y: Optional[Sequence[Sequence[int]]] = None,
              conf_y: Optional[Sequence[Sequence[float]]] = None,
              cond_id: Optional[Sequence[int]] = None,
              cf_group: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Numpy bundle for CGRVGate (labels optional; cond_id/cf_group, conf_y and
    both relation label arrays are LOSS-ONLY — the forward pass consumes emb,
    absent and obs exclusively)."""
    data: Dict[str, np.ndarray] = {
        "emb": np.asarray(emb, dtype=np.float32),
        "absent": np.asarray(absent, dtype=np.float32),
        "obs": np.asarray(obs, dtype=np.float32),
        "mask": np.asarray(action_mask, dtype=bool),
    }
    if primary_y is not None:
        data["primary_y"] = np.asarray(primary_y, dtype=np.float32)
    if recovery_y is not None:
        data["recovery_y"] = np.asarray(
            [ACTIONS.index(a) for a in recovery_y], dtype=np.int64)
    if relation_y is not None:
        data["relation_y"] = np.asarray(relation_y, dtype=np.int64)
    if matcher_relation_y is not None:
        data["matcher_relation_y"] = np.asarray(matcher_relation_y,
                                                dtype=np.int64)
    if conf_y is not None:
        data["conf_y"] = np.asarray(conf_y, dtype=np.float32)
    if cond_id is not None:
        data["cond_id"] = np.asarray(cond_id, dtype=np.int64)
    if cf_group is not None:
        data["cf_group"] = np.asarray(cf_group, dtype=np.int64)
    return data


subset_data = rel.subset_data  # same semantics


# ---------------------------------------------------------------------------
# Decision-layer scalar vocabulary
# ---------------------------------------------------------------------------

Z_FEATURES = (
    "g_c", "g_r", "g_p", "c_c", "c_r", "c_p",
    "eq_cr", "comp_cr", "contra_cr",
    "eq_cp", "comp_cp", "contra_cp",
    "eq_rp", "comp_rp", "contra_rp",
    "veto", "support", "defer", "suspicion",
)
Z_DIM = len(Z_FEATURES)
Z_IDX = {name: i for i, name in enumerate(Z_FEATURES)}
# Sign constraints for the PRIMARY head (DESIGN_CGRV: monotone in
# equivalent-mass/support, anti-monotone in veto/suspicion; p_contra_CR is
# also constrained non-positive so that increasing p_contra_CR*g_R can NEVER
# raise p_answer through the direct term either).
Z_POSITIVE = ("support", "eq_cr")
Z_NEGATIVE = ("veto", "suspicion", "contra_cr")
N_STATES = 8  # presence pattern (c_C, c_R, c_P) in {0,1}^3
MIN_STATE_ROWS = 30  # per-state threshold needs this many dev rows


# ---------------------------------------------------------------------------
# Torch model
# ---------------------------------------------------------------------------

if rel.torch_available():

    def _sign_vector() -> "torch.Tensor":
        sign = torch.zeros(Z_DIM)
        for name in Z_POSITIVE:
            sign[Z_IDX[name]] = 1.0
        for name in Z_NEGATIVE:
            sign[Z_IDX[name]] = -1.0
        return sign

    class StateHead(nn.Module):
        """Per-state linear head over the scalar vector z.

        monotone=True (out_dim must be 1): constrained weights are
        sign * softplus(raw) so the primary logit is non-decreasing in every
        Z_POSITIVE feature and non-increasing in every Z_NEGATIVE feature,
        for every state. Free features keep raw weights."""

        def __init__(self, n_states: int, out_dim: int = 1,
                     monotone: bool = False) -> None:
            super().__init__()
            if monotone and out_dim != 1:
                raise ValueError("monotone constraints defined for out_dim=1")
            self.n_states = int(n_states)
            self.out_dim = int(out_dim)
            self.monotone = bool(monotone)
            self.weight = nn.Parameter(
                torch.randn(n_states, out_dim, Z_DIM) * 0.05)
            self.bias = nn.Parameter(torch.zeros(n_states, out_dim))
            self.register_buffer("sign", _sign_vector().view(1, 1, Z_DIM))

        def effective_weight(self) -> "torch.Tensor":
            if not self.monotone:
                return self.weight
            constrained = self.sign * F.softplus(self.weight)
            return torch.where(self.sign == 0, self.weight, constrained)

        def forward(self, z: "torch.Tensor",
                    state: "torch.Tensor") -> "torch.Tensor":
            """z (B, Z_DIM), state (B,) long -> (B, out_dim) logits."""
            w = self.effective_weight()[state]        # (B, out, Z)
            b = self.bias[state]                      # (B, out)
            return torch.einsum("boz,bz->bo", w, z) + b

    class CGRV(nn.Module):
        """Competence gates + typed relation head + per-state decision layer."""

        def __init__(self, embed_dim: int, proj_dim: int = 64,
                     rel_hidden: int = 64, gate_hidden: int = 16,
                     dropout: float = 0.1, gates: bool = True,
                     typed_relations: bool = True, state_mixture: bool = True,
                     monotone: bool = True,
                     gate_standardize: bool = False,
                     gates_off: Sequence[str] = ()) -> None:
            super().__init__()
            self.embed_dim = int(embed_dim)
            self.gates = bool(gates)
            # Per-source gate ablation: each named source's gate is pinned to
            # 1.0, the same neutral value the whole-gate ablation uses, while
            # the other gates keep learning.  Default () leaves every path
            # untouched, so the frozen grid reproduces bit for bit.
            bad = [s for s in gates_off if s not in SOURCES]
            if bad:
                raise ValueError(f"gates_off names must be in {SOURCES}: {bad}")
            self.gates_off = tuple(gates_off)
            self._gate_off_idx = [SOURCES.index(s) for s in self.gates_off]
            # EXP7 switch (B): z-score the gate observables with TRAIN-split
            # statistics.  Kept as plain tensors (NOT buffers/parameters) so
            # the state_dict and the torch RNG stream are unchanged; they are
            # persisted separately by CGRVGate.save under "obs_stats".
            self.gate_standardize = bool(gate_standardize)
            self.obs_mean: Optional["torch.Tensor"] = None
            self.obs_std: Optional["torch.Tensor"] = None
            self.typed_relations = bool(typed_relations)
            self.state_mixture = bool(state_mixture)
            self.monotone = bool(monotone)
            self.n_rel_classes = N_TYPED if self.typed_relations else N_BINARY
            self.n_states = N_STATES if self.state_mixture else 1
            # competence gate MLPs (per-source observable blocks only)
            def gate_mlp(n_in: int) -> nn.Sequential:
                return nn.Sequential(nn.Linear(n_in, gate_hidden), nn.GELU(),
                                     nn.Linear(gate_hidden, 1))
            self.gate_c = gate_mlp(len(OBS_C))
            self.gate_r = gate_mlp(len(OBS_R))
            self.gate_p = gate_mlp(len(OBS_P))
            # typed relation head: shared MLP over projected embedding pairs
            self.proj = nn.Linear(embed_dim, proj_dim)
            self.rel_mlp = nn.Sequential(
                nn.Linear(4 * proj_dim, rel_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(rel_hidden, self.n_rel_classes))
            # decision layer
            self.primary_head = StateHead(self.n_states, out_dim=1,
                                          monotone=self.monotone)
            self.recovery_head = StateHead(self.n_states, out_dim=len(ACTIONS),
                                           monotone=False)

        # -- pieces ---------------------------------------------------------
        def set_obs_stats(self, mean: Sequence[float],
                          std: Sequence[float]) -> None:
            """Freeze the TRAIN-split observable mean/std used by switch (B).

            Columns whose train std is below _STD_FLOOR (constant observables,
            e.g. an always-0 missing indicator) get std=1 so standardization
            centres them to exactly 0 instead of exploding."""
            m = torch.as_tensor(list(mean), dtype=torch.float32)
            s = torch.as_tensor(list(std), dtype=torch.float32)
            if m.shape != (len(OBS_FEATURES),) or s.shape != (len(OBS_FEATURES),):
                raise ValueError("obs stats must have one entry per observable")
            s = torch.where(s < _STD_FLOOR, torch.ones_like(s), s)
            dev = next(self.parameters()).device
            self.obs_mean, self.obs_std = m.to(dev), s.to(dev)

        def standardize(self, obs: "torch.Tensor") -> "torch.Tensor":
            """Identity unless switch (B) is on AND train stats were frozen."""
            if not self.gate_standardize or self.obs_mean is None:
                return obs
            return (obs - self.obs_mean.to(obs.device)) / self.obs_std.to(obs.device)

        def gate_logits(self, obs: "torch.Tensor") -> "torch.Tensor":
            """(B, n_obs) -> (B, 3) PRE-sigmoid competence-gate logits.

            These are also the per-source confidence logits: switch (A) ties
            the auxiliary source-confidence head to the gate by supervising
            THIS tensor, so no extra parameters exist and g_s is literally a
            calibrated P(source s is alias-correct) estimate."""
            x = self.standardize(obs)
            return torch.cat([
                self.gate_c(x[:, OBS_SLICES["context"]]),
                self.gate_r(x[:, OBS_SLICES["reference"]]),
                self.gate_p(x[:, OBS_SLICES["parametric"]]),
            ], dim=-1)

        def apply_gate_mask(self, g: "torch.Tensor") -> "torch.Tensor":
            """Pin the gates named by gates_off to one. Identity by default.

            This is the ONLY place the per-source ablation is applied, and both
            forward() and gate_values() route through it, so the switch cannot
            silently miss the training path."""
            if not self._gate_off_idx:
                return g
            keep = torch.ones_like(g)
            keep[:, self._gate_off_idx] = 0.0
            return g * keep + (1.0 - keep)

        def gate_values(self, obs: "torch.Tensor") -> "torch.Tensor":
            """(B, n_obs) -> (B, 3) competence gates in (0, 1); all-ones when
            the competence-gate ablation is active, and pinned to one in the
            columns named by gates_off."""
            if not self.gates:
                return obs.new_ones(obs.shape[0], len(SOURCES))
            return self.apply_gate_mask(torch.sigmoid(self.gate_logits(obs)))

        def relation_logits(self, emb: "torch.Tensor") -> "torch.Tensor":
            """(B, 3, D) -> (B, 3 pairs, n_rel_classes) in PAIRS order."""
            h = self.proj(emb)
            outs = []
            for i, j, _tag in PAIRS:
                hi, hj = h[:, i], h[:, j]
                pair = torch.cat([hi, hj, hi * hj, (hi - hj).abs()], dim=-1)
                outs.append(self.rel_mlp(pair))
            return torch.stack(outs, dim=1)

        @staticmethod
        def rel_triples(rel_probs: "torch.Tensor",
                        typed: bool) -> "torch.Tensor":
            """Map relation probabilities to (eq, comp, contra) per pair.
            typed: first three classes directly; binary: (support, 0,
            contradict) — the compatible slot is structurally zero."""
            if typed:
                return rel_probs[..., :3]
            zeros = torch.zeros_like(rel_probs[..., :1])
            return torch.cat([rel_probs[..., 0:1], zeros,
                              rel_probs[..., 1:2]], dim=-1)

        @staticmethod
        def assemble_z(g: "torch.Tensor", commit: "torch.Tensor",
                       triples: "torch.Tensor") -> "torch.Tensor":
            """Scalar bottleneck: g (B,3), commit (B,3), triples (B,3,3)
            [(eq, comp, contra) per pair in PAIRS order] -> z (B, Z_DIM)."""
            g_r, g_p = g[:, 1], g[:, 2]
            eq_cr, contra_cr = triples[:, 0, 0], triples[:, 0, 2]
            inter = torch.stack([
                contra_cr * g_r,          # veto
                eq_cr * g_r,              # support
                (1.0 - g_r) * g_p,        # defer
                g_r * (1.0 - commit[:, 1]),  # suspicion
            ], dim=-1)
            return torch.cat([g, commit, triples.reshape(g.shape[0], 9),
                              inter], dim=-1)

        def state_index(self, commit: "torch.Tensor") -> "torch.Tensor":
            """Presence pattern -> sigma in [0, 8); constant 0 when the
            state-mixture ablation is active (single shared head)."""
            if not self.state_mixture:
                return torch.zeros(commit.shape[0], dtype=torch.long,
                                   device=commit.device)
            bits = (commit > 0.5).long()
            return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]

        def forward(self, emb: "torch.Tensor", absent: "torch.Tensor",
                    obs: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            """emb (B,3,D), absent (B,3) {0,1}, obs (B, n_obs)."""
            commit = 1.0 - absent
            gate_logit = self.gate_logits(obs) if self.gates else None
            # The mask applies to the gates that enter z, not to gate_logit,
            # which stays the per-source confidence target of switches A and C.
            g = (self.apply_gate_mask(torch.sigmoid(gate_logit))
                 if gate_logit is not None
                 else obs.new_ones(obs.shape[0], len(SOURCES)))
            rel_logits = self.relation_logits(emb)
            rel_probs = rel_logits.softmax(dim=-1)
            triples = self.rel_triples(rel_probs, self.typed_relations)
            z = self.assemble_z(g, commit, triples)
            state = self.state_index(commit)
            primary_logit = self.primary_head(z, state).squeeze(-1)
            rec_logits = self.recovery_head(z, state)
            out = {"primary_logit": primary_logit, "rec_logits": rec_logits,
                   "rel_logits": rel_logits, "g": g, "z": z, "state": state}
            if gate_logit is not None:
                # = per-source confidence logits (switch A) / L2 target (C).
                out["gate_logit"] = gate_logit
            return out


# ---------------------------------------------------------------------------
# Trainer / predictor wrapper (RelationalGate-compatible surface)
# ---------------------------------------------------------------------------

class CGRVGate:
    """Fit (AdamW + early stop on dev action loss), per-state dev temperature
    calibration, dev-frozen c=2 thresholds, gate-compatible predict,
    save/load. Counterfactual pairing reuses relational.counterfactual_loss
    with group-sampled batches. NO Group-DRO (design decision)."""

    MIN_STATE_ROWS = 8  # dev rows needed to fit a per-state temperature

    def __init__(self, embed_dim: int, proj_dim: int = 64, rel_hidden: int = 64,
                 gate_hidden: int = 16, dropout: float = 0.1,
                 gates: bool = True, typed_relations: bool = True,
                 state_mixture: bool = True, monotone: bool = True,
                 gate_standardize: bool = False,
                 gates_off: Sequence[str] = (),
                 loss_weights: Optional[Dict[str, float]] = None,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, max_epochs: int = 100,
                 patience: int = 10, min_epochs: int = 1, seed: int = 13,
                 device: str = "auto",
                 cf_margin: float = DEFAULT_CF_MARGIN,
                 per_state_threshold: bool = False) -> None:
        if not rel.torch_available():
            raise RuntimeError("method kind 'cgrv' needs torch")
        self.hparams = {
            "embed_dim": int(embed_dim), "proj_dim": int(proj_dim),
            "rel_hidden": int(rel_hidden), "gate_hidden": int(gate_hidden),
            "dropout": float(dropout), "gates": bool(gates),
            "typed_relations": bool(typed_relations),
            "state_mixture": bool(state_mixture), "monotone": bool(monotone),
            "gate_standardize": bool(gate_standardize),
            "gates_off": tuple(gates_off),
            "loss_weights": {**DEFAULT_LOSS_WEIGHTS, **(loss_weights or {})},
            "lr": float(lr), "weight_decay": float(weight_decay),
            "batch_size": int(batch_size), "max_epochs": int(max_epochs),
            "patience": int(patience), "min_epochs": int(min_epochs),
            "seed": int(seed), "cf_margin": float(cf_margin),
            "per_state_threshold": bool(per_state_threshold),
        }
        # Per-state dev-frozen thresholds (dead-state fix): states with fewer
        # than MIN_STATE_ROWS dev rows fall back to the global threshold.
        self.per_state_threshold = bool(per_state_threshold)
        self.state_thresholds: Dict[int, float] = {}
        self.loss_weights = self.hparams["loss_weights"]
        self.batch_size = self.hparams["batch_size"]
        self.seed = self.hparams["seed"]
        self.cf_margin = self.hparams["cf_margin"]
        self.device = rel.resolve_device(device)
        torch.manual_seed(self.seed)
        self.model = CGRV(
            embed_dim=self.hparams["embed_dim"], proj_dim=self.hparams["proj_dim"],
            rel_hidden=self.hparams["rel_hidden"],
            gate_hidden=self.hparams["gate_hidden"],
            dropout=self.hparams["dropout"], gates=self.hparams["gates"],
            typed_relations=self.hparams["typed_relations"],
            state_mixture=self.hparams["state_mixture"],
            monotone=self.hparams["monotone"],
            gate_standardize=self.hparams["gate_standardize"],
            gates_off=self.hparams["gates_off"]).to(self.device)
        self.thresholds: Dict[str, float] = {}
        # per-state primary temperatures (len n_states) + global recovery T
        self.state_temperatures: List[float] = [1.0] * self.model.n_states
        self.recovery_temperature: float = 1.0
        self.history: List[Dict[str, float]] = []
        # TRAIN-split observable statistics for switch (B); None until fit().
        self.obs_stats: Optional[Dict[str, Any]] = None

    # -- loss --------------------------------------------------------------
    def _relation_key(self) -> str:
        return ("relation_y" if self.model.typed_relations
                else "matcher_relation_y")

    def _batch_losses(self, out: Dict[str, "torch.Tensor"],
                      batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        w = self.loss_weights
        zero = out["primary_logit"].new_zeros(())
        losses = {"primary": zero, "recovery": zero, "relation": zero,
                  "counterfactual": zero, "source_conf": zero,
                  "gate_l2": zero}
        if w.get("primary", 0) > 0 and "primary_y" in batch:
            losses["primary"] = F.binary_cross_entropy_with_logits(
                out["primary_logit"], batch["primary_y"])
        if w.get("recovery", 0) > 0 and "recovery_y" in batch:
            masked = out["rec_logits"].masked_fill(~batch["mask"], _NEG_INF)
            losses["recovery"] = F.cross_entropy(masked, batch["recovery_y"])
        rel_key = self._relation_key()
        if w.get("relation", 0) > 0 and rel_key in batch:
            y = batch[rel_key].reshape(-1)
            logits = out["rel_logits"].reshape(-1, self.model.n_rel_classes)
            losses["relation"] = F.cross_entropy(logits, y, ignore_index=-1)
            if bool(torch.all(y == -1)):
                losses["relation"] = zero  # all-masked batch -> defined 0
        if (w.get("counterfactual", 0) > 0 and "cond_id" in batch
                and "cf_group" in batch):
            losses["counterfactual"] = rel.counterfactual_loss(
                torch.sigmoid(out["primary_logit"]), batch["cond_id"],
                batch["cf_group"], margin=self.cf_margin)
        # --- EXP7 switch (A): supervised per-source confidence -------------
        # The gate logit IS the confidence logit (tied head, no new weights).
        # Target conf_y[i, s] = 1 iff source s's answer alias-matches gold
        # (LabelBuilder.candidate_matches); gold is a TARGET, never an input.
        # Absent sources are masked: competence is a property of the source,
        # not of whether it happened to emit an answer (DESIGN_CGRV routes
        # SILENCE by g, so silence must not be trained to g=0).
        if (w.get("source_conf", 0) > 0 and "conf_y" in batch
                and "gate_logit" in out):
            present = 1.0 - batch["absent"]
            per = F.binary_cross_entropy_with_logits(
                out["gate_logit"], batch["conf_y"], reduction="none")
            losses["source_conf"] = ((per * present).sum()
                                     / present.sum().clamp(min=1.0))
        # --- EXP7 switch (C): L2 on the gate logits ------------------------
        if w.get("gate_l2", 0) > 0 and "gate_logit" in out:
            losses["gate_l2"] = out["gate_logit"].pow(2).mean()
        losses["action"] = (w.get("primary", 0) * losses["primary"]
                            + w.get("recovery", 0) * losses["recovery"])
        losses["total"] = (losses["action"]
                           + w.get("relation", 0) * losses["relation"]
                           + w.get("counterfactual", 0) * losses["counterfactual"]
                           + w.get("source_conf", 0) * losses["source_conf"]
                           + w.get("gate_l2", 0) * losses["gate_l2"])
        return losses

    def _to_device(self, data: Dict[str, np.ndarray],
                   idx=None) -> Dict[str, "torch.Tensor"]:
        sub = data if idx is None else subset_data(data, idx)
        return {k: torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
                for k, v in sub.items()}

    def _forward_batch(self, batch: Dict[str, "torch.Tensor"]
                       ) -> Dict[str, "torch.Tensor"]:
        return self.model(batch["emb"], batch["absent"], batch["obs"])

    def _eval_loss(self, data: Dict[str, np.ndarray]) -> Tuple[float, float]:
        self.model.eval()
        n = len(data["emb"])
        tot_action = tot_total = 0.0
        with torch.no_grad():
            for lo in range(0, n, self.batch_size):
                idx = np.arange(lo, min(lo + self.batch_size, n))
                batch = self._to_device(data, idx)
                losses = self._batch_losses(self._forward_batch(batch), batch)
                tot_action += float(losses["action"]) * len(idx)
                tot_total += float(losses["total"]) * len(idx)
        return tot_action / max(n, 1), tot_total / max(n, 1)

    # -- training ----------------------------------------------------------
    def fit(self, train: Dict[str, np.ndarray],
            dev: Optional[Dict[str, np.ndarray]] = None,
            verbose: bool = False) -> "CGRVGate":
        """AdamW; early stop on dev action loss; restores the best epoch.
        With an active counterfactual weight + cond_id/cf_group arrays,
        batches keep whole cf_group blocks together (relational semantics)."""
        hp = self.hparams
        if hp["gate_standardize"]:
            # TRAIN-split statistics only, frozen before the first update and
            # reused unchanged for dev/test/external scoring.
            obs = np.asarray(train["obs"], dtype=np.float64)
            self.obs_stats = {"mean": obs.mean(axis=0).tolist(),
                              "std": obs.std(axis=0).tolist(),
                              "n": int(obs.shape[0])}
            self.model.set_obs_stats(self.obs_stats["mean"],
                                     self.obs_stats["std"])
        opt = torch.optim.AdamW(self.model.parameters(), lr=hp["lr"],
                                weight_decay=hp["weight_decay"])
        rng = np.random.default_rng(self.seed)
        use_cf = (self.loss_weights.get("counterfactual", 0) > 0
                  and "cond_id" in train and "cf_group" in train)
        best_loss, best_state, best_epoch, bad = np.inf, None, -1, 0
        monitor = dev if dev is not None and len(dev["emb"]) else train
        batcher = rel.RelationalGate._train_batches
        for epoch in range(hp["max_epochs"]):
            self.model.train()
            for idx in batcher(self, train, rng, group_sample=use_cf):
                batch = self._to_device(train, idx)
                losses = self._batch_losses(self._forward_batch(batch), batch)
                opt.zero_grad()
                losses["total"].backward()
                opt.step()
            action_loss, total_loss = self._eval_loss(monitor)
            self.history.append({"epoch": epoch,
                                 "monitor_action_loss": action_loss,
                                 "monitor_total_loss": total_loss})
            if verbose:
                print(f"[cgrv] epoch={epoch} action={action_loss:.4f} "
                      f"total={total_loss:.4f}")
            if action_loss < best_loss - 1e-6:
                best_loss, best_epoch, bad = action_loss, epoch, 0
                best_state = copy.deepcopy(
                    {k: v.detach().cpu()
                     for k, v in self.model.state_dict().items()})
            else:
                bad += 1
                if epoch + 1 >= hp["min_epochs"] and bad >= hp["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_epoch = best_epoch
        self.model.to(self.device)
        return self

    # -- raw logits --------------------------------------------------------
    def _logits_all(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.model.eval()
        n = len(data["emb"])
        chunks: Dict[str, List[np.ndarray]] = {}
        with torch.no_grad():
            for lo in range(0, n, self.batch_size):
                idx = np.arange(lo, min(lo + self.batch_size, n))
                batch = self._to_device(data, idx)
                out = self._forward_batch(batch)
                for k in ("primary_logit", "rec_logits", "state", "g",
                          "rel_logits"):
                    chunks.setdefault(k, []).append(out[k].cpu().numpy())
        return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}

    # -- calibration (dev-only, BEFORE threshold selection) ----------------
    def calibrate(self, dev: Dict[str, np.ndarray],
                  heads: Sequence[str] = ("primary", "recovery"),
                  max_iter: int = 100) -> Dict[str, Any]:
        """Per-state temperature scaling of the PRIMARY head on dev (states
        with < MIN_STATE_ROWS labeled dev rows keep T=1) + one global
        recovery temperature. Stored temperatures are applied inside
        head_scores, so the frozen threshold sees calibrated probabilities."""
        logits = self._logits_all(dev)
        report: Dict[str, Any] = {}
        if "primary" in heads and "primary_y" in dev:
            states = logits["state"]
            per_state: Dict[str, Dict[str, float]] = {}
            for s in range(self.model.n_states):
                rows = np.nonzero(states == s)[0]
                if len(rows) < self.MIN_STATE_ROWS:
                    self.state_temperatures[s] = 1.0
                    continue
                z = torch.from_numpy(
                    logits["primary_logit"][rows]).to(self.device)
                y = torch.from_numpy(np.asarray(
                    dev["primary_y"][rows], dtype=np.float32)).to(self.device)
                t, nll_b, nll_a = rel.fit_temperature(
                    lambda tt: F.binary_cross_entropy_with_logits(z / tt, y),
                    self.device, max_iter=max_iter)
                self.state_temperatures[s] = t
                per_state[str(s)] = {"temperature": t, "n": int(len(rows)),
                                     "nll_before": nll_b, "nll_after": nll_a}
            report["primary"] = {"per_state": per_state,
                                 "temperatures": list(self.state_temperatures)}
        if "recovery" in heads and "recovery_y" in dev:
            z = torch.from_numpy(logits["rec_logits"]).to(self.device)
            mask = torch.from_numpy(
                np.asarray(dev["mask"], dtype=bool)).to(self.device)
            y = torch.from_numpy(np.asarray(
                dev["recovery_y"], dtype=np.int64)).to(self.device)
            t, nll_b, nll_a = rel.fit_temperature(
                lambda tt: F.cross_entropy(
                    (z / tt).masked_fill(~mask, _NEG_INF), y),
                self.device, max_iter=max_iter)
            self.recovery_temperature = t
            report["recovery"] = {"temperature": t, "nll_before": nll_b,
                                  "nll_after": nll_a}
        return report

    # -- scoring -----------------------------------------------------------
    def head_scores(self, head: str,
                    data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """(p_answer, proposed action) per row with stored temperatures
        applied (per-state for primary, global for recovery)."""
        logits = self._logits_all(data)
        n = len(data["emb"])
        if head == "primary":
            temps = np.asarray(self.state_temperatures, dtype=np.float64)
            t_row = temps[logits["state"]]
            p = 1.0 / (1.0 + np.exp(-logits["primary_logit"] / t_row))
            return p, ["answer_context"] * n
        if head != "recovery":
            raise ValueError(f"unknown head {head!r}")
        z = torch.from_numpy(logits["rec_logits"]) / self.recovery_temperature
        mask_t = torch.from_numpy(np.asarray(data["mask"], dtype=bool))
        probs = z.masked_fill(~mask_t, _NEG_INF).softmax(dim=-1).numpy()
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
        if self.per_state_threshold and head == "primary":
            states = self._logits_all(data)["state"]
            self.state_thresholds = {}
            for s in range(int(states.max()) + 1 if len(states) else 0):
                idx = [i for i, st in enumerate(states) if st == s]
                if len(idx) < MIN_STATE_ROWS:
                    continue          # too few dev rows -> keep global
                self.state_thresholds[s] = sweep_threshold(
                    p[idx], [proposed[i] for i in idx],
                    [candidate_matches[i] for i in idx], c=c)
        return thr

    # -- prediction --------------------------------------------------------
    def predict(self, head: str, data: Dict[str, np.ndarray],
                views: Sequence[InputView]) -> List[Dict[str, Any]]:
        """Gate-compatible decisions (same contract as RelationalGate)."""
        p, proposed = self.head_scores(head, data)
        thr = self.thresholds.get(head, DEFAULT_THRESHOLD)
        row_thr = None
        if self.per_state_threshold and head == "primary" and self.state_thresholds:
            states = self._logits_all(data)["state"]
            row_thr = [self.state_thresholds.get(int(s), thr) for s in states]
        out: List[Dict[str, Any]] = []
        answer_attr = {"answer_context": "context_answer",
                       "answer_reference": "reference_answer",
                       "answer_parametric": "parametric_answer"}
        for i_row, (view, pi, act) in enumerate(zip(views, p, proposed)):
            thr = row_thr[i_row] if row_thr is not None else self.thresholds.get(
                head, DEFAULT_THRESHOLD)
            action, answer = "abstain", None
            if act != "abstain" and pi >= thr:
                candidate = getattr(view, answer_attr[act])
                if isinstance(candidate, str) and normalize(candidate):
                    action, answer = act, candidate
            out.append({"action": action, "answer": answer,
                        "p_answer": float(pi), "proposed_action": act})
        return out

    # -- interpretability --------------------------------------------------
    def explain(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Per-row scalar bottleneck for inspection: gates, states, z."""
        logits = self._logits_all(data)
        return {"g": logits["g"], "state": logits["state"]}

    def gate_values(self, data: Dict[str, np.ndarray]
                    ) -> Dict[str, np.ndarray]:
        """Per-row competence gates of a fitted model, READ-ONLY.

        Returns {'g_C', 'g_R', 'g_P'} -> float64 arrays of shape (N,) in
        GATE_KEYS order (= SOURCES order), each in (0, 1) — or exactly 1.0
        for every row when the gates=False ablation is active. Pure view on
        explain()['g']: no state is written, nothing about predict/fit
        changes. Intended for validating the gates against EVAL-ONLY ground
        truth (corpus coverage for g_R, familiarity for g_P); those targets
        must never be fed back into the model.
        """
        g = np.asarray(self.explain(data)["g"], dtype=np.float64)
        if g.ndim != 2 or g.shape[1] != len(GATE_KEYS):
            raise ValueError(f"unexpected gate array shape {g.shape}")
        return {key: g[:, i].copy() for i, key in enumerate(GATE_KEYS)}

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": {k: v.cpu()
                           for k, v in self.model.state_dict().items()},
            "hparams": dict(self.hparams),
            "thresholds": self.thresholds,
            "state_temperatures": list(self.state_temperatures),
            "recovery_temperature": self.recovery_temperature,
            "actions": list(ACTIONS),
            "z_features": list(Z_FEATURES),
            "obs_features": list(OBS_FEATURES),
            "obs_stats": self.obs_stats,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "CGRVGate":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("actions") != list(ACTIONS):
            raise ValueError(f"{path}: action vocabulary drift")
        if payload.get("z_features") != list(Z_FEATURES):
            raise ValueError(f"{path}: z-feature vocabulary drift")
        if payload.get("obs_features", list(OBS_FEATURES)) != list(OBS_FEATURES):
            raise ValueError(f"{path}: observable vocabulary drift")
        gate = cls(device=device, **payload["hparams"])
        gate.model.load_state_dict(payload["state_dict"])
        gate.model.to(gate.device)
        stats = payload.get("obs_stats")
        if stats is not None:
            gate.obs_stats = dict(stats)
            gate.model.set_obs_stats(stats["mean"], stats["std"])
        gate.thresholds = dict(payload["thresholds"])
        gate.state_temperatures = list(payload["state_temperatures"])
        gate.recovery_temperature = float(payload["recovery_temperature"])
        return gate


# ---------------------------------------------------------------------------
# Bundle builder (script-side; mirrors train._relational_bundle)
# ---------------------------------------------------------------------------

def cgrv_bundle(rows: Sequence[Any], encoder, cache,
                relation_labels: Optional[Dict[Tuple[str, str, str],
                                               Dict[str, Optional[str]]]] = None,
                familiarity: Optional[Dict[str, float]] = None,
                with_labels: bool = True):
    """(views, data, matches) for CGRV: answer-only cached embeddings +
    observable scalars + masks + (optionally) gold action labels and the
    LOSS-ONLY typed/matcher relation targets."""
    from experiments.src.evaluation.labels import LabelBuilder

    views = [InputView(r) for r in rows]
    inputs = build_cgrv_inputs(views, familiarity=familiarity)
    emb = rel.encode_node_texts(inputs["texts"], encoder, cache)
    matches = None
    primary_y = recovery_y = relation_y = matcher_y = conf_y = None
    if with_labels:
        builder = LabelBuilder()
        matches = [builder.candidate_matches(r) for r in rows]
        primary_y = [builder.primary_label(r) for r in rows]
        recovery_y = [
            nf.remap_recovery_label(builder.recovery_label(r), canon)
            for r, canon in zip(rows, inputs["canonical"])]
        relation_y = relation_targets(rows, inputs["absent"], relation_labels)
        matcher_y = inputs["matcher_relation_y"]
        # LOSS-ONLY per-source confidence targets in SOURCES order (switch A).
        # LabelBuilder is the only gold-reading component; this array is passed
        # like cond_id/dro_group and is never seen by the forward pass.
        conf_y = [[float(m[s]) for s in SOURCES] for m in matches]
    data = pack_data(emb, inputs["absent"], inputs["obs"],
                     inputs["action_mask"], primary_y=primary_y,
                     recovery_y=recovery_y, relation_y=relation_y,
                     matcher_relation_y=matcher_y, conf_y=conf_y)
    return views, data, matches
