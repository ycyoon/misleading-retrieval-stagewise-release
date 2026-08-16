"""M1 gate features: leakage-safe flat feature dict, one per candidates row.

Consumes rows ONLY through data.views.InputView (which structurally omits
gold_answers / a_false / condition / dataset); agreement and token-F1 follow the
frozen matching semantics. A module-level guard pins the exact feature set.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from experiments.src.data.views import InputView
from experiments.src.features.matching import (
    agreement,
    embedding_similarity,
    normalize,
    token_f1,
)

ABSTAIN_SUBSTRINGS = ("i don't know", "i dont know")

# Value imputed for a missing (None) embedding similarity; paired with a
# *_missing indicator so the model can tell imputed from observed.
EMBED_IMPUTE = 0.5

_SOURCES = ("context", "reference", "parametric")
_PAIRS = (("context", "reference", "cr"), ("context", "parametric", "cp"),
          ("reference", "parametric", "rp"))

# The exact permitted feature set. build_features asserts its output keys equal
# this set; anything gold/condition/dataset-derived is forbidden by name below.
FEATURE_NAMES = (
    # presence of each candidate (non-empty, non-abstain)
    "has_context", "has_reference", "has_parametric",
    # pairwise agreement via frozen matching.agreement (abstain -> absent)
    "agree_cr", "agree_cp", "agree_rp",
    # alias-free token F1 between candidate pairs
    "token_f1_cr", "token_f1_cp", "token_f1_rp",
    # embedding similarities (nullable -> imputed EMBED_IMPUTE + indicator)
    "emb_sim_cr", "emb_sim_cp", "emb_sim_rp",
    "emb_sim_cr_missing", "emb_sim_cp_missing", "emb_sim_rp_missing",
    # retrieval score stats (nullable -> zeros + indicator)
    "retrieval_mean", "retrieval_max", "retrieval_std", "retrieval_count",
    "retrieval_missing",
    "n_reference_passages",
    # per-source abstain-substring flags (on the raw answer text)
    "abstain_context", "abstain_reference", "abstain_parametric",
    # answer length stats (tokens of the normalized raw answer)
    "len_context", "len_reference", "len_parametric",
    # numeric-answer indicators and their pairwise agreement
    "numeric_context", "numeric_reference", "numeric_parametric",
    "numeric_agree_cr", "numeric_agree_cp", "numeric_agree_rp",
)

# No feature name may reference a gold/eval-only concept.
FORBIDDEN_NAME_SUBSTRINGS = ("gold", "a_false", "condition", "dataset",
                             "covered", "label", "poison")

for _name in FEATURE_NAMES:
    for _bad in FORBIDDEN_NAME_SUBSTRINGS:
        assert _bad not in _name, f"forbidden feature name: {_name}"
assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def _is_abstain_text(text: Optional[str]) -> bool:
    if not isinstance(text, str):
        return False
    nt = normalize(text)
    return any(sub in nt for sub in ABSTAIN_SUBSTRINGS)


def _effective(answer: Optional[str]) -> Optional[str]:
    """Answer usable for cross-candidate comparison: non-empty, non-abstain."""
    if not isinstance(answer, str) or not normalize(answer):
        return None
    if _is_abstain_text(answer):
        return None
    return answer


def _n_tokens(answer: Optional[str]) -> int:
    if not isinstance(answer, str):
        return 0
    return len(normalize(answer).split())


def _has_digit(answer: Optional[str]) -> float:
    if not isinstance(answer, str):
        return 0.0
    return 1.0 if any(ch.isdigit() for ch in answer) else 0.0


def build_features(
    view: InputView,
    embedding_sims: Optional[Dict[str, Optional[float]]] = None,
    compute_embeddings: bool = False,
) -> Dict[str, float]:
    """Build the flat feature dict for one row.

    embedding_sims optionally supplies precomputed pair similarities under keys
    'cr'/'cp'/'rp' (None values stay missing). With compute_embeddings=True the
    optional bge encoder is tried; it degrades gracefully to missing.
    """
    if not isinstance(view, InputView):
        raise TypeError(
            f"build_features requires InputView, got {type(view).__name__}; "
            "raw rows are forbidden (leakage guard)")

    raw = {
        "context": view.context_answer,
        "reference": view.reference_answer,
        "parametric": view.parametric_answer,
    }
    eff = {src: _effective(ans) for src, ans in raw.items()}

    feats: Dict[str, float] = {}
    for src in _SOURCES:
        feats[f"has_{src}"] = 1.0 if eff[src] is not None else 0.0
        feats[f"abstain_{src}"] = 1.0 if _is_abstain_text(raw[src]) else 0.0
        feats[f"len_{src}"] = float(_n_tokens(raw[src]))
        feats[f"numeric_{src}"] = _has_digit(eff[src])

    sims = dict(embedding_sims or {})
    for a, b, tag in _PAIRS:
        ea, eb = eff[a], eff[b]
        both = ea is not None and eb is not None
        feats[f"agree_{tag}"] = 1.0 if (both and agreement(ea, eb)) else 0.0
        feats[f"token_f1_{tag}"] = token_f1(ea, eb) if both else 0.0
        sim = sims.get(tag)
        if sim is None and compute_embeddings and both:
            sim = embedding_similarity(ea, eb)
        if sim is None or (isinstance(sim, float) and math.isnan(sim)):
            feats[f"emb_sim_{tag}"] = EMBED_IMPUTE
            feats[f"emb_sim_{tag}_missing"] = 1.0
        else:
            feats[f"emb_sim_{tag}"] = float(sim)
            feats[f"emb_sim_{tag}_missing"] = 0.0
        feats[f"numeric_agree_{tag}"] = (
            1.0 if (both and _has_digit(ea) == _has_digit(eb)) else 0.0)

    scores = view.retrieval_scores
    if scores:
        vals = [float(s) for s in scores]
        mean = sum(vals) / len(vals)
        feats["retrieval_mean"] = mean
        feats["retrieval_max"] = max(vals)
        feats["retrieval_std"] = math.sqrt(
            sum((v - mean) ** 2 for v in vals) / len(vals))
        feats["retrieval_count"] = float(len(vals))
        feats["retrieval_missing"] = 0.0
    else:
        feats["retrieval_mean"] = 0.0
        feats["retrieval_max"] = 0.0
        feats["retrieval_std"] = 0.0
        feats["retrieval_count"] = 0.0
        feats["retrieval_missing"] = 1.0

    feats["n_reference_passages"] = float(len(view.reference_passages or []))

    assert set(feats) == set(FEATURE_NAMES), (
        "feature guard: produced features differ from FEATURE_NAMES: "
        f"{sorted(set(feats) ^ set(FEATURE_NAMES))}")
    return feats
