"""M2 node inputs: candidate-blind per-source structure for the relational
verifier. One node per source in {C, R, P}; each node carries a text to embed
(question [SEP] answer [SEP] evidence-snippet), an absent flag, silver relation
labels between node pairs (from frozen matching.agreement), and the RECOVERY
action mask after dedupe_candidates merging of alias-equal answers.

Consumes rows ONLY through data.views.InputView (which structurally omits
gold_answers / a_false / condition / dataset). No gold-derived quantity is
produced here; gold labels live exclusively in evaluation.labels.LabelBuilder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from experiments.src.data.views import InputView
from experiments.src.features.matching import dedupe_candidates, normalize

ABSTAIN_SUBSTRINGS = ("i don't know", "i dont know")

# Node order is fixed: index 0=C(context), 1=R(reference), 2=P(parametric).
SOURCES = ("context", "reference", "parametric")
ROLES = ("C", "R", "P")
_ANSWER_ATTR = {
    "context": "context_answer",
    "reference": "reference_answer",
    "parametric": "parametric_answer",
}

ACTIONS = ("answer_context", "answer_reference", "answer_parametric", "abstain")
ACTION_OF_SOURCE = {s: f"answer_{s}" for s in SOURCES}
SOURCE_OF_ACTION = {a: s for s, a in ACTION_OF_SOURCE.items()}

# Unordered node pairs (i, j, tag) used for relation supervision.
PAIRS = ((0, 1, "cr"), (0, 2, "cp"), (1, 2, "rp"))
RELATIONS = ("support", "contradict", "unknown")
REL_SUPPORT, REL_CONTRADICT, REL_UNKNOWN = 0, 1, 2

DEFAULT_MAX_EVIDENCE_CHARS = 512
SEP = " [SEP] "


def _require_view(view: Any) -> None:
    if not isinstance(view, InputView):
        raise TypeError(
            f"node_features requires InputView, got {type(view).__name__}; "
            "raw rows are forbidden (leakage guard)")


def usable_answer(answer: Optional[str]) -> Optional[str]:
    """Answer usable as a candidate: a non-empty, non-abstain string."""
    if not isinstance(answer, str):
        return None
    na = normalize(answer)
    if not na or any(sub in na for sub in ABSTAIN_SUBSTRINGS):
        return None
    return answer


def evidence_snippet(view: InputView, source: str,
                     max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS) -> str:
    """First max_chars chars of the node's evidence: supplied_context text for
    C, reference_passages text for R, empty for P (parametric has none)."""
    _require_view(view)
    if source == "context":
        passages = view.supplied_context or []
    elif source == "reference":
        passages = view.reference_passages or []
    elif source == "parametric":
        return ""
    else:
        raise ValueError(f"unknown source {source!r}; want one of {SOURCES}")
    parts: List[str] = []
    total = 0
    for p in passages:
        text = str(p.get("text") or "")
        if not text:
            continue
        parts.append(text)
        total += len(text) + 1
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def node_texts(view: InputView,
               max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS) -> List[Optional[str]]:
    """Per-node embed text in SOURCES order; None for absent candidates
    (missing / empty / abstain answers get a zero embedding + absent flag)."""
    _require_view(view)
    out: List[Optional[str]] = []
    for source in SOURCES:
        answer = usable_answer(getattr(view, _ANSWER_ATTR[source]))
        if answer is None:
            out.append(None)
        else:
            out.append(view.question + SEP + answer
                       + SEP + evidence_snippet(view, source, max_chars))
    return out


def relation_silver(view: InputView) -> List[int]:
    """Silver relation label per PAIRS: both answers usable and agreement()
    -> support; both usable and no agreement -> contradict; else unknown."""
    _require_view(view)
    from experiments.src.features.matching import agreement
    answers = [usable_answer(getattr(view, _ANSWER_ATTR[s])) for s in SOURCES]
    labels: List[int] = []
    for i, j, _tag in PAIRS:
        ai, aj = answers[i], answers[j]
        if ai is None or aj is None:
            labels.append(REL_UNKNOWN)
        elif agreement(ai, aj):
            labels.append(REL_SUPPORT)
        else:
            labels.append(REL_CONTRADICT)
    return labels


def recovery_structure(view: InputView) -> Dict[str, Any]:
    """Dedupe alias-equal candidates (matching.dedupe_candidates) and build the
    RECOVERY head structure: action_mask over ACTIONS (True = selectable;
    abstain always selectable; only the canonical first-seen source of each
    merged group is selectable) and canonical source mapping."""
    _require_view(view)
    answers = {s: usable_answer(getattr(view, _ANSWER_ATTR[s])) for s in SOURCES}
    groups = dedupe_candidates(answers)
    canonical: Dict[str, str] = {}
    for g in groups:
        canon = g["sources"][0]
        for s in g["sources"]:
            canonical[s] = canon
    mask = [False, False, False, True]  # abstain always allowed
    for g in groups:
        mask[SOURCES.index(g["sources"][0])] = True
    return {"action_mask": mask, "canonical": canonical}


def remap_recovery_label(label: str, canonical: Dict[str, str]) -> str:
    """Map a RECOVERY action label onto the canonical merged candidate (the
    first-seen source of its dedupe group); abstain is unchanged."""
    if label == "abstain":
        return label
    source = SOURCE_OF_ACTION[label]
    return ACTION_OF_SOURCE[canonical.get(source, source)]


def build_node_inputs(views: Sequence[InputView],
                      max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS) -> Dict[str, list]:
    """Batch node inputs for the relational verifier.

    Returns dict of aligned lists (one entry per view):
      texts:        [t_C, t_R, t_P] with None for absent nodes
      absent:       [0/1] * 3
      relation_y:   silver relation label per PAIRS (ints into RELATIONS)
      action_mask:  [bool] * 4 over ACTIONS
      canonical:    {source: canonical_source}
    """
    out = {"texts": [], "absent": [], "relation_y": [],
           "action_mask": [], "canonical": []}
    for view in views:
        _require_view(view)
        texts = node_texts(view, max_chars)
        struct = recovery_structure(view)
        out["texts"].append(texts)
        out["absent"].append([1.0 if t is None else 0.0 for t in texts])
        out["relation_y"].append(relation_silver(view))
        out["action_mask"].append(struct["action_mask"])
        out["canonical"].append(struct["canonical"])
    return out
