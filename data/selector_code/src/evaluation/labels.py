"""LabelBuilder for the learned gate M1 (EVAL-ONLY).

The ONLY component besides metrics allowed to read gold_answers. Builds the
PRIMARY label (context answer alias-matches gold) and the RECOVERY label
(which candidate alias-matches gold, preference C > R > P, else abstain).
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from experiments.src.features.matching import alias_match, normalize

ABSTAIN_SUBSTRINGS = ("i don't know", "i dont know")

# (candidate key, answer field, recovery action) in preference order C > R > P.
_CANDIDATES = (
    ("context", "context_answer", "answer_context"),
    ("reference", "reference_answer", "answer_reference"),
    ("parametric", "parametric_answer", "answer_parametric"),
)

RECOVERY_ACTIONS = ("answer_context", "answer_reference", "answer_parametric", "abstain")


def _field(row: Union[dict, Any], name: str) -> Any:
    """Read a field from a Row dataclass or a raw row dict."""
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _usable(answer: Any) -> bool:
    """A candidate is usable if non-empty and not an abstain string."""
    if not isinstance(answer, str):
        return False
    na = normalize(answer)
    if not na:
        return False
    return not any(sub in na for sub in ABSTAIN_SUBSTRINGS)


class LabelBuilder:
    """Gold-reading label construction for gate training/threshold selection."""

    def candidate_matches(self, row: Union[dict, Any]) -> Dict[str, bool]:
        """{'context': bool, 'reference': bool, 'parametric': bool}: usable AND
        alias-matches gold_answers (bidirectional normalized substring)."""
        gold: List[str] = _field(row, "gold_answers") or []
        out: Dict[str, bool] = {}
        for key, answer_field, _ in _CANDIDATES:
            answer = _field(row, answer_field)
            out[key] = _usable(answer) and alias_match(answer, gold)
        return out

    def primary_label(self, row: Union[dict, Any]) -> int:
        """1 iff the context answer alias-matches gold (else 0)."""
        return int(self.candidate_matches(row)["context"])

    def recovery_label(self, row: Union[dict, Any]) -> str:
        """First alias-matching candidate in preference order C > R > P;
        'abstain' if no candidate matches gold."""
        matches = self.candidate_matches(row)
        for key, _, action in _CANDIDATES:
            if matches[key]:
                return action
        return "abstain"
