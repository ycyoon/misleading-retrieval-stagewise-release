"""M0 rule-based deciders over per-row candidate answers (a_C, a_R, a_P).

Every rule is pure: decide(row, mode) -> {action, answer, rule}, no I/O.
Presence/agreement follow the frozen semantics in experiments.src.features.matching;
a candidate is absent if answered=False or its text contains an abstain substring.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from experiments.src.features.matching import agreement, dedupe_candidates, normalize

ABSTAIN_SUBSTRINGS = ("i don't know", "i dont know")

# Action taken when answering with the candidate from each source.
_SOURCE_ACTION = {
    "C": "answer_context",
    "R": "answer_reference",
    "P": "answer_parametric",
}
_ANSWER_KEYS = {
    "C": ("context_answer", "context_generation_scores"),
    "R": ("reference_answer", "reference_generation_scores"),
    "P": ("parametric_answer", "parametric_generation_scores"),
}


def is_abstain_text(text: str) -> bool:
    """True if the normalized text contains an abstain substring."""
    nt = normalize(text)
    return any(sub in nt for sub in ABSTAIN_SUBSTRINGS)


def candidate(row: dict, source: str) -> Optional[str]:
    """Return the answer string for source 'C'|'R'|'P', or None if absent.

    Absent = answer missing/empty, generation scores say answered=False, or the
    answer text contains an abstain substring ("i don't know"/"i dont know").
    """
    answer_key, scores_key = _ANSWER_KEYS[source]
    answer = row.get(answer_key)
    if answer is None or not normalize(answer):
        return None
    scores = row.get(scores_key) or {}
    if scores.get("answered") is False:
        return None
    if is_abstain_text(answer):
        return None
    return answer


def _abstain(rule: str) -> dict:
    return {"action": "abstain", "answer": None, "rule": rule}


def _answer(source: str, answer: str, rule: str) -> dict:
    return {"action": _SOURCE_ACTION[source], "answer": answer, "rule": rule}


def _majority(row: dict, rule: str) -> dict:
    """Strict-majority vote over deduped present candidates; tie/empty -> abstain."""
    groups = dedupe_candidates({s: candidate(row, s) for s in ("C", "R", "P")})
    if not groups:
        return _abstain(rule)
    top = max(len(g["sources"]) for g in groups)
    winners = [g for g in groups if len(g["sources"]) == top]
    if len(winners) != 1:
        return _abstain(rule)
    sources = winners[0]["sources"]
    src = next(s for s in ("C", "R", "P") if s in sources)
    return _answer(src, winners[0]["answer"], rule)


def context_only(row: dict, mode: str = "primary") -> dict:
    """Answer a_C whenever present; otherwise abstain."""
    a_c = candidate(row, "C")
    return _answer("C", a_c, "context_only") if a_c else _abstain("context_only")


def always_abstain(row: dict, mode: str = "primary") -> dict:
    """Always abstain."""
    return _abstain("always_abstain")


def parametric_only(row: dict, mode: str = "primary") -> dict:
    """Answer a_P whenever present; otherwise abstain (baseline; ignores mode)."""
    a_p = candidate(row, "P")
    return _answer("P", a_p, "parametric_only") if a_p else _abstain("parametric_only")


def majority_vote(row: dict, mode: str = "primary") -> dict:
    """Strict majority over deduped candidates; tie -> abstain."""
    return _majority(row, "majority_vote")


def _pair_agreement(row: dict, other: str, mode: str, rule: str) -> dict:
    """Shared core: answer a_C iff the other candidate agrees, else abstain.

    Recovery mode policy (documented, kept simple): on disagreement return the
    strict-majority answer over all present candidates, else abstain.
    """
    a_c = candidate(row, "C")
    a_o = candidate(row, other)
    if a_c is None or a_o is None:
        return _abstain(rule)
    if agreement(a_c, a_o):
        return _answer("C", a_c, rule)
    if mode == "recovery":
        return _majority(row, rule)
    return _abstain(rule)


def context_parametric_agreement(row: dict, mode: str = "primary") -> dict:
    """Answer a_C iff a_P agrees; else abstain (recovery: majority fallback)."""
    return _pair_agreement(row, "P", mode, "context_parametric_agreement")


def context_reference_agreement(row: dict, mode: str = "primary") -> dict:
    """Answer a_C iff a_R agrees; else abstain (recovery: majority fallback)."""
    return _pair_agreement(row, "R", mode, "context_reference_agreement")


def _pair_agreement_silence_returns(row: dict, other: str, mode: str,
                                    rule: str) -> dict:
    """Answer a_C iff the other candidate agrees OR is silent; else abstain.

    The silence-default sibling of _pair_agreement. It exists because a
    single-auxiliary rule that abstains whenever its one source is quiet is a
    poor rule, not because the channel is uninformative: on these datasets it
    scores below the rule that ignores the auxiliary source entirely. A
    value-of-learning decomposition that uses the abstaining variant as THE
    fixed rule for a one-auxiliary channel is measuring the silence default,
    not the channel.
    """
    a_c = candidate(row, "C")
    if a_c is None:
        return _abstain(rule)
    a_o = candidate(row, other)
    if a_o is None:
        return _answer("C", a_c, rule)          # silence -> unchecked return
    if agreement(a_c, a_o):
        return _answer("C", a_c, rule)
    if mode == "recovery":
        return _majority(row, rule)
    return _abstain(rule)


def context_reference_or_silent(row: dict, mode: str = "primary") -> dict:
    """C+R: answer a_C iff a_R agrees or is silent; else abstain."""
    return _pair_agreement_silence_returns(row, "R", mode,
                                           "context_reference_or_silent")


def context_parametric_or_silent(row: dict, mode: str = "primary") -> dict:
    """C+P: answer a_C iff a_P agrees or is silent; else abstain."""
    return _pair_agreement_silence_returns(row, "P", mode,
                                           "context_parametric_or_silent")


def current_paper_rule(row: dict, mode: str = "primary") -> dict:
    """EXACT replication of the frozen wiki-hybrid decision path from candidates.

    a_C absent -> abstain. a_R present: agree -> answer a_C, disagree -> abstain.
    a_R absent: a_P present -> agree ? answer a_C : abstain; a_P absent -> ANSWER
    a_C (the documented silence-default).
    """
    rule = "current_paper_rule"
    a_c = candidate(row, "C")
    if a_c is None:
        return _abstain(rule)
    a_r = candidate(row, "R")
    if a_r is not None:
        return _answer("C", a_c, rule) if agreement(a_c, a_r) else _abstain(rule)
    a_p = candidate(row, "P")
    if a_p is not None:
        return _answer("C", a_c, rule) if agreement(a_c, a_p) else _abstain(rule)
    return _answer("C", a_c, rule)  # both silent: silence-default answers


def strict_corroboration(row: dict, mode: str = "primary") -> dict:
    """Like current_paper_rule but requires corroboration: both-silent -> ABSTAIN.

    Answer a_C only when a_R agrees, or (a_R absent and) a_P agrees; everything
    else — including both verifiers silent — abstains.
    """
    rule = "strict_corroboration"
    a_c = candidate(row, "C")
    if a_c is None:
        return _abstain(rule)
    a_r = candidate(row, "R")
    if a_r is not None:
        return _answer("C", a_c, rule) if agreement(a_c, a_r) else _abstain(rule)
    a_p = candidate(row, "P")
    if a_p is not None:
        return _answer("C", a_c, rule) if agreement(a_c, a_p) else _abstain(rule)
    return _abstain(rule)  # both silent: no corroboration, abstain


def current_paper_rule_recover(row: dict, mode: str = "recovery") -> dict:
    """current_paper_rule with wiki-recovery: on a_R disagreement answer a_R."""
    rule = "current_paper_rule_recover"
    a_c = candidate(row, "C")
    if a_c is None:
        return _abstain(rule)
    a_r = candidate(row, "R")
    if a_r is not None:
        return _answer("C", a_c, rule) if agreement(a_c, a_r) else _answer("R", a_r, rule)
    a_p = candidate(row, "P")
    if a_p is not None:
        return _answer("C", a_c, rule) if agreement(a_c, a_p) else _abstain(rule)
    return _answer("C", a_c, rule)  # silence-default, as in current_paper_rule


RULES: Dict[str, Callable[..., dict]] = {
    "context_only": context_only,
    "always_abstain": always_abstain,
    "parametric_only": parametric_only,
    "majority_vote": majority_vote,
    "context_parametric_agreement": context_parametric_agreement,
    "context_reference_agreement": context_reference_agreement,
    "current_paper_rule": current_paper_rule,
    "strict_corroboration": strict_corroboration,
    "current_paper_rule_recover": current_paper_rule_recover,
    "context_reference_or_silent": context_reference_or_silent,
    "context_parametric_or_silent": context_parametric_or_silent,
}
