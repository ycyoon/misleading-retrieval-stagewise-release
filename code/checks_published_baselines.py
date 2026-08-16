"""Recompute the published-baseline family's numbers from the released rows.

Four external systems were run against the frozen 400-cell primary panel this
session: TrustMargin (frozen-candidate reimplementation and the official code
run natively), SABER, a SURE-RAG-style sufficiency verifier of our own, and
CARE. This module recomputes, from the per-cell rows only, every number the
manuscript quotes for them. Nothing here reads a result CSV or a summary JSON:
delete every `*_primary.csv`, `*_bootstrap.csv` and `*_summary.json` under
`data/published_baselines/` and all of these checks still pass.

What is recomputed, and from what:

  TrustMargin (frozen candidates)  `tm_frozen_logprobs.jsonl`
    The six official likelihoods per cell give M_prior, M_bind and the margin
    M = M_prior + lambda_bind * M_bind; the two candidate strings give the
    agreement test under TrustMargin's own `normalize_answer`. The official
    three-branch rule (`external/trustmargin/src/trustmargin.py:155-166`) is
    then replayed at the published defaults lambda_bind = 0.5, tau = -1.5 and
    its source counts are checked.

  TrustMargin (native, official code)  `tm_native_predictions.redacted.json`
    The official EM is the mean of the per-record `score.em` field; the source
    counts come from each record's `method_debug.selected_source`.

  SABER  `panel_decisions_selected.jsonl`, `behavior_wp4_panel.jsonl`
    The probe's two per-cell probabilities and the official threshold tau = 0.5
    give the 4-cell label; the official `cell_counts.py` action map (C01 -> CK,
    C10 -> PK, C11 -> either -> PK, C00 -> abstain) is replayed against the
    routed answer, which is matched back to the backbone's own CK/PK answers.

  SURE-RAG-style  `srstyle_verification_predictions.jsonl`
    The aggregator's three-way posterior and the per-cost (tau, beta) give the
    selective score s = P(supported) - beta * u and the deployed rule
    "return a_C iff argmax = supported and s >= tau"; both the score identity
    and the resulting decision at all four costs are recomputed.

  CARE  `care_panel_{care,closed,care_rawq}.json`, `care_panel_conditions.csv`
    CARE's own headline open-QA metric (`substring_match`) is reimplemented
    here from `external/care/src/eval/utils.py` (`has_answer` /
    `get_substring_match_score`) rather than imported, and the supportive
    vs conflicting split is recomputed under the house lexical criterion
    (`alias_ok`) using the evidence condition and gold answers carried by the
    shipped per-cell CSV.

Two implementation notes on the reimplemented CARE metric. (1) CARE normalises
with the `regex` module and NFD-normalises first; this module is restricted to
the standard library plus numpy/pandas, so the tokenizer and the SQuAD-style
normaliser are transliterated with `str.isalnum` / `str.isprintable` in place
of `\\p{L}\\p{N}\\p{M}` / `\\p{Z}\\p{C}`. The transliteration was verified
character-for-character against CARE's implementation on all 7,332 prediction
and gold strings shipped here (0 differences in the normalised string, 0 in the
token sequence, 0 in the `has_answer` outcome). (2) NFD is skipped: no shipped
string contains a combining mark, so every accented character is precomposed on
both sides of the comparison and decomposition cannot change any outcome.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FAMILY = "published_baselines"

# published hyperparameters, held fixed (prespecified_comparisons.md sec. 0)
TM_LAMBDA_BIND = 0.5
TM_TAU = -1.5
SABER_TAU = 0.5
SR_COSTS = ("0.5", "1", "2", "4")

_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")  # == string.punctuation


# --------------------------------------------------------------- normalisers
def _is_word_char(ch):
    """Stand-in for the regex module's \\w inside a \\b assertion."""
    return ch.isalnum() or ch == "_"


def _remove_articles(text):
    """re.sub(r'\\b(a|an|the)\\b', ' ', text), left-to-right, alternation order."""
    out = []
    i, n = 0, len(text)
    while i < n:
        prev_w = _is_word_char(text[i - 1]) if i > 0 else False
        hit = None
        if _is_word_char(text[i]) and not prev_w:
            for word in ("a", "an", "the"):
                j = i + len(word)
                if text[i:j] == word:
                    nxt_w = _is_word_char(text[j]) if j < n else False
                    if not nxt_w:
                        hit = word
                        break
        if hit is not None:
            out.append(" ")
            i += len(hit)
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _normalize_answer(s, tm_a_guard=False):
    """SQuAD normalisation: lower, drop punctuation, drop articles, fix space.

    `tm_a_guard` reproduces TrustMargin's extra `if text == "a": return text`
    branch (external/trustmargin/src/evaluate.py:9-12); CARE has no such branch
    (external/care/src/eval/utils.py:normalize_answer).
    """
    t = "".join(ch for ch in str(s).lower() if ch not in _PUNCT)
    if not (tm_a_guard and t == "a"):
        t = _remove_articles(t)
    return " ".join(t.split())


def _tokenize(text):
    """CARE's SimpleTokenizer: runs of \\p{L}\\p{N}\\p{M}, else one non-space char."""
    tokens, run = [], []
    for ch in text:
        if ch.isalnum():
            run.append(ch)
            continue
        if run:
            tokens.append("".join(run))
            run = []
        if not (ch.isspace() or not ch.isprintable()):  # not in \p{Z} u \p{C}
            tokens.append(ch)
    if run:
        tokens.append("".join(run))
    return [t.lower() for t in tokens]


def _has_answer(answers, text):
    """CARE's has_answer: any gold token sequence occurs inside the prediction."""
    toks = _tokenize(_normalize_answer(text))
    for answer in answers:
        gold = _tokenize(_normalize_answer(answer))
        for i in range(0, len(toks) - len(gold) + 1):
            if gold == toks[i:i + len(gold)]:
                return True
    return False


def _house_norm(x):
    """The house matcher's normaliser (gpt_semantic_closure/run_all.py:36)."""
    return " ".join(str(x or "").casefold().split())


def _alias_ok(ans, gold):
    """The house lexical criterion (gpt_semantic_closure/run_all.py:38-41)."""
    a = _house_norm(ans)
    gs = [_house_norm(g) for g in (gold or [])]
    if not a or not gs:
        return False
    return any(a == g or a in g or g in a for g in gs)


def _read_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -------------------------------------------------------------------- checks
def run(D, check):
    B = Path(D) / FAMILY

    # =================================================== TrustMargin, frozen
    print("== published_baselines: TrustMargin on the frozen candidates ==")
    tm = _read_jsonl(B / "trustmargin/tm_frozen_logprobs.jsonl")
    check("tm-frozen cells scored", float(len(tm)), 400.0, tol=0)
    check("tm-frozen rows at the published (lambda_bind, tau)",
          float(sum(r["lambda_bind"] == TM_LAMBDA_BIND and r["tau"] == TM_TAU
                    for r in tm)), 400.0, tol=0)

    # the official margin, rebuilt from the six likelihoods
    m_prior = np.array([r["direct_yR"] - r["direct_yD"] for r in tm])
    m_bind = np.array([(r["rag_yR"] - r["context_only_yR"])
                       - (r["rag_yD"] - r["context_only_yD"]) for r in tm])
    margin = m_prior + TM_LAMBDA_BIND * m_bind
    check("tm-frozen margin identity, max |recomputed - stored|",
          float(np.abs(margin - np.array([r["margin"] for r in tm])).max()),
          0.0, tol=1e-6)

    # the agreement branch, rebuilt from the two candidate strings
    agree = np.array([_normalize_answer(r["y_D"], tm_a_guard=True)
                      == _normalize_answer(r["y_R"], tm_a_guard=True)
                      for r in tm])
    check("tm-frozen both_agree from y_D/y_R, mismatches",
          float((agree != np.array([r["both_agree"] for r in tm])).sum()),
          0.0, tol=0)

    # the official three-branch rule at the published tau
    source = np.where(agree, "both_agree",
                      np.where(margin > TM_TAU, "rag", "direct"))
    counts = Counter(source.tolist())
    check("tm-frozen selected_source = direct", float(counts["direct"]),
          158.0, tol=0)
    check("tm-frozen selected_source = rag", float(counts["rag"]), 172.0, tol=0)
    check("tm-frozen selected_source = both_agree",
          float(counts["both_agree"]), 70.0, tol=0)
    check("tm-frozen source matches the recorded decision, mismatches",
          float((source != np.array([r["selected_source"] for r in tm])).sum()),
          0.0, tol=0)
    # rag -> the context candidate a_C, every other branch -> the parametric a_P
    want_cand = np.where(source == "rag", "a_C", "a_P")
    check("tm-frozen returned candidate a_C (context answer)",
          float((want_cand == "a_C").sum()), 172.0, tol=0)
    check("tm-frozen returned candidate consistent, mismatches",
          float((want_cand
                 != np.array([r["returned_candidate"] for r in tm])).sum()),
          0.0, tol=0)

    # =================================================== TrustMargin, native
    print("== published_baselines: TrustMargin native, official code ==")
    nat = json.loads((B / "trustmargin/tm_native_predictions.redacted.json")
                     .read_text(encoding="utf-8"))
    recs = nat["datasets"]["2wikimultihopqa"]["records"]
    check("tm-native records", float(len(recs)), 400.0, tol=0)
    check("tm-native official EM",
          100.0 * float(np.mean([r["score"]["em"] for r in recs])), 45.0)
    nc = Counter(r["method_debug"]["selected_source"] for r in recs)
    check("tm-native selected_source = direct", float(nc["direct"]), 165.0, tol=0)
    check("tm-native selected_source = rag", float(nc["rag"]), 155.0, tol=0)
    check("tm-native selected_source = both_agree", float(nc["both_agree"]),
          80.0, tol=0)

    # ================================================================ SABER
    print("== published_baselines: SABER 4-cell routing on the panel ==")
    sb = _read_jsonl(B / "saber/panel_decisions_selected.jsonl")
    check("saber panel cells", float(len(sb)), 400.0, tol=0)

    # the 4-cell label, rebuilt from the probe's two probabilities at tau = 0.5
    cell = np.array(["C%d%d" % (int(r["pk_proba"] >= SABER_TAU),
                                int(r["ck_proba"] >= SABER_TAU)) for r in sb])
    check("saber predicted_cell from probe probabilities, mismatches",
          float((cell != np.array([r["predicted_cell"] for r in sb])).sum()),
          0.0, tol=0)
    cc = Counter(cell.tolist())
    check("saber cell C01 (PK unreliable, CK reliable)", float(cc["C01"]),
          252.0, tol=0)
    check("saber cell C11 (both reliable -> either)", float(cc["C11"]),
          136.0, tol=0)
    check("saber cell C00 (neither reliable -> abstain)", float(cc["C00"]),
          6.0, tol=0)
    check("saber cell C10 (PK reliable, CK unreliable)", float(cc["C10"]),
          6.0, tol=0)

    # the official cell -> action map, replayed against the routed answer
    want = {"C11": "pk", "C10": "pk", "C01": "ck", "C00": "abstain"}
    routed = Counter()
    bad_route = 0
    for c, r in zip(cell, sb):
        target = want[c]
        got = r["routed_answer"]
        ok = (got == r["pk_answer"] if target == "pk" else
              got == r["ck_answer"] if target == "ck" else
              not str(got or "").strip())
        routed[(c, target, ok)] += 1
        bad_route += 0 if ok else 1
    check("saber C11 cells returning the PK answer (either->PK)",
          float(routed[("C11", "pk", True)]), 136.0, tol=0)
    check("saber C01 cells returning the CK answer",
          float(routed[("C01", "ck", True)]), 252.0, tol=0)
    check("saber C10 cells returning the PK answer",
          float(routed[("C10", "pk", True)]), 6.0, tol=0)
    check("saber C00 cells abstaining", float(routed[("C00", "abstain", True)]),
          6.0, tol=0)
    check("saber routing violations of the official map", float(bad_route),
          0.0, tol=0)

    # the routed answers are the backbone's own CK/PK generations
    beh = {r["qid"]: r
           for r in _read_jsonl(B / "saber/behavior_wp4_panel.jsonl")}
    drift = sum(1 for r in sb
                if r["qid"] not in beh
                or beh[r["qid"]]["pk_answer"] != r["pk_answer"]
                or beh[r["qid"]]["ck_answer"] != r["ck_answer"]
                or bool(beh[r["qid"]]["pk_correct"]) != bool(r["pk_correct"])
                or bool(beh[r["qid"]]["ck_correct"]) != bool(r["ck_correct"]))
    check("saber decisions match the backbone behaviour rows, mismatches",
          float(drift), 0.0, tol=0)

    # ====================================================== SURE-RAG-style
    print("== published_baselines: SURE-RAG-style verifier on the panel ==")
    sr = _read_jsonl(B / "surerag_style/srstyle_verification_predictions.jsonl")
    check("sr-style panel cells", float(len(sr)), 400.0, tol=0)

    argmax = [max(r["pi"], key=lambda k, _r=r: _r["pi"][k]) for r in sr]
    check("sr-style argmax matches the recorded label, mismatches",
          float(sum(a != r["argmax"] for a, r in zip(argmax, sr))), 0.0, tol=0)
    check("sr-style cells whose argmax is 'supported'",
          float(sum(a == "supported" for a in argmax)), 0.0, tol=0)

    resid, returns = 0.0, {}
    for c in SR_COSTS:
        n_ret = 0
        for a, r in zip(argmax, sr):
            t = r["per_cost"][c]
            s = r["pi"]["supported"] - t["beta"] * r["u"]
            resid = max(resid, abs(s - t["s"]))
            if a == "supported" and s >= t["tau"]:
                n_ret += 1
        returns[c] = n_ret
    check("sr-style selective-score identity, max |recomputed - stored|",
          float(resid), 0.0, tol=1e-12)
    for c in SR_COSTS:
        check(f"sr-style abstentions at c={float(c):g} (of 400)",
              float(len(sr) - returns[c]), 400.0, tol=0)

    # ================================================================= CARE
    print("== published_baselines: CARE end-to-end on the panel ==")
    cond = pd.read_csv(B / "care/care_panel_conditions.csv")
    cond["gold"] = cond.gold_answers.map(json.loads)
    gold_by_qid = dict(zip(cond.qid, cond.gold))
    cond_by_qid = dict(zip(cond.qid, cond.condition))
    check("care panel cells, external_supportive",
          float((cond.condition == "external_supportive").sum()), 175.0, tol=0)
    check("care panel cells, external_conflicting",
          float((cond.condition == "external_conflicting").sum()), 225.0, tol=0)

    arms = {}
    for arm in ("care", "closed", "care_rawq"):
        arms[arm] = json.loads((B / f"care/care_panel_{arm}.json")
                               .read_text(encoding="utf-8"))
    check("care gold answers match the frozen panel, mismatches",
          float(sum(list(r["answer"]) != list(gold_by_qid[r["id"]])
                    for rows in arms.values() for r in rows)), 0.0, tol=0)

    for arm, want_sm in (("care", 0.4425), ("closed", 0.4050),
                         ("care_rawq", 0.4550)):
        rows = arms[arm]
        hits = [float(_has_answer(gold_by_qid[r["id"]], r["pred"])) for r in rows]
        check(f"care substring_match, {arm} arm",
              float(sum(hits) / len(rows)), want_sm)

    split = Counter()
    for r in arms["care"]:
        if _alias_ok(r["pred"], gold_by_qid[r["id"]]):
            split[cond_by_qid[r["id"]]] += 1
    check("care alias_ok hits on supportive cells (of 175)",
          float(split["external_supportive"]), 114.0, tol=0)
    check("care alias_ok hits on conflicting cells (of 225)",
          float(split["external_conflicting"]), 60.0, tol=0)
