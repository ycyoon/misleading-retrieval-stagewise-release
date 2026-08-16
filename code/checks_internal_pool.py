"""internal_pool: the pooled internal fitting inventory the selectors are fit on.

Recomputes, from one compact per-record table alone, every count the manuscript
states about the internal pool: the pooled size, the four source collections,
the distinct question strings, the canonical question keys those strings
collapse to, the ClashEval diagnostic set, and the structural claim that no
reference candidate is available on any record in the inventory.

Nothing is read back from a summary. `internal_pool_records.parquet` carries one
row per pooled candidate record with its identifiers, its raw question string,
and the three per-channel flags; every number below is a group-by over those
rows. Delete every other file in this family and the checks still run.

The candidate answers, the supplied contexts and the reference passages are
withheld under the release's redaction policy, so the per-channel flags are
derived from them before redaction, exactly as the pipeline defines them:

  present_a_X  the raw answer field for channel X was non-null and not blank.
  av_a_X       channel X is *available* under the house rule
               (experiments/src/models/rules.py, `candidate()`): the answer is
               missing or normalizes to empty, or its generation metadata
               records `answered=False`, or its normalized text contains an
               abstain substring ("i don't know" / "i dont know") -> unavailable.
               Normalization is strip, lowercase, collapse internal whitespace.
  ok_a_X       channel X is available AND its answer satisfies the house lexical
               matcher against the record's accepted-answer set (`alias_ok`:
               casefold, collapse whitespace, then equality or containment in
               either direction against any accepted answer).

The three flags are nested by construction (ok <= av <= present); that nesting
is checked below on every record before any count is taken, so the flag table is
not accepted on trust.

The canonical question key is rebuilt here rather than shipped, so the
2,094 -> 2,090 collapse is recomputed from the raw question strings.
"""

from __future__ import annotations

import pandas as pd

# The four collections the manuscript names, in manuscript order, mapped to the
# dataset identifiers the pipeline writes.
SOURCES = (("ClashEval", "clash_eval"), ("RAGuard", "raguard"),
           ("RAMDocs", "ramdocs"), ("RGB", "rgb"))
CHANNELS = ("a_C", "a_R", "a_P")


def _canonical_qkey(dataset: pd.Series, question: pd.Series) -> pd.Series:
    """The house canonical question key, dataset-scoped.

    Reimplementation of experiments/src/data/canonical.py
    (`CANONICAL_QUESTION_VERSION = "v2-unicode-casefold-symbol-space"`):
    Unicode Normalization Form KC, case folding, every character that is not a
    letter or a number replaced by a space, then whitespace collapsed. The house
    version tests `unicodedata.category(c)[0] in {"L", "N"}`; `str.isalnum()` is
    the same predicate for these strings (verified character-for-character on
    all 13,228 questions) and keeps this module inside the release's import set.
    """
    folded = question.str.normalize("NFKC").str.casefold()
    stripped = folded.map(
        lambda t: "".join(c if c.isalnum() else " " for c in t))
    return dataset + "::" + stripped.str.replace(r"\s+", " ", regex=True).str.strip()


def run(D, check):
    P = D / "internal_pool"
    rec = pd.read_parquet(P / "internal_pool_records.parquet")

    # ---- the flag table is grounded before anything is counted -----------
    print("== internal_pool: per-record flag integrity ==")
    viol = 0
    for ch in CHANNELS:
        viol += int((rec[f"ok_{ch}"] > rec[f"av_{ch}"]).sum())
        viol += int((rec[f"av_{ch}"] > rec[f"present_{ch}"]).sum())
        viol += int((~rec[f"present_{ch}"].isin([0, 1])).sum())
    check("channel flags nested ok <= av <= present, all records",
          float(viol), 0.0, tol=0)
    # A row key must identify a record: (dataset, qid, condition) is the
    # pipeline's cell key on this inventory.
    check("duplicate (dataset, qid, condition) record keys",
          float(rec.duplicated(["dataset", "qid", "condition"]).sum()),
          0.0, tol=0)
    # The availability rule must be exercised, or "no available a_R" would be an
    # artifact of an all-zero table rather than a property of the inventory.
    if not (0 < int(rec.av_a_C.sum()) < len(rec)) \
            or not (0 < int(rec.av_a_P.sum()) < len(rec)):
        raise SystemExit("internal_pool: availability flags are degenerate on "
                         "a_C or a_P; the a_R result would be uninterpretable")

    # ---- pooled size and the four source collections ---------------------
    # Manuscript, Section "Data, splits, and evaluation panels":
    #   "We pool $13{,}228$ candidate records from ClashEval, RAGuard,
    #    RAMDocs, and RGB to fit one set of selectors across heterogeneous
    #    conflict settings"
    print("== internal_pool: pooled records and source collections ==")
    check("pooled internal candidate records", float(len(rec)), 13228, tol=0)
    check("source collections in the pool", float(rec.dataset.nunique()),
          len(SOURCES), tol=0)
    by_source = rec.dataset.value_counts()
    check("manuscript-named collections present in the pool",
          float(sum(1 for _, ds in SOURCES if by_source.get(ds, 0) > 0)),
          len(SOURCES), tol=0)
    # Every record belongs to one of the four collections the manuscript names:
    # the subtotals of the four named identifiers exhaust the printed 13,228, so
    # no record comes from an unnamed collection.
    check("records in the four named collections",
          float(sum(by_source.get(ds, 0) for _, ds in SOURCES)), 13228, tol=0)
    # The per-source subtotals themselves are NOT printed in the manuscript --
    # the paper states the four names and the total -- so they are reported here
    # as the release's own record of the pool composition, never asserted.
    print("  note  pool composition: "
          + ", ".join(f"{label} {int(by_source.get(ds, 0))}"
                      for label, ds in SOURCES))

    # ---- question strings and the canonical keys they collapse to --------
    # Manuscript, same subsection:
    #   "The records contain $2{,}094$ distinct question strings, which become
    #    $2{,}090$ canonical question keys after Unicode Normalization Form KC
    #    and case folding."
    print("== internal_pool: question strings and canonical question keys ==")
    check("distinct question strings in the pool",
          float(rec.question.nunique()), 2094, tol=0)
    qkey = _canonical_qkey(rec.dataset, rec.question)
    check("canonical question keys after normalization",
          float(qkey.nunique()), 2090, tol=0)
    # The size of the collapse is the arithmetic difference of the two numbers
    # checked above and is not printed in the manuscript, so it is reported.
    print(f"  note  {rec.question.nunique() - qkey.nunique()} question strings "
          f"collapse onto another string's canonical key")
    # Diagnostic, not a manuscript number: the manuscript's one-line description
    # of the key names only NFKC and case folding. Those two steps alone leave
    # 2,093 keys; the shipped 2,090 is the full house key, which additionally
    # replaces every non-letter, non-number character with a space.
    nfkc_only = (rec.dataset + "::"
                 + rec.question.str.normalize("NFKC").str.casefold()
                 .str.replace(r"\s+", " ", regex=True).str.strip())
    print(f"  note  NFKC + case folding alone leave {nfkc_only.nunique()} keys; "
          f"the house key also maps non-alphanumeric characters to spaces")

    # Manuscript, Results, "Sufficiency does not establish credibility":
    #   "On the $2{,}400$-record ClashEval diagnostic set, ..."
    # The diagnostic set is one record per pooled ClashEval question identifier.
    ce = rec[rec.dataset == "clash_eval"]
    check("ClashEval diagnostic records (distinct ClashEval question ids)",
          float(ce.qid.nunique()), 2400, tol=0)

    # ---- the reference channel is empty on this inventory ----------------
    # Manuscript, Section "Methods", verification-only fixed comparator:
    #   "No reference candidate is available on any of the $13{,}228$ internal
    #    records, so the two corroboration rules reduce on this inventory to
    #    testing $\ac$ against $\ap$ when both are available; they differ when
    #    no parametric answer is available."
    # and Limitations:
    #   "its original candidate inventory contains no reference candidates."
    print("== internal_pool: reference-channel availability ==")
    check("records with an available reference candidate a_R",
          float(rec.av_a_R.sum()), 0, tol=0)
    check("records checked for reference availability",
          float(rec.av_a_R.notna().sum()), 13228, tol=0)
    # Stronger than the manuscript's claim and reported for completeness: the
    # reference answer field is not merely an abstention, it is absent outright.
    check("records with any nonempty reference answer text",
          float(rec.present_a_R.sum()), 0, tol=0)
    check("records with a correct reference candidate",
          float(rec.ok_a_R.sum()), 0, tol=0)
    # Context for the sentence that follows the claim: with a_R never available,
    # the corroboration rules test a_C against a_P where both are available and
    # part company on the records where a_P is not. Both populations are
    # nonempty, so the branch distinction is live on this inventory. Neither
    # count is printed in the manuscript, so both are reported, not checked.
    both = int(((rec.av_a_C == 1) & (rec.av_a_P == 1)).sum())
    only_c = int(((rec.av_a_C == 1) & (rec.av_a_P == 0)).sum())
    print(f"  note  a_C and a_P both available on {both} records; a_C available "
          f"with no a_P on {only_c} records (corroboration rules differ there)")
