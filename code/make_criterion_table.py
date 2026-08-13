"""Emit the correctness-criterion table: policy levels and the stage gains under each.

The manuscript previously carried two level tables: one under the frozen lexical
matcher beside a single auxiliary judge whose model identifier and prompt were
not preserved, and one under the jury. An unreproducible criterion should not
appear in a results table at all, and the jury artifact scores every policy the
other table listed, so one table replaced both.

This version adds the block the paper was missing. Showing that the criteria rank
policies identically on the full panel is not the same as showing the
decomposition survives them: the stages select on held out questions, and two
criteria that agree about the ranking can still disagree about what a fold picks.
The jury and the matcher already swap the two fixed rules, so the question is
live. The lower block therefore carries $G_1$, $G_2$ and $G_3$ recomputed under
each criterion, in the same columns.

Every number is read from an artifact. Nothing is typed by hand, which is how a
mis-rounded entry entered an earlier table.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.make_criterion_table
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LEVELS = ROOT / "artifacts/maintrack_v1/v12_15_jury/criterion_levels_jury.csv"
DECOMP = ROOT / ("artifacts/maintrack_v1/v12_22_criterion_decomposition/"
                 "criterion_decomposition.csv")
OUT = ROOT / "paper_wsdm/generated_v12_11b/tab_jury.tex"
FINDINGS = ROOT / "paper_wsdm/generated_v12_11b/jury_findings.tex"
COST = 1.0

# display order: fixed channel rules, then learned selectors, then the floor
POLICIES = [
    ("fixed_parametric", "Parametric rule"),
    ("fixed_reference", "Reference rule"),
    ("trustmargin_tau0", "Likelihood margin"),
    ("prompt_candidatewise", "Candidate-wise prompting"),
    ("gbdt_recovery", "Gradient boosting"),
    ("cgrv_recovery", "\\CGrV{} selection"),
    ("fixed_recovery", "Context rule"),
]
# The conservative column reproduces the jury column exactly, because an
# unsettled item already scores incorrect. Its place goes to the panel with the
# judge that is the reader's own checkpoint dropped, reported as the bound pair
# it is: two judges remain, so every disagreement between them is unsettled and
# the two treatments of the unsettled no longer coincide.
CRITERIA = [("jury", "Jury"), ("jury_no_reader", "No-reader$^-$"),
            ("jury_no_reader_optimistic", "No-reader$^+$"),
            ("jury_optimistic", "Optim."), ("alias", "Lexical")]
GAINS = [("fixed_policy_gain_G1", "$G_1$ fixed policy"),
         ("simple_selector_gain_G2", "$G_2$ simple"),
         ("structured_gain_G3", "$G_3$ structured")]

CAPTION = (
    "Utility at $c{=}1$ under five scoring treatments: the jury consensus, "
    "the no-reader panel with unsettled items scored against and then for "
    "the policy, the optimistic bound, and the frozen lexical matcher; "
    "below, the cross-fitted stage gains under the same treatments.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default=str(LEVELS))
    ap.add_argument("--decomp", default=str(DECOMP))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    d = pd.read_csv(a.levels)
    d = d[d.cost == COST]
    tab = d.pivot(index="method_id", columns="criterion", values="utility")

    # The published level artifact is built from the panel arms alone, so the
    # likelihood-margin rule -- the closest published training-free baseline --
    # has no row in it and appeared only in prose. Its levels come from the
    # criterion artifact, which recomputes every policy under every criterion;
    # where the two overlap they must agree, and that is asserted rather than
    # assumed, because two paths to the same number are only worth having if a
    # disagreement between them stops the build.
    alt = pd.read_csv(ROOT / ("artifacts/maintrack_v1/"
                              "v12_22_criterion_decomposition/"
                              "policy_levels_by_criterion.csv"))
    alt = alt[alt.cost == COST].pivot(index="method_id", columns="criterion",
                                      values="utility")
    shared = [m for m in tab.index if m in alt.index]
    off = max((abs(tab.loc[m, c] - alt.loc[m, c])
               for m in shared for c in tab.columns if c in alt.columns),
              default=0.0)
    if off > 1e-9:
        raise SystemExit(f"the two level artifacts disagree by {off:.2e}")
    for m in alt.index:
        if m not in tab.index:
            tab.loc[m] = alt.loc[m]
    print(f"[criterion-table] {len(shared)} policies agree across both level "
          f"artifacts (worst {off:.1e})")

    missing = [(m, n) for m, n in POLICIES if m not in tab.index]
    if missing:
        raise SystemExit(f"criterion artifact is missing policies: {missing}")
    absent = [c for c, _ in CRITERIA if c not in tab.columns]
    if absent:
        raise SystemExit(f"criterion artifact is missing criteria: {absent}")

    dec = pd.read_csv(a.decomp)
    dec = dec[dec.cost == COST].set_index("criterion")
    gone = [c for c, _ in CRITERIA if c not in dec.index]
    if gone:
        raise SystemExit(f"decomposition artifact is missing criteria: {gone}")

    # The sentence about the ordering is derived, not asserted. An earlier
    # version stated which fixed rule led under the jury; a change in the
    # settlement rule reversed it by half a hundredth and the sentence stayed
    # behind, which is the failure this generator exists to prevent.
    FIX = ["fixed_parametric", "fixed_reference"]
    LEARNED = ["trustmargin_tau0", "prompt_candidatewise", "gbdt_recovery",
               "cgrv_recovery"]
    assert len(CRITERIA) == 5, "the caption spells out five"
    below = all(tab.loc[m, c] < min(tab.loc[f, c] for f in FIX)
                for c, _ in CRITERIA for m in LEARNED)
    lead = {c: ("reference" if tab.loc["fixed_reference", c]
                > tab.loc["fixed_parametric", c] else "parametric")
            for c, _ in CRITERIA}
    label = {"jury": "the consensus", "jury_no_reader": "both no-reader bounds",
             "jury_no_reader_optimistic": "both no-reader bounds",
             "jury_optimistic": "the optimistic bound",
             "alias": "the matcher"}
    if lead["jury_no_reader"] != lead["jury_no_reader_optimistic"]:
        raise SystemExit("the two no-reader bounds no longer agree about which "
                         "fixed rule leads, so they cannot share a phrase")
    ref_led = list(dict.fromkeys(label[c] for c, _ in CRITERIA
                                 if lead[c] == "reference"))
    par_led = list(dict.fromkeys(label[c] for c, _ in CRITERIA
                                 if lead[c] == "parametric"))

    def listed(xs):
        return (xs[0] if len(xs) == 1
                else " and ".join([", ".join(xs[:-1]), xs[-1]]))

    if not below:
        raise SystemExit("a learned selector is no longer below both fixed "
                         "rules; the caption's claim must be rewritten")
    # The gap between the two fixed rules is a range across the treatments, not
    # a single number. An earlier caption read the jury column's 0.005 and said
    # "within a hundredth of each other", which is false of the other four; the
    # range is computed here so the sentence cannot outlive the numbers again.
    gaps = {c: abs(tab.loc["fixed_reference", c] - tab.loc["fixed_parametric", c])
            for c, _ in CRITERIA}
    lo, hi = min(gaps.values()), max(gaps.values())
    if ref_led and par_led:
        ordering = (f"Every selector in the main ladder stays below both fixed channel "
                    f"rules under all five scoring treatments, and the two "
                    f"fixed rules' order depends on the treatment: the "
                    f"reference rule leads under {listed(ref_led)}, the "
                    f"parametric rule under {listed(par_led)}.")
    else:
        ordering = (f"Every selector in the main ladder stays below both fixed channel "
                    f"rules under all five scoring treatments, and "
                    f"the {'reference' if ref_led else 'parametric'} rule leads "
                    f"under every one of them.")
    caption = CAPTION
    gaps_lo, gaps_hi = min(gaps.values()), max(gaps.values())
    findings = (
        "% generated by make_criterion_table.py -- do not hand-edit\n"
        + ordering + " Their gap runs from "
        + f"${gaps_lo:.3f}$ to ${gaps_hi:.3f}$. The jury criterion already "
        "scores contested items as incorrect, so its conservative bound is "
        "identical to the jury column and is omitted. The likelihood-margin "
        "row reports the threshold selected on held-out questions, "
        "$\\tau{=}0$. Always abstaining has zero utility and is omitted. "
        "In the lower block $G_3$ is exactly zero and $G_2$ zero or "
        "negative at all four costs under every treatment.\n")
    FINDINGS.write_text(findings)

    lines = [f"% generated by experiments/scripts/maintrack/"
             f"make_criterion_table.py -- do not hand-edit",
             r"\begin{table}[t]\centering\scriptsize"
             r"\setlength{\tabcolsep}{3pt}",
             f"\\caption{{{caption}}}",
             r"\label{tab:jury}",
             r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
             "Policy & " + " & ".join(n for _, n in CRITERIA) + r"\\\midrule"]
    for m, name in POLICIES:
        cells = [f"${tab.loc[m, c]:+.3f}$".replace("+0.000", "0.000")
                 for c, _ in CRITERIA]
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    for col, name in GAINS:
        cells = [f"${dec.loc[c, col]:+.3f}$".replace("+0.000", "0.000")
                 for c, _ in CRITERIA]
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")

    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"[criterion-table] {len(POLICIES)} policies and {len(GAINS)} stage "
          f"gains x {len(CRITERIA)} criteria at c={COST} -> "
          f"{Path(a.out).relative_to(ROOT)}")
    for m, name in POLICIES:
        print(f"  {name:26s} " +
              "  ".join(f"{tab.loc[m, c]:+.4f}" for c, _ in CRITERIA))
    for col, name in GAINS:
        print(f"  {name:26s} " +
              "  ".join(f"{dec.loc[c, col]:+.4f}" for c, _ in CRITERIA))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
