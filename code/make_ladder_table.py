"""Emit the stagewise decomposition table from the decomposition artifact.

The table was previously assembled by hand from the same artifact, which is how
a mis-rounded entry entered an earlier version of a different table. Generating
it fixes that, but generating it into a single column did not fix the width: the
last version ran $145$pt past the column and the $c{=}2$ and $c{=}4$ entries were
clipped off the page, so the headline result was unreadable in the PDF. It spans
both columns now, which is the only honest place for four costs each carrying an
interval.

The caption is generated too. Its previous text asserted that the selected policy
was the same at every stage while the body of the same table printed three
different stage rows, because the sentence was written for the branch where the
stages agree and the code emitted the branch where they do not.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.make_ladder_table
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/maintrack_v1"
SRC = ROOT / ("artifacts/maintrack_v1/v12_13_telescoping/"
              "telescoping_decomposition.csv")
OUT = ROOT / "paper_wsdm/generated_v12_11b/tab_ladder.tex"

ROWS = [("headroom_H", r"Headroom $H_c$"),
        ("fixed_policy_gain_G1", r"$G_1$ fixed policy"),
        ("simple_selector_gain_G2", r"$G_2$ simple selectors"),
        ("structured_gain_G3", r"$G_3$ structured"),
        ("residual_after_selected_policy_R", r"$R_c$ residual")]

SHORT = {"trustmargin_tau0": "margin", "trustmargin_tau0.05": "margin .05",
         "trustmargin_tau0.1": "margin .1", "trustmargin_tau0.2": "margin .2",
         "trustmargin_tau0.4": "margin .4", "trustmargin_tau0.8": "margin .8",
         "closure_agree_any": "any pair",
         "closure_agree_any_else_P": "any pair${\\to}\\ap$",
         "closure_agree_RP": "$\\ar{=}\\ap$",
         "closure_agree_CR": "$\\ac{=}\\ar$",
         "closure_agree_CP": "$\\ac{=}\\ap$",
         "closure_cascade_RP": "$\\ar{\\to}\\ap$",
         "closure_cascade_PR": "$\\ap{\\to}\\ar$",
         "closure_cascade_CR": "$\\ac{\\to}\\ar$",
         "closure_cascade_CP": "$\\ac{\\to}\\ap$",
         "closure_cascade_PC": "$\\ap{\\to}\\ac$",
         "closure_cascade_RC": "$\\ar{\\to}\\ac$",
         "closure_cascade_RPC": "$\\ar{\\to}\\ap{\\to}\\ac$",
         "closure_cascade_PRC": "$\\ap{\\to}\\ar{\\to}\\ac$",
         "closure_cascade_CRP": "$\\ac{\\to}\\ar{\\to}\\ap$",
         "closure_cascade_CPR": "$\\ac{\\to}\\ap{\\to}\\ar$",
         "closure_cascade_RCP": "$\\ar{\\to}\\ac{\\to}\\ap$",
         "closure_cascade_PCR": "$\\ap{\\to}\\ac{\\to}\\ar$",
         "fixed_parametric": "$\\ap$", "fixed_reference": "$\\ar$",
         "fixed_recovery": "$\\ac$", "prompt_candidatewise": "prompting",
         "gbdt_recovery": "GBDT", "always_abstain": "abstain",
         "cgrv_recovery": r"\CGrV{}", "p_true_gate": r"$p_{\mathrm{true}}$"}


def names(cell: str) -> str:
    parts = [SHORT.get(x, x) for x in str(cell).split("|")]
    return parts[0] if len(set(parts)) == 1 else " / ".join(dict.fromkeys(parts))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    d = pd.read_csv(a.src).sort_values("cost")
    worst = d.identity_residual.abs().max()
    if worst > 1e-9:
        raise SystemExit(f"identity does not hold in the artifact: {worst}")
    r70 = pd.read_csv(A / ("v12_27_second_reader_70b/"
                           "second_reader_decomposition.csv")).sort_values("cost")
    worst70 = r70.identity_residual.abs().max()
    if worst70 > 1e-9:
        raise SystemExit(f"identity does not hold under the 70B: {worst70}")
    up70 = float(pd.read_csv(A / ("v12_27_second_reader_70b/"
                                  "stagewise_simultaneous_upper.csv"))
                 .simultaneous_hi.max())
    b70 = pd.read_csv(A / "v12_27_second_reader_70b/stagewise_bands.csv")
    b70map = {(r.term, r.cost): (r.lo, r.hi)
              for r in b70.itertuples(index=False)}

    costs = list(d.cost)
    picks = [[names(v) for v in d[f"stage{k}_choice"]] for k in (1, 2, 3)]
    same = picks[0] == picks[1] == picks[2]

    # The caption states what the table shows, so it is written from the table.
    caption = (
        "Stagewise decomposition over the nested policy classes under the frozen "
        "lexical criterion, cross-fitted: each question-keyed fold selects each "
        "class's best member on the other fold and is scored out of fold. "
        "$G_1$ is the fixed-policy gain over the context rule, $G_2$ and "
        "$G_3$ the simple- and structured-selector gains, and $R_c$ the "
        "residual to the oracle (Section~\\ref{sec:setup}). Brackets are "
        "per-cell percentile intervals, the stage selection repeated inside "
        "each draw. The displayed values are rounded; "
        "the four terms sum to headroom up to rounding. "
        + ("The last rows name the policy each stage selected, both folds "
           "shown where they differ."
           if not same else
           "The last row names the policy the stages selected, which was the "
           "same at every stage; where the two folds disagree, both are shown.")
        + " Here $x{\\to}y$ is the cascade returning $x$ where available and $y$ otherwise, $x{=}y$ the rule answering only where those channels agree, and a slash separates the folds. Table~\\ref{tab:jury} gives the three stage gains at $c{=}1$ under every scoring treatment.")

    lines = ["% generated file -- do not hand-edit",
             r"\begin{table*}[t]\centering\scriptsize"
             r"\setlength{\tabcolsep}{4pt}",
             f"\\caption{{{caption}}}", r"\label{tab:ladder}",
             r"\begin{tabular}{@{}l" + "r" * len(costs) + r"@{}}\toprule",
             " & " + " & ".join(f"$c{{=}}{c:g}$" for c in costs) + r"\\\midrule"]
    bands = pd.read_csv(A / "v12_13_telescoping/stagewise_bands.csv")
    bmap = {(r.term, r.cost): (r.lo, r.hi) for r in bands.itertuples(index=False)}
    TERM = {"headroom_H": "H", "fixed_policy_gain_G1": "G1",
            "simple_selector_gain_G2": "G2", "structured_gain_G3": "G3",
            "residual_after_selected_policy_R": "R"}
    for col, label in ROWS:
        cells = []
        for v, c in zip(d[col], d.cost):
            b_ = bmap.get((TERM[col], float(c)))
            cells.append(f"${v:+.3f}$" if not b_ else
                         f"${v:+.3f}$ \\tiny$[{b_[0]:+.3f},{b_[1]:+.3f}]$")
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    if same:
        lines.append("Selected policy & " + " & ".join(picks[0]) + r"\\")
    elif picks[1] == picks[2]:
        # stages 2 and 3 select identically, so two rows carry all three
        lines.append("Stage 1 selected & " + " & ".join(picks[0]) + r"\\")
        lines.append("Stages 2--3 selected & " + " & ".join(picks[1]) + r"\\")
    else:
        for k in (1, 2, 3):
            lines.append(f"Stage {k} selected & " + " & ".join(picks[k - 1])
                         + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{Second reader, "
                 r"Llama-3.1-70B-Instruct, its own candidate inventory}}\\")
    L70 = [("headroom_H", "H", r"Headroom $H_c$"),
           ("fixed_policy_gain_G1", "G1", r"$G_1$ fixed policy"),
           ("simple_selector_gain_G2", "G2", r"$G_2$ simple selectors"),
           ("structured_gain_G3", "G3", r"$G_3$ structured"),
           ("residual_after_selected_policy_R", "R", r"$R_c$ residual")]
    for col, term, label in L70:
        cells = []
        for v, c in zip(r70[col], r70.cost):
            b_ = b70map.get((term, float(c)))
            cells.append(f"${v:+.3f}$" if not b_ else
                         f"${v:+.3f}$ \\tiny$[{b_[0]:+.3f},{b_[1]:+.3f}]$")
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")

    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"[ladder] identity max residual {worst:.2e}; stages "
          f"{'agree' if same else 'differ, so all three are printed'} -> "
          f"{Path(a.out).relative_to(ROOT)}")
    print(d[["cost", "headroom_H", "fixed_policy_gain_G1",
             "simple_selector_gain_G2", "structured_gain_G3",
             "residual_after_selected_policy_R"]].to_string(index=False))
    for k in (1, 2, 3):
        print(f"  stage {k}: " + " | ".join(picks[k - 1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
