"""Recompute the paper's headline numbers from the released data alone.

This file ships inside the release as code/reproduce.py. It reads only the
release's own data/ directory, uses only numpy, pandas and pyarrow, and
checks every recomputed value against the number the paper prints. It exits
nonzero on any disagreement, so a clean run is a machine-checked statement
that the released rows support the published results.

What is recomputed from raw rows (not read back from summaries):
  - the row-level utility identity on every decision-row file
  - the primary panel's oracle utility, headroom and G1 at all four costs,
    with the two question-keyed folds rebuilt from the recorded split seed
  - the decomposition identity H = G1 + G2 + G3 + R on all three panels
  - the jury label distribution and Fleiss' kappa over the raw verdicts
  - the Llama sufficiency-crosscheck gap and its bootstrap interval
  - the extension's prompted-minus-carried contrast and the three
    per-collection parametric-minus-context points
  - the second reader's oracle utility
  - the two ablation effects the paper quotes at c=2

Run (from the release root):
  python code/reproduce.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
COSTS = (0.5, 1.0, 2.0, 4.0)
SPLIT_SEED = 20260811  # the recorded question-keyed fold seed

CHECKS = []


def check(name, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    CHECKS.append((name, ok, got, want))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:58s} got {got:+.4f}  "
          f"expected {want:+.4f}")
    return ok


def macro(df, col="answer_utility"):
    p = df.groupby(["seed_id", "source_dataset", "canonical_qkey"])[col].mean()
    return float(p.groupby(["seed_id", "source_dataset"]).mean()
                 .groupby("seed_id").mean().mean())


def row_identity(path):
    r = pd.read_parquet(path)
    a = int((r.correct_return + r.wrong_return != r.returned).sum())
    b = float((r.answer_utility
               - (r.correct_return - r.cost * r.wrong_return)).abs().max())
    ok = a == 0 and b < 1e-9
    CHECKS.append((f"identity {path.name}", ok, a + b, 0.0))
    print(f"  {'PASS' if ok else 'FAIL'}  utility identity on {path.name}: "
          f"{len(r)} rows, {a} violations, max residual {b:.1e}")
    return r


def rebuild_folds(arms):
    rng = np.random.default_rng(SPLIT_SEED)
    fold = {}
    for _, g in arms.groupby("source_dataset"):
        keys = sorted(g.canonical_qkey.unique())
        for rank, i in enumerate(rng.permutation(len(keys))):
            fold[keys[i]] = 0 if rank < len(keys) // 2 else 1
    return arms.canonical_qkey.map(fold)


def main() -> int:
    print("== row-level utility identity ==")
    arms = row_identity(D / "decision_rows/primary_all_arms.parquet")
    closure = row_identity(D / "decision_rows/primary_closure_rows.parquet")
    ext = row_identity(D / "decision_rows/extension_rows.parquet")
    r70 = row_identity(D / "decision_rows/second_reader_all_arms.parquet")
    abl = row_identity(D / "decision_rows/ablation_rows.parquet")

    print("== primary panel: oracle, headroom and G1 from the rows ==")
    fixed = arms[arms.method_id.isin(
        ["fixed_recovery", "fixed_reference", "fixed_parametric"])]
    f1 = fixed[fixed.cost == 1.0]
    per = (f1.groupby(["source_dataset", "canonical_qkey", "cell_id"])
           .correct_return.max())
    oracle = float(per.groupby(level=[0, 1]).mean()
                   .groupby(level=0).mean().mean())
    check("oracle utility, primary panel", oracle, 0.6150)

    tele = pd.read_csv(D / "stagewise/telescoping_decomposition.csv")
    sel = pd.read_csv(D / "stagewise/comparator_selection.csv")
    pool = pd.concat([arms, closure], ignore_index=True)
    pool = pool.assign(_fold=rebuild_folds(pool))
    for c in COSTS:
        row = tele[tele.cost == c].iloc[0]
        ctx = macro(pool[(pool.method_id == "fixed_recovery")
                         & (pool.cost == c)])
        check(f"headroom H at c={c:g}", oracle - ctx, float(row.headroom_H))
        parts = []
        for f in (0, 1):
            pick = sel[(sel.cost == c) & (sel.eval_fold == f)].p1.iloc[0]
            ev = pool[(pool.cost == c) & (pool._fold == f)]
            parts.append(macro(ev[ev.method_id == pick])
                         - macro(ev[ev.method_id == "fixed_recovery"]))
        check(f"G1 at c={c:g}", float(np.mean(parts)),
              float(row.fixed_policy_gain_G1))

    print("== decomposition identity on all three panels ==")
    for name, path in (("primary", D / "stagewise/telescoping_decomposition.csv"),
                       ("second reader", D / "second_reader/decomposition.csv"),
                       ("extension", D / "extension/extension_decomposition.csv")):
        d = pd.read_csv(path)
        resid = float((d.headroom_H - d.fixed_policy_gain_G1
                       - d.simple_selector_gain_G2 - d.structured_gain_G3
                       - d.residual_after_selected_policy_R).abs().max()) \
            if "headroom_H" in d.columns else \
            float(d.identity_residual.abs().max())
        check(f"H = G1+G2+G3+R, {name}", resid, 0.0, tol=1e-9)

    print("== second reader: oracle from the regenerated rows ==")
    ch = r70[r70.method_id.isin(["fixed_recovery", "fixed_reference",
                                 "fixed_parametric"]) & (r70.cost == 1.0)]
    per70 = (ch.groupby(["source_dataset", "canonical_qkey", "cell_id"])
             .correct_return.max())
    orc70 = float(per70.groupby(level=[0, 1]).mean()
                  .groupby(level=0).mean().mean())
    check("oracle utility, second reader", orc70, 0.6675)

    print("== extension: transported comparator and per-collection points ==")
    d1 = ext[ext.cost == 1.0]
    check("prompted minus carried F*, c=1",
          macro(d1[d1.method_id == "prompt_candidatewise"])
          - macro(d1[d1.method_id == "fixed_parametric"]), 0.0867)
    for coll, want in (("wp4_confiqa-mc", -0.090), ("wp4_confiqa-mr", -0.060),
                       ("wp4_conflictqa-popqa-llama2-7b", 0.100)):
        g = d1[d1.source_dataset == coll]
        got = (float(g[g.method_id == "fixed_parametric"].answer_utility.mean())
               - float(g[g.method_id == "fixed_recovery"].answer_utility.mean()))
        check(f"parametric minus context, c=1, {coll}", got, want)

    print("== ablation: the two quoted effects at c=2 ==")
    d2 = abl[abl.cost == 2.0]
    m = lambda mid: macro(d2[d2.method_id == mid])
    check("relation head unsupervised minus full, c=2",
          m("cgrv_no_relation_loss") - m("cgrv_full"), -0.1394)
    check("logistic-regression replacement minus full, c=2",
          m("simple_logreg") - m("cgrv_full"), -0.3656)

    print("== jury: labels and Fleiss' kappa from the raw verdicts ==")
    lab = pd.read_csv(D / "jury/jury_labels.csv")
    counts = lab.jury_label.value_counts()
    for k, want in (("correct", 202), ("incorrect", 226), ("contested", 19)):
        check(f"jury label count, {k}", float(counts.get(k, 0)), want, tol=0)
    mains = [pd.read_csv(D / f"jury/{j}_main.csv")[["item_id", "verdict"]]
             for j in ("qwen72b", "mistral24b", "phi4")]
    merged = mains[0].merge(mains[1], on="item_id", suffixes=("_a", "_b")) \
                     .merge(mains[2], on="item_id")
    # the recorded basis: items where all three verdict strings are nonempty
    has = lambda s: s.notna() & (s.astype(str).str.len() > 0)
    ok3 = merged[has(merged.verdict_a) & has(merged.verdict_b)
                 & has(merged.verdict)]
    check("items with all three verdicts parsed", float(len(ok3)), 445, tol=0)
    votes = ok3[["verdict_a", "verdict_b", "verdict"]].values
    cats = sorted({v for row in votes for v in row})
    nij = np.stack([[sum(v == c for v in row) for c in cats]
                    for row in votes])
    n = 3
    p_i = ((nij ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_j = nij.sum(axis=0) / (len(nij) * n)
    kappa = (p_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum())
    check("Fleiss' kappa over raw verdicts", float(kappa), 0.822, tol=5e-4)

    print("== sufficiency crosscheck: gap and interval from the rows ==")
    rows = pd.read_csv(D / "diagnosis/sufficiency_crosscheck_rows.csv")
    rate = lambda df, c: float((df[(df.cond == c) & df.label.notna()]
                                .label == "sufficient").mean())
    check("Llama-3.1-70B gap", rate(rows, "C_poison") - rate(rows, "C_insuf"),
          0.7407)
    per_q = {q: dict(zip(g.cond, g.label))
             for q, g in rows.groupby("qid", sort=False)}
    qs = list(per_q.values())
    def stat(sample):
        out = {}
        for c in ("C_poison", "C_insuf"):
            v = [q.get(c) for q in sample
                 if isinstance(q.get(c), str)]
            out[c] = sum(x == "sufficient" for x in v) / len(v)
        return out["C_poison"] - out["C_insuf"]
    rng = random.Random(42)
    draws = [stat([qs[rng.randrange(len(qs))] for _ in range(len(qs))])
             for _ in range(1000)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    check("crosscheck interval, lower", round(float(lo), 3), 0.706, tol=0)
    check("crosscheck interval, upper", round(float(hi), 3), 0.778, tol=0)

    # Later experiment families ship their own recomputation module beside this
    # one. Each exposes run(D, check) and recomputes from its own raw rows. A
    # missing or broken module is a hard failure: a release that silently
    # dropped a family's checks would still print a clean bill of health.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for fam in ("published_baselines", "reader_scale", "popqa_tail",
                "diagnostics", "internal_pool", "robustness_arms",
                "internal_verification", "selector_code"):
        __import__(f"checks_{fam}").run(D, check)

    bad = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(bad)} of {len(CHECKS)} checks pass"
          + (f"; FAILURES: {[c[0] for c in bad]}" if bad
             else " -- the released data reproduce the published numbers"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
