"""popqa_tail: the popularity-stratified boundary-condition experiment.

Recomputes, from the released decision rows and per-cell tables alone, every
number the manuscript quotes for the PopQA head / long-tail audit: the two
oracles, headroom and the fixed-policy gain at all four costs, the learned
increment L = G2 + G3, the decomposition identity, the two channel utilities,
retrieval feasibility, and the retrieval-supported slice that closes the
retriever-bottleneck alternative explanation.

Nothing is read back from a summary file. The stagewise CSVs under
`popqa_tail/ladder_*/` and `popqa_tail/out/` are shipped for reference only;
delete them all and every check below still runs, because each value is
rebuilt from `panel_all_arms.parquet`, `closure_decision_rows.parquet`,
`trustmargin_decision_rows.parquet` and `out/cells_*.parquet`, with the two
question-keyed folds regenerated from the recorded split seed 20260811 and
the 23/32/33 policy registries rebuilt from the method identifiers present in
the rows.

The retrieved passages are withheld under the release's redaction policy, so
retrieval feasibility is recomputed from the per-cell `support_top{1,3,10}`
flags derived from them before redaction; those flags' companion columns
(`ok_a_*`, `av_a_*`) are checked against the decision rows first, so the
per-cell table is not taken on trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COSTS = (0.5, 1.0, 2.0, 4.0)
SPLIT_SEED = 20260811          # the recorded question-keyed fold seed
CONTEXT = "fixed_recovery"
CHANNEL_ARM = {"a_C": "fixed_recovery", "a_R": "fixed_reference",
               "a_P": "fixed_parametric"}
SIMPLE_BASE = ["gbdt_recovery", "prompt_candidatewise", "p_true_gate"]
STRUCTURED = "cgrv_recovery"
STRATA = ("head", "tail")
# the decision-row columns every statistic below needs
KEEP = ["source_dataset", "canonical_qkey", "cell_id", "method_id", "seed_id",
        "cost", "returned", "correct_return", "wrong_return", "answer_utility"]


def _macro(df, col="answer_utility"):
    """House macro: cells within question, questions within collection, seeds."""
    p = df.groupby(["seed_id", "source_dataset", "canonical_qkey"])[col].mean()
    return float(p.groupby(["seed_id", "source_dataset"]).mean()
                 .groupby("seed_id").mean().mean())


def _folds(dec):
    """Rebuild the two question-keyed folds from the recorded split seed."""
    rng = np.random.default_rng(SPLIT_SEED)
    fold = {}
    for _, g in dec.groupby("source_dataset"):
        keys = sorted(g.canonical_qkey.unique())
        for r, i in enumerate(rng.permutation(len(keys))):
            fold[keys[i]] = 0 if r < len(keys) // 2 else 1
    return dec.canonical_qkey.map(fold)


def _macro_table(d, col="answer_utility"):
    """macro(method, cost, fold) for every arm in one pass; same nesting."""
    g = d.groupby(["method_id", "cost", "_fold", "seed_id", "source_dataset",
                   "canonical_qkey"], sort=False)[col].mean()
    g = g.groupby(level=[0, 1, 2, 3, 4]).mean()   # questions within collection
    g = g.groupby(level=[0, 1, 2, 3]).mean()      # collections within seed
    return g.groupby(level=[0, 1, 2]).mean().to_dict()   # seeds


def _oracle(dec):
    """Best of the three channels per cell, then the house macro."""
    ch = dec[dec.method_id.isin(CHANNEL_ARM.values()) & (dec.cost == 1.0)]
    return float(ch.groupby(["source_dataset", "canonical_qkey", "cell_id"])
                 .correct_return.max().groupby(level=[0, 1]).mean()
                 .groupby(level=0).mean().mean())


def _registries(dec, closure_ids, margin_ids):
    """P1/P2/P3 exactly as the audit builds them; sizes asserted at 23/32/33."""
    p1 = ["fixed_recovery", "fixed_reference", "fixed_parametric",
          "always_abstain"] + sorted(closure_ids)
    p2 = p1 + SIMPLE_BASE + sorted(margin_ids)
    p3 = p2 + [STRUCTURED]
    have = set(dec.method_id)
    p1, p2, p3 = ([m for m in p if m in have] for p in (p1, p2, p3))
    if (len(p1), len(p2), len(p3)) != (23, 32, 33):
        raise SystemExit(f"popqa_tail registry sizes {(len(p1), len(p2), len(p3))}"
                         " != (23, 32, 33)")
    return p1, p2, p3


def _decompose(dec, p1, p2, p3, oracle):
    """Cross-fitted stage gains: select on the opposite fold, score held out."""
    d = dec.assign(_fold=_folds(dec))
    tab = _macro_table(d)
    rows = []
    for c in COSTS:
        parts = {k: [] for k in ("F", "p1", "p2", "p3")}
        for f in (0, 1):
            def u(fold, m):
                return tab.get((m, c, fold), float("-inf"))
            for k, reg in (("p1", p1), ("p2", p2), ("p3", p3)):
                # first maximum in registry order, as the canonical generator
                best = reg[int(np.argmax([u(1 - f, m) for m in reg]))]
                parts[k].append(u(f, best))
            parts["F"].append(u(f, CONTEXT))
        uF, u1, u2, u3 = (float(np.mean(parts[k]))
                          for k in ("F", "p1", "p2", "p3"))
        rows.append({"cost": c, "U_context": uF, "H": oracle - uF,
                     "G1": u1 - uF, "G2": u2 - u1, "G3": u3 - u2,
                     "L": u3 - u1, "R": oracle - u3,
                     "identity_residual": (oracle - uF)
                     - ((u1 - uF) + (u2 - u1) + (u3 - u2) + (oracle - u3))})
    return pd.DataFrame(rows).set_index("cost")


def _channel_utility(dec, arm, c):
    m = dec[(dec.method_id == arm) & (dec.cost == c)]
    return _macro(m)


def run(D, check):
    P = D / "popqa_tail"

    # ---- the rows themselves ------------------------------------------
    print("== popqa_tail: row-level utility identity on both strata ==")
    dec, cells = {}, {}
    for s in STRATA:
        parts = []
        for name in ("panel_all_arms", "closure_decision_rows",
                     "trustmargin_decision_rows"):
            r = pd.read_parquet(P / f"ladder_{s}" / f"{name}.parquet")
            viol = int((r.correct_return + r.wrong_return != r.returned).sum())
            resid = float((r.answer_utility
                           - (r.correct_return - r.cost * r.wrong_return))
                          .abs().max())
            check(f"utility identity, {s}/{name} ({len(r)} rows)",
                  viol + resid, 0.0, tol=1e-9)
            parts.append(r[KEEP])
        dec[s] = pd.concat(parts, ignore_index=True)
        cells[s] = pd.read_parquet(P / "out" / f"cells_{s}.parquet")

    # ---- the two panels, and the per-cell table grounded in the rows ---
    print("== popqa_tail: panel size and the per-cell table against the rows ==")
    for s, n_cell, n_q in (("head", 400, 199), ("tail", 400, 200)):
        c = cells[s]
        check(f"{s} panel cells", float(len(c)), n_cell, tol=0)
        check(f"{s} panel unique questions", float(c.qkey.nunique()), n_q, tol=0)
        a1 = dec[s][(dec[s].cost == 1.0)]
        bad = 0
        for ch, arm in CHANNEL_ARM.items():
            g = a1[a1.method_id == arm].groupby("cell_id")
            ok = g.correct_return.max().reindex(c.cell_id).to_numpy()
            av = g.returned.max().reindex(c.cell_id).to_numpy()
            bad += int((ok != c[f"ok_{ch}"].to_numpy()).sum())
            bad += int((av != c[f"av_{ch}"].to_numpy()).sum())
        check(f"{s} per-cell correctness/availability vs decision rows",
              float(bad), 0.0, tol=0)

    # ---- oracle, headroom, fixed-policy gain, learned increment -------
    print("== popqa_tail: oracle, headroom, G1 and L = G2+G3 from the rows ==")
    reg, res, orc = {}, {}, {}
    for s in STRATA:
        closure_ids = pd.read_parquet(
            P / f"ladder_{s}" / "closure_decision_rows.parquet").method_id.unique()
        margin_ids = pd.read_parquet(
            P / f"ladder_{s}" / "trustmargin_decision_rows.parquet").method_id.unique()
        reg[s] = _registries(dec[s], closure_ids, margin_ids)
        orc[s] = _oracle(dec[s])
        res[s] = _decompose(dec[s], *reg[s], orc[s])

    check("oracle utility, PopQA head", orc["head"], 0.71357)
    check("oracle utility, PopQA long-tail", orc["tail"], 0.4925)

    want_H = {"head": (0.5075, 0.7362, 1.1935, 2.1081),
              "tail": (0.3613, 0.5975, 1.0700, 2.0150)}
    want_G1 = {"head": (0.2437, 0.3343, 0.7415, 1.5155),
               "tail": (-0.0213, 0.1650, 0.6225, 1.5175)}
    want_L = {"head": (-0.007576, 0.0, 0.0, 0.0),
              "tail": (0.0, 0.0, 0.0, 0.0)}
    for s in STRATA:
        for j, c in enumerate(COSTS):
            r = res[s].loc[c]
            check(f"headroom H, {s}, c={c:g}", float(r.H), want_H[s][j])
            check(f"fixed-policy gain G1, {s}, c={c:g}", float(r.G1),
                  want_G1[s][j])
            check(f"learned increment L = G2+G3, {s}, c={c:g}", float(r.L),
                  want_L[s][j])
            check(f"identity H = G1+G2+G3+R, {s}, c={c:g}",
                  float(r.identity_residual), 0.0, tol=1e-9)

    # ---- the two candidate channels ------------------------------------
    print("== popqa_tail: reference and parametric channel utility at c=1 ==")
    want_ch = {("head", "a_R"): 0.3015, ("head", "a_P"): 0.2663,
               ("tail", "a_R"): -0.0300, ("tail", "a_P"): -0.0450}
    u1 = {}
    for s in STRATA:
        for ch in ("a_R", "a_P"):
            u1[(s, ch)] = _channel_utility(dec[s], CHANNEL_ARM[ch], 1.0)
            check(f"U1({ch}), {s}", u1[(s, ch)], want_ch[(s, ch)])
    check("Delta_channel = D(tail) - D(head), c=1",
          (u1[("tail", "a_R")] - u1[("tail", "a_P")])
          - (u1[("head", "a_R")] - u1[("head", "a_P")]), -0.0202)

    # ---- retrieval feasibility, from the derived per-cell support flags -
    print("== popqa_tail: retrieval feasibility from the per-cell support flags ==")
    want_sup = {"head": (0.4422, 0.6030, 0.7186),
                "tail": (0.2350, 0.3000, 0.3500)}
    for s in STRATA:
        q = cells[s].drop_duplicates("qkey")
        for j, k in enumerate((1, 3, 10)):
            check(f"gold alias in retrieved top-{k}, {s}",
                  float(q[f"support_top{k}"].mean()), want_sup[s][j])

    # ---- the retrieval-supported slice ---------------------------------
    print("== popqa_tail: retrieval-supported slice (retriever-bottleneck test) ==")
    for s, n_q, want_orc in (("head", 143, 0.8042), ("tail", 70, 0.7143)):
        keep = set(cells[s].loc[cells[s].support_top10 == 1, "qkey"])
        sub = dec[s][dec[s].canonical_qkey.isin(keep)].copy()
        check(f"supported-slice questions, {s}",
              float(sub.canonical_qkey.nunique()), n_q, tol=0)
        o = _oracle(sub)
        check(f"supported-slice oracle, {s}", o, want_orc)
        r = _decompose(sub, *reg[s], o)
        for c in COSTS:
            check(f"supported-slice L = G2+G3, {s}, c={c:g}",
                  float(r.loc[c, "L"]), 0.0, tol=1e-9)
