"""Recompute the three-reader ladder audit from the released decision rows.

Everything below is derived from row-level decisions. No summary CSV in this
family is read: the ladder (oracle, H, G1, G2, G3, R), the exact policy each
nested class selects on each fold, and the channel-level reader comparison are
all rebuilt from the parquets, with the two question-keyed folds regenerated
from the recorded split seed 20260811 and the P1/P2/P3 registries assembled
from the shipped rows exactly as the pipeline assembles them. Deleting every
CSV under data/reader_scale/ leaves these checks passing.

The Llama-3.1-8B block reads only data/reader_scale/llama8_ladder/. The
Qwen2.5-72B and Llama-3.1-70B rows of the channel comparison are recomputed
from the release's existing primary and second-reader decision rows under
data/decision_rows/, so the three-reader table is checked end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COSTS = (0.5, 1.0, 2.0, 4.0)
SPLIT_SEED = 20260811          # the primary panel's partition seed, reused
CONTEXT = "fixed_recovery"
CHANNELS = ("fixed_recovery", "fixed_reference", "fixed_parametric")
BASE = ["fixed_recovery", "fixed_reference", "fixed_parametric",
        "always_abstain"]
SIMPLE_BASE = ["gbdt_recovery", "prompt_candidatewise", "p_true_gate"]
STRUCTURED = "cgrv_recovery"
KEEP = ["source_dataset", "canonical_qkey", "cell_id", "method_id", "seed_id",
        "cost", "returned", "correct_return", "wrong_return", "answer_utility"]

# the stage-1 member each (cost, eval fold) cell selects under Llama-3.1-70B,
# as recorded in llama70_exact_selections.csv; re-derived here from the rows
LLAMA70_P1 = ("closure_agree_any_else_P", "closure_agree_any_else_P",
              "closure_agree_any_else_P", "closure_agree_any_else_P",
              "closure_agree_any", "closure_agree_any",
              "closure_agree_any", "closure_agree_CP")


def run(D, check):
    R = D / "reader_scale"
    L = R / "llama8_ladder"

    def rows(path):
        return pd.read_parquet(path, columns=KEEP)

    def macro(df):
        p = (df.groupby(["seed_id", "source_dataset", "canonical_qkey"])
             .answer_utility.mean())
        return float(p.groupby(["seed_id", "source_dataset"]).mean()
                     .groupby("seed_id").mean().mean())

    def identity(name, path):
        r = pd.read_parquet(path)
        bad = int((r.correct_return + r.wrong_return != r.returned).sum())
        res = float((r.answer_utility
                     - (r.correct_return - r.cost * r.wrong_return))
                    .abs().max())
        check(f"utility identity, {name} ({len(r)} rows)", bad + res, 0.0,
              tol=1e-9)

    def oracle_of(dec):
        """The ceiling of this reader's own candidate set, at cost 1."""
        ch = dec[dec.method_id.isin(CHANNELS) & (dec.cost == 1.0)]
        per = (ch.groupby(["source_dataset", "canonical_qkey", "cell_id"])
               .correct_return.max())
        return float(per.groupby(level=[0, 1]).mean()
                     .groupby(level=0).mean().mean())

    def rebuild_folds(dec):
        rng = np.random.default_rng(SPLIT_SEED)
        f = {}
        for _, g in dec.groupby("source_dataset"):
            keys = sorted(g.canonical_qkey.unique())
            for rank, i in enumerate(rng.permutation(len(keys))):
                f[keys[i]] = 0 if rank < len(keys) // 2 else 1
        return dec.canonical_qkey.map(f)

    def ladder(dec, cls, oracle):
        """Cross-fitted stage gains: select on one fold, score on the other."""
        d = dec.assign(_fold=rebuild_folds(dec))
        out = []
        for c in COSTS:
            um = {}
            for f in (0, 1):
                s = d[(d.cost == c) & (d._fold == f)]
                um[f] = (s.groupby(["method_id", "seed_id", "source_dataset",
                                    "canonical_qkey"]).answer_utility.mean()
                         .groupby(level=[0, 1, 2]).mean()
                         .groupby(level=[0, 1]).mean()
                         .groupby(level=0).mean().to_dict())
            parts = {k: [] for k in ("F", "p1", "p2", "p3")}
            picks = {k: [] for k in ("p1", "p2", "p3")}
            for f in (0, 1):
                sel, ev = um[1 - f], um[f]
                for k in ("p1", "p2", "p3"):
                    m = max(cls[k], key=lambda z: sel.get(z, -np.inf))
                    picks[k].append(m)
                    parts[k].append(ev.get(m, -np.inf))
                parts["F"].append(ev[CONTEXT])
            uF, u1, u2, u3 = (float(np.mean(parts[k]))
                              for k in ("F", "p1", "p2", "p3"))
            out.append({"cost": c, "H": oracle - uF, "G1": u1 - uF,
                        "G2": u2 - u1, "G3": u3 - u2, "R": oracle - u3,
                        "picks": picks, "held": parts})
        return out

    def agreement(closure_rows):
        """How often any two candidate channels agree, and how often that is
        right; the closure arm that answers exactly on agreement carries it."""
        g = closure_rows[(closure_rows.method_id == "closure_agree_any")
                         & (closure_rows.cost == 1.0)]
        return (float(g.returned.mean()),
                float(g[g.returned == 1].correct_return.mean()))

    def given_available(arms, method):
        a = arms[(arms.method_id == method) & (arms.cost == 1.0)]
        return float(a[a.returned == 1].correct_return.mean())

    # ---------------------------------------------------------------- rows
    print("== reader_scale: row-level utility identity on the 8B ladder ==")
    for name, path in (("panel_all_arms", L / "panel_all_arms.parquet"),
                       ("closure_decision_rows",
                        L / "closure_decision_rows.parquet"),
                       ("trustmargin_decision_rows",
                        L / "trustmargin_decision_rows.parquet")):
        identity(name, path)

    arms8 = rows(L / "panel_all_arms.parquet")
    clo8 = rows(L / "closure_decision_rows.parquet")
    tm8 = rows(L / "trustmargin_decision_rows.parquet")
    dec8 = pd.concat([arms8, clo8, tm8], ignore_index=True)

    # --------------------------------------------- 8B oracle and registries
    print("== reader_scale: Llama-3.1-8B oracle and nested policy classes ==")
    oracle8 = oracle_of(dec8)
    check("oracle utility, Llama-3.1-8B", oracle8, 0.5875)

    P1 = BASE + sorted(clo8.method_id.unique())
    P2 = P1 + SIMPLE_BASE + sorted(tm8.method_id.unique())
    P3 = P2 + [STRUCTURED]
    have = set(dec8.method_id)
    P1, P2, P3 = ([m for m in p if m in have] for p in (P1, P2, P3))
    check("registry size, P1 deterministic", float(len(P1)), 23, tol=0)
    check("registry size, P2 through simple", float(len(P2)), 32, tol=0)
    check("registry size, P3 through structured", float(len(P3)), 33, tol=0)

    # --------------------------------------------------- 8B stagewise ladder
    print("== reader_scale: Llama-3.1-8B cross-fitted stagewise ladder ==")
    lad = ladder(dec8, {"p1": P1, "p2": P2, "p3": P3}, oracle8)
    for r, h in zip(lad, (0.4950, 0.7850, 1.3650, 2.5250)):
        check(f"headroom H at c={r['cost']:g}, 8B", r["H"], h)
    for r, g in zip(lad, (0.1255, 0.3275, 0.8575, 1.9100)):
        check(f"G1 fixed-policy gain at c={r['cost']:g}, 8B", r["G1"], g)
    for r, g in zip(lad, (0.035156, 0.0, 0.0, 0.0)):
        check(f"G2 simple-selector gain at c={r['cost']:g}, 8B", r["G2"], g)
    check("G3 structured gain, largest over the four costs",
          max(abs(r["G3"]) for r in lad), 0.0, tol=1e-12)
    check("H = G1+G2+G3+R, 8B, largest residual over costs",
          max(abs(r["H"] - (r["G1"] + r["G2"] + r["G3"] + r["R"]))
              for r in lad), 0.0, tol=1e-12)

    # ------------------------------------------------- exact selections, 8B
    print("== reader_scale: exact policy selections in the nested classes ==")
    cells = [(r["picks"]["p1"][f], r["picks"]["p2"][f], r["picks"]["p3"][f])
             for r in lad for f in (0, 1)]
    check("8B cost/fold cells where P1 == P2 (of 8)",
          float(sum(a == b for a, b, _ in cells)), 7, tol=0)
    check("8B cost/fold cells where P2 == P3 (of 8)",
          float(sum(b == c for _, b, c in cells)), 8, tol=0)
    check("8B cost/fold cells where P1 == P2 == P3 (of 8)",
          float(sum(a == b == c for a, b, c in cells)), 7, tol=0)
    c05 = lad[0]
    check("8B c=0.5 fold 0, P1 selects closure_cascade_RC",
          float(c05["picks"]["p1"][0] == "closure_cascade_RC"), 1.0, tol=0)
    check("8B c=0.5 fold 0, P2 selects trustmargin_tau0",
          float(c05["picks"]["p2"][0] == "trustmargin_tau0"), 1.0, tol=0)
    check("8B c=0.5 fold 0, P1 held-out utility",
          c05["held"]["p1"][0], 0.213437)
    check("8B c=0.5 fold 0, P2 held-out utility",
          c05["held"]["p2"][0], 0.283750)
    check("8B c=0.5 fold 0, P2 minus P1 out of fold",
          c05["held"]["p2"][0] - c05["held"]["p1"][0], 0.070313)
    check("8B c=0.5 fold 1, P2 minus P1 out of fold",
          c05["held"]["p2"][1] - c05["held"]["p1"][1], 0.0, tol=1e-12)

    # ------------------------------------------------ exact selections, 70B
    arms70 = rows(D / "decision_rows/second_reader_all_arms.parquet")
    clo70 = rows(D / "decision_rows/second_reader_closure_rows.parquet")
    tm70 = rows(D / "decision_rows/second_reader_trustmargin_rows.parquet")
    dec70 = pd.concat([arms70, clo70, tm70], ignore_index=True)
    Q1 = [m for m in BASE + sorted(clo70.method_id.unique())
          if m in set(dec70.method_id)]
    Q2 = [m for m in Q1 + SIMPLE_BASE + sorted(tm70.method_id.unique())
          if m in set(dec70.method_id)]
    Q3 = [m for m in Q2 + [STRUCTURED] if m in set(dec70.method_id)]
    check("registry size, P1 deterministic, 70B", float(len(Q1)), 23, tol=0)
    check("registry size, P2 simple selectors, 70B", float(len(Q2)), 32, tol=0)
    check("registry size, P3 structured, 70B", float(len(Q3)), 33, tol=0)
    l70 = ladder(dec70, {"p1": Q1, "p2": Q2, "p3": Q3}, oracle_of(dec70))
    got70 = tuple(r["picks"]["p1"][f] for r in l70 for f in (0, 1))
    check("70B stage-1 selections matching the recorded table (of 8)",
          float(sum(a == b for a, b in zip(got70, LLAMA70_P1))), 8, tol=0)
    for r, g in zip(l70, (0.2800, 0.4550, 0.8650, 1.8075)):
        check(f"G1 fixed-policy gain at c={r['cost']:g}, 70B", r["G1"], g)
    # TASK A of the three-reader report: under the 70B the three nested classes
    # select the SAME member at every cost and fold, checked by policy id and
    # not inferred from the gains being zero. 8 cost/fold cells per comparison.
    check("70B cells where P2 selects P1's member (of 8)",
          float(sum(r["picks"]["p1"][f] == r["picks"]["p2"][f]
                    for r in l70 for f in (0, 1))), 8, tol=0)
    check("70B cells where P3 selects P2's member (of 8)",
          float(sum(r["picks"]["p2"][f] == r["picks"]["p3"][f]
                    for r in l70 for f in (0, 1))), 8, tol=0)

    # ------------------------------------------------- exact selections, Qwen
    # The main reader's block is the one place where the deterministic class and
    # the wider class part company: at c=0.5 one fold's P2 takes the
    # likelihood-margin policy and loses on the opposite fold, which is the whole
    # content of the paper's negative G2 there. Reproducing it needs the primary
    # panel's margin arms, so they are released alongside the closure rows.
    print("== reader_scale: main-reader nested classes and the negative G2 ==")
    armsQ0 = rows(D / "decision_rows/primary_all_arms.parquet")
    cloQ0 = rows(D / "decision_rows/primary_closure_rows.parquet")
    tmQ0 = rows(D / "decision_rows/primary_trustmargin_rows.parquet")
    decQ = pd.concat([armsQ0, cloQ0, tmQ0], ignore_index=True)
    W1 = [m for m in BASE + sorted(cloQ0.method_id.unique())
          if m in set(decQ.method_id)]
    W2 = [m for m in W1 + SIMPLE_BASE + sorted(tmQ0.method_id.unique())
          if m in set(decQ.method_id)]
    W3 = [m for m in W2 + [STRUCTURED] if m in set(decQ.method_id)]
    check("registry size, P1 deterministic, Qwen", float(len(W1)), 23, tol=0)
    check("registry size, P2 simple selectors, Qwen", float(len(W2)), 32, tol=0)
    check("registry size, P3 structured, Qwen", float(len(W3)), 33, tol=0)
    lQ = ladder(decQ, {"p1": W1, "p2": W2, "p3": W3}, oracle_of(decQ))
    for r, g in zip(lQ, (-0.0256, 0.0, 0.0, 0.0)):
        check(f"G2 simple-selector gain at c={r['cost']:g}, Qwen", r["G2"], g)
    c05 = lQ[0]
    check("Qwen c=0.5 fold 0 selects a margin policy in P2 but not in P1",
          float(c05["picks"]["p1"][0] != c05["picks"]["p2"][0]
                and c05["picks"]["p2"][0].startswith("trustmargin")), 1.0, tol=0)
    check("Qwen c=0.5 fold 1 selects the same member in P1 and P2",
          float(c05["picks"]["p1"][1] == c05["picks"]["p2"][1]), 1.0, tol=0)
    check("Qwen cells where P3 selects P2's member (of 8)",
          float(sum(r["picks"]["p2"][f] == r["picks"]["p3"][f]
                    for r in lQ for f in (0, 1))), 8, tol=0)

    check("Qwen ladder identity, largest residual over the four costs",
          max(abs(r["H"] - (r["G1"] + r["G2"] + r["G3"] + r["R"])) for r in lQ),
          0.0, tol=1e-12)
    for r in l70:
        check(f"learned increment L = G2+G3 at c={r['cost']:g}, 70B",
              r["G2"] + r["G3"], 0.0, tol=1e-12)

    # ------------------------------------------------- channel-level reader
    print("== reader_scale: channel-level comparison across three readers ==")
    armsQ = rows(D / "decision_rows/primary_all_arms.parquet")
    cloQ = rows(D / "decision_rows/primary_closure_rows.parquet")
    table = (("Qwen2.5-72B", armsQ, cloQ, 0.2250, 0.2600, -0.0350,
              0.3550, 0.845),
             ("Llama-3.1-70B", arms70, clo70, 0.2150, 0.2850, -0.0700,
              0.3900, 0.814),
             ("Llama-3.1-8B", arms8, clo8, 0.0900, 0.1300, -0.0400,
              0.2900, 0.759))
    for name, arms, clo, wR, wP, wD, wAgr, wAcc in table:
        d1 = arms[arms.cost == 1.0]
        uR = macro(d1[d1.method_id == "fixed_reference"])
        uP = macro(d1[d1.method_id == "fixed_parametric"])
        check(f"U(a_R) at c=1, {name}", uR, wR)
        check(f"U(a_P) at c=1, {name}", uP, wP)
        check(f"a_R minus a_P at c=1, {name}", uR - uP, wD)
        agr, acc = agreement(clo)
        check(f"P(any two channels agree), {name}", agr, wAgr)
        check(f"correct given agreement, {name}", acc, wAcc)
    for name, arms, wR, wP in (("Qwen2.5-72B", armsQ, 0.706, 0.745),
                               ("Llama-3.1-8B", arms8, 0.576, 0.616)):
        check(f"a_R correct given available, {name}",
              given_available(arms, "fixed_reference"), wR)
        check(f"a_P correct given available, {name}",
              given_available(arms, "fixed_parametric"), wP)
