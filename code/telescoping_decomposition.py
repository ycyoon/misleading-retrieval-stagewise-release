"""A stagewise decomposition over nested policy classes, with cross-fitting.

Two defects in the earlier version are fixed here.

**The residual was not what it was called.** With A = U(M) - U(S) negative at
three of the four costs, the simple selector beats the structured one there, so
U(O) - U(M) counts value a policy we evaluated had already recovered. Calling it
"the part no evaluated policy recovers" was wrong, and the "roughly two thirds"
figure derived from it was wrong at every cost (the ratio runs 98.5%, 90.1%,
54.5%, 29.8%). The stages are now nested policy classes,

    P1 = fixed policies (three channel rules and always abstain)
    P2 = P1 + simple selectors
    P3 = P2 + the structured selector

with the best member of each class chosen on development questions. Each gain is
what widening the class buys, and the residual is what the finally selected
policy leaves on the table. A negative gain is meaningful: it says a wider class
was selected on development data and failed to carry.

**One arbitrary half-split decided everything.** Comparators were chosen on one
half and reported on the other, so a single split drove the headline and half the
questions never entered the estimate. Selection and evaluation are now
cross-fitted: each fold selects on the other, and every question contributes an
out-of-fold utility.

The identity H = G1 + G2 + G3 + R is asserted numerically at every cost.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.telescoping_decomposition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ARMS = ROOT / ("artifacts/maintrack_v1/v12_10_verification_arms/"
               "panel_all_arms.parquet")
CANDSTATS = ROOT / ("artifacts/maintrack_v1/v12_9a_panel_rederived/"
                    "candidate_statistics.json")
OUT = ROOT / "artifacts/maintrack_v1/v12_13_telescoping"
COSTS = (0.5, 1.0, 2.0, 4.0)
SPLIT_SEED = 20260811

# The fixed class is every label-free deterministic policy we can build from the
# frozen candidates, not only the three single channel rules. A cascade or an
# agreement rule fits no score and sees no label, so admitting them costs the
# learned side nothing it is entitled to and makes the first stage the comparator
# a deployer would actually reach for.
FIXED = ["fixed_recovery", "fixed_reference", "fixed_parametric",
         "always_abstain"]
CLOSURE = ROOT / ("artifacts/maintrack_v1/v12_17_fixed_closure/"
                  "closure_decision_rows.parquet")
# the closest published training-free selector, recovered by teacher-forced
# scoring of the frozen candidates; each margin threshold is its own policy so
# the cross-fitted selection chooses one on held out questions
TRUSTMARGIN = ROOT / ("artifacts/maintrack_v1/v12_19_trustmargin/"
                      "trustmargin_decision_rows.parquet")
SIMPLE = ["gbdt_recovery", "prompt_candidatewise", "p_true_gate"]
STRUCTURED = "cgrv_recovery"
CONTEXT = "fixed_recovery"


def macro(df: pd.DataFrame, col: str = "answer_utility") -> float:
    """Equal weight over collections, then over question keys, then seeds."""
    p = df.groupby(["seed_id", "source_dataset", "canonical_qkey"])[col].mean()
    return float(p.groupby(["seed_id", "source_dataset"]).mean()
                 .groupby("seed_id").mean().mean())




def stagewise_bands(dec, P1, P2, P3, oracle, draws=2000, seed=20260808):
    """Percentile bands for the stage gains, re-selecting inside each draw."""
    costs = list(COSTS)
    pols = list(dict.fromkeys(P3 + [CONTEXT]))
    colls = sorted(dec.source_dataset.unique())
    qs = {(cl, f): sorted(dec.loc[(dec.source_dataset == cl) & (dec._fold == f),
                                  "canonical_qkey"].unique())
          for cl in colls for f in (0, 1)}
    seeds = sorted(dec.seed_id.unique(), key=str)
    maxq = max(len(v) for v in qs.values())

    # arr[policy, seed, collection, fold, question, cost]
    arr = np.full((len(pols), len(seeds), len(colls), 2, maxq, len(costs)),
                  np.nan)
    pi = {m: i for i, m in enumerate(pols)}
    si = {s_: i for i, s_ in enumerate(seeds)}
    ci = {c_: i for i, c_ in enumerate(colls)}
    g = dec[dec.method_id.isin(pols)].groupby(
        ["method_id", "seed_id", "source_dataset", "_fold", "canonical_qkey",
         "cost"], sort=False).answer_utility.mean()
    for (m, s_, cl, f, q, c), v in g.items():
        arr[pi[m], si[s_], ci[cl], int(f), qs[(cl, int(f))].index(q),
            costs.index(float(c))] = v

    idx = {"p1": [pi[m] for m in P1], "p2": [pi[m] for m in P2],
           "p3": [pi[m] for m in P3]}
    rng = np.random.default_rng(seed)
    out = {(t, c): [] for t in ("G1", "G2", "G3", "R", "H") for c in costs}
    for _ in range(draws):
        six = rng.integers(0, len(seeds), len(seeds))
        qix = {k: rng.integers(0, len(v), len(v)) for k, v in qs.items()}
        # mean over resampled seeds and questions, per collection and fold
        mu = np.full((len(pols), len(colls), 2, len(costs)), np.nan)
        for cl in colls:
            for f in (0, 1):
                sub = arr[:, six][:, :, ci[cl], f][:, :, qix[(cl, f)], :]
                mu[:, ci[cl], f, :] = np.nanmean(sub, axis=(1, 2))
        macro_ = np.nanmean(mu, axis=1)          # [policy, fold, cost]
        for j, c in enumerate(costs):
            vals = {}
            for f in (0, 1):
                sel, ev = macro_[:, 1 - f, j], macro_[:, f, j]
                for k in ("p1", "p2", "p3"):
                    best = idx[k][int(np.nanargmax(sel[idx[k]]))]
                    vals.setdefault(k, []).append(ev[best])
                vals.setdefault("F", []).append(ev[pi[CONTEXT]])
            uF, u1, u2, u3 = (float(np.mean(vals[k]))
                              for k in ("F", "p1", "p2", "p3"))
            out[("H", c)].append(oracle - uF)
            out[("G1", c)].append(u1 - uF)
            out[("G2", c)].append(u2 - u1)
            out[("G3", c)].append(u3 - u2)
            out[("R", c)].append(oracle - u3)

    rows = []
    for (t, c), v in out.items():
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({"term": t, "cost": c, "point": float(np.mean(v)),
                     "lo": float(lo), "hi": float(hi), "draws": draws,
                     "bootstrap_seed": seed,
                     "selection_repeated_in_draw": True,
                     "crosses_zero": bool(lo * hi <= 0)})
    band = pd.DataFrame(rows).sort_values(["cost", "term"])

    # A one-sided simultaneous upper band over the eight cells the headline
    # actually quantifies over, the two learned gains crossed with the four
    # costs. The per-cell intervals above are not a family, and the paper had
    # been making a family-wide statement from them; this is the band that
    # statement needs. One-sided because the claim is only that neither gain is
    # positive, and max-T over the studentised draws so the correction is the
    # same machinery the panel contrasts use.
    fam = [(t, c) for t in ("G2", "G3") for c in costs]
    mat = np.column_stack([np.asarray(out[k]) for k in fam])
    pt = mat.mean(axis=0)
    se = np.maximum(mat.std(axis=0, ddof=1), 1e-12)
    crit = float(np.percentile(((mat - pt) / se).max(axis=1), 95))
    sim = pd.DataFrame([{"term": t, "cost": c, "point": float(pt[i]),
                         "simultaneous_hi": float(pt[i] + crit * se[i]),
                         "maxT_critical": crit, "family": "G2,G3 x 4 costs",
                         "sided": "upper", "draws": draws}
                        for i, (t, c) in enumerate(fam)])
    return band, sim



def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=str(ARMS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260808)
    a = ap.parse_args(argv)

    dec = pd.read_parquet(a.arms)
    if CLOSURE.exists():
        extra = pd.read_parquet(CLOSURE)
        dec = pd.concat([dec, extra], ignore_index=True)
        FIXED.extend(sorted(extra.method_id.unique()))
        print(f"[stagewise] fixed class widened to {len(FIXED)} label-free "
              f"deterministic policies")
    if TRUSTMARGIN.exists():
        tm = pd.read_parquet(TRUSTMARGIN)
        dec = pd.concat([dec, tm], ignore_index=True)
        SIMPLE.extend(sorted(tm.method_id.unique()))
        print(f"[stagewise] simple class now includes {tm.method_id.nunique()} "
              f"likelihood-margin thresholds")
    oracle = json.loads(CANDSTATS.read_text())["candidate_headroom"]

    # two folds of question keys, stratified by collection
    rng = np.random.default_rng(a.split_seed)
    fold = {}
    for coll, g in dec.groupby("source_dataset"):
        keys = sorted(g.canonical_qkey.unique())
        perm = rng.permutation(len(keys))
        for r, i_ in enumerate(perm):
            fold[keys[i_]] = 0 if r < len(keys) // 2 else 1
    dec = dec.assign(_fold=dec.canonical_qkey.map(fold))
    print(f"[tele] {dec.canonical_qkey.nunique()} question keys; "
          f"fold sizes {dec.groupby('_fold').canonical_qkey.nunique().to_dict()}")

    P1 = [m for m in FIXED if m in set(dec.method_id)]
    P2 = P1 + [m for m in SIMPLE if m in set(dec.method_id)]
    P3 = P2 + ([STRUCTURED] if STRUCTURED in set(dec.method_id) else [])

    rows, sel_log, per_fold = [], [], {}
    for c in COSTS:
        # cross-fitting: select on the other fold, score on this one
        parts = {k: [] for k in ("F", "p1", "p2", "p3")}
        for f in (0, 1):
            sel = dec[(dec.cost == c) & (dec._fold != f)]
            ev = dec[(dec.cost == c) & (dec._fold == f)]

            def u(frame, m):
                x = frame[frame.method_id == m]
                return macro(x) if len(x) else float("-inf")

            pick = {"p1": max(P1, key=lambda m: u(sel, m)),
                    "p2": max(P2, key=lambda m: u(sel, m)),
                    "p3": max(P3, key=lambda m: u(sel, m))}
            sel_log.append({"cost": c, "eval_fold": f, **pick})
            for k, m in pick.items():
                parts[k].append(u(ev, m))
            parts["F"].append(u(ev, CONTEXT))

        # equal weight over folds; each question contributes out of fold
        uF, u1, u2, u3 = (float(np.mean(parts[k])) for k in ("F", "p1", "p2", "p3"))
        H = oracle - uF
        G1, G2, G3 = u1 - uF, u2 - u1, u3 - u2
        R = oracle - u3
        rows.append({
            "cost": c, "oracle_utility": oracle, "U_context": uF,
            "U_stage1_fixed": u1, "U_stage2_simple": u2,
            "U_stage3_structured": u3,
            "headroom_H": H, "fixed_policy_gain_G1": G1,
            "simple_selector_gain_G2": G2, "structured_gain_G3": G3,
            "residual_after_selected_policy_R": R,
            "sum_G1_G2_G3_R": G1 + G2 + G3 + R,
            "identity_residual": H - (G1 + G2 + G3 + R),
            "stage1_choice": "|".join(x["p1"] for x in sel_log if x["cost"] == c),
            "stage2_choice": "|".join(x["p2"] for x in sel_log if x["cost"] == c),
            "stage3_choice": "|".join(x["p3"] for x in sel_log if x["cost"] == c),
        })

    res = pd.DataFrame(rows)
    worst = float(np.max(np.abs(res.identity_residual)))
    if worst > 1e-9:
        raise SystemExit(f"identity violated, max residual {worst}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "telescoping_decomposition.csv", index=False)
    pd.DataFrame(sel_log).to_csv(out / "comparator_selection.csv", index=False)

    # Bands that carry the uncertainty of the selection itself. Each draw
    # resamples the training runs and, within each collection and fold, the
    # question keys; the stage member is then chosen again on the resampled
    # opposite fold before being scored on the resampled held-out fold. Holding
    # the policy identities fixed at their full-sample values would give an
    # interval conditional on a choice the data made, which is not the quantity
    # the decomposition reports.
    bands, sim = stagewise_bands(dec, P1, P2, P3, oracle, draws=a.draws,
                                 seed=a.bootstrap_seed)
    bands.to_csv(out / "stagewise_bands.csv", index=False)
    sim.to_csv(out / "stagewise_simultaneous_upper.csv", index=False)
    worst_hi = float(sim.simultaneous_hi.max())
    print(f"\none-sided simultaneous upper band over {len(sim)} cells "
          f"(G2 and G3 x four costs), max-T critical "
          f"{sim.maxT_critical.iloc[0]:.3f}")
    for r in sim.itertuples(index=False):
        print(f"  c={r.cost:<4} {r.term:<3} {r.point:+.4f}  upper "
              f"{r.simultaneous_hi:+.4f}")
    print(f"  largest simultaneous upper bound {worst_hi:+.4f}")
    print("\nstagewise bands, selection repeated inside every draw")
    for r in bands.itertuples(index=False):
        print(f"  c={r.cost:<4} {r.term:<3} {r.point:+.4f} "
              f"[{r.lo:+.4f}, {r.hi:+.4f}]"
              + ("" if r.crosses_zero else "  excludes zero"))
    (out / "manifest.json").write_text(json.dumps({
        "arms": str(Path(a.arms).relative_to(ROOT)),
        "candidate_statistics": str(CANDSTATS.relative_to(ROOT)),
        "oracle_utility": oracle,
        "policy_classes": {"P1_fixed": P1, "P2_plus_simple": P2,
                           "P3_plus_structured": P3},
        "cross_fitting": "two question-keyed folds stratified by collection; "
                         "each fold selects on the other and is scored out of "
                         "fold, so every question enters the estimate",
        "split_seed": a.split_seed,
        "identity": "H = G1 + G2 + G3 + R, asserted numerically at every cost",
        "max_identity_residual": worst,
        "residual_meaning": "what the development-selected policy leaves "
                            "relative to the oracle; NOT what no policy recovers",
        "source_script":
            "experiments/scripts/maintrack/telescoping_decomposition.py",
    }, indent=1))

    print(f"\noracle {oracle:.4f}; identity residual max {worst:.2e}\n")
    print(f"{'cost':>5} {'H':>9} {'G1 fixed':>10} {'G2 simple':>10} "
          f"{'G3 struct':>10} {'R':>9}  stage choices (fold0|fold1)")
    for r in res.itertuples(index=False):
        print(f"{r.cost:>5} {r.headroom_H:>+9.4f} {r.fixed_policy_gain_G1:>+10.4f} "
              f"{r.simple_selector_gain_G2:>+10.4f} {r.structured_gain_G3:>+10.4f} "
              f"{r.residual_after_selected_policy_R:>+9.4f}  "
              f"{r.stage1_choice} / {r.stage2_choice} / {r.stage3_choice}")
    print(f"\nR/H: " + ", ".join(f"c={r.cost}: {r.residual_after_selected_policy_R/r.headroom_H:.1%}"
                                 for r in res.itertuples(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
