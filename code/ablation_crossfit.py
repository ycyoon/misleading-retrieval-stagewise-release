"""Cross-fit the architecture ablation, as the decomposition is cross-fit.

The ablation was reported on one half of the panel while the decomposition uses
both halves, each selecting on the other. Two analyses of the same panel under
two protocols invite the reading that the halves were chosen to suit, and the
rows for the other half exist, so there is no reason to leave them unused.

Each variant is scored out of fold on both halves and the two are averaged, which
is what the decomposition does. The comparison against the registered model is
then a paired difference on every question rather than on half of them, and the
bootstrap resamples the training runs and the question keys within each
collection exactly as elsewhere.

Run (from <repository root>):
  python -m \\
      experiments.scripts.maintrack.ablation_crossfit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/maintrack_v1"
ROWS = A / "v12_14_architecture_ablation/ablation_rows.parquet"
OUT = A / "v12_21_ablation_crossfit"
COSTS = (0.5, 1.0, 2.0, 4.0)
BASE = "cgrv_full"
B, BSEED, SPLIT_SEED = 10000, 20260808, 20260811


def macro(d: pd.DataFrame) -> float:
    p = d.groupby(["seed_id", "source_dataset", "canonical_qkey"]).answer_utility.mean()
    return float(p.groupby(["seed_id", "source_dataset"]).mean()
                 .groupby("seed_id").mean().mean())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    dec = pd.read_parquet(a.rows)
    variants = sorted(set(dec.method_id) - {BASE})
    print(f"[ablation] {dec.canonical_qkey.nunique()} question keys, "
          f"{len(variants)} variants against {BASE}")

    rng = np.random.default_rng(SPLIT_SEED)
    fold = {}
    for coll, g in dec.groupby("source_dataset"):
        keys = sorted(g.canonical_qkey.unique())
        perm = rng.permutation(len(keys))
        for r, i in enumerate(perm):
            fold[keys[i]] = 0 if r < len(keys) // 2 else 1
    dec = dec.assign(_fold=dec.canonical_qkey.map(fold))

    # levels, averaging the two out-of-fold halves
    levels = []
    for m in [BASE] + variants:
        for c in COSTS:
            vals = [macro(dec[(dec.method_id == m) & (dec.cost == c)
                              & (dec._fold == f)]) for f in (0, 1)]
            levels.append({"method_id": m, "cost": c,
                           "utility": float(np.mean(vals)),
                           "fold0": vals[0], "fold1": vals[1]})
    lv = pd.DataFrame(levels)

    # paired bootstrap over every question, both halves
    colls = sorted(dec.source_dataset.unique())
    q_by = {c_: sorted(dec.loc[dec.source_dataset == c_, "canonical_qkey"].unique())
            for c_ in colls}
    seeds = sorted(dec.seed_id.unique(), key=str)
    maxq = max(len(v) for v in q_by.values())
    pols = [BASE] + variants
    arr = np.full((len(pols), len(seeds), len(colls), maxq, len(COSTS)), np.nan)
    pi = {m: i for i, m in enumerate(pols)}
    si = {s_: i for i, s_ in enumerate(seeds)}
    ci = {c_: i for i, c_ in enumerate(colls)}
    for (m, s_, c_, q, c), v in dec.groupby(
            ["method_id", "seed_id", "source_dataset", "canonical_qkey",
             "cost"], sort=False).answer_utility.mean().items():
        arr[pi[m], si[s_], ci[c_], q_by[c_].index(q), COSTS.index(float(c))] = v

    def observed(m):
        return np.asarray([
            float(np.mean([np.nanmean(arr[pi[m], :, k, :, j])
                           - np.nanmean(arr[pi[BASE], :, k, :, j])
                           for k in range(len(colls))]))
            for j in range(len(COSTS))])

    pts = {m: observed(m) for m in variants}
    draws = {m: np.empty((B, len(COSTS))) for m in variants}
    rng = np.random.default_rng(BSEED)
    for z in range(B):
        six = rng.integers(0, len(seeds), len(seeds))
        tot = {m: np.zeros(len(COSTS)) for m in variants}
        for k, c_ in enumerate(colls):
            qix = rng.integers(0, len(q_by[c_]), len(q_by[c_]))
            sm = np.take(np.take(arr[:, :, k, :, :], six, axis=1), qix, axis=2)
            mu = np.nanmean(sm, axis=2)
            for m in variants:
                tot[m] += np.nanmean(mu[pi[m]] - mu[pi[BASE]], axis=0) / len(colls)
        for m in variants:
            draws[m][z] = tot[m]

    rows = []
    zs_all = []
    ses = {}
    for m in variants:
        se = draws[m].std(axis=0, ddof=1)
        ses[m] = se
        zs = np.abs((draws[m] - pts[m]) / np.maximum(se, 1e-12))
        zs_all.append(zs)
        crit = float(np.percentile(zs.max(axis=1), 95))
        for j, c in enumerate(COSTS):
            lo, hi = pts[m][j] - crit * se[j], pts[m][j] + crit * se[j]
            rows.append({"variant": m, "cost": c, "point": pts[m][j],
                         "lo": lo, "hi": hi, "maxT_critical": crit,
                         "draws": B, "bootstrap_seed": BSEED,
                         "crosses_zero": bool(lo * hi <= 0)})
    bands = pd.DataFrame(rows)

    # The same draws corrected once over all 28 cells, so the seven variants
    # are also corrected against each other rather than only within themselves.
    crit28 = float(np.percentile(np.concatenate(zs_all, axis=1).max(axis=1), 95))
    rows28 = []
    for m in variants:
        for j, c in enumerate(COSTS):
            lo = pts[m][j] - crit28 * ses[m][j]
            hi = pts[m][j] + crit28 * ses[m][j]
            rows28.append({"variant": m, "cost": c, "point": pts[m][j],
                           "lo": lo, "hi": hi, "maxT_critical": crit28,
                           "draws": B, "bootstrap_seed": BSEED,
                           "crosses_zero": bool(lo * hi <= 0)})
    bands28 = pd.DataFrame(rows28)

    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    lv.to_csv(outd / "ablation_crossfit_levels.csv", index=False)
    bands.to_csv(outd / "ablation_crossfit_bands.csv", index=False)
    bands28.to_csv(outd / "ablation_crossfit_bands28.csv", index=False)
    (outd / "manifest.json").write_text(json.dumps({
        "design": "every question scored out of fold on both halves and the two "
                  "averaged, matching the stagewise decomposition",
        "previous": "the earlier ablation used the held-out half only",
        "n_question_keys": int(dec.canonical_qkey.nunique()),
        "family": "per-variant max-T over the four costs",
        "family28": "the same draws corrected once over all 28 variant-cost "
                    "cells (ablation_crossfit_bands28.csv)",
        "source_script": "experiments/scripts/maintrack/ablation_crossfit.py",
    }, indent=1))

    sig = bands[~bands.crosses_zero]
    print(f"\n{len(sig)} of {len(bands)} contrasts exclude zero "
          f"(per-variant families)")
    print(sig[["variant", "cost", "point", "lo", "hi"]].round(4)
          .to_string(index=False) if len(sig) else "  none")
    sig28 = bands28[~bands28.crosses_zero]
    print(f"\n{len(sig28)} of {len(bands28)} exclude zero under the single "
          f"28-cell family (critical {crit28:.3f})")
    print(sig28[["variant", "cost", "point", "lo", "hi"]].round(4)
          .to_string(index=False) if len(sig28) else "  none")
    print(f"\n[ablation] -> {outd.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
