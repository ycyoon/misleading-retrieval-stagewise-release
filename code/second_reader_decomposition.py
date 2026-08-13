"""The full stagewise decomposition under the second reader.

The paper's largest stated limitation is that the decomposition rests on one
primary reader, with an 8B ordering check standing in for the preregistered
comparable-capacity replication. This runs the identical procedure on the
candidates the 70B reader generated: the same 23-policy deterministic family,
the same simple and structured classes, the same two question-keyed folds, the
same bootstrap that repeats the selection inside every draw, the same frozen
lexical matcher.

Everything is recomputed from the second reader's own decision rows, including
the oracle, which is a property of its candidate set and not of the first
reader's. The learned arms are refit per training seed on the internal
population exactly as the primary panel's were, so their parameters never see a
panel label under either reader.

Run (from <repository root>):
  python -m \\
      experiments.scripts.maintrack.second_reader_decomposition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from experiments.scripts.maintrack.criterion_decomposition import (
    SIMPLE_BASE, STRUCTURED, CHANNEL_ARM, decompose)
from experiments.scripts.maintrack.telescoping_decomposition import (
    stagewise_bands)

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/maintrack_v1"
BASE = A / "v12_27_second_reader_70b"
SPLIT_SEED = 20260811          # the primary panel's partition seed, reused
TERMS = ["headroom_H", "fixed_policy_gain_G1", "simple_selector_gain_G2",
         "structured_gain_G3", "residual_after_selected_policy_R"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args(argv)
    base = Path(a.base)

    dec = pd.concat([
        pd.read_parquet(base / "verification_arms/panel_all_arms.parquet"),
        pd.read_parquet(base / "closure/closure_decision_rows.parquet"),
        pd.read_parquet(base / "trustmargin/trustmargin_decision_rows.parquet"),
    ], ignore_index=True)

    # the ceiling belongs to this reader's candidates
    ch = dec[dec.method_id.isin(CHANNEL_ARM.values()) & (dec.cost == 1.0)]
    per = (ch.groupby(["source_dataset", "canonical_qkey", "cell_id"])
           .correct_return.max())
    oracle = float(per.groupby(level=[0, 1]).mean()
                   .groupby(level=0).mean().mean())

    P1 = (["fixed_recovery", "fixed_reference", "fixed_parametric",
           "always_abstain"]
          + sorted(pd.read_parquet(base / "closure/closure_decision_rows"
                                          ".parquet").method_id.unique()))
    P2 = P1 + SIMPLE_BASE + sorted(
        pd.read_parquet(base / "trustmargin/trustmargin_decision_rows"
                               ".parquet").method_id.unique())
    P3 = P2 + [STRUCTURED]
    P1, P2, P3 = ([m for m in p if m in set(dec.method_id)]
                  for p in (P1, P2, P3))
    print(f"[70b-decomp] {len(P1)} fixed, {len(P2)} simple, {len(P3)} "
          f"structured; oracle {oracle:.4f}", flush=True)

    res = decompose(dec, P1, P2, P3, oracle, "answer_utility", SPLIT_SEED)
    worst = float(res.identity_residual.abs().max())
    if worst > 1e-9:
        raise SystemExit(f"identity violated: {worst}")

    # the same bands, selection repeated inside every draw
    rng = np.random.default_rng(SPLIT_SEED)
    fold = {}
    for _, g in dec.groupby("source_dataset"):
        keys = sorted(g.canonical_qkey.unique())
        for r, i_ in enumerate(rng.permutation(len(keys))):
            fold[keys[i_]] = 0 if r < len(keys) // 2 else 1
    dec = dec.assign(_fold=dec.canonical_qkey.map(fold))
    bands, sim = stagewise_bands(dec, P1, P2, P3, oracle, draws=a.draws)

    res.to_csv(base / "second_reader_decomposition.csv", index=False)
    bands.to_csv(base / "stagewise_bands.csv", index=False)
    sim.to_csv(base / "stagewise_simultaneous_upper.csv", index=False)
    (base / "decomposition_manifest.json").write_text(json.dumps({
        "reader": "meta-llama/Llama-3.1-70B-Instruct",
        "oracle_utility": oracle,
        "identity_max_residual": worst,
        "split_seed": SPLIT_SEED, "draws": a.draws,
        "classes": {"P1": len(P1), "P2": len(P2), "P3": len(P3)},
        "protocol": "identical to the primary panel: same folds rule, same "
                    "bootstrap with in-draw reselection, same lexical matcher",
        "simultaneous_upper_max": float(sim.simultaneous_hi.max()),
        "source_script":
            "experiments/scripts/maintrack/second_reader_decomposition.py",
    }, indent=1))

    print(f"\noracle {oracle:.4f}; identity residual {worst:.1e}")
    print(res[["cost"] + TERMS].round(4).to_string(index=False))
    print("\nstage selections:")
    for r in res.itertuples(index=False):
        print(f"  c={r.cost}: {r.stage1_choice} / {r.stage2_choice} / "
              f"{r.stage3_choice}")
    print(f"\nsimultaneous upper bound on the learned gains: "
          f"{float(sim.simultaneous_hi.max()):+.4f}")
    print(f"[70b-decomp] -> {base.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
