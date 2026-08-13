"""Does the symmetric fallback change anything, or only the appearance of it?

The deterministic comparator family was enumerated after the candidates were
frozen, and an earlier enumeration gave the any-pair agreement rule a fallback to
the parametric channel alone -- which is one of the policies the first stage
selects. A reader is entitled to ask whether the family was shaped, however
unconsciously, around its own winner.

The check is to run the decomposition both ways: with the 21-policy family that
has the parametric fallback only, and with the 23-policy family that has all
three. If the selected policy and every stage gain are identical at every cost,
the asymmetry was an oversight rather than a thumb on the scale.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.family_symmetry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from experiments.scripts.maintrack.criterion_decomposition import (
    ARMS, CLOSURE, TRUSTMARGIN, CANDSTATS, SPLIT_SEED, SIMPLE_BASE,
    STRUCTURED, decompose)

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/maintrack_v1"
OUT = A / "v12_26_family_symmetry"
ADDED = ("closure_agree_any_else_C", "closure_agree_any_else_R")
TERMS = ["headroom_H", "fixed_policy_gain_G1", "simple_selector_gain_G2",
         "structured_gain_G3", "residual_after_selected_policy_R"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    dec = pd.concat([pd.read_parquet(ARMS), pd.read_parquet(CLOSURE),
                     pd.read_parquet(TRUSTMARGIN)], ignore_index=True)
    oracle = json.loads(CANDSTATS.read_text())["candidate_headroom"]
    full = (["fixed_recovery", "fixed_reference", "fixed_parametric",
             "always_abstain"]
            + sorted(pd.read_parquet(CLOSURE).method_id.unique()))
    full = [m for m in full if m in set(dec.method_id)]
    narrow = [m for m in full if m not in ADDED]
    print(f"[symmetry] {len(narrow)} policies without the added fallbacks, "
          f"{len(full)} with them")

    res = {}
    for name, P1 in (("family_21", narrow), ("family_23", full)):
        P2 = P1 + [m for m in SIMPLE_BASE if m in set(dec.method_id)] + sorted(
            pd.read_parquet(TRUSTMARGIN).method_id.unique())
        P3 = P2 + [STRUCTURED]
        res[name] = decompose(dec, P1, P2, P3, oracle, "answer_utility",
                              SPLIT_SEED).assign(family=name)

    a21, a23 = res["family_21"].set_index("cost"), res["family_23"].set_index("cost")
    worst = max(abs(float(a21.loc[c, t]) - float(a23.loc[c, t]))
                for c in a21.index for t in TERMS)
    same = all(a21.loc[c, f"stage{k}_choice"] == a23.loc[c, f"stage{k}_choice"]
               for c in a21.index for k in (1, 2, 3))
    print(f"  worst stage-gain difference {worst:.2e}; selections "
          f"{'identical' if same else 'DIFFER'}")
    print(pd.concat(res.values()).round(4)
          .set_index(["family", "cost"])[TERMS].to_string())

    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    pd.concat(res.values()).to_csv(outd / "family_symmetry.csv", index=False)
    (outd / "manifest.json").write_text(json.dumps({
        "question": "whether the symmetric fallback templates change the "
                    "decomposition or only remove an asymmetry",
        "added": list(ADDED),
        "worst_stage_gain_difference": worst,
        "selections_identical": bool(same),
        "grammar_reads": "availability and exact agreement only; never "
                         "candidate content or correctness",
        "source_script": "experiments/scripts/maintrack/family_symmetry.py",
    }, indent=1))
    print(f"\n[symmetry] -> {outd.relative_to(ROOT)}")
    return 0 if (worst <= 1e-12 and same) else 1


if __name__ == "__main__":
    raise SystemExit(main())
