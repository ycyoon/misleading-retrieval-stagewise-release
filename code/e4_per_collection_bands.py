"""Per-collection extension contrast: parametric minus context, with bands.

The manuscript justified reporting the panels separately with "fails on two of
the three new collections", but no per-collection number appeared anywhere.
This emits the parametric-rule-minus-context-rule contrast inside each of the
three extension collections, through the same crossed seed/question bootstrap
every other band uses. Each collection's contrast is its own max-T family over
the four costs, which is the family shape Section 3.2 declares for later
analyses; the three collections are not corrected against each other.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.e4_per_collection_bands
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
E4 = ROOT / "artifacts/maintrack_v1/v12_13_e4_panel_extension"
OUT = E4 / "e4_per_collection_bands.csv"

from experiments.scripts.maintrack.e2_reference_depth_arms import band  # noqa: E402

COMPS = [("fixed_parametric", "fixed_recovery", "parametric_minus_context")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    dec = pd.read_parquet(E4 / "panel_decision_rows.parquet")
    frames = []
    for coll, g in dec.groupby("source_dataset"):
        b = band(g, COMPS)
        b.insert(0, "collection", coll)
        frames.append(b)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(a.out, index=False)

    print(f"[e4-per-collection] {out.collection.nunique()} collections x "
          f"{out.cost.nunique()} costs -> {Path(a.out).relative_to(ROOT)}")
    for r in out[out.cost == 1.0].itertuples(index=False):
        mark = "crosses zero" if r.crosses_zero else "EXCLUDES zero"
        print(f"  c=1 {r.collection:32s} {r.point:+.4f} "
              f"[{r.simultaneous_lo:+.4f}, {r.simultaneous_hi:+.4f}] {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
