"""E4 readouts: does the ordering hold on collections the panel never contained?

The pre-registered primary readout is on the **new cells only**, so it is not
diluted by the cells already reported, and the fixed policy it compares against
is chosen on the **original panel** rather than on the new cells. That makes the
comparator choice genuinely prospective: the original panel is prior data with
respect to these three collections, so nothing about the new cells informs which
channel is called best.

`always_abstain` is constructed here, since it is utility 0 by definition.

Secondary readouts, all reported whichever way they point: the same contrasts
pooled over all seven collections, per-collection availability and utility, and
candidate headroom on the new cells.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.e4_extension_bands
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
E4 = ROOT / "artifacts/maintrack_v1/v12_13_e4_panel_extension"
ORIG_ARMS = ROOT / ("artifacts/maintrack_v1/v12_10_verification_arms/"
                    "panel_all_arms.parquet")
ORIG_ROWS = ROOT / ("artifacts/maintrack_v1/v12_9a_panel_rederived/"
                    "panel_input.candidates.jsonl")
COSTS = (0.5, 1.0, 2.0, 4.0)
FIXED = ["fixed_recovery", "fixed_reference", "fixed_parametric",
         "always_abstain"]
CONTEXT = "fixed_recovery"

from experiments.scripts.maintrack.e2_reference_depth_arms import band  # noqa: E402
from experiments.scripts.maintrack.gpt_semantic_closure.run_all import (  # noqa: E402
    alias_ok,
)
from experiments.scripts.maintrack.rerun_panel_decisions import (  # noqa: E402
    base_record,
)
from experiments.scripts.maintrack.telescoping_decomposition import (  # noqa: E402
    macro,
)

ABST = {"i don't know", "i dont know", "unknown", "n/a", ""}


def sub(v) -> bool:
    return bool(v) and str(v).strip().lower().rstrip(".").strip() not in ABST


def headroom(rows) -> float:
    fields = ("context_answer", "reference_answer", "parametric_answer")
    return sum(1 for r in rows if any(
        sub(r.get(f)) and alias_ok(r.get(f), r.get("gold_answers"))
        for f in fields)) / len(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e4", default=str(E4))
    a = ap.parse_args(argv)

    e4 = Path(a.e4)
    new_rows = [json.loads(l) for l in (e4 / "panel_input.candidates.jsonl").open()]
    dec = pd.read_parquet(e4 / "panel_decision_rows.parquet")
    seeds = sorted(dec.seed_id.unique(), key=int)
    abstain = pd.DataFrame([
        base_record(r, s, c, "always_abstain", "trivial", "NONE", "", None,
                    "constant")
        for r in new_rows for c in COSTS for s in seeds])
    new = pd.concat([dec, abstain], ignore_index=True)

    orig = pd.read_parquet(ORIG_ARMS)
    orig_rows = [json.loads(l) for l in ORIG_ROWS.open()]

    # F* chosen on the ORIGINAL panel: prior data with respect to these cells
    sel, comps, seen = [], [], set()
    for c in COSTS:
        o_c = orig[orig.cost == c]
        avail = [m for m in FIXED if m in set(o_c.method_id)]
        f_star = max(avail, key=lambda m: macro(o_c[o_c.method_id == m]))
        sel.append({"cost": c, "F_star_selected_on_original_panel": f_star,
                    "original_panel_utility": macro(o_c[o_c.method_id == f_star])})
        for lo, tag in ((CONTEXT, "minus_context"),
                        ("always_abstain", "minus_abstain")):
            name = f"{f_star}_{tag}"
            if name not in seen and f_star != lo:
                seen.add(name)
                comps.append((f_star, lo, name))

    bands_new = band(new, comps)
    bands_new["population"] = "new_cells_only"

    pooled = pd.concat([orig, new], ignore_index=True)
    bands_pooled = band(pooled, comps)
    bands_pooled["population"] = "pooled_seven_collections"
    bands = pd.concat([bands_new, bands_pooled], ignore_index=True)

    lv = (new[new.cost == 1.0]
          .groupby(["method_id", "source_dataset", "canonical_qkey"])
          .answer_utility.mean()
          .groupby(["method_id", "source_dataset"]).mean()
          .groupby("method_id").mean())

    per_coll = (new[new.cost == 1.0]
                .groupby(["source_dataset", "method_id"])
                .agg(utility=("answer_utility", "mean"),
                     answer_rate=("returned", "mean"))
                .reset_index())

    by_coll_head = {}
    for d in sorted({r["source_dataset"] for r in new_rows}):
        rs = [r for r in new_rows if r["source_dataset"] == d]
        by_coll_head[d] = {
            "n_cells": len(rs), "headroom": headroom(rs),
            "a_C": sum(1 for r in rs if sub(r.get("context_answer"))) / len(rs),
            "a_R": sum(1 for r in rs if sub(r.get("reference_answer"))) / len(rs),
            "a_P": sum(1 for r in rs if sub(r.get("parametric_answer"))) / len(rs)}

    bands.to_csv(e4 / "e4_bands.csv", index=False)
    lv.to_csv(e4 / "e4_levels_c1.csv")
    per_coll.to_csv(e4 / "e4_per_collection_c1.csv", index=False)
    (e4 / "e4_readout.json").write_text(json.dumps({
        "comparator_selection": sel,
        "comparator_rule": "F* chosen on the original 400-cell panel, which is "
                           "prior data with respect to these three collections",
        "headroom_new_cells": headroom(new_rows),
        "headroom_original_panel": 0.6150,
        "per_collection": by_coll_head,
        "n_new_cells": len(new_rows),
        "contract_failures": int((new.correct_return + new.wrong_return
                                  != new.returned).sum()),
        "utility_failures": int((new.answer_utility != new.correct_return
                                 - new.cost * new.wrong_return).sum()),
        "source_script":
            "experiments/scripts/maintrack/e4_extension_bands.py",
    }, indent=1))

    print(f"headroom on the new cells {headroom(new_rows):.4f} "
          f"(original panel 0.6150)")
    print("\nper collection (new cells)")
    for d, v in by_coll_head.items():
        print(f"  {d:28s} n={v['n_cells']:3d} headroom={v['headroom']:.4f} "
              f"a_C={v['a_C']:.4f} a_R={v['a_R']:.4f} a_P={v['a_P']:.4f}")
    print("\nlevels at c=1 (new cells)")
    for m in lv.index:
        print(f"  {m:24s} {lv[m]:+.4f}")
    print("\nF* selected on the original panel:",
          {r["cost"]: r["F_star_selected_on_original_panel"] for r in sel})
    for pop in ("new_cells_only", "pooled_seven_collections"):
        print(f"\n{pop}, c=1")
        for r in bands[(bands.population == pop)
                       & (bands.cost == 1.0)].itertuples(index=False):
            mark = "crosses zero" if r.crosses_zero else "EXCLUDES zero"
            print(f"  {r.comparison:44s} {r.point:+.4f} "
                  f"[{r.simultaneous_lo:+.4f}, {r.simultaneous_hi:+.4f}] {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
