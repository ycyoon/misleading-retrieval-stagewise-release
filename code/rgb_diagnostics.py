"""Per-source diagnostics behind the RGB exception, from cached rows only.

Section 6.6 reports that CGrV minus gradient boosting excludes zero on RGB
alone and, until now, said only that no account was available. These are the
checks actually run, per held-out source, all at c=1 from the LOSO decision
rows and the internal candidate files: channel availability, the pairwise
agreement profile among substantive answers, each selector's answer rate,
its return-source decomposition, and its accuracy among answers.

What they show is written to the artifact as data, not as a conclusion; the
manuscript sentence is derived from it and claim-checked against it.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.rgb_diagnostics
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOSO = ROOT / "artifacts/maintrack_v1/v12_14_leave_one_source_out/loso_rows.parquet"
CAND = ROOT / "experiments/outputs/candidates"
OUT = ROOT / "artifacts/maintrack_v1/v12_29_rgb_diagnostics"

ABST = {"i don't know", "i dont know", "unknown", "n/a", ""}
SOURCES = {"clash_eval": "clash_eval", "raguard": "raguard",
           "ramdocs": "ramdocs", "rgb": "rgb"}


def norm(v) -> str:
    return " ".join(str(v).lower().split())


def sub(v) -> bool:
    return v is not None and norm(v).rstrip(".").strip() not in ABST


def main() -> int:
    d1 = pd.read_parquet(LOSO)
    d1 = d1[d1.cost == 1.0]

    prof = {}
    for src in sorted(SOURCES):
        e = d1[d1.source_dataset == src]
        row = {"availability": {}, "selectors": {}}
        for m, ch in (("fixed_context", "a_C"), ("fixed_parametric", "a_P")):
            row["availability"][ch] = round(
                float(e[e.method_id == m].returned.mean()), 4)
        for m in ("cgrv_recovery", "gbdt_recovery"):
            x = e[e.method_id == m]
            ret = x[x.returned == 1]
            srcs = (ret.returned_source.value_counts(normalize=True)
                    .round(4).to_dict())
            row["selectors"][m] = {
                "answer_rate": round(float(x.returned.mean()), 4),
                "accuracy_given_answer":
                    round(float(ret.correct_return.mean()), 4),
                "return_source_shares": srcs}
        prof[src] = row

    # pairwise agreement among substantive answers, from the candidate files;
    # the internal pool's own reference field is null, so C-P is the only pair
    for src, fname in SOURCES.items():
        rows = [json.loads(l) for l in (CAND / f"{fname}.candidates.jsonl").open()]
        both = [r for r in rows
                if sub(r.get("context_answer")) and sub(r.get("parametric_answer"))]
        agree = [norm(r["context_answer"]) == norm(r["parametric_answer"])
                 for r in both]
        prof[src]["agreement_CP"] = {
            "n_both_available": len(both),
            "rate": round(sum(agree) / len(both), 4) if both else None}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rgb_diagnostics.json").write_text(json.dumps({
        "cost": 1.0,
        "per_source": prof,
        "reading": {
            "rgb_context_availability": prof["rgb"]["availability"]["a_C"],
            "other_context_availability_range": [
                min(prof[s]["availability"]["a_C"] for s in prof if s != "rgb"),
                max(prof[s]["availability"]["a_C"] for s in prof if s != "rgb")],
            "rgb_cgrv_answer_rate":
                prof["rgb"]["selectors"]["cgrv_recovery"]["answer_rate"],
            "rgb_gbdt_answer_rate":
                prof["rgb"]["selectors"]["gbdt_recovery"]["answer_rate"],
            "rgb_cgrv_parametric_share":
                prof["rgb"]["selectors"]["cgrv_recovery"]
                    ["return_source_shares"].get("a_P"),
            "rgb_gbdt_parametric_share":
                prof["rgb"]["selectors"]["gbdt_recovery"]
                    ["return_source_shares"].get("a_P"),
        },
        "source_script": "experiments/scripts/maintrack/rgb_diagnostics.py",
    }, indent=1))

    for src in sorted(prof):
        p = prof[src]
        print(f"{src:12s} a_C={p['availability']['a_C']:.3f} "
              f"a_P={p['availability']['a_P']:.3f} "
              f"agree_CP={p['agreement_CP']['rate']} "
              f"cgrv ans={p['selectors']['cgrv_recovery']['answer_rate']:.3f} "
              f"gbdt ans={p['selectors']['gbdt_recovery']['answer_rate']:.3f}")
    print(f"[rgb-diag] -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
