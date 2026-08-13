"""Row-level replay of the Llama-3.1-70B sufficiency crosscheck, with a CI.

The manuscript quoted the crosscheck gap (0.741) from a summary that carried no
interval and no per-condition counts, because the run wrote only aggregates.
Every judged unit is still in the deterministic disk cache, keyed by the exact
request, so the row-level verdicts are recoverable offline: rebuild each prompt
with the original code path, look it up in the cache, and never touch the
network. A unit with no cache entry is one of the run's recorded failures and
stays missing, exactly as the original aggregation treated it.

Later analyses (the S3 truncation study runs the same autorater on the same
records) retried units that failed during the crosscheck and cached them, so
the cache today holds more entries than the crosscheck saw. The run-time state
is recovered by mtime: the crosscheck wrote its summary last, so exactly the
entries no newer than `crosscheck.json` existed when it aggregated.

The replay must reproduce the recorded aggregates to 4 decimals before the CI
is computed; a replay that disagrees with the summary it claims to reconstruct
proves the reconstruction wrong, and the script exits nonzero instead of
writing an artifact.

The interval is the same construction as the other Section-4 gaps: resample the
600 question records with replacement (conditions stay paired within a record),
1000 draws, percentile 2.5/97.5, `random.Random(42)` -- a literal mirror of the
diagnostic's `_bootstrap`.

Subset rule, from the stage-2 driver that built `clasheval_cc600.jsonl`: the
first 100 records per ClashEval domain, in the frozen scaled file's order.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.sufficiency_crosscheck_ci
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WRC = ROOT / "web_rag_confound"
CACHE = WRC / "data/llm_cache/meta-llama__Llama-3.1-70B-Instruct"
CC600 = WRC / "data/poisoned/clasheval_cc600.jsonl"
RECORDED = WRC / "results/20260703_235131_diagnostic_f88b057056/crosscheck.json"
OUT = ROOT / "artifacts/maintrack_v1/v12_28_sufficiency_crosscheck"

MODEL = "meta-llama/Llama-3.1-70B-Instruct"
CONDS = ["C_suf_clean", "C_poison", "C_mixed", "C_insuf"]
PARAMS = {"temperature": 0.0, "max_tokens": 8, "top_p": 1.0, "seed": 42}

# The prompt construction, verbatim from web_rag_confound/src/detectors/.
SUFF_SYS = "You judge whether a context contains enough information to answer a question."
SUFF_TMPL = """Question: {question}

Context:
{context}

Does this context contain enough information for a diligent reader to derive a PLAUSIBLE answer
to the question, using ONLY the context? Judge sufficiency of information ONLY — do NOT judge
whether the answer is factually correct or true.

Answer with exactly one word: "sufficient" or "insufficient"."""


def format_context(passages):
    if not passages:
        return "(no documents retrieved)"
    return "\n\n".join(f"[Doc {i+1}] {p['text']}" for i, p in enumerate(passages))


def parse_label(text, labels):
    t = text.lower()
    for lab in labels:
        if re.search(rf"\b{re.escape(lab.lower())}\b", t):
            return lab
    return None


def cache_key(question, passages):
    prompt = SUFF_TMPL.format(question=question, context=format_context(passages))
    messages = [{"role": "system", "content": SUFF_SYS},
                {"role": "user", "content": prompt}]
    blob = json.dumps({"model": MODEL, "messages": messages, "params": PARAMS},
                      sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def replay():
    cutoff = RECORDED.stat().st_mtime
    recs = [json.loads(l) for l in CC600.open()]
    rows, hits, misses = [], 0, 0
    for r in recs:
        for cond in CONDS:
            p = CACHE / f"{cache_key(r['question'], r['conditions'][cond])}.json"
            if p.exists() and p.stat().st_mtime <= cutoff:
                hits += 1
                lab = parse_label(json.loads(p.read_text())["text"],
                                  ["insufficient", "sufficient"])
            else:
                misses += 1
                lab = None
            rows.append({"qid": r["qid"], "domain": r["domain"], "cond": cond,
                         "label": lab})
    return recs, pd.DataFrame(rows), hits, misses


def rate(df, cond):
    v = df[(df.cond == cond) & df.label.notna()].label
    return float((v == "sufficient").mean()) if len(v) else float("nan")


def main() -> int:
    recs, df, hits, misses = replay()
    rec = json.loads(RECORDED.read_text())

    got = {c: round(rate(df, c), 4) for c in CONDS}
    gap = round(got["C_poison"] - got["C_insuf"], 4)
    want, want_gap = rec["P_sufficient_B"], rec["H1_suffA_B"]
    bad = [c for c in CONDS if got[c] != want[c]] + (["gap"] if gap != want_gap else [])
    print(f"[crosscheck-ci] replay: {hits} cache hits, {misses} misses "
          f"(recorded failures: {rec['llm_stats']['failures']})")
    print(f"[crosscheck-ci] rates {got} gap {gap} vs recorded {want} {want_gap}")
    if bad or misses != rec["llm_stats"]["failures"]:
        print(f"[crosscheck-ci] REPLAY DISAGREES: {bad or 'failure count'}")
        return 1

    # Paired-by-question percentile bootstrap, mirroring diagnostic._bootstrap.
    per_q = {}
    for r in recs:
        per_q[r["qid"]] = {c: None for c in CONDS}
    for row in df.itertuples():
        per_q[row.qid][row.cond] = row.label
    qs = [per_q[r["qid"]] for r in recs]

    def stat(sample):
        out = {}
        for c in ("C_poison", "C_insuf"):
            v = [q[c] for q in sample if q[c] is not None]
            out[c] = sum(x == "sufficient" for x in v) / len(v)
        return out["C_poison"] - out["C_insuf"]

    rng = random.Random(42)
    n = len(qs)
    draws = [stat([qs[rng.randrange(n)] for _ in range(n)]) for _ in range(1000)]
    lo, hi = (round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5]))

    n_by = {c: int(df[(df.cond == c) & df.label.notna()].shape[0])
            for c in ("C_poison", "C_insuf")}
    dom = Counter(r["domain"] for r in recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "crosscheck_rows.csv", index=False)
    summary = {
        "model": MODEL,
        "n_questions": len(recs),
        "subset_rule": "first 100 records per ClashEval domain, in the frozen "
                       "scaled file's order (run_wsdm_stage2.sh)",
        "per_domain": dict(sorted(dom.items())),
        "n_judged": n_by,
        "P_sufficient_C_poison": got["C_poison"],
        "P_sufficient_C_insuf": got["C_insuf"],
        "gap": gap,
        "gap_ci95": [lo, hi],
        "bootstrap": {"iters": 1000, "seed": 42, "scheme":
                      "resample question records, conditions paired"},
        "replay": {"cache_hits": hits, "cache_misses": misses,
                   "recorded_failures": rec["llm_stats"]["failures"]},
    }
    (OUT / "crosscheck_ci.json").write_text(json.dumps(summary, indent=2))
    print(f"[crosscheck-ci] n=600 ({dict(sorted(dom.items()))})")
    print(f"[crosscheck-ci] judged: poison {n_by['C_poison']}, insuf {n_by['C_insuf']}")
    print(f"[crosscheck-ci] gap {gap} ci95 [{lo}, {hi}] -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
