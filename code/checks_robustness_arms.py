"""robustness_arms: is the reference channel robust to corpus, retrieval and reader?

Recomputes, from the released arm rows alone, every number the manuscript
prints in the reference-robustness table and its prose: the reference
candidate's availability, correctness and decision score under the original
Wikipedia index, under a CC-News index and under E5 reranking; the four paired
effects at c=1 (CC-News corpus, retrieval depth 10->3, E5 reranking, E5 exact
retrieval) with their simultaneous intervals; the CC-News rule's contrasts
against the context rule and against abstention; the number of cells on which
reranking changes the reference answer; and the Llama-3.1-8B
reference-minus-context interval.

Nothing is read back from a summary file. The band tables under `bands/`, the
level tables under `levels/` and the run manifests under `provenance/` are
shipped as the frozen record of the generating runs; delete all of them and
every check below still runs, because every point estimate is re-derived from
`arms/*.parquet` and every interval is re-drawn here with the paper's own
bootstrap: 10,000 draws, seed 20260808, the twelve seed replicas and the
question keys inside each collection resampled jointly, collections weighted
equally, and a max-T critical value calibrated over the four costs separately
for each comparison. That construction is the one used for every band in the
paper, so the intervals below are recomputations, not copies.

Each variant arm carries only the rows the knob can move -- the reference rule
and the LLM candidate selector -- and is compared against the corresponding
Wikipedia-original arm in `arms/wikipedia_baseline_arms.parquet`, which is the
same slice of `data/decision_rows/primary_all_arms.parquet` the release
already ships. The two copies are reconciled row by row below, so the
baseline side of every contrast is not taken on trust either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COSTS = (0.5, 1.0, 2.0, 4.0)
DRAWS, BOOT_SEED = 10000, 20260808   # the recorded band settings

# the arm rows this family ships, and the variant reference rule in each
ARM_FILES = {
    "wikipedia_baseline": "wikipedia_baseline_arms.parquet",
    "depth_k3": "depth_k3_arms.parquet",
    "rerank": "rerank_arms.parquet",
    "news": "news_arms.parquet",
    "e5exact": "e5exact_arms.parquet",
    "llama8_channel": "llama8_channel_arms.parquet",
}
REFERENCE_ARM = {"wikipedia_baseline": "fixed_reference",
                 "news": "fixed_reference_news",
                 "rerank": "fixed_reference_rerank",
                 "depth_k3": "fixed_reference_k3",
                 "e5exact": "fixed_reference_e5exact"}


def _macro(df, col="answer_utility"):
    """House macro: cells within question, questions within collection, seeds."""
    p = df.groupby(["seed_id", "source_dataset", "canonical_qkey"])[col].mean()
    return float(p.groupby(["seed_id", "source_dataset"]).mean()
                 .groupby("seed_id").mean().mean())


def _cells(rows, arm, cost=1.0):
    """One row per panel cell for a deterministic arm (seed-invariant)."""
    d = rows[(rows.method_id == arm) & (rows.cost == cost)]
    return d.groupby("cell_id")[["returned", "correct_return"]].max()


def _bands(rows, comps):
    """The paper's band construction, re-run here on the shipped rows.

    Crossed bootstrap: within a draw the twelve seed replicas are resampled
    with replacement, then the question keys of each collection are resampled
    with replacement; the arm difference is averaged over questions, over
    seeds, and finally over collections with equal weight. The simultaneous
    band is `point +/- crit * se`, with `crit` the 95th percentile of the
    max-over-costs absolute studentised draw for that comparison.
    """
    methods = sorted({m for c in comps for m in c[:2]})
    d = rows[rows.method_id.isin(methods)]
    seeds = sorted(d.seed_id.unique(), key=int)
    datasets = sorted(d.source_dataset.unique())
    q_by_d = {x: sorted(d.loc[d.source_dataset == x, "canonical_qkey"].unique())
              for x in datasets}
    si = {s: i for i, s in enumerate(seeds)}
    di = {x: i for i, x in enumerate(datasets)}
    qi = {x: {q: j for j, q in enumerate(q_by_d[x])} for x in datasets}
    mi = {m: i for i, m in enumerate(methods)}
    ci = {c: j for j, c in enumerate(COSTS)}

    arr = np.full((len(seeds), len(datasets), max(map(len, q_by_d.values())),
                   len(methods), len(COSTS)), np.nan)
    cell = d.groupby(["seed_id", "source_dataset", "canonical_qkey",
                      "method_id", "cost"], sort=False).answer_utility.mean()
    for (s, x, q, m, c), v in cell.items():
        arr[si[s], di[x], qi[x][q], mi[m], ci[float(c)]] = v

    # per comparison, per collection: the arm difference on the full grid
    diff = {n: [arr[:, di[x], :len(q_by_d[x]), mi[a], :]
                - arr[:, di[x], :len(q_by_d[x]), mi[b], :] for x in datasets]
            for a, b, n in comps}
    if any(np.isnan(blk).any() for blks in diff.values() for blk in blks):
        raise SystemExit("robustness_arms: an arm is missing seed/question cells")
    point = {n: np.mean([blk.mean(axis=1).mean(axis=0) for blk in diff[n]],
                        axis=0) for _, _, n in comps}

    rng = np.random.default_rng(BOOT_SEED)
    draw = {n: np.empty((DRAWS, len(COSTS))) for _, _, n in comps}
    for z in range(DRAWS):
        w_seed = np.bincount(rng.integers(0, len(seeds), len(seeds)),
                             minlength=len(seeds)) / len(seeds)
        acc = {n: np.zeros(len(COSTS)) for _, _, n in comps}
        for k, x in enumerate(datasets):
            nq = len(q_by_d[x])
            w_q = np.bincount(rng.integers(0, nq, nq), minlength=nq) / nq
            for _, _, n in comps:
                acc[n] += np.einsum("s,q,sqc->c", w_seed, w_q,
                                    diff[n][k]) / len(datasets)
        for n in acc:
            draw[n][z] = acc[n]

    out = {}
    for _, _, n in comps:
        p, dr = point[n], draw[n]
        se = dr.std(axis=0, ddof=1)
        crit = np.percentile(
            np.max(np.abs((dr - p) / np.maximum(se, 1e-12)), axis=1), 95)
        out[n] = {c: (float(p[j]), float(p[j] - crit * se[j]),
                      float(p[j] + crit * se[j])) for j, c in enumerate(COSTS)}
    return out


def run(D, check):
    P = D / "robustness_arms"

    # ---- the rows themselves -------------------------------------------
    print("== robustness_arms: row-level utility identity on the shipped arms ==")
    rows = {}
    for tag, name in ARM_FILES.items():
        r = pd.read_parquet(P / "arms" / name)
        viol = int((r.correct_return + r.wrong_return != r.returned).sum())
        resid = float((r.answer_utility
                       - (r.correct_return - r.cost * r.wrong_return))
                      .abs().max())
        check(f"utility identity, {name} ({len(r)} rows)", viol + resid, 0.0,
              tol=1e-9)
        rows[tag] = r

    # the manuscript sizes every robustness statement on the 400-cell panel:
    # "changes the reference answer on $84$ of $400$ cells" (line 1065)
    print("== robustness_arms: the 400-cell panel behind every robustness arm ==")
    for tag in ARM_FILES:
        check(f"panel cells, {tag}", float(rows[tag].cell_id.nunique()), 400,
              tol=0)

    # the Wikipedia-original side of every contrast is the release's own
    # primary decision rows; prove the shipped copy is that same table
    prim = pd.read_parquet(D / "decision_rows/primary_all_arms.parquet")
    key = ["cell_id", "seed_id", "method_id", "cost"]
    lhs = rows["wikipedia_baseline"].copy()
    lhs["seed_id"] = lhs.seed_id.astype(str)
    rhs = prim[prim.method_id.isin(lhs.method_id.unique())].copy()
    rhs["seed_id"] = rhs.seed_id.astype(str)
    j = lhs.merge(rhs, on=key, suffixes=("_s", "_p"))
    # both row counts, so neither a shipped subset nor a duplicated key can
    # cancel against a column mismatch and leave the total at zero
    bad = abs(float(len(lhs) - len(j))) + abs(float(len(rhs) - len(j))) + sum(
        float((j[f"{c}_s"] - j[f"{c}_p"]).abs().max())
        for c in ("returned", "correct_return", "wrong_return",
                  "answer_utility"))
    check("Wikipedia baseline rows identical to data/decision_rows/", bad, 0.0,
          tol=0)

    # ---- availability, correctness and decision score at c=1 ------------
    # "reduces reference availability from $0.545$ to $0.460$, correctness
    #  from $0.385$ to $0.245$, and the decision score at $c{=}1$ from
    #  $+0.225$ to $+0.030$" (lines 1050-1052; table lines 1004-1012)
    # "raises availability to $0.580$, while correctness remains $0.385$"
    #  (lines 1065-1066; table lines 1022-1023)
    print("== robustness_arms: reference availability, correctness and score at c=1 ==")
    want_cell = {"wikipedia_baseline": ("Wikipedia, original", 0.545, 0.385, 0.225),
                 "news": ("CC-News corpus", 0.460, 0.245, 0.030),
                 "rerank": ("E5 reranking", 0.580, 0.385, None)}
    for tag, (label, av, ok, sc) in want_cell.items():
        arm = REFERENCE_ARM[tag]
        c = _cells(rows[tag], arm)
        check(f"reference availability, {label}", float(c.returned.mean()), av)
        check(f"reference correctness, {label}", float(c.correct_return.mean()),
              ok)
        if sc is not None:
            r = rows[tag]
            check(f"reference decision score at c=1, {label}",
                  _macro(r[(r.method_id == arm) & (r.cost == 1.0)]), sc)

    # ---- the paired effects, re-drawn from the rows ---------------------
    print("== robustness_arms: paired effects at c=1, simultaneous over the four costs ==")
    pool = pd.concat(
        [rows["wikipedia_baseline"].drop(columns=["returned_answer"]),
         rows["depth_k3"], rows["rerank"].drop(columns=["returned_answer"]),
         rows["news"], rows["e5exact"],
         rows["llama8_channel"].assign(
             method_id="llama8_" + rows["llama8_channel"].method_id)],
        ignore_index=True)
    comps = [
        ("fixed_reference_news", "fixed_reference", "news_minus_bge"),
        ("fixed_reference_k3", "fixed_reference", "k3_minus_k10"),
        ("fixed_reference_rerank", "fixed_reference", "rerank_minus_bge"),
        ("fixed_reference_e5exact", "fixed_reference", "e5exact_minus_bge"),
        ("fixed_reference_news", "fixed_recovery", "news_minus_context"),
        ("fixed_reference_news", "always_abstain", "news_minus_abstain"),
        ("prompt_candidatewise_news", "prompt_candidatewise",
         "prompted_news_minus_bge"),
        ("llama8_fixed_reference", "llama8_fixed_recovery",
         "llama8_reference_minus_context"),
    ]
    band = _bands(pool, comps)

    # table lines 1013, 1019, 1025, 1031 and prose lines 1053-1057
    want_band = [
        ("news_minus_bge", "CC-News corpus", -0.1950, -0.3074, -0.0826),
        ("k3_minus_k10", "retrieval depth 10->3", -0.0450, -0.1019, +0.0119),
        ("rerank_minus_bge", "E5 reranking", -0.0350, -0.1053, +0.0353),
        ("e5exact_minus_bge", "E5 exact retrieval", -0.0150, -0.1121, +0.0821),
        ("news_minus_context", "CC-News reference minus context",
         +0.1425, +0.0375, +0.2475),
        ("news_minus_abstain", "CC-News reference minus abstention",
         +0.0300, -0.0715, +0.1315),
    ]
    for name, label, pt, lo, hi in want_band:
        g = band[name][1.0]
        check(f"effect at c=1, {label}", g[0], pt)
        check(f"interval lower, {label}", g[1], lo)
        check(f"interval upper, {label}", g[2], hi)

    # "The corpus change does not measurably affect the LLM-based candidate
    #  selector at any evaluated cost." (lines 1057-1059)
    excl = sum(band["prompted_news_minus_bge"][c][1]
               * band["prompted_news_minus_bge"][c][2] > 0 for c in COSTS)
    check("costs where the CC-News selector effect excludes zero (of 4)",
          float(excl), 0, tol=0)

    # "Changing retrieval depth, reranking the retrieved passages, or replacing
    #  the retriever produces no decision-score difference whose interval
    #  excludes zero." (lines 1061-1064), at the cost the table reports
    excl = sum(band[n][1.0][1] * band[n][1.0][2] > 0
               for n in ("k3_minus_k10", "rerank_minus_bge",
                         "e5exact_minus_bge"))
    check("retrieval-method effects at c=1 excluding zero (of 3)", float(excl),
          0, tol=0)

    # ---- how much reranking moved the reference answer ------------------
    # "E5 reranking changes the reference answer on $84$ of $400$ cells"
    #  (lines 1064-1065)
    print("== robustness_arms: cells whose reference answer reranking changed ==")
    def answers(r, arm):
        d = r[(r.method_id == arm) & (r.cost == 1.0)]
        return (d.drop_duplicates("cell_id").set_index("cell_id")
                .returned_answer.fillna("").astype(str).str.strip())
    a = answers(rows["wikipedia_baseline"], "fixed_reference")
    b = answers(rows["rerank"], "fixed_reference_rerank")
    check("cells where reranking changes the reference answer",
          float((a != b.reindex(a.index)).sum()), 84, tol=0)

    # ---- the 8B reader's channel interval -------------------------------
    # "8B reader: reference $-$ context ... $+0.2875$ $[+0.1855,+0.3895]$"
    #  (table line 1038)
    print("== robustness_arms: Llama-3.1-8B reference minus context ==")
    g = band["llama8_reference_minus_context"][1.0]
    check("8B reference minus context at c=1", g[0], +0.2875)
    check("8B interval lower, reference minus context", g[1], +0.1855)
    check("8B interval upper, reference minus context", g[2], +0.3895)
