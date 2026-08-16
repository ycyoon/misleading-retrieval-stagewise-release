"""internal_verification: CGrV against the fixed verification comparator.

Recomputes, from the released row-level decisions alone, every number the
manuscript quotes for the internal verification comparison (Section
"Does verification help?"):

  "For each training run and cost, this comparator selects one rule from a
   four-rule registry using the development split and then applies that rule
   unchanged to the corresponding test questions. Over the 1,806 canonical
   question keys that enter a held-out test fold in at least one of the
   twelve runs, CGrV differs from this comparator by +0.0057
   [-0.0018, +0.0131], again with the interval crossing zero."
                                     -- manuscript lines 832-839

Nothing is read back from a summary file. `reference/` ships the published
crossed-bootstrap row and the per-seed deltas for a reader's convenience;
delete both and every check below still runs, because each value is rebuilt
from `rows/verification_rowlevel.parquet`:

  * the +0.0057 point is the mean paired difference between the CGrV
    verification arm and the fixed comparator arm over the 12 x 316 seed by
    canonical-question cells, after the recorded reduction (mean utility
    within seed, canonical question key and arm), and is checked twice --
    once over all cells and once as the mean of the twelve per-seed deltas;
  * the [-0.0018, +0.0131] band is re-run here: the recorded crossed
    multinomial seed x question bootstrap, 10,000 draws at rng seed
    20260807, with max-T simultaneous bounds over the four costs;
  * the 1,806 canonical question keys and the twelve runs are counted off
    the rows;
  * the registry member the comparator applied in each run is *identified
    from the rows*, not read off the provenance column: for every run and
    cost exactly one of the four registry arms reproduces the comparator's
    per-row utilities exactly, and the recorded `dev_selected_policy_c*`
    column is then checked against that identification.

Two literals below are not manuscript numbers and are labelled as such in
their check names: the 12 x 316 = 3,792 cell count, which is the shape of
the crossed universe, and the per-cost composition of the dev selection
(at c=1, m0_context_only in 9 runs and m0_current_paper_rule in 3), which is
the concrete content of the manuscript's "selects one rule from a four-rule
registry ... and then applies that rule unchanged" sentence.

The development split is not part of this release, so the *selection* itself
is provenance, not a recomputation; what is recomputed is that the shipped
comparator arm is bit-for-bit one fixed registry rule per run and cost.

Estimator convention, stated plainly because it differs from the rest of the
release. The point below is an *unweighted* mean over the 3,792 run-by-
question cells, and the bootstrap resamples the 1,806 question keys with a
single multinomial over all four collections at once. That is the estimator
that produced the published artifact and the manuscript's +0.0057. It is not
the collection-macro estimator that `reproduce.py`'s `macro()` applies to the
primary panel, and it does not implement the letter of the manuscript's
inference section ("resamples question keys within each source collection;
collections receive equal macro weight", lines 341-343): the collections are
unequal here (960/296/212/338 question keys), so weighting them equally moves
the c=1 point to +0.0088. The checks below reproduce what the paper prints,
+0.0057, rather than what its inference section describes; the discrepancy is
a property of the published pipeline, not of this release, and is recorded
here and in README_SECTION.md so that a reader is not misled by it.

Note also that the two point checks coincide exactly rather than
independently: every run contributes exactly 316 cells, so the mean over all
cells and the mean of the twelve per-run means are algebraically equal. The
second is kept because it is the estimand of the published per-seed delta
table, rebuilt from the rows, but it is not an independent confirmation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COSTS = (0.5, 1.0, 2.0, 4.0)
CGRV = "confirmatory_selector_verification"
FIXED = "fixed_dev_selected_verification"
# the verification-head registry: always abstain, the context rule, and the
# two corroboration rules (manuscript, "Verification-only fixed comparator")
ABSTAIN = "always_abstain"
REGISTRY = (ABSTAIN, "m0_context_only", "m0_current_paper_rule",
            "m0_strict_corroboration")
# the recorded crossed-bootstrap protocol
DRAWS = 10000
BOOT_SEED = 20260807
# a decision row is unique within a run at this key
ROWKEY = ["question_key", "dataset", "qid", "condition"]


def _u(c):
    return f"utility_c{c:g}"


def _reduce(rows, arms):
    """Mean utility within (run, canonical question key, arm): the recorded
    reduction of the four evidence conditions into one question cluster."""
    cols = [_u(c) for c in COSTS]
    d = rows[rows.method.isin(arms)]
    return d.groupby(["seed", "question_key", "method"], as_index=False) \
            .agg(**{c: (c, "mean") for c in cols})


def _paired(grouped):
    """CGrV minus the fixed comparator on the shared seed x question grid."""
    cols = [_u(c) for c in COSTS]
    a = grouped[grouped.method == CGRV].set_index(["seed", "question_key"])[cols]
    b = grouped[grouped.method == FIXED].set_index(["seed", "question_key"])[cols]
    if set(a.index) != set(b.index):
        raise SystemExit("internal_verification: the two arms do not share "
                         "the seed/question universe")
    common = a.index.sort_values()
    a, b = a.loc[common], b.loc[common]
    return common, (a.to_numpy() - b.to_numpy()).astype(float)


def _crossed_bootstrap(common, values):
    """Crossed seed x question multinomial bootstrap with max-T bands."""
    seeds = common.get_level_values("seed").to_numpy()
    qs = common.get_level_values("question_key").to_numpy()
    s_lev, s_inv = np.unique(seeds, return_inverse=True)
    q_lev, q_inv = np.unique(qs, return_inverse=True)
    point = values.mean(axis=0)
    rng = np.random.default_rng(BOOT_SEED)
    p_s = np.full(len(s_lev), 1 / len(s_lev))
    p_q = np.full(len(q_lev), 1 / len(q_lev))
    boot = np.empty((DRAWS, values.shape[1]), dtype=float)
    for i in range(DRAWS):
        sw = rng.multinomial(len(s_lev), p_s)
        qw = rng.multinomial(len(q_lev), p_q)
        w = sw[s_inv] * qw[q_inv]
        denom = w.sum()
        boot[i] = (w[:, None] * values).sum(axis=0) / denom if denom else point
    sd = boot.std(axis=0, ddof=1)
    z = np.divide(boot - point[None, :], sd[None, :],
                  out=np.zeros_like(boot), where=sd[None, :] > 0)
    max_t = np.quantile(np.max(np.abs(z), axis=1), 0.95)
    return point, point - max_t * sd, point + max_t * sd


def _identify_registry_rule(rows, seed, cost):
    """Which registry rule does the comparator arm reproduce in this run?

    Returns every registry member whose per-row utilities equal the shipped
    comparator's, so an ambiguous match is visible rather than hidden.
    """
    run = rows[rows.seed == seed]
    fx = run[run.method == FIXED].set_index(ROWKEY).sort_index()
    if not fx.index.is_unique:
        raise SystemExit(f"internal_verification: comparator rows for seed "
                         f"{seed} are not unique at {ROWKEY}")
    target = fx[_u(cost)].to_numpy()
    hits = []
    for name in REGISTRY:
        if name == ABSTAIN:
            # abstaining returns nothing, so A = W = 0 and u = 0 at every cost
            cand = np.zeros_like(target)
        else:
            arm = run[run.method == name].set_index(ROWKEY).sort_index()
            if list(arm.index) != list(fx.index):
                raise SystemExit(f"internal_verification: arm {name} does not "
                                 f"cover the comparator rows for seed {seed}")
            cand = arm[_u(cost)].to_numpy()
        if np.array_equal(cand, target):
            hits.append(name)
    return hits


def run(D, check):
    P = D / "internal_verification"
    rows = pd.read_parquet(P / "rows/verification_rowlevel.parquet")
    seeds = sorted(int(s) for s in rows.seed.unique())

    # ---- the rows themselves -------------------------------------------
    print("== internal_verification: row-level integrity of the twelve runs ==")
    for c in COSTS:
        a = rows[f"correct_c{c:g}"].to_numpy().astype(float)
        w = rows[f"wrong_c{c:g}"].to_numpy().astype(float)
        r = rows[f"returned_c{c:g}"].to_numpy().astype(float)
        resid = float(np.abs(rows[_u(c)].to_numpy() - (a - c * w)).max())
        viol = float((a + w > r + 1e-9).sum())
        check(f"utility identity u = A - cW and A+W<=R, c={c:g} "
              f"({len(rows)} rows)", resid + viol, 0.0, tol=1e-9)
    # "in at least one of the twelve runs" -- manuscript line 837
    check("twelve training runs in the row table", float(len(seeds)), 12, tol=0)
    check("verification-head arms shipped (structural)",
          float(rows.method.nunique()), 5, tol=0)
    check("all arms share one row universe per run (structural)",
          float(rows.groupby(["seed", "method"]).size()
                .unstack().nunique(axis=1).ne(1).sum()), 0, tol=0)

    # ---- the four-rule registry ----------------------------------------
    print("== internal_verification: the four-rule fixed comparator registry ==")
    shipped = {m for m in rows.method.unique() if m.startswith("m0_")}
    # "selects one rule from a four-rule registry" -- manuscript line 834
    check("registry members for the verification head",
          float(len(shipped | {ABSTAIN})), 4, tol=0)

    selected, unique_hits = {}, 0
    for c in COSTS:
        for s in seeds:
            hits = _identify_registry_rule(rows, s, c)
            unique_hits += int(len(hits) == 1)
            selected[(s, c)] = hits[0] if len(hits) == 1 else "|".join(hits)
    check("runs x costs with exactly one matching registry rule",
          float(unique_hits), len(seeds) * len(COSTS), tol=0)

    # the identification above never touched dev_selected_policy_c*; check it
    recorded = 0
    for c in COSTS:
        col = f"dev_selected_policy_c{c:g}"
        got = (rows[rows.method == FIXED].groupby("seed")[col]
               .agg(lambda v: sorted(set(v.dropna()))))
        for s in seeds:
            recorded += int(got.loc[s] != [selected[(s, c)]])
    check("recorded dev_selected_policy disagreements with the rows",
          float(recorded), 0, tol=0)

    # composition of the dev selection (not a manuscript literal)
    want_sel = {0.5: {"m0_context_only": 12},
                1.0: {"m0_context_only": 9, "m0_current_paper_rule": 3},
                2.0: {"m0_strict_corroboration": 7, "m0_current_paper_rule": 5},
                4.0: {"m0_strict_corroboration": 12}}
    for c in COSTS:
        for rule in REGISTRY:
            n = sum(selected[(s, c)] == rule for s in seeds)
            if n == 0 and rule not in want_sel[c]:
                continue
            check(f"runs applying {rule} at c={c:g} (not a manuscript literal)",
                  float(n), want_sel[c].get(rule, 0), tol=0)

    # ---- the comparison itself ------------------------------------------
    print("== internal_verification: CGrV minus the fixed comparator at c=1 ==")
    grouped = _reduce(rows, (CGRV, FIXED))
    common, values = _paired(grouped)
    # "the 1,806 canonical question keys that enter a held-out test fold in at
    # least one of the twelve runs" -- manuscript lines 836-837
    check("canonical question keys entering a test fold",
          float(len(set(common.get_level_values("question_key")))), 1806, tol=0)
    check("seed x question cells compared (structural)",
          float(len(common)), 3792, tol=0)

    j = COSTS.index(1.0)
    # "CGrV differs from this comparator by +0.0057" -- manuscript line 838
    check("CGrV minus fixed comparator, c=1",
          float(values[:, j].mean()), 0.0057)
    per_seed = pd.Series(values[:, j]).groupby(
        common.get_level_values("seed").to_numpy()).mean()
    check("CGrV minus fixed comparator, c=1, mean of twelve per-run deltas",
          float(per_seed.mean()), 0.0057)

    point, sim_lo, sim_hi = _crossed_bootstrap(common, values)
    # "[-0.0018, +0.0131]" -- manuscript line 839
    check("simultaneous lower bound, c=1", float(sim_lo[j]), -0.0018)
    check("simultaneous upper bound, c=1", float(sim_hi[j]), 0.0131)
