"""Widen the fixed policy class to every label-free deterministic rule we can build.

The first stage held four policies: three single channel rules and abstention.
That is not the set a deployer would consider, and it is not the set that makes
the negative result strong. A cascade that returns the parametric answer when it
exists and the reference answer otherwise needs no supervision, no fitted score
and no development data beyond what the single channel rules already use; so does
a rule that answers only where two channels agree.

If learned selection cannot beat that larger class either, the finding is much
harder to dismiss. If it can, the finding was an artefact of a thin comparator
set, which is exactly the failure this paper is about.

The class enumerated here is:

  the three single channel rules and abstention, as before
  every ordered cascade over a subset of the channels, falling through to
  abstention, and every cascade ending in the context rule
  agreement rules that answer only where a given pair, or any pair, agrees

Nothing here fits a score or sees a label. Answers come from the frozen
candidates and correctness from the frozen decision rows, so no new generation
is needed.

Run (from <repository root>):
  python -m \\
      experiments.scripts.maintrack.fixed_policy_closure
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/maintrack_v1"
ARMS = A / "v12_10_verification_arms/panel_all_arms.parquet"
OUT = A / "v12_17_fixed_closure"
COSTS = (0.5, 1.0, 2.0, 4.0)
CHANNELS = {"C": "fixed_recovery", "R": "fixed_reference", "P": "fixed_parametric"}


def channel_table(arms: pd.DataFrame) -> pd.DataFrame:
    """Per cell and seed: availability, correctness and answer for each channel.

    A single channel rule returns exactly when its channel is substantive and
    does not depend on the cost, so one cost slice carries everything.
    """
    d = arms[arms.cost == 1.0]
    out = None
    for tag, arm in CHANNELS.items():
        s = (d[d.method_id == arm]
             .set_index(["cell_id", "seed_id"])
             [["returned", "correct_return", "returned_answer",
               "source_dataset", "canonical_qkey"]]
             .rename(columns={"returned": f"av_{tag}",
                              "correct_return": f"ok_{tag}",
                              "returned_answer": f"ans_{tag}"}))
        keep = ["source_dataset", "canonical_qkey"] if out is None else []
        s = s[[f"av_{tag}", f"ok_{tag}", f"ans_{tag}"] + keep]
        out = s if out is None else out.join(s, how="outer")
    return out.reset_index()


def norm(x) -> str:
    return " ".join(str(x).lower().split())


def policies():
    """(name, decide) where decide maps a row to a channel tag or None."""
    out = []
    for tag in CHANNELS:
        out.append((f"fixed_{tag}", lambda r, t=tag: t if r[f"av_{t}"] else None))
    out.append(("abstain", lambda r: None))

    order_pool = [t for t in CHANNELS]
    for k in (2, 3):
        for order in itertools.permutations(order_pool, k):
            name = "cascade_" + "".join(order)

            def decide(r, o=order):
                for t in o:
                    if r[f"av_{t}"]:
                        return t
                return None
            out.append((name, decide))

    def agree_pair(r, a, b):
        if r[f"av_{a}"] and r[f"av_{b}"] and norm(r[f"ans_{a}"]) == norm(r[f"ans_{b}"]):
            return a
        return None
    for a, b in itertools.combinations(order_pool, 2):
        out.append((f"agree_{a}{b}", lambda r, x=a, y=b: agree_pair(r, x, y)))

    def agree_any(r):
        for x, y in itertools.combinations(order_pool, 2):
            t = agree_pair(r, x, y)
            if t:
                return t
        return None
    out.append(("agree_any", agree_any))

    # Agreement with a fallback. This used to exist for the parametric channel
    # alone, and the parametric variant is what the first stage selects at two
    # of the four costs, which is exactly the shape of a family fitted to its
    # own winner. All three fallbacks are here now, and abstaining is the fourth
    # since that is what agree_any already does.
    def agree_any_else(r, tag):
        t = agree_any(r)
        if t:
            return t
        return tag if r[f"av_{tag}"] else None
    for tag in CHANNELS:
        out.append((f"agree_any_else_{tag}",
                    lambda r, x=tag: agree_any_else(r, x)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=str(ARMS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    arms = pd.read_parquet(a.arms)
    tab = channel_table(arms)
    print(f"[closure] {tab.cell_id.nunique()} cells x {tab.seed_id.nunique()} seeds")

    recs = []
    for name, decide in policies():
        pick = tab.apply(decide, axis=1)
        ok = pd.Series(
            [False if t is None else bool(r[f"ok_{t}"])
             for t, (_, r) in zip(pick, tab.iterrows())], index=tab.index)
        ret = pick.notna()
        for c in COSTS:
            u = ok.astype(float) * ret - c * ((~ok) & ret)
            g = (tab.assign(u=u)
                 .groupby(["source_dataset", "canonical_qkey"]).u.mean()
                 .groupby("source_dataset").mean().mean())
            recs.append({"policy": name, "cost": c, "utility": float(g),
                         "answer_rate": float(ret.mean())})
    lv = pd.DataFrame(recs)

    # decision rows in the panel schema, so the stagewise decomposition can take
    # the widened class without knowing how these policies were built
    rows = []
    base = arms[arms.method_id == CHANNELS["C"]].set_index(["cell_id", "seed_id"])
    for name, decide in policies():
        if name in ("fixed_C", "fixed_R", "fixed_P", "abstain"):
            continue                      # already in the panel under their own ids
        pick = tab.apply(decide, axis=1)
        for c in COSTS:
            for t, (_, r) in zip(pick, tab.iterrows()):
                ret = int(t is not None)
                ok = int(ret and bool(r[f"ok_{t}"]))
                rows.append({
                    "panel_id": "closure", "source_dataset": r.source_dataset,
                    "qid": None, "canonical_qkey": r.canonical_qkey,
                    "cell_id": r.cell_id, "method_id": f"closure_{name}",
                    "method_family": "fixed_closure", "head": "recovery",
                    "seed_id": r.seed_id, "cost": c,
                    "action": "return" if ret else "abstain",
                    "returned_source": {"C": "a_C", "R": "a_R", "P": "a_P"}.get(t, "NONE"),
                    "returned_answer": r[f"ans_{t}"] if t else "",
                    "returned": ret, "correct_return": ok,
                    "wrong_return": ret - ok,
                    "answer_utility": ok - c * (ret - ok),
                    "score_optional": None, "calibrated_score_optional": None,
                    "selected_on": "deterministic_rule",
                    "technical_status": "ok"})
    closure = pd.DataFrame(rows)

    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    # coverage and accuracy-given-coverage are what make a fixed channel rule
    # strong, and the paper quotes them, so they are written here rather than
    # recomputed in prose
    cond = []
    for tag in CHANNELS:
        per = tab.groupby("cell_id")[[f"av_{tag}", f"ok_{tag}"]].max()
        av = float(per[f"av_{tag}"].mean())
        ok = float(per[f"ok_{tag}"].mean())
        cond.append({"channel": tag, "availability": av,
                     "correct_over_all_cells": ok,
                     "correct_given_available": ok / av if av else float("nan")})
    pd.DataFrame(cond).to_csv(outd / "channel_conditional_accuracy.csv",
                              index=False)
    lv.to_csv(outd / "fixed_closure_levels.csv", index=False)
    closure.to_parquet(outd / "closure_decision_rows.parquet", index=False)
    # How large the class would be if it were closed rather than prespecified.
    # A policy that sees only which channels are available chooses one action per
    # availability state, and a state with k channels available admits k+1 (each
    # channel, or abstention). The paper contrasts its 21 against this number, so
    # the number is computed rather than asserted.
    all_maps = 1
    for k in (1, 1, 1, 2, 2, 2, 3):
        all_maps *= k + 1
    (outd / "manifest.json").write_text(json.dumps({
        "n_policies": int(lv.policy.nunique()),
        "deterministic_maps_over_availability_alone": all_maps,
        "why_not_all_of_them": "the family is prespecified from rules a "
                               "deployer would write; enumerating every "
                               "mapping would fit the panel rather than "
                               "describe a rule",
        "class": "single channel rules, abstention, ordered cascades over every "
                 "subset of the three channels, pairwise and any-pair agreement "
                 "rules, and agreement falling through to the parametric rule",
        "fits_no_score": True,
        "sees_no_label": True,
        "source": a.arms,
        "source_script":
            "experiments/scripts/maintrack/fixed_policy_closure.py",
    }, indent=1))

    print(f"\n{lv.policy.nunique()} label-free deterministic policies\n")
    for c in COSTS:
        s = lv[lv.cost == c].sort_values("utility", ascending=False)
        best = s.iloc[0]
        old = s[s.policy.isin(["fixed_C", "fixed_R", "fixed_P", "abstain"])
                ].iloc[0]
        print(f"  c={c}: best of the widened class {best.policy} "
              f"{best.utility:+.4f}   best of the original four {old.policy} "
              f"{old.utility:+.4f}   gain {best.utility - old.utility:+.4f}")
    print(f"\n[closure] -> {outd.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
