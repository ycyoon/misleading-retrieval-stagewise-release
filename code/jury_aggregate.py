"""Aggregate the jury, validate it on its own controls, and bound the unresolved.

The rule is fixed before the numbers are looked at, and it is the same one the
earlier two-judge analysis used: a judge arbitrates a question only where it
passed the constructed controls for that question. Controls have answers known by
construction -- the accepted answer itself must be called correct, another item's
answer must be called incorrect -- so no opinion enters the validation.

An item's jury label is the majority of the judges entitled to vote on it.
Anything without a majority, and anything a judge marks `unresolved`, stays
contested; contested items are not resolved by the authors but carried into
conservative and optimistic bounds, which is the same treatment abstention gets
elsewhere in this paper.

Reported per judge: positive-control recall, foreign-control rejection, verdict
mix, and agreement with the rest of the panel. Reported for the panel: unanimous
and majority coverage, contested rate, pairwise and Fleiss agreement, and
leave-one-family-out labels, which matter here because one judge shares a model
family with the reader whose answers are being judged.

Run (from <repository root>):
  python -m \
      experiments.scripts.maintrack.jury_aggregate
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
JURY = ROOT / "artifacts/maintrack_v1/v12_15_jury"
JUDGES = ("qwen72b", "mistral24b", "phi4")
READER_FAMILY = "qwen72b"      # shares a family with the reader under test


def load(judge: str, cond: str, base: Path = None):
    p = (base or JURY) / f"{judge}_{cond}.csv"
    if not p.exists():
        return {}
    return {r["item_id"]: r for r in csv.DictReader(open(p))}


def fleiss(labels_per_item):
    """Fleiss' kappa over items rated by the same number of judges."""
    cats = sorted({v for row in labels_per_item for v in row})
    n = len(labels_per_item[0]) if labels_per_item else 0
    if n < 2 or not labels_per_item:
        return None
    N = len(labels_per_item)
    P_i, p_j = [], {c: 0 for c in cats}
    for row in labels_per_item:
        cnt = Counter(row)
        for c in cats:
            p_j[c] += cnt[c]
        P_i.append((sum(v * v for v in cnt.values()) - n) / (n * (n - 1)))
    for c in cats:
        p_j[c] /= (N * n)
    Pbar = sum(P_i) / N
    Pe = sum(v * v for v in p_j.values())
    return None if Pe == 1 else (Pbar - Pe) / (1 - Pe)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jury", default=str(JURY))
    ap.add_argument("--controls", default=None,
                    help="directory holding the controls; defaults to the jury "
                         "directory, and every control found there is used")
    ap.add_argument("--two-control", action="store_true",
                    help="the superseded pair of an accepted and a foreign "
                         "answer only, kept so the earlier artifact can be "
                         "reproduced; the paper reports the full set")
    a = ap.parse_args(argv)

    base = Path(a.jury)
    present = [j for j in JUDGES if (base / f"{j}_main.csv").exists()]
    if not present:
        raise SystemExit("no judge output found")
    main_v = {j: load(j, "main", base) for j in present}
    cbase = Path(a.controls) if a.controls else base
    # A judge is entitled on a question only where it passed every control it
    # saw there. The original pair is one accepted answer, which must be called
    # correct, and one foreign answer, which must be called incorrect. A foreign
    # answer can be accidentally right, so the harder set adds a false conjunct
    # and a perturbed number, both wrong by construction, and a second accepted
    # answer, which is right without being the string a matcher would key on.
    # The full set is the default. It used to be reached only by passing
    # --controls, so running the script bare silently rebuilt the artifact under
    # the superseded two-control rule and changed 44 entitlements without saying
    # so. Falling back quietly to a weaker rule is the failure mode this whole
    # campaign is about, so a missing control file is now an error, and the
    # summary records which controls the labels came from.
    POS = ["positive"] if a.two_control else ["positive", "alias"]
    NEG = ["foreign"] if a.two_control else ["foreign", "conjunct", "perturbed"]
    absent = [f"{j}_{c}.csv" for j in present for c in POS + NEG
              if not (cbase / f"{j}_{c}.csv").exists()]
    if absent:
        raise SystemExit(f"controls missing from {cbase}: {absent}; pass "
                         f"--two-control to reproduce the superseded artifact")
    pos_v = {j: {c: load(j, c, cbase) for c in POS} for j in present}
    for_v = {j: {c: load(j, c, cbase) for c in NEG} for j in present}

    # A negative control is only known-wrong if its constructed answer is not
    # itself accepted for the target item. The number-shift and conjunct
    # constructions check that at build time; the permuted foreign answer does
    # not, and two of the 447 land on an accepted answer. A judge that calls
    # those correct is right, so scoring them against it would disqualify it for
    # being correct. Those item-control pairs are dropped from entitlement here
    # and counted, rather than left to shave a judge's entitlement silently.
    def _n(x):
        return re.sub(r"\W+", " ", str(x).lower()).strip()

    inv = {r["item_id"]: r for r in
           csv.DictReader(open(base / "inventory_447.csv"))}

    def _accepted(row):
        v = row.get("accepted_answers", "")
        try:
            x = ast.literal_eval(v)
            if isinstance(x, (list, tuple)):
                return [str(i).strip() for i in x if str(i).strip()]
        except Exception:
            pass
        return [t.strip() for t in str(v).split(";") if t.strip()]

    excused = set()
    for j in present:
        for cond, d in for_v[j].items():
            for it, r in d.items():
                row = inv.get(it)
                if row is None:
                    continue
                ans = r.get("candidate_shown") or r.get("candidate_answer") or ""
                if ans and _n(ans) in {_n(x) for x in _accepted(row)}:
                    excused.add((cond, it))
    if excused:
        print(f"  {len(set(i for _, i in excused))} item(s) excused from a "
              f"negative control whose constructed answer is accepted there: "
              f"{sorted({c for c, _ in excused})}")
    items = sorted(set(main_v[present[0]]))

    # ---- per-judge control diagnostics ------------------------------------
    diag = {}
    for j in present:
        pos = [r for d in pos_v[j].values() for r in d.values() if r["verdict"]]
        neg = [r for d in for_v[j].values() for r in d.values() if r["verdict"]]
        m = [r for r in main_v[j].values() if r["verdict"]]
        diag[j] = {
            "positive_control_recall":
                sum(1 for r in pos if r["verdict"] == "correct") / max(len(pos), 1),
            "foreign_control_rejection":
                sum(1 for r in neg if r["verdict"] == "incorrect") / max(len(neg), 1),
            "n_positive": len(pos), "n_foreign": len(neg),
            "main_verdicts": dict(Counter(r["verdict"] for r in m)),
            "unparsed_main": sum(1 for r in main_v[j].values() if not r["verdict"]),
        }

    # ---- entitlement: a judge votes on a question only where it passed there
    def entitled(j, item):
        # at least one control of each polarity must exist for the question, and
        # every control that does exist must have been passed
        seen_pos = seen_neg = False
        for d in pos_v[j].values():
            r = d.get(item)
            if r is not None and r["verdict"]:
                seen_pos = True
                if r["verdict"] != "correct":
                    return False
        for cond, d in for_v[j].items():
            if (cond, item) in excused:
                continue
            r = d.get(item)
            if r is not None and r["verdict"]:
                seen_neg = True
                if r["verdict"] != "incorrect":
                    return False
        return seen_pos and seen_neg

    # A label is settled only when at least two entitled families return the same
    # firm verdict. A plain majority of the entitled would let one judge decide an
    # item alone wherever the controls disqualified the other two, which is a
    # single grader wearing the panel's name; eleven of the 447 items sat in that
    # position. Two agreeing families is also the rule the leave-one-family-out
    # column needs, where only two judges remain by construction.
    def settle(votes):
        firm = Counter(v for v in votes.values() if v in ("correct", "incorrect"))
        t = firm.most_common()
        return (t[0][0] if t and t[0][1] >= 2 and t[0][1] * 2 > len(votes)
                else "contested")

    labels, rows = {}, []
    for it in items:
        votes = {j: main_v[j][it]["verdict"] for j in present
                 if entitled(j, it) and main_v[j].get(it, {}).get("verdict")}
        firm = {j: v for j, v in votes.items() if v in ("correct", "incorrect")}
        cnt = Counter(firm.values())
        label = settle(votes)
        labels[it] = label
        rows.append({"item_id": it, "n_entitled": len(votes),
                     "n_firm": len(firm),
                     **{f"vote_{j}": votes.get(j, "") for j in present},
                     "jury_label": label,
                     "unanimous": int(len(firm) == len(votes) and len(cnt) == 1
                                      and len(votes) == len(present))})

    # ---- panel-level agreement -------------------------------------------
    full = [[main_v[j][it]["verdict"] for j in present]
            for it in items
            if all(main_v[j].get(it, {}).get("verdict") for j in present)]
    pairwise = {}
    for x, y in itertools.combinations(present, 2):
        both = [it for it in items
                if main_v[x].get(it, {}).get("verdict")
                and main_v[y].get(it, {}).get("verdict")]
        agree = sum(1 for it in both
                    if main_v[x][it]["verdict"] == main_v[y][it]["verdict"])
        pairwise[f"{x}|{y}"] = agree / max(len(both), 1)

    # ---- leave-one-family-out --------------------------------------------
    loo = {}
    for drop in present:
        keep = [j for j in present if j != drop]
        diff = 0
        for it in items:
            votes = {j: main_v[j][it]["verdict"] for j in keep
                     if entitled(j, it) and main_v[j].get(it, {}).get("verdict")}
            diff += int(settle(votes) != labels[it])
        loo[f"without_{drop}"] = {"labels_changed": diff,
                                  "fraction": diff / max(len(items), 1)}

    outp = base / "jury_labels.csv"
    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    dist = Counter(labels.values())
    summary = {
        "judges": present, "n_items": len(items),
        "positive_controls": POS, "negative_controls": NEG,
        "settlement_rule": "at least two entitled families returning the same "
                           "firm verdict; anything else is contested",
        "entitlement_distribution":
            dict(sorted(Counter(r["n_entitled"] for r in rows).items())),
        "per_judge": diag,
        "label_distribution": dict(dist),
        "contested_rate": dist["contested"] / max(len(items), 1),
        "unanimous_coverage": sum(r["unanimous"] for r in rows) / max(len(rows), 1),
        "pairwise_agreement": pairwise,
        "fleiss_kappa": fleiss(full) if full else None,
        "fleiss_kappa_items": len(full),
        "fleiss_kappa_basis": "raw three-family verdicts on items all three "
                              "parsed, before entitlement, unresolved as its "
                              "own category",
        "leave_one_family_out": loo,
        "reader_family_judge": READER_FAMILY,
        "entitlement_rule": "a judge votes on a question only where it called "
                            "the accepted answer correct and a foreign answer "
                            "incorrect for that question",
        "contested_treatment": "carried into conservative and optimistic "
                               "bounds, not resolved by the authors",
        "source_script": "experiments/scripts/maintrack/jury_aggregate.py",
    }
    (base / "jury_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"[jury] {len(items)} items, judges {present}")
    print(f"  controls: positive {POS}, negative {NEG}")
    for j in present:
        d = diag[j]
        print(f"  {j:12s} positive {d['positive_control_recall']:.4f}  "
              f"foreign {d['foreign_control_rejection']:.4f}  "
              f"main {d['main_verdicts']}")
    print(f"\n  labels {dict(dist)}  contested {summary['contested_rate']:.4f}")
    print(f"  unanimous coverage {summary['unanimous_coverage']:.4f}  "
          f"Fleiss kappa {summary['fleiss_kappa']}")
    print(f"  pairwise {pairwise}")
    print(f"  leave-one-out {loo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
