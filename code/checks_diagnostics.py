"""diagnostics: the Section 5.1 sufficiency-versus-credibility claims.

Recomputes, from the released per-record verdict tables alone, every number
the manuscript prints in "Sufficiency does not establish credibility": the
poisoned-minus-insufficient sufficiency gap and its interval for Qwen2.5-72B
on the 2,400-record ClashEval diagnostic set, for Mistral-Small-24B on the
2,396-record replication, and for Qwen2.5-72B on RAMDocs and on RAGuard; plus
the rate at which the context-trusting reader returns the planted false answer
on poisoned ClashEval contexts, with its interval.

Nothing is read back from a summary file: this family ships no summaries at
all. Each table holds one row per benchmark record, carrying only the
sufficiency verdict per condition and, for ClashEval/Qwen, one derived boolean
flag. Delete anything else in the release and every check below still runs.

Two bootstrap conventions are in play, because the manuscript's five numbers
come from two generating drivers, and the interval endpoints do not agree to
three decimals across them:

  * the reversal-diagnostic driver (ClashEval/Qwen, RAMDocs) resamples whole
    records with CPython's `random.Random(42).randrange(n)`, computes the two
    conditional pass rates inside each resample -- so records whose verdict is
    missing for a condition drop out of that condition's denominator, resample
    by resample -- and reports the bootstrap MEAN as the point estimate.
  * the third-judge driver (Mistral, RAGuard) forms one signed gap per record
    over the records judged under both conditions, resamples those per-record
    gaps with `numpy.random.default_rng(42)`, and reports the plug-in mean as
    the point estimate.

Both use 1,000 draws and the percentile method. The first convention needs
CPython's Mersenne Twister stream, so `_MT19937` below reproduces it exactly
(seeding, tempering and the `getrandbits`-with-rejection path that
`randrange` uses) on numpy alone, rather than importing `random`. The
reproduction is exact, not approximate: the released tables return the paper's
digits, not merely values near them, so every comparison here is made against
the printed three-decimal value with zero tolerance.

The reader-side flag `emits_planted` is derived before redaction, because the
passage text and the planted answer string are withheld: it is true when the
reader's answer on the poisoned context contains the planted false answer
after casefolding and whitespace collapse. It agrees with the run's own
`outputs_a_false` detector on all 2,400 records, so only one of the two is
shipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOOT_ITERS = 1000
BOOT_SEED = 42
SUF = "sufficient"

# ---------------------------------------------------------------------------
# CPython's random.Random stream, on numpy only.
# ---------------------------------------------------------------------------
_N, _M = 624, 397
_A = np.uint32(0x9908B0DF)
_UPPER, _LOWER = np.uint32(0x80000000), np.uint32(0x7FFFFFFF)


class _MT19937:
    """The exact generator behind `random.Random(seed)` for integer seeds.

    `random.seed(42)` runs init_by_array over the seed's 32-bit little-endian
    words; `randrange(n)` draws `getrandbits(n.bit_length())` and rejects until
    the draw is below n. Both are reproduced here so that the resample indices
    match the diagnostic driver's index for index.
    """

    def __init__(self, seed: int):
        key = []
        s = int(seed)
        while s:
            key.append(s & 0xFFFFFFFF)
            s >>= 32
        key = key or [0]
        mt = [19650218] + [0] * (_N - 1)
        for i in range(1, _N):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) \
                & 0xFFFFFFFF
        i, j = 1, 0
        for _ in range(max(_N, len(key))):
            mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525))
                     + key[j] + j) & 0xFFFFFFFF
            i, j = i + 1, j + 1
            if i >= _N:
                mt[0], i = mt[_N - 1], 1
            if j >= len(key):
                j = 0
        for _ in range(_N - 1):
            mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941))
                     - i) & 0xFFFFFFFF
            i += 1
            if i >= _N:
                mt[0], i = mt[_N - 1], 1
        mt[0] = 0x80000000
        self._mt = np.array(mt, dtype=np.uint32)

    def _twist(self):
        old, new = self._mt, self._mt.copy()

        def step(hi, lo, xor):
            y = (hi & _UPPER) | (lo & _LOWER)
            return xor ^ (y >> np.uint32(1)) ^ np.where(y & np.uint32(1),
                                                        _A, np.uint32(0))

        # the sequential twist reads mt[i+1] before it is rewritten and
        # mt[i+397-624] after; three slices keep both facts true
        for lo, hi in ((0, _N - _M), (_N - _M, 2 * (_N - _M)),
                       (2 * (_N - _M), _N - 1)):
            i = np.arange(lo, hi)
            src = old[i + _M] if lo == 0 else new[i - (_N - _M)]
            new[i] = step(old[i], old[i + 1], src)
        new[_N - 1] = step(old[_N - 1:], new[:1], new[_M - 1:_M])[0]
        self._mt = new

    def u32(self, count: int) -> np.ndarray:
        """The next `count` tempered outputs, i.e. getrandbits(32) repeated."""
        out, need = [], count
        while need > 0:
            self._twist()
            y = self._mt.copy()
            y ^= y >> np.uint32(11)
            y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
            y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
            y ^= y >> np.uint32(18)
            out.append(y[:need])
            need -= _N
        return np.concatenate(out)

    def randbelow(self, bound: int, count: int) -> np.ndarray:
        """`[randrange(bound) for _ in range(count)]`, as an int array."""
        k = int(bound).bit_length()
        shift = np.uint32(32 - k)
        keep, have = [], 0
        while have < count:
            block = max(4 * (count - have), 1024)
            r = self.u32(block) >> shift
            r = r[r < bound]
            keep.append(r)
            have += len(r)
        return np.concatenate(keep)[:count].astype(np.int64)


def _record_resamples(n: int) -> np.ndarray:
    """The driver's `[rows[rng.randrange(n)] for _ in range(n)]`, 1000 times.

    A fresh `random.Random(42)` is created per metric there, so every metric on
    a table of n records sees this same index matrix.
    """
    idx = _MT19937(BOOT_SEED).randbelow(n, BOOT_ITERS * n)
    return idx.reshape(BOOT_ITERS, n)


def _pctl(draws):
    lo, hi = np.percentile(np.asarray(draws, float), [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


# ---------------------------------------------------------------------------
# the two statistics, each in its generating driver's convention
# ---------------------------------------------------------------------------
def _rate_gap(df):
    """Pass-rate difference and its bootstrap, resampling whole records.

    Missing verdicts leave that condition's denominator inside each resample,
    exactly as the driver's rate helper drops `None` before dividing.
    """
    ok_p = df.suffA_C_poison.notna().to_numpy()
    ok_i = df.suffA_C_insuf.notna().to_numpy()
    hit_p = (df.suffA_C_poison == SUF).to_numpy()
    hit_i = (df.suffA_C_insuf == SUF).to_numpy()
    idx = _record_resamples(len(df))
    gaps = (hit_p[idx].sum(1) / ok_p[idx].sum(1)
            - hit_i[idx].sum(1) / ok_i[idx].sum(1))
    return round(float(gaps.mean()), 3), _pctl(gaps)


def _paired_gap(df):
    """Mean signed per-record gap and its bootstrap over those gaps.

    Only records judged under both conditions carry a gap, which is how the
    third-judge driver builds its resampling population.
    """
    both = df[df.suffA_C_poison.notna() & df.suffA_C_insuf.notna()]
    g = ((both.suffA_C_poison == SUF).to_numpy(float)
         - (both.suffA_C_insuf == SUF).to_numpy(float))
    rng = np.random.default_rng(BOOT_SEED)
    draws = g[rng.integers(0, len(g), size=(BOOT_ITERS, len(g)))].mean(axis=1)
    return round(float(g.mean()), 3), _pctl(draws)


def _emission(df):
    """Planted-answer emission over all records, same record resamples."""
    e = df.emits_planted.to_numpy(bool)
    idx = _record_resamples(len(df))
    draws = e[idx].mean(axis=1)
    return round(float(draws.mean()), 3), _pctl(draws)


def run(D, check):
    P = D / "diagnostics"
    tab = {n: pd.read_parquet(P / f"{n}_per_record.parquet") for n in
           ("clasheval_qwen72b", "clasheval_mistral24b",
            "ramdocs_qwen72b", "raguard_qwen72b")}
    # the Llama crosscheck rows, released by an earlier family: used below for
    # that leg's stated basis, and here as the outside anchor that keeps the
    # two ClashEval inventories from being counts of this family's own files
    rows = pd.read_csv(D / "diagnosis" / "sufficiency_crosscheck_rows.csv")

    print("== diagnostics: record inventories the manuscript names ==")
    cq, cm = tab["clasheval_qwen72b"], tab["clasheval_mistral24b"]
    # "On the 2,400-record ClashEval diagnostic set" (submission line 524)
    check("ClashEval diagnostic records, Qwen2.5-72B", float(len(cq)),
          2400, tol=0)
    # "on the 2,396 records with a nonempty insufficient context" (line 528)
    check("ClashEval replication records, Mistral-Small-24B", float(len(cm)),
          2396, tol=0)
    # "On a fixed 600-question subset" (line 529) -- of the set counted above,
    # so every crosscheck question is one of these records and carries the same
    # domain. Recomputed against rows this family did not write, which is what
    # makes the two counts falsifiable rather than self-reported.
    dom = cq.set_index("qid").domain
    sub = rows.drop_duplicates("qid").set_index("qid").domain
    check("crosscheck questions found in the diagnostic set, same domain",
          float((sub == dom.reindex(sub.index)).sum()), 600, tol=0)
    # the replication is judged on the same records, minus those it dropped
    check("replication records outside the diagnostic set",
          float((~cm.qid.isin(set(cq.qid))).sum()), 0, tol=0)

    print("== diagnostics: sufficiency gap, poisoned minus insufficient ==")
    # "the difference between the pass rates on poisoned and insufficient
    #  contexts is 0.943 [0.934, 0.953] for Qwen2.5-72B" (lines 524-526)
    pt, (lo, hi) = _rate_gap(tab["clasheval_qwen72b"])
    check("ClashEval gap, Qwen2.5-72B", pt, 0.943, tol=0)
    check("ClashEval gap interval, lower", lo, 0.934, tol=0)
    check("ClashEval gap interval, upper", hi, 0.953, tol=0)

    # "Mistral-Small-24B yields 0.851 [0.837, 0.866]" (lines 526-528)
    pt, (lo, hi) = _paired_gap(tab["clasheval_mistral24b"])
    check("ClashEval gap, Mistral-Small-24B", pt, 0.851, tol=0)
    check("Mistral gap interval, lower", lo, 0.837, tol=0)
    check("Mistral gap interval, upper", hi, 0.866, tol=0)

    # "the corresponding gaps are 0.693 [0.623, 0.763] on RAMDocs and
    #  0.248 [0.212, 0.279] on RAGuard" (lines 532-534)
    pt, (lo, hi) = _rate_gap(tab["ramdocs_qwen72b"])
    check("RAMDocs gap, Qwen2.5-72B", pt, 0.693, tol=0)
    check("RAMDocs gap interval, lower", lo, 0.623, tol=0)
    check("RAMDocs gap interval, upper", hi, 0.763, tol=0)

    pt, (lo, hi) = _paired_gap(tab["raguard_qwen72b"])
    check("RAGuard gap, Qwen2.5-72B", pt, 0.248, tol=0)
    check("RAGuard gap interval, lower", lo, 0.212, tol=0)
    check("RAGuard gap interval, upper", hi, 0.279, tol=0)

    print("== diagnostics: planted-answer emission on poisoned contexts ==")
    # "returns the false answer stated in the context on 0.740 [0.722, 0.757]
    #  of poisoned ClashEval records" (lines 536-538)
    pt, (lo, hi) = _emission(tab["clasheval_qwen72b"])
    check("planted-answer emission, Qwen2.5-72B reader", pt, 0.740, tol=0)
    check("emission interval, lower", lo, 0.722, tol=0)
    check("emission interval, upper", hi, 0.757, tol=0)

    print("== diagnostics: basis of the 600-question Llama-3.1-70B crosscheck ==")
    # The gap itself (0.741 [0.706, 0.778]) is recomputed by the driver from
    # data/diagnosis/sufficiency_crosscheck_rows.csv and is not repeated here.
    # Its stated basis is not otherwise checked, so it is checked here from the
    # same rows: "On a fixed 600-question subset, 100 per ClashEval domain,
    # ... from 597 poisoned and 430 insufficient context judgments" (529-532).
    check("crosscheck subset questions", float(rows.qid.nunique()), 600, tol=0)
    per_dom = rows.drop_duplicates("qid").domain.value_counts()
    check("crosscheck questions per ClashEval domain (min)",
          float(per_dom.min()), 100, tol=0)
    check("crosscheck questions per ClashEval domain (max)",
          float(per_dom.max()), 100, tol=0)
    judged = rows[rows.label.notna()].cond.value_counts()
    check("crosscheck poisoned-context judgments",
          float(judged.get("C_poison", 0)), 597, tol=0)
    check("crosscheck insufficient-context judgments",
          float(judged.get("C_insuf", 0)), 430, tol=0)
