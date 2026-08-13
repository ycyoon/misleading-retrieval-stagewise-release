# What Learning Buys under Misleading Retrieval — code and data

Anonymous release accompanying the WSDM 2027 submission *What Learning Buys
under Misleading Retrieval: A Stagewise Decomposition of Answer Selection
over Nested Policy Classes*.

## Contents

| path | contents |
|---|---|
| `data/panel/` | panel manifests and the frozen candidate rows for the 400-cell primary panel, the 300-cell extension and the second-reader inventory. The rows are redacted: `supplied_context` and `reference_passages` are removed, because the paper does not release the poisoned passages themselves. Answers, gold-answer sets, generation scores, identifiers and evidence conditions are unchanged. |
| `data/decision_rows/` | row-level decisions for every policy arm on all inventories. Every utility in the paper recomputes from these rows: per-cell utility is `correct_return - cost * wrong_return`, macro-averaged over question keys within collection, collections within seed, then seeds. |
| `data/jury/` | the 447-item adjudication inventory, each judge's raw outputs for the main task and every control family, per-judge manifests (model snapshot identifiers, decoding settings, prompts), the settled labels and the summary. |
| `data/stagewise/` | the stagewise decomposition, per-cell percentile bands, the one-sided simultaneous upper bound, per-fold comparator selections, the exact max-T statistic definition (`max_t_definition.json`), and the subset (composition) analysis. |
| `data/policy_family/` | the restricted-21 versus symmetric-23 deterministic-family sensitivity. |
| `data/second_reader/` | the Llama-3.1-70B-Instruct repetition of the stagewise evaluation. |
| `data/extension/` | extension bands, per-collection contrasts and the in-extension closure. |
| `data/verification/`, `data/ablation/`, `data/loso/`, `data/recalibration/`, `data/reference_robustness/`, `data/diagnosis/` | the remaining analyses, one directory per family; `recalibration/` carries the per-source calibrated probabilities and thresholds, `diagnosis/` the sufficiency-crosscheck replay and its interval. |
| `code/` | the analysis scripts exactly as run. They are the authoritative implementation of every statistic in the paper. They reference the original repository layout, so they are provided as reference implementations; every number they produced is present under `data/` and recomputable from the decision rows. |
| `MANIFEST.json` | sha256 of every file and the artifact path each was copied from. |

## Reproducing the numbers

One command recomputes the paper's headline numbers from the released rows
and checks each against the published value:

```bash
python -m pip install -r code/requirements.txt   # numpy, pandas, pyarrow
python code/reproduce.py                          # run from this directory
```

The script exits nonzero on any disagreement. Expected final line:

```
32 of 32 checks pass -- the released data reproduce the published numbers
```

It verifies, from raw rows only: the utility identity on every decision-row
file; the primary panel's oracle (0.6150), headroom and G1 at all four
costs, with the two question-keyed folds rebuilt from the recorded split
seed 20260811; the decomposition identity on all three panels; the second
reader's oracle (0.6675); the extension's prompted-minus-carried contrast
(+0.0867) and the three per-collection points; the two ablation effects at
c=2; the jury label distribution (202/226/19) and Fleiss' kappa (0.822)
recomputed from the per-judge raw verdicts; and the Llama sufficiency gap
(0.741) with its bootstrap interval ([0.706, 0.778], seed 42). Runtime is
about one minute on a laptop.

Beyond the driver:

* Utilities: recompute from `data/decision_rows/` with the macro-average
  recipe above; the row-level identity `correct + wrong = returned` holds on
  every row.
* Intervals: every band file records its `draws` and `bootstrap_seed`
  columns; the scripts repeat policy selection inside each draw. The max-T
  construction is stated in `data/stagewise/max_t_definition.json` and
  implemented in `code/telescoping_decomposition.py`.
* Jury labels: `code/jury_aggregate.py` reproduces the settled labels, the
  entitlement rule and both no-reader bounds from the per-judge raw outputs.

## Redaction policy

The only removed content is passage text (`supplied_context`,
`reference_passages`) in the candidate rows, consistent with the paper's
ethics statement. No decision, label, score or identifier is altered.

## License

To be set by the authors before the public (de-anonymised) release.

## Citation

Added at camera-ready; the release is anonymous during review.
