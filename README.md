# Code and data

Release accompanying the WSDM 2027 submission *What Learning Buys under
Misleading Retrieval: A Stagewise Decomposition of Answer Selection over
Nested Policy Classes*.

## Run

```bash
python -m pip install -r code/requirements.txt   # numpy, pandas, pyarrow
python code/reproduce.py                          # from this directory
```

`code/reproduce.py` recomputes the paper's numbers from the rows in `data/`
and compares each against the printed value. It exits nonzero on any
disagreement and ends with `N of N checks pass`. Runtime is a few minutes.

## Data

| path | contents |
|---|---|
| `data/panel/` | candidate rows and panel manifests for the 400-cell primary panel and the 300-cell extension. Passage text is removed; answers, gold-answer sets, generation scores, identifiers and evidence conditions are unchanged. |
| `data/decision_rows/` | row-level decisions for every policy arm. Per-cell utility is `correct_return - cost * wrong_return`, macro-averaged over question keys within collection, collections within seed, then seeds. |
| `data/stagewise/` | the stagewise decomposition, percentile bands, the one-sided simultaneous upper bound, per-fold comparator selections, and `max_t_definition.json`. |
| `data/second_reader/`, `data/reader_scale/` | the Llama-3.1-70B and Llama-3.1-8B inventories and their ladders. |
| `data/extension/` | extension bands, per-collection contrasts and the in-extension closure. |
| `data/popqa_tail/` | the popularity-stratified head and long-tail panels: decision rows, per-cell tables with retrieval-support flags, ladders and interaction bands. |
| `data/diagnostics/` | per-record sufficiency verdicts and planted-answer flags behind Section 5.1. |
| `data/internal_pool/` | per-record availability and correctness flags for the 13,228-record fitting pool. |
| `data/internal_verification/` | row-level decisions of the twelve internal runs and the fixed comparator registry. |
| `data/robustness_arms/` | decision rows for the corpus-swap, retrieval-depth, reranking and exact-retrieval arms. |
| `data/published_baselines/` | per-cell outputs of the published-system comparisons. |
| `data/jury/` | the 447-item adjudication inventory, each judge's raw outputs and manifests, the settled labels and the summary. |
| `data/selector_code/` | the selector source (`src/models/`, `src/features/`) and `configs/protocol_full.yaml`. |
| `data/policy_family/`, `data/ablation/`, `data/loso/`, `data/recalibration/`, `data/reference_robustness/`, `data/verification/`, `data/diagnosis/` | the remaining analyses, one directory per family. |
| `code/` | the analysis scripts as run, plus `reproduce.py` and the per-family `checks_*.py` it calls. |

Band files record their `draws` and `bootstrap_seed`; the question-keyed fold
seed is 20260811. `reproduce.py` checks point estimates, not bands.

## Redaction

Passage text (`supplied_context`, `reference_passages`, and nested `passages`)
is removed everywhere. No decision, label, score or identifier is altered.

## License

To be set by the authors before the public release.

## External resources this release does not redistribute

The release ships our own code, our decision rows and our derived per-record
tables. It does not redistribute upstream benchmarks, retrieval corpora or
model weights. Each is named below with where to obtain it and what in the
paper depends on it, so a reviewer can rebuild the parts we cannot ship.

### Benchmarks behind the evaluation panels

Every panel cell records the upstream file it came from in its `source_file`
field, and the panel manifests under `data/panel/` record the sha256 of each
source file at the time the panel was frozen.

| resource | where | what depends on it |
|---|---|---|
| SABER benchmark redistribution (`nq-huang`, `triviaqa-huang`, `confiqa-qa`, `confiqa-mc`, `confiqa-mr`, `conflictbank-pilot10k`, `conflictqa-popqa-llama2-7b`) | <https://github.com/xizhu1022/SABER>, directory `data/datasets/` | the 400-cell primary panel and the 300-cell extension; `data/panel/*_panel_manifest.json` lists the exact file and hash per collection |
| ConflictQA (Adaptive Chameleon or Stubborn Sloth, ICLR 2024) | <https://huggingface.co/datasets/osunlp/ConflictQA>, config `ConflictQA-popQA-llama2-7b` | the popularity-stratified head and long-tail panels; supplies the evidence passages we do not release |
| PopQA | <https://huggingface.co/datasets/akariasai/PopQA>, file `test.tsv` | the `s_pop` subject-popularity measure that defines the head and long-tail strata; the quartile thresholds are recorded in `data/popqa_tail/popularity_thresholds.json` |
| ClashEval, RAGuard, RAMDocs, RGB | the sources cited in the paper | the internal fitting pool. `data/internal_pool/` ships per-record availability and correctness flags, not the passages |

To rebuild a panel from scratch, fetch the upstream file named in the manifest,
confirm its sha256 against the manifest, and apply the cell-construction rule
recorded there (for the popularity panels the rule and its frozen
pre-registration are in `data/popqa_tail/DESIGN.md`, whose own sha256 the
builder asserts before sampling).

### Retrieval corpora, encoder and index

| resource | where | what depends on it |
|---|---|---|
| Wikipedia passage collection released with Dense Passage Retrieval | the DPR release (Karpukhin et al., EMNLP 2020) | the reference channel `a_R` on every panel. The exact index size, 21,015,324 passages, is recorded in `data/robustness_arms/provenance/e5exact_manifest.json` |
| CC-News | the CC-News collection as distributed by its maintainers | the corpus-swap robustness row. We record the arm's decisions in `data/robustness_arms/`, not the corpus; the passage count quoted in the paper is a property of the snapshot we indexed and is not recoverable from the released rows |
| `BAAI/bge-base-en-v1.5` | <https://huggingface.co/BAAI/bge-base-en-v1.5> | passage and query encoding for the reference channel |
| `intfloat/e5-base` family | <https://huggingface.co/intfloat> | the reranking and exact-retrieval robustness arms |
| FAISS | <https://github.com/facebookresearch/faiss> | approximate search over the encoded corpus |

Retrieved passage text is withheld release-wide. Where a statistic needs it,
we ship the derived quantity instead: `data/popqa_tail/` carries per-cell
`support_top{1,3,10}` flags, and `data/robustness_arms/` carries the per-cell
decisions of each retrieval arm.

### Readers and judges

| resource | where | what depends on it |
|---|---|---|
| `Qwen/Qwen2.5-72B-Instruct` | <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct> | the main candidate inventory, one sufficiency judge, one jury member |
| `meta-llama/Llama-3.1-70B-Instruct` | <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct> | the second-reader inventory and the sufficiency crosscheck |
| `meta-llama/Llama-3.1-8B-Instruct` | <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> | the smaller-reader inventory |
| `mistralai/Mistral-Small-24B-Instruct-2501` | <https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501> | one sufficiency judge, one jury member |
| `microsoft/phi-4` | <https://huggingface.co/microsoft/phi-4> | one jury member |

Decoding settings and, where recorded, the model snapshot identifier are in the
candidate manifests under `data/panel/` and in the per-judge manifests under
`data/jury/`. Regenerating an inventory requires serving the reader; scoring an
existing inventory does not, because every candidate answer we generated ships
in the decision rows.

### What still cannot be checked from the release alone

Two quantities in the paper depend on resources above and are not derivable
from anything we ship: the CC-News index passage count, and any statistic over
retrieved passage text beyond the derived flags listed above. Both are stated
here so a reviewer knows the boundary rather than discovering it.
