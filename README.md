# RGBS

Code and recorded experimental results for the paper on deidentification-amplified semantic collisions in shared semantic caches. Historical artifacts use the name **RKGB**; the current manuscript uses **RGBS**. The historical filenames and result values are preserved.

## Paper results

The main results use BGE-small-en-v1.5, unit-normalized embeddings, cosine threshold 0.85, and seed 42. Query construction and final selection use Train; the selected strings are frozen before Test evaluation.

| Dataset | Query budget | Covered / Test requests | Test coverage |
|---|---:|---:|---:|
| Banking77 | 15 | 1,109 / 1,485 | 74.68% |
| ABCD | 5 | 308 / 476 | 64.71% |
| MultiDoGO | 10 | 909 / 1,094 | 83.09% |

The source reports and frozen texts are in [`results/main`](results/main), with a machine-readable [`coverage.csv`](results/main/coverage.csv). Coverage means cosine-threshold union coverage. Task-relation audit results and actual cache admission/returns are reported separately in their respective experiments.

**MultiDoGO provenance:** the paper's 83.09% result comes from the historical `original__gradient_residual_retention` route, preserved in `results/main/dogo/final_report.json`. The separate 32×50 run in `results/rq2/search_32x50/runs/dogo` obtained 888/1,094 = 81.17%. These are different frozen sets. Use the main directory for the paper's headline results; the 32×50 run is retained as a separate experiment.

## Repository contents

| Path | Contents |
|---|---|
| `code/search/` | Archived configurable K-means → Riemannian guides → HotFlip and dual-API beam search → portfolio selection implementation |
| `config/api_providers.example.json` | API configuration using environment variables |
| `results/main/` | Paper's three main reports and frozen query sets |
| `results/rq1/` | Request collision matrices, document collision statistics, edit/template controls, code and audit results |
| `results/rq2/` | Construction runs, baseline reports/code, vector-space comparison, ablations, parameter sweeps, transfer and admission evidence |
| `results/rq3/` | Fixed-key paired evaluation and task-relation audits, including unresolved judgments |
| `results/gptcache/` | GPTCache evaluation scripts, configuration and recorded results |
| `results/paper_analysis/` | Analysis tables and evidence copied from the current paper project |
| `paper_tables/` | Eight tables extracted directly from the current `main.tex`, avoiding stale standalone copies |
| `paper_figures/` | Figure PDFs, editable vector sources and available plotting data |
| `UPLOAD_PROVENANCE.json` | Source locations, source/output SHA-256 hashes and export exclusions |
| `ARTIFACT_SHA256.json` | Checksums for all published artifact files |

Historical experiment READMEs describe their own configurations and may mention earlier paper layouts or intermediate runs. The current manuscript tables are in `paper_tables/`. The paper-analysis copies and experimental reports provide the underlying values; this upload does not rerun model inference or API experiments.

## Running construction

Python dependencies are listed in `requirements.txt`. The recorded environment uses PyTorch 2.1.2 and Transformers 4.48.3; a compatible Python/CUDA environment and locally downloaded encoder and language-model weights are needed for a new run. API calls incur the provider's normal usage charges.

```bash
python -m pip install -r requirements.txt
export QWEN_API_KEY='your-key'
export DEEPSEEK_API_KEY='your-key'
python code/search/run_configurable.py \
  --dataset banking77 --representation desensitized \
  --encoder bge --threshold 0.85 --seed 42 --k 15 \
  --data-dir /path/to/derived/banking77/desensitized \
  --model-dir /path/to/bge-small-en-v1.5 \
  --lm-dir /path/to/Qwen2.5-1.5B-Instruct \
  --api-config config/api_providers.example.json \
  --steps-per-slot 32 --beam-width 50 \
  --device cuda --out-dir runs/banking77_new
```

For ABCD use `--dataset abcd --k 5`; for MultiDoGO use `--dataset dogo --k 10`. These commands start new experiments. They do not reproduce the historical MultiDoGO 83.09% route merely by changing the dataset argument. See the source reports, route configuration and `results/rq2/dogo_main_module_controls` for that frozen result.

Data directories need the JSONL splits expected by the archived entry point. Available experiment-specific derived inputs are retained beside their reports; they are not consolidated into a newly generated dataset. Request identifying fields are synthetic in the paired-request experiments. Original dataset provenance and scope are described in the archived input manifests and experiment READMEs. Model weights, API response caches, raw document corpora, and large intermediate embeddings are omitted. Historical paths in archived code must be supplied or adjusted for the local environment.

To prepare targets without text generation, add `--prepare-only`. Keep the dataset-domain restrictions in the archived scripts. Regenerate target vectors whenever the data, encoder, threshold, budget, or seed changes, and use a fresh output directory. Never use Test to select or rewrite queries.

## Checking the artifact

```bash
python tools/verify_artifact.py
```

This performs local checksum verification, Python syntax checks, and checks the three headline coverage counts and frozen-set sizes without loading models or calling APIs. It verifies the exported files; it is not a fresh execution of the experiments. Recorded API model names reflect the historical configuration and depend on provider availability for new runs.
