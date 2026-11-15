# RGBS: paper code and results

This artifact contains the construction code and the results reported in the manuscript on deidentification-amplified semantic collisions in shared semantic caches. Historical implementation names use RKGB; the paper uses RGBS.

## Main results

| Dataset | Query budget | Covered / Test requests | Test coverage |
|---|---:|---:|---:|
| Banking77 | 15 | 1,109 / 1,485 | 74.68% |
| ABCD | 5 | 308 / 476 | 64.71% |
| MultiDoGO | 10 | 909 / 1,094 | 83.09% |

These are cosine-threshold union coverage results with BGE-small-en-v1.5 at threshold 0.85. Train is used for construction and selection; Test is evaluated after the query set is frozen. The reports and frozen strings are in [`results/main`](results/main). The MultiDoGO report contains only the paper's `original__gradient_residual_retention` route.

## Results in the paper

[`PAPER_RESULTS_INDEX.json`](PAPER_RESULTS_INDEX.json) maps manuscript tables, figures and body analyses to the included data. [`paper_tables`](paper_tables) contains all eight tables extracted directly from the current manuscript. [`paper_figures`](paper_figures) contains only figures cited by that manuscript, their available editable sources, and plotting data.

| Paper content | Data |
|---|---|
| Derived collections | `paper_tables/01_*` |
| RQ1 collision changes and editing/template controls | `results/rq1/`, `paper_tables/02_*`, `paper_tables/03_*` |
| Representation and retained-text budget comparisons | `results/rq2/representation_comparison.csv`, `results/rq2/budget_coverage.csv` |
| External methods and perplexity | `paper_tables/04_*`, `results/rq2/ppl_medians.csv` |
| Vector-space comparison and text realization | `paper_tables/05_*`, `results/rq2/vector_validation.json`, `results/rq2/stage_coverage.csv` |
| Module ablations | `paper_tables/06_*` |
| Calibrated cross-encoder evaluation | `paper_tables/07_*`, `results/rq2/cross_encoder_calibrated.csv` |
| Sequential admission | `paper_tables/08_*`, `results/rq2/sequential_admission.csv` |
| Fixed-key Top-1 and offline cross-task rates | `paper_figures/Figure_09_task_relationships_data.json`, `paper_figures/Figure_RQ3_offline_cross_task_rates_data.json`, `results/rq3/` |

Tables preserve the reported rounded values; separate source-data files retain available counts and precision. Coverage, cross-task pair proportions and Top-1 return proportions have different denominators. Unresolved audit judgments are excluded where indicated in the paper's figure data. The artifact does not infer answer correctness from a task-relation label.

## Code and credentials

`code/search/` contains target generation, gradient-guided text search, language screening and complementary portfolio selection. `code/experiments/` contains calculation programs for the included experiment families. Their historical experiment layouts and model/data paths need configuring before use; this compact artifact does not include the full original experiment workspaces, dataset corpora, model weights, API response caches, launchers or intermediate checkpoints.

No personal API credentials are included. Search and audit code read credentials only from environment variables named in the caller's configuration. [`config/api_providers.example.json`](config/api_providers.example.json) contains placeholder model names and endpoints. Replace them with your own providers' compatible settings and set `RGBS_API_KEY_1` and `RGBS_API_KEY_2` in your local environment. Keep real configuration and secrets outside Git. Changing providers or model versions can change generated results.

Example for a new Banking77 search, with locally prepared derived Train/Test JSONL splits and locally downloaded models:

```bash
python -m pip install -r requirements.txt
python code/search/run_configurable.py \
  --dataset banking77 --representation desensitized \
  --encoder bge --threshold 0.85 --seed 42 --k 15 \
  --data-dir /path/to/banking77/desensitized \
  --model-dir /path/to/bge-small-en-v1.5 \
  --lm-dir /path/to/Qwen2.5-1.5B-Instruct \
  --api-config /path/to/private-api-config.json \
  --steps-per-slot 32 --beam-width 50 \
  --device cuda --out-dir runs/banking77_new
```

Use `--dataset abcd --k 5` or `--dataset dogo --k 10` for a new search on the other collections. These commands initiate new experiments; the archived MultiDoGO main set was produced by its recorded historical route. Add `--prepare-only` to generate guides without API text search. Regenerate guides when data, encoder, threshold, budget or seed changes; preserve dataset scope restrictions and freeze queries before opening Test.

## Verification

```bash
python tools/verify_artifact.py
```

This checks file hashes, Python syntax, main-result counts, frozen-set sizes and paper-index references without calling providers or loading models. `SOURCE_HASHES.json` records hashes of retained source artifacts; metadata and credential-loading code have been cleaned during export. `ARTIFACT_SHA256.json` hashes the current published files.
