# Run from prepared data

This repository supports a reproducible run that skips Instagram crawling and manual binary-gold selection.
Use this path when the prepared datasets already exist.

## Required prepared inputs

Place these files in `data/prepared/`:

| File | Role |
|---|---|
| `normalized_posts.csv` | Full normalized corpus after preprocessing. Kept for audit/reference. |
| `silver_pool.csv` | Silver/weak-labeling candidate pool after binary gold has been removed. |
| `binary_gold_eval.csv` | Human-verified held-out binary evaluation set. |
| `ner_re_gold_annotated_subset.json` | Label Studio NER + relation-signal annotations for extractor training. |

The current prepared-data run uses `silver_pool.csv`, `binary_gold_eval.csv`, and `ner_re_gold_annotated_subset.json` directly.
`normalized_posts.csv` is included for auditability and for rebuilding splits if needed.

## Recommended full experimental command

Windows PowerShell:

```powershell
.\scripts\run_prepared_data_pipeline_windows.ps1
```

Cross-platform Python:

```bash
python scripts/run_prepared_data_pipeline.py
```

This runs:

1. `src/extraction/train_ner_re_extractor.py`
2. `src/weak_labeling/build_weak_labels.py`
3. `src/modeling/train_classifier.py`

## Smoke test

Use this only to check that the environment and file paths work:

```powershell
.\scripts\run_prepared_data_pipeline_windows.ps1 -QuickTest
```

or:

```bash
python scripts/run_prepared_data_pipeline.py --quick-test
```

Do not report smoke-test metrics as final research results.

## Outputs

| Directory | Content |
|---|---|
| `data/processed/ner_re/` | NER/RE CV metrics and silver/binary-gold NER/RE predictions. |
| `data/processed/weak_labeling/` | Weak-labeled silver training set, uncertain posts, and debug reports. |
| `outputs/classifier_evaluation/` | Classifier metrics, model ranking, predictions, and schema audit. |

## Data-release note

For a public GitHub repository, avoid committing raw screenshots, Instagram session files, and crawler state.
Prepared CSV/JSON files may still contain public captions, URLs, account names, and person names from testimonial posts.
Only commit the full prepared data if your repository is private or your data-release plan permits it.
For a public artifact, prefer a small `data/sample/` subset plus a data statement explaining how the full data can be accessed for review.

### Dependency note

The NER/RE extractor uses Hugging Face `Trainer`, so `accelerate>=1.1.0` is required. If you see `ImportError: Using the Trainer with PyTorch requires accelerate`, run:

```powershell
.\.venv\Scripts\python.exe -m pip install "accelerate>=1.1.0"
```

Then rerun `scripts\run_prepared_data_pipeline.py`.


## CPU/GPU device selection

The prepared-data runner supports explicit device selection:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device auto --quick-test
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu --fp16
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device cpu
```

Use `--device gpu` for full experiments. It fails early if CUDA is unavailable, which prevents accidentally running the full NER/RE training on CPU.

Before using GPU, install CUDA PyTorch with:

```powershell
.\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu128 --allow-nightly-fallback
```
