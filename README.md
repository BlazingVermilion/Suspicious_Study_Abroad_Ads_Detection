# Suspicious Educational Ads Detection

This repository contains the research pipeline for detecting suspicious English-language Instagram educational advertisements about studying in Germany.

The repo is organized so that a reviewer can either:

1. **Run from prepared data** without crawling Instagram again; or
2. **Run the full data-construction workflow** from collection to modeling.

For most reviewers and collaborators, use the **prepared-data workflow** below.

---

## Prepared-data quick start

This repository includes the curated input files in `data/prepared/`, so you can skip Instagram crawling, manual NER/RE annotation, and binary-gold selection.

```text
data/prepared/
├── normalized_posts.csv
├── silver_pool.csv
├── binary_gold_eval.csv
├── ner_re_gold_annotated_subset.json
└── PREPARED_DATA_MANIFEST.json
```

The prepared-data pipeline runs:

```text
NER/RE training + prediction
        ↓
weak-label construction
        ↓
classifier training and gold-set evaluation
```

---

## 1. Create a virtual environment

Run all commands from the repository root.

Windows PowerShell:

```powershell
cd "C:\path\to\GitHub_Ready_v3"
python -m venv .venv
```

You do **not** need to activate the environment. All commands below call the venv Python directly:

```powershell
.\.venv\Scripts\python.exe ...
```

This avoids PowerShell execution-policy issues with `Activate.ps1`.

---

## 2. Choose CPU or GPU setup

`requirements.txt` intentionally **does not install PyTorch**. PyTorch is installed separately because CPU and CUDA/GPU builds use different package indexes. This avoids accidentally installing `torch+cpu` when you want GPU training.

### Option A — CPU setup

Use this only for smoke tests or debugging. Full NER/RE DistilBERT training on CPU is slow.

```powershell
.\.venv\Scripts\python.exe scripts\setup_environment.py --device cpu
```

Check the installation:

```powershell
.\.venv\Scripts\python.exe scripts\check_torch_device.py
```

Expected CPU output contains:

```text
cuda available: False
NO CUDA
```

### Option B — GPU setup, recommended for full training

Use this when the machine has an NVIDIA GPU and a recent NVIDIA driver.

```powershell
.\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu128 --allow-nightly-fallback
```

This script will:

```text
1. upgrade pip/setuptools/wheel
2. install base dependencies from requirements.txt
3. uninstall any existing torch/torchvision/torchaudio
4. install a CUDA PyTorch build
5. verify torch.cuda.is_available()
```

Check the GPU:

```powershell
.\.venv\Scripts\python.exe scripts\check_torch_device.py
```

Expected GPU output contains:

```text
cuda available: True
NVIDIA GeForce RTX ...
```

If this still prints `cuda available: False`, do not run the full pipeline yet. Fix the PyTorch CUDA installation first.

### Python 3.13 note

Python 3.13 can work, but PyTorch CUDA wheels are more sensitive to Python/CUDA compatibility. For Python 3.13, start with:

```powershell
.\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu128 --allow-nightly-fallback
```

If CUDA setup fails, try:

```powershell
.\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu126 --allow-nightly-fallback
```

For the most stable ML/NLP setup, Python 3.11 or 3.12 is usually easier, but the project does not force you to use them.

---

## 3. Run a quick smoke test

Before a full run, verify that paths, dependencies, and data files work.

CPU-safe smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device cpu --quick-test
```

GPU smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu --quick-test
```

The quick test uses fewer folds/epochs and is only for checking that the pipeline runs. Do **not** report quick-test metrics as research results.

---

## 4. Run full training/evaluation

### Full GPU run, recommended

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu --fp16
```

If `--fp16` is unstable on your GPU/driver, run without fp16:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu
```

The script will stop early if `--device gpu` is requested but CUDA is not available, so you do not accidentally run a full experiment on CPU.

### Full CPU run

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device cpu
```

This is valid but slow. Use CPU mainly for debugging or small smoke tests.

---

## 5. Main outputs

After a successful prepared-data run, check:

```text
data/processed/ner_re/
├── cv_results/
│   ├── ner_per_label_average.csv
│   └── re_per_label_average.csv
└── predictions/
    ├── silver_with_ner_re_oof_ensemble.csv
    └── binary_gold_with_ner_re_ensemble.csv

data/processed/weak_labeling/
├── silver_train.csv
├── uncertain_posts.csv
├── weak_labeling_report.csv
└── weak_labeling_scored_posts_debug.csv

outputs/classifier_evaluation/
├── metrics_by_run.csv
├── metrics_summary.csv
├── gold_model_ranking.csv
├── gold_predictions_all_models.csv
├── classifier_results.json
└── classifier_schema_audit.json
```

The most important files for writing the results section are:

```text
outputs/classifier_evaluation/gold_model_ranking.csv
outputs/classifier_evaluation/metrics_summary.csv
outputs/classifier_evaluation/classifier_results.json
```

---

## 6. Useful prepared-data commands

Print commands without running them:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --dry-run
```

Reuse existing NER/RE outputs and rerun only weak labeling + classifier:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu --skip-nerre
```

Reuse existing NER/RE and weak-labeling outputs, rerun only classifier:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu --skip-nerre --skip-weak-labeling
```

Run GPU without fp16:

```powershell
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu
```

---

## Repository structure

```text
project/
├── src/
│   ├── collection/          # Instagram account discovery and post crawling
│   ├── preprocessing/       # Normalize raw Instagram records
│   ├── annotation/          # Build binary gold/silver split
│   ├── extraction/          # Train/predict NER + relation-signal extractor
│   ├── weak_labeling/       # Critical-risk + co-occurrence weak labeling
│   ├── modeling/            # Classifier baselines, ablations, gold evaluation
│   └── utils/
├── data/
│   ├── prepared/            # Curated inputs for reproducible prepared-data runs
│   ├── raw/instagram/       # Raw metadata and screenshots; usually ignored by Git
│   └── processed/           # Normalized, annotated, split, NER/RE, weak-label data
├── outputs/                 # Evaluation outputs and reports
├── notebooks/               # Step-by-step executable research notebooks
├── docs/                    # Pipeline, schema, migration, CPU/GPU notes
├── scripts/                 # Setup and pipeline runners
└── secrets/                 # Local session/cookie files; ignored by Git
```

---

## Clean data naming convention

| Stage | Canonical file/folder | Role |
|---|---|---|
| Prepared normalized corpus | `data/prepared/normalized_posts.csv` | Auditable normalized corpus |
| Prepared silver pool | `data/prepared/silver_pool.csv` | Corpus used for NER/RE prediction and weak labeling |
| Prepared binary gold | `data/prepared/binary_gold_eval.csv` | Held-out human-verified evaluation set |
| Prepared NER/RE gold | `data/prepared/ner_re_gold_annotated_subset.json` | Label Studio NER/RE annotation export |
| NER/RE outputs | `data/processed/ner_re/` | CV metrics and enriched prediction CSVs |
| Weak-labeled train | `data/processed/weak_labeling/silver_train.csv` | Final silver training data for classifier |
| Classifier eval | `outputs/classifier_evaluation/` | Model metrics, predictions, ranking, error analysis |

---

## Optional full data-construction workflow

Only use this if you want to rebuild the dataset from raw Instagram collection.

```powershell
# 1) Optional Instagram session for crawling
.\.venv\Scripts\python.exe src\collection\save_instagram_session.py

# 2) Discover suspicious accounts and crawl candidate posts
.\.venv\Scripts\python.exe src\collection\discover_suspicious_accounts.py
.\.venv\Scripts\python.exe src\collection\crawl_suspicious_posts.py

# 3) Crawl legitimate seed posts
.\.venv\Scripts\python.exe src\collection\crawl_legitimate_posts.py

# 4) Normalize raw metadata
.\.venv\Scripts\python.exe src\preprocessing\build_normalized_dataset.py `
  --input-dir data\raw\instagram\metadata `
  --output data\processed\normalized\normalized_posts.csv

# 5) Build binary gold evaluation split and silver pool
.\.venv\Scripts\python.exe src\annotation\build_binary_gold_split.py `
  --normalize data\processed\normalized\normalized_posts.csv `
  --annotation-subset data\processed\annotations\ner_re_gold_annotated_subset.json `
  --out-dir data\processed\splits
```

After this, use the NER/RE, weak-labeling, and classifier stages through `scripts\run_prepared_data_pipeline.py` or the individual scripts.

---

## Privacy and GitHub upload

Raw screenshots, session files, model checkpoints, and large output folders should not be committed unless intentionally publishing a private/review dataset.

For a private or reviewer-only repository, keeping `data/prepared/` is useful because it makes the experiment reproducible without re-crawling Instagram.

For a public repository, consider moving full data to a controlled release location and keeping only a small sample in GitHub.

---

## Troubleshooting

### `torch.cuda.is_available()` is `False`

Check the active environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_torch_device.py
```

If it shows `torch ... +cpu`, reinstall using the GPU setup script:

```powershell
.\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu128 --allow-nightly-fallback
```

### Hugging Face `Trainer` says `accelerate` is missing

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install "accelerate>=1.1.0"
```

Then rerun the pipeline.


### Classifier error: `np.hstack` row mismatch after cached embeddings

If you rerun weak labeling or change `silver_train.csv`, old cached DistilBERT embeddings may no longer match the current training rows. Version v3.4 automatically detects stale cache shapes and recomputes embeddings. You can also manually clear classifier caches:

```powershell
Remove-Item outputs\classifier_evaluation\cache_embeddings_*.npy -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu --skip-nerre --skip-weak-labeling
```

For a fully fresh end-to-end run, remove the whole classifier output folder:

```powershell
Remove-Item -Recurse -Force outputs\classifier_evaluation -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\run_prepared_data_pipeline.py --device gpu
```

### PowerShell blocks activation

You do not need to activate the venv. Use:

```powershell
.\.venv\Scripts\python.exe ...
```

Alternatively, for the current PowerShell window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
