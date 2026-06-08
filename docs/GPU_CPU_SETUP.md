# CPU/GPU setup

The project supports both CPU and GPU training. GPU is strongly recommended for the NER/RE extractor and the final DistilBERT-based classifier.

## Why `requirements.txt` does not install `torch`

`requirements.txt` intentionally excludes `torch`, `torchvision`, and `torchaudio`. PyTorch has different builds for CPU and CUDA, so installing `torch` inside the normal requirements can accidentally install a CPU-only build. Install PyTorch through the setup helper instead.

## CPU setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe scripts\setup_environment.py --device cpu
```

Run the prepared-data pipeline on CPU:

```powershell
.\.venv\Scripts\python.exe scriptsun_prepared_data_pipeline.py --device cpu --quick-test
```

CPU is useful for smoke tests, but full NER/RE training will be slow.

## GPU setup on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu128 --allow-nightly-fallback
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected:

```text
cuda available: True
NVIDIA GeForce RTX ...
```

Run a quick test on GPU:

```powershell
.\.venv\Scripts\python.exe scriptsun_prepared_data_pipeline.py --device gpu --quick-test
```

Run the full experiment on GPU:

```powershell
.\.venv\Scripts\python.exe scriptsun_prepared_data_pipeline.py --device gpu --fp16
```

If `--fp16` causes instability, remove it:

```powershell
.\.venv\Scripts\python.exe scriptsun_prepared_data_pipeline.py --device gpu
```

## Device options

`run_prepared_data_pipeline.py` supports:

- `--device auto`: use GPU if `torch.cuda.is_available()` is true, otherwise CPU.
- `--device gpu`: require CUDA; fail early if CUDA is unavailable.
- `--device cpu`: force CPU even if CUDA exists.

## If GPU setup fails on Python 3.13

Try:

```powershell
.\.venv\Scripts\python.exe scripts\setup_environment.py --device gpu --cuda cu128 --allow-nightly-fallback
```

If CUDA still cannot be installed, Python 3.11 or 3.12 is usually a more stable ML/NLP environment for PyTorch CUDA.
