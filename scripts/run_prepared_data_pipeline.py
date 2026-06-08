#!/usr/bin/env python3
"""
Run the full experimental pipeline from prepared data.

This script skips Instagram data collection and manual annotation. It assumes that
these prepared inputs already exist:

  data/prepared/ner_re_gold_annotated_subset.json
  data/prepared/silver_pool.csv
  data/prepared/binary_gold_eval.csv

It then runs:
  1) NER/RE extraction training + prediction
  2) weak-label construction
  3) final classifier training/evaluation

Use --dry-run to print commands without executing them.
Use --quick-test for a very small smoke test configuration; do not report quick-test
metrics as research results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(cmd: List[str], dry_run: bool = False) -> None:
    printable = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)
    print("\n" + "=" * 100)
    print(printable)
    print("=" * 100)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device_arg: str, cpu_alias: bool, quick_test: bool) -> str:
    if cpu_alias:
        return "cpu"
    if quick_test and device_arg == "auto":
        return "cpu"
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "gpu":
        if not torch_cuda_available():
            raise RuntimeError(
                "--device gpu was requested, but torch.cuda.is_available() is False.\n"
                "Install a CUDA PyTorch build first, for example:\n"
                "  .\\.venv\\Scripts\\python.exe scripts\\setup_environment.py --device gpu --cuda cu128 --allow-nightly-fallback\n"
                "Then verify with:\n"
                "  .\\.venv\\Scripts\\python.exe -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\""
            )
        return "gpu"
    return "gpu" if torch_cuda_available() else "cpu"


def main() -> None:
    root = project_root()
    os.environ.setdefault("PROJECT_ROOT", str(root))

    parser = argparse.ArgumentParser(description="Run NER/RE -> weak labeling -> classifier from prepared data.")
    parser.add_argument("--prepared-dir", default=str(root / "data" / "prepared"))
    parser.add_argument("--nerre-json", default=None)
    parser.add_argument("--silver-csv", default=None)
    parser.add_argument("--binary-gold-csv", default=None)
    parser.add_argument("--nerre-out-dir", default=str(root / "data" / "processed" / "ner_re"))
    parser.add_argument("--weak-out-dir", default=str(root / "data" / "processed" / "weak_labeling"))
    parser.add_argument("--classifier-out-dir", default=str(root / "outputs" / "classifier_evaluation"))

    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--nerre-max-length", type=int, default=384)
    parser.add_argument("--classifier-max-length", type=int, default=256)
    parser.add_argument("--epochs-ner", type=float, default=5)
    parser.add_argument("--epochs-re", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pred-batch-size", type=int, default=8)
    parser.add_argument("--classifier-seeds", default="1,7,13,21,42")
    parser.add_argument("--split-mode", choices=["stratified", "group", "auto"], default="stratified")
    parser.add_argument("--threshold-mode", choices=["fixed_05", "val_f1", "recall_at_precision", "balanced_accuracy"], default="val_f1")
    parser.add_argument("--imbalance-mode", choices=["pos_weight", "focal", "sampler", "none"], default="pos_weight")

    parser.add_argument("--silver-text-col", default="clean_text")
    parser.add_argument("--binary-text-col", default="clean_text")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device policy for training: auto uses CUDA when available; gpu fails if CUDA is unavailable; cpu forces CPU.")
    parser.add_argument("--cpu", action="store_true", help="Deprecated alias for --device cpu.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for NER/RE training if CUDA supports it. Only valid with GPU.")
    parser.add_argument("--reuse-existing-nerre", action="store_true", help="Reuse existing NER/RE fold models if present.")
    parser.add_argument("--skip-nerre", action="store_true", help="Skip NER/RE training/prediction and use existing NER/RE outputs.")
    parser.add_argument("--skip-weak-labeling", action="store_true", help="Skip weak labeling and use existing silver_train.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--quick-test", action="store_true", help="Fast smoke test: 2 folds, 1 NER epoch, 1 RE epoch, 1 classifier seed, CPU-safe settings.")

    args = parser.parse_args()

    effective_device = resolve_device(args.device, args.cpu, args.quick_test)
    print(f"[device-policy] requested={args.device} effective={effective_device} cuda_available={torch_cuda_available()}")
    if args.fp16 and effective_device != "gpu":
        raise RuntimeError("--fp16 requires GPU/CUDA. Use --device gpu after installing CUDA PyTorch, or remove --fp16.")

    prepared_dir = Path(args.prepared_dir)
    nerre_json = Path(args.nerre_json) if args.nerre_json else prepared_dir / "ner_re_gold_annotated_subset.json"
    silver_csv = Path(args.silver_csv) if args.silver_csv else prepared_dir / "silver_pool.csv"
    binary_gold_csv = Path(args.binary_gold_csv) if args.binary_gold_csv else prepared_dir / "binary_gold_eval.csv"

    require_file(nerre_json, "NER/RE annotated subset")
    require_file(silver_csv, "prepared silver pool")
    require_file(binary_gold_csv, "prepared binary gold evaluation set")

    nerre_out = Path(args.nerre_out_dir)
    weak_out = Path(args.weak_out_dir)
    clf_out = Path(args.classifier_out_dir)
    nerre_out.mkdir(parents=True, exist_ok=True)
    weak_out.mkdir(parents=True, exist_ok=True)
    clf_out.mkdir(parents=True, exist_ok=True)

    epochs_ner = args.epochs_ner
    epochs_re = args.epochs_re
    num_folds = 5
    classifier_seeds = args.classifier_seeds
    batch_size = args.batch_size
    pred_batch_size = args.pred_batch_size

    if args.quick_test:
        print("[quick-test] Running smoke-test settings. Do not report these metrics as final results.")
        epochs_ner = 1
        epochs_re = 1
        num_folds = 2
        classifier_seeds = "42"
        batch_size = min(batch_size, 2)
        pred_batch_size = min(pred_batch_size, 4)

    silver_with_nerre = nerre_out / "predictions" / "silver_with_ner_re_oof_ensemble.csv"
    gold_with_nerre = nerre_out / "predictions" / "binary_gold_with_ner_re_ensemble.csv"
    ner_metrics = nerre_out / "cv_results" / "ner_per_label_average.csv"
    re_metrics = nerre_out / "cv_results" / "re_per_label_average.csv"
    silver_train = weak_out / "silver_train.csv"

    py = sys.executable

    if not args.skip_nerre:
        cmd = [
            py, str(root / "src" / "extraction" / "train_ner_re_extractor.py"),
            "--nerre-json", str(nerre_json),
            "--silver-csv", str(silver_csv),
            "--binary-gold-csv", str(binary_gold_csv),
            "--out-dir", str(nerre_out),
            "--model-name", args.model_name,
            "--silver-text-col", args.silver_text_col,
            "--binary-text-col", args.binary_text_col,
            "--num-folds", str(num_folds),
            "--epochs-ner", str(epochs_ner),
            "--epochs-re", str(epochs_re),
            "--batch-size", str(batch_size),
            "--pred-batch-size", str(pred_batch_size),
            "--max-length", str(args.nerre_max_length),
        ]
        if effective_device == "cpu":
            cmd.append("--cpu")
        if args.fp16 and effective_device == "gpu":
            cmd.append("--fp16")
        if args.reuse_existing_nerre:
            cmd.append("--reuse-existing-fold-models")
        run_command(cmd, dry_run=args.dry_run)

    if not args.skip_weak_labeling:
        for p, label in [(silver_with_nerre, "silver NER/RE prediction"), (ner_metrics, "NER metrics"), (re_metrics, "RE metrics")]:
            if not args.dry_run:
                require_file(p, label)
        cmd = [
            py, str(root / "src" / "weak_labeling" / "build_weak_labels.py"),
            "--silver-csv", str(silver_with_nerre),
            "--gold-json", str(nerre_json),
            "--ner-metrics", str(ner_metrics),
            "--re-metrics", str(re_metrics),
            "--out-dir", str(weak_out),
        ]
        run_command(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        require_file(silver_train, "weak-labeled silver training set")
        require_file(gold_with_nerre, "binary gold with NER/RE features")

    cmd = [
        py, str(root / "src" / "modeling" / "train_classifier.py"),
        "--train", str(silver_train),
        "--gold", str(gold_with_nerre),
        "--output_dir", str(clf_out),
        "--bert_model", args.model_name,
        "--max_length", str(args.classifier_max_length),
        "--bert_batch_size", "16",
        "--mlp_batch_size", "32",
        "--hidden_dim", "256",
        "--dropout", "0.35",
        "--lr", "0.0005",
        "--weight_decay", "0.0001",
        "--epochs", "40" if not args.quick_test else "2",
        "--patience", "8" if not args.quick_test else "1",
        "--split_mode", args.split_mode,
        "--threshold_mode", args.threshold_mode,
        "--imbalance_mode", args.imbalance_mode,
        "--seeds", classifier_seeds,
        "--device", effective_device,
    ]
    run_command(cmd, dry_run=args.dry_run)

    print("\nPipeline finished.")
    print(f"NER/RE outputs:       {nerre_out}")
    print(f"Weak-labeling outputs:{weak_out}")
    print(f"Classifier outputs:   {clf_out}")


if __name__ == "__main__":
    main()
