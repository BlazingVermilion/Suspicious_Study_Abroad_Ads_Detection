#!/usr/bin/env python3
"""Copy known files from the old local project layout into the v2 GitHub-ready layout.

Usage:
    python scripts/migrate_existing_project.py --old-root "C:/Users/Huy Vu/Documents/Suspicious_Educational_Ads_Detection" --new-root .
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MAPPINGS = [
    ("instagram_session.json", "secrets/instagram_session.json"),
    ("data/raw/metadata/account_registry.json", "data/raw/instagram/metadata/account_registry.json"),
    ("data/raw/metadata/legitimate_seed_data.json", "data/raw/instagram/metadata/legitimate_seed_posts.json"),
    ("data/raw/metadata/legitimate_seed_posts_scope_cleaned.json", "data/raw/instagram/metadata/legitimate_seed_posts.json"),
    ("data/raw/metadata/pre_labeled_data.json", "data/raw/instagram/metadata/suspicious_candidate_posts.json"),
    ("data/raw/metadata/suspicious_crawl_audit.jsonl", "data/raw/instagram/metadata/crawl_audit_suspicious_posts.jsonl"),
    ("data/raw/metadata/suspicious_crawl_state.json", "data/raw/instagram/metadata/crawl_state_suspicious_posts.json"),
    ("data/processed/normalize_dataset.csv", "data/processed/normalized/normalized_posts.csv"),
    ("data/processed/gold_NER_RE_annotated_subset.json", "data/processed/annotations/ner_re_gold_annotated_subset.json"),
    ("data/processed/binary_gold_subset_173.csv", "data/processed/splits/binary_gold_eval.csv"),
    ("data/processed/pre_labeled_silver_subset.csv", "data/processed/splits/silver_pool.csv"),
    ("data/processed/silver_subset.csv", "data/processed/splits/silver_pool.csv"),
    ("data/processed/ner_re_extraction_outputs", "data/processed/ner_re"),
    ("data/processed/silver_training_dataset.csv", "data/processed/weak_labeling/silver_train.csv"),
    ("data/processed/balance_silver_training_dataset.csv", "data/processed/weak_labeling/silver_train.csv"),
    ("data/processed/uncertain.csv", "data/processed/weak_labeling/uncertain_posts.csv"),
    ("data/processed/weak_labeling_statistics.csv", "data/processed/weak_labeling/weak_labeling_report.csv"),
    ("data/processed/all_scored_dataset_debug.csv", "data/processed/weak_labeling/weak_labeling_scored_posts_debug.csv"),
    ("data/processed/phase9_10_11_outputs_multiseed", "outputs/classifier_evaluation"),
]

def copy_any(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", required=True, type=Path)
    parser.add_argument("--new-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    copied = 0
    for old_rel, new_rel in MAPPINGS:
        src = args.old_root / old_rel
        dst = args.new_root / new_rel
        if copy_any(src, dst):
            copied += 1
            print(f"COPIED: {old_rel} -> {new_rel}")
        else:
            print(f"MISS:   {old_rel}")
    print(f"Done. Copied {copied}/{len(MAPPINGS)} known paths.")

if __name__ == "__main__":
    main()
