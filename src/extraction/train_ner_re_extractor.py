#!/usr/bin/env python
# =============================================================================
# GitHub-ready refactor note
# =============================================================================
# Folder layout is intentionally simple: scripts live under src/<pipeline_module>/
# instead of src/suspicious_ads/<pipeline_module>/.
#
# Data paths are project-relative and can be overridden with PROJECT_ROOT.
# The original research logic is preserved, but filenames and output folders were
# renamed so the pipeline is easier to read on GitHub.
# =============================================================================

# -*- coding: utf-8 -*-
"""
Post-ID-routed OOF / ensemble NER+RE extraction pipeline
for the suspicious educational ads project.

Purpose
-------
This version is designed for the corrected clean NER/RE-to-classifier methodology:

1) Parse a Label Studio JSON export with manually annotated NER spans and
   caption-level relation-signal labels.
2) Check ONLY whether:
      G = NER/RE annotated set
      S = silver / weak-labeling pool
   has the expected subset relationship using post_id.
   Binary gold B is NOT overlap-checked in this script; it is treated as an
   already-audited held-out evaluation set.
3) Train ONE shared 5-fold NER/RE system:
      - For G examples: create out-of-fold (OOF) NER/RE predictions.
      - For S rows whose post_id belongs to G: use the OOF prediction.
      - For S rows outside G: use 5-fold ensemble prediction.
      - For B rows: use 5-fold ensemble prediction.
4) Export enriched silver and binary-gold CSVs with identical NER/RE feature
   columns.
5) Export CV quality reports in out_dir/cv_results, especially:
      - ner_per_label_average.csv
      - re_per_label_average.csv

Notes
-----
Folds are still grouped by normalized caption_fingerprint when possible, so
near-duplicate annotated captions stay in the same fold. However, routing from
G to S is done by post_id, because G has been confirmed to be a post_id subset
of S while G.data.text and S.clean_text may differ after preprocessing.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import inspect
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------
# Project schema
# ---------------------------------------------------------------------

NER_LABELS = [
    "COST_CLAIM",
    "COST_DETAIL",
    "REQUIREMENT",
    "NEGATION_CUE",
    "PROGRAM_OR_INTAKE",
    "FIELD_OF_STUDY",
    "GENERIC_INSTITUTION",
    "SPECIFIC_EDU_ORG",
    "SERVICE_PROVIDER",
    "SUPPORT_SERVICE",
    "OUTCOME",
    "GUARANTEE_CUE",
    "VAGUE_BENEFIT",
    "PRESSURE_CUE",
    "TESTIMONIAL_ACTOR",
    "DESTINATION",
]

SCRIPT_VERSION = "2026-06-06-oof-postid-fix2"

RELATION_LABELS = [
    "OUTCOME_GUARANTEED",
    "REQUIREMENT_WAIVED",
    "TESTIMONIAL_SUCCESS_CLAIM",
    "PROGRAM_OR_OUTCOME_HAS_VERIFIABLE_ORG",
]

ORIGIN_PRIORITY = {
    "manual": 4,
    "prediction-changed": 3,
    "prediction": 2,
    None: 1,
}

TEXT_COL_CANDIDATES = ["clean_text", "model_text", "caption_text", "core_caption", "text", "caption"]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ---------------------------------------------------------------------
# Caption fingerprinting and relationship audit
# ---------------------------------------------------------------------


def normalize_caption_for_matching(text: Any) -> str:
    """Normalize caption for exact content matching across post ids.

    This is intentionally stronger than a raw string comparison but still
    deterministic and auditable. It removes common crawl artifacts while
    preserving semantic content.
    """
    s = safe_str(text)
    s = re.sub(r"^\s*\[CAPTION\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = s.lower()
    # Normalize quotes/dashes lightly.
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def caption_fingerprint(text: Any) -> str:
    norm = normalize_caption_for_matching(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def choose_text_column(df: pd.DataFrame, requested: Optional[str], dataset_name: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Requested {dataset_name} text column '{requested}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer text column for {dataset_name}. Pass it explicitly.")


def add_caption_fingerprints_to_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    out["caption_match_text"] = out[text_col].fillna("").astype(str).map(normalize_caption_for_matching)
    out["caption_fingerprint"] = out["caption_match_text"].map(caption_fingerprint)
    return out


def set_stats(name: str, fps: Sequence[str]) -> Dict[str, Any]:
    c = Counter(fps)
    return {
        "name": name,
        "rows": int(len(fps)),
        "unique_caption_fingerprints": int(len(c)),
        "duplicate_caption_groups": int(sum(1 for v in c.values() if v > 1)),
        "duplicate_caption_rows": int(sum(v for v in c.values() if v > 1)),
    }


def sample_overlap_rows(
    overlap_fps: set,
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_df: pd.DataFrame,
    n: int = 20,
) -> List[Dict[str, Any]]:
    rows = []
    for fp in sorted(list(overlap_fps))[:n]:
        left_rows = left_df[left_df["caption_fingerprint"] == fp]
        right_rows = right_df[right_df["caption_fingerprint"] == fp]
        left_post_ids = left_rows["post_id"].astype(str).tolist() if "post_id" in left_rows.columns else []
        right_post_ids = right_rows["post_id"].astype(str).tolist() if "post_id" in right_rows.columns else []
        text = left_rows["caption_match_text"].iloc[0] if len(left_rows) else right_rows["caption_match_text"].iloc[0]
        rows.append(
            {
                "caption_fingerprint": fp,
                f"{left_name}_rows": int(len(left_rows)),
                f"{right_name}_rows": int(len(right_rows)),
                f"{left_name}_post_ids": json.dumps(left_post_ids, ensure_ascii=False),
                f"{right_name}_post_ids": json.dumps(right_post_ids, ensure_ascii=False),
                "caption_preview": text[:300],
            }
        )
    return rows



def audit_g_subset_s_by_post_id(g_df: pd.DataFrame, s_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Audit whether G is a subset of S using post_id only.

    Binary gold is intentionally not checked here. The user already audited B
    at the content level, and this script treats B as a held-out evaluation set.
    """
    audit_dir = out_dir / "post_id_subset_audit"
    ensure_dir(audit_dir)

    if "post_id" not in g_df.columns:
        raise ValueError("G / annotated examples must contain a post_id column.")
    if "post_id" not in s_df.columns:
        raise ValueError("S / silver CSV must contain a post_id column for post_id-based OOF routing.")

    g_ids_series = g_df["post_id"].fillna("").astype(str)
    s_ids_series = s_df["post_id"].fillna("").astype(str)
    g_ids = set(g_ids_series)
    s_ids = set(s_ids_series)
    missing_ids = sorted(g_ids - s_ids)
    overlap_ids = sorted(g_ids & s_ids)

    report = {
        "matching_method": "post_id_for_G_subset_S_and_OOF_routing",
        "binary_gold_overlap_check": "skipped_by_design",
        "G_rows": int(len(g_df)),
        "S_rows": int(len(s_df)),
        "G_unique_post_ids": int(len(g_ids)),
        "S_unique_post_ids": int(len(s_ids)),
        "G_intersect_S_unique_post_ids": int(len(overlap_ids)),
        "G_missing_from_S_unique_post_ids": int(len(missing_ids)),
        "G_is_post_id_subset_of_S": bool(g_ids <= s_ids),
        "G_duplicate_post_id_count": int(g_ids_series.duplicated().sum()),
        "S_duplicate_post_id_count": int(s_ids_series.duplicated().sum()),
        "extraction_plan": {
            "silver_rows_with_post_id_in_G": "use OOF NER/RE prediction from held-out fold",
            "silver_rows_without_post_id_in_G": "use 5-fold ensemble NER/RE prediction",
            "binary_gold_rows": "use 5-fold ensemble NER/RE prediction; no B overlap check performed here",
        },
    }

    (audit_dir / "post_id_subset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if missing_ids:
        g_df[g_df["post_id"].astype(str).isin(missing_ids)].to_csv(
            audit_dir / "G_post_ids_missing_from_S.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=list(g_df.columns)).to_csv(
            audit_dir / "G_post_ids_missing_from_S.csv", index=False, encoding="utf-8-sig"
        )

    print(f"[{now()}] Post-id subset audit saved to {audit_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_annotation(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled", False)]
    if not annotations:
        return None
    annotations.sort(key=lambda a: safe_str(a.get("updated_at") or a.get("created_at")), reverse=True)
    return annotations[0]


def spans_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def resolve_overlapping_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for s in spans:
        key = (int(s["start"]), int(s["end"]), s["label"])
        if key not in dedup or ORIGIN_PRIORITY.get(s.get("origin"), 1) > ORIGIN_PRIORITY.get(dedup[key].get("origin"), 1):
            dedup[key] = s

    candidates = list(dedup.values())
    candidates.sort(
        key=lambda s: (
            ORIGIN_PRIORITY.get(s.get("origin"), 1),
            int(s["end"]) - int(s["start"]),
        ),
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    for s in candidates:
        if any(spans_overlap(s, k) for k in kept):
            continue
        kept.append(s)
    kept.sort(key=lambda s: (s["start"], s["end"], s["label"]))
    return kept


def parse_labelstudio_export(path: Path, keep_overlapping: bool = False) -> List[Dict[str, Any]]:
    raw = read_json(path)
    if isinstance(raw, dict) and "tasks" in raw:
        tasks = raw["tasks"]
    elif isinstance(raw, list):
        tasks = raw
    else:
        raise ValueError("Unsupported Label Studio JSON structure. Expected a list of tasks or {'tasks': [...]}.")

    examples: List[Dict[str, Any]] = []
    skipped = 0

    for idx, task in enumerate(tasks):
        data = task.get("data", {})
        text = safe_str(data.get("text") or data.get("clean_text") or data.get("caption_text"))
        post_id = safe_str(data.get("post_id") or task.get("id") or f"task_{idx}")
        account_name = safe_str(data.get("account_name"))
        seed_label = safe_str(data.get("seed_label"))
        post_url = safe_str(data.get("post_url"))

        ann = choose_annotation(task)
        if ann is None:
            skipped += 1
            continue

        spans: List[Dict[str, Any]] = []
        rel_choices: List[str] = []

        for r in ann.get("result", []):
            r_type = r.get("type")
            val = r.get("value", {})
            from_name = r.get("from_name")
            origin = r.get("origin")

            if r_type == "labels" and "start" in val and "end" in val and val.get("labels"):
                label = val.get("labels", [None])[0]
                if label not in NER_LABELS:
                    continue
                start, end = int(val["start"]), int(val["end"])
                if start < 0 or end <= start or end > len(text):
                    continue
                span_text = text[start:end] or safe_str(val.get("text"))
                spans.append(
                    {
                        "start": start,
                        "end": end,
                        "text": span_text,
                        "label": label,
                        "origin": origin,
                    }
                )

            elif r_type == "choices" and from_name == "relation_signals":
                for c in val.get("choices", []):
                    if c in RELATION_LABELS and c not in rel_choices:
                        rel_choices.append(c)

        if not keep_overlapping:
            spans = resolve_overlapping_spans(spans)

        caption_match_text = normalize_caption_for_matching(text)
        examples.append(
            {
                "post_id": post_id,
                "text": text,
                "caption_match_text": caption_match_text,
                "caption_fingerprint": caption_fingerprint(text),
                "account_name": account_name,
                "seed_label": seed_label,
                "post_url": post_url,
                "ner_spans": spans,
                "relation_labels": sorted(rel_choices),
            }
        )

    print(f"[{now()}] Parsed {len(examples)} annotated examples from {path}. Skipped {skipped} tasks without annotations.")
    return examples


def examples_to_df(examples: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "post_id": e.get("post_id", ""),
                "account_name": e.get("account_name", ""),
                "seed_label": e.get("seed_label", ""),
                "post_url": e.get("post_url", ""),
                "caption_match_text": e.get("caption_match_text", ""),
                "caption_fingerprint": e.get("caption_fingerprint", ""),
                "text": e.get("text", ""),
                "num_ner_spans": len(e.get("ner_spans", [])),
                "relation_labels": json.dumps(e.get("relation_labels", []), ensure_ascii=False),
            }
            for e in examples
        ]
    )


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------


def build_bio_label_maps() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    labels = ["O"]
    for lab in NER_LABELS:
        labels.append(f"B-{lab}")
        labels.append(f"I-{lab}")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label


BIO_LABELS, BIO_LABEL2ID, BIO_ID2LABEL = build_bio_label_maps()


class NERDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex["text"]
        spans = ex.get("ner_spans", [])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        labels = []
        previous_entity_key: Optional[Tuple[int, int, str]] = None

        for start, end in offsets:
            if start == end:
                labels.append(-100)
                previous_entity_key = None
                continue

            token_label = "O"
            entity_key = None
            for sp in spans:
                sp_start, sp_end, sp_label = sp["start"], sp["end"], sp["label"]
                if start < sp_end and end > sp_start:
                    entity_key = (sp_start, sp_end, sp_label)
                    prefix = "B" if start <= sp_start or previous_entity_key != entity_key else "I"
                    token_label = f"{prefix}-{sp_label}"
                    break
            labels.append(BIO_LABEL2ID[token_label])
            previous_entity_key = entity_key

        enc["labels"] = labels
        return {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}


class REDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(ex["text"], truncation=True, max_length=self.max_length, padding=False)
        y = np.zeros(len(RELATION_LABELS), dtype=np.float32)
        for lab in ex.get("relation_labels", []):
            if lab in RELATION_LABELS:
                y[RELATION_LABELS.index(lab)] = 1.0
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.float)
        return item


# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------


def make_training_args(output_dir: Path, epochs: float, batch_size: int, lr: float, seed: int, fp16: bool) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        seed=seed,
        fp16=fp16,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**kwargs)


def build_trainer(model: Any, args: TrainingArguments, train_dataset: Dataset, data_collator: Any, tokenizer: Any) -> Trainer:
    sig = inspect.signature(Trainer.__init__)
    params = set(sig.parameters.keys())
    kwargs = dict(model=model, args=args, train_dataset=train_dataset, data_collator=data_collator)
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return Trainer(**kwargs)


def train_ner_model(
    train_examples: List[Dict[str, Any]],
    model_name: str,
    output_dir: Path,
    epochs: float,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    fp16: bool,
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(BIO_LABELS),
        id2label=BIO_ID2LABEL,
        label2id=BIO_LABEL2ID,
    )
    dataset = NERDataset(train_examples, tokenizer, max_length)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = build_trainer(
        model=model,
        args=make_training_args(output_dir, epochs, batch_size, lr, seed, fp16),
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    return model, tokenizer


def train_re_model(
    train_examples: List[Dict[str, Any]],
    model_name: str,
    output_dir: Path,
    epochs: float,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    fp16: bool,
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(RELATION_LABELS),
        id2label={i: lab for i, lab in enumerate(RELATION_LABELS)},
        label2id={lab: i for i, lab in enumerate(RELATION_LABELS)},
        problem_type="multi_label_classification",
    )
    dataset = REDataset(train_examples, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = build_trainer(
        model=model,
        args=make_training_args(output_dir, epochs, batch_size, lr, seed, fp16),
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    return model, tokenizer


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------


def merge_token_predictions_to_spans(
    text: str,
    offsets: List[Tuple[int, int]],
    pred_ids: List[int],
    probs: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    cur_label: Optional[str] = None
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_scores: List[float] = []

    def close_current() -> None:
        nonlocal cur_label, cur_start, cur_end, cur_scores
        if cur_label is not None and cur_start is not None and cur_end is not None and cur_end > cur_start:
            spans.append(
                {
                    "start": int(cur_start),
                    "end": int(cur_end),
                    "text": text[cur_start:cur_end],
                    "label": cur_label,
                    "score": float(np.mean(cur_scores)) if cur_scores else None,
                }
            )
        cur_label = None
        cur_start = None
        cur_end = None
        cur_scores = []

    for i, ((start, end), pred_id) in enumerate(zip(offsets, pred_ids)):
        if start == end:
            continue
        lab = BIO_ID2LABEL[int(pred_id)]
        score = probs[i] if probs is not None else None
        if lab == "O":
            close_current()
            continue
        prefix, ent_label = lab.split("-", 1)
        if prefix == "B" or cur_label != ent_label:
            close_current()
            cur_label = ent_label
            cur_start = start
            cur_end = end
            cur_scores = [float(score)] if score is not None else []
        else:
            cur_end = end
            if score is not None:
                cur_scores.append(float(score))
    close_current()
    return spans


@torch.no_grad()
def predict_ner_spans(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    max_length: int,
    device: str,
    batch_size: int = 16,
) -> List[List[Dict[str, Any]]]:
    model.to(device)
    model.eval()
    all_spans: List[List[Dict[str, Any]]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting NER", leave=False):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_batch = enc.pop("offset_mapping").cpu().tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).tolist()
        max_probs = probs.max(dim=-1).values.tolist()

        for text, offsets, ids, scores in zip(batch_texts, offsets_batch, pred_ids, max_probs):
            all_spans.append(
                merge_token_predictions_to_spans(
                    text,
                    [(int(a), int(b)) for a, b in offsets],
                    ids,
                    scores,
                )
            )
    return all_spans


@torch.no_grad()
def predict_re_probs(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    max_length: int,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    model.to(device)
    model.eval()
    all_probs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting RE", leave=False):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu()
        all_probs.append(torch.sigmoid(logits).numpy())
    return np.vstack(all_probs) if all_probs else np.empty((0, len(RELATION_LABELS)))


def get_re_y(examples: List[Dict[str, Any]]) -> np.ndarray:
    y = np.zeros((len(examples), len(RELATION_LABELS)), dtype=int)
    for i, ex in enumerate(examples):
        for lab in ex.get("relation_labels", []):
            if lab in RELATION_LABELS:
                y[i, RELATION_LABELS.index(lab)] = 1
    return y


def relation_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, lab in enumerate(RELATION_LABELS):
        p, r, f1, support = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
        rows.append({"label": lab, "support": int(y_true[:, i].sum()), "precision": p, "recall": r, "f1": f1})
    per_label = pd.DataFrame(rows)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true.reshape(-1), y_pred.reshape(-1), average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    summary = {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "total_positive_labels": int(y_true.sum()),
        "threshold": threshold,
    }
    return per_label, summary


def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def exact_span_metrics(gold_examples: List[Dict[str, Any]], pred_examples: List[List[Dict[str, Any]]]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    label_rows = []
    total_tp = total_fp = total_fn = 0
    for lab in NER_LABELS:
        tp = fp = fn = 0
        for ex, pred_spans in zip(gold_examples, pred_examples):
            gold_set = {(s["start"], s["end"], s["label"]) for s in ex.get("ner_spans", []) if s["label"] == lab}
            pred_set = {(s["start"], s["end"], s["label"]) for s in pred_spans if s["label"] == lab}
            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)
        m = prf_from_counts(tp, fp, fn)
        support = tp + fn
        label_rows.append({"label": lab, "support": support, "tp": tp, "fp": fp, "fn": fn, **m})
        total_tp += tp
        total_fp += fp
        total_fn += fn
    per_label = pd.DataFrame(label_rows)
    micro = prf_from_counts(total_tp, total_fp, total_fn)
    summary = {
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_precision": float(per_label["precision"].mean()) if len(per_label) else 0.0,
        "macro_recall": float(per_label["recall"].mean()) if len(per_label) else 0.0,
        "macro_f1": float(per_label["f1"].mean()) if len(per_label) else 0.0,
        "total_support": int(per_label["support"].sum()) if len(per_label) else 0,
    }
    return per_label, summary


# ---------------------------------------------------------------------
# Fold creation, training and OOF prediction
# ---------------------------------------------------------------------


def make_caption_group_folds(examples: List[Dict[str, Any]], num_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.array([e["caption_fingerprint"] for e in examples])
    unique_groups = np.unique(groups)
    indices = np.arange(len(examples))
    if len(unique_groups) >= num_folds:
        print(f"[{now()}] Using GroupKFold by caption_fingerprint for NER/RE folds.")
        splitter = GroupKFold(n_splits=num_folds)
        return [(tr, te) for tr, te in splitter.split(indices, groups=groups)]
    print(f"[{now()}] Not enough unique caption groups for GroupKFold; using shuffled KFold.")
    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(indices)]



def train_folds_and_create_oof(args: argparse.Namespace, examples: List[Dict[str, Any]], out_dir: Path) -> Tuple[List[Dict[str, Path]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    fp16 = bool(torch.cuda.is_available() and not args.cpu and args.fp16)

    folds_dir = out_dir / "models" / "caption_group_5fold"
    cv_dir = out_dir / "cv_results"
    if folds_dir.exists() and not args.reuse_existing_fold_models:
        shutil.rmtree(folds_dir)
    if cv_dir.exists() and not args.reuse_existing_fold_models:
        shutil.rmtree(cv_dir)
    ensure_dir(folds_dir)
    ensure_dir(cv_dir)

    fold_splits = make_caption_group_folds(examples, args.num_folds, args.seed)
    fold_model_dirs: List[Dict[str, Path]] = []
    oof_by_post_id: Dict[str, Dict[str, Any]] = {}
    fold_summaries: List[Dict[str, Any]] = []
    all_ner_per_label = []
    all_re_per_label = []

    for fold, (train_idx, test_idx) in enumerate(fold_splits, start=1):
        fold_dir = folds_dir / f"fold_{fold}"
        ner_dir = fold_dir / "ner"
        re_dir = fold_dir / "re"
        ensure_dir(fold_dir)

        train_ex = [examples[i] for i in train_idx]
        test_ex = [examples[i] for i in test_idx]
        test_texts = [e["text"] for e in test_ex]
        print(f"\n[{now()}] ===== Caption-group fold {fold}/{args.num_folds}: train={len(train_ex)}, heldout={len(test_ex)} =====")

        if args.reuse_existing_fold_models and ner_dir.exists() and re_dir.exists():
            print(f"[{now()}] Reusing existing fold models from {fold_dir}")
            ner_tok = AutoTokenizer.from_pretrained(ner_dir, use_fast=True)
            ner_model = AutoModelForTokenClassification.from_pretrained(ner_dir)
            re_tok = AutoTokenizer.from_pretrained(re_dir, use_fast=True)
            re_model = AutoModelForSequenceClassification.from_pretrained(re_dir)
        else:
            if ner_dir.exists():
                shutil.rmtree(ner_dir)
            if re_dir.exists():
                shutil.rmtree(re_dir)
            ner_model, ner_tok = train_ner_model(
                train_ex,
                args.model_name,
                fold_dir / "ner_tmp",
                args.epochs_ner,
                args.batch_size,
                args.learning_rate,
                args.max_length,
                args.seed + fold,
                fp16,
            )
            ner_model.save_pretrained(ner_dir)
            ner_tok.save_pretrained(ner_dir)
            (ner_dir / "schema_labels.json").write_text(json.dumps({"ner_labels": NER_LABELS, "bio_labels": BIO_LABELS}, indent=2), encoding="utf-8")

            re_model, re_tok = train_re_model(
                train_ex,
                args.model_name,
                fold_dir / "re_tmp",
                args.epochs_re,
                args.batch_size,
                args.learning_rate,
                args.max_length,
                args.seed + 100 + fold,
                fp16,
            )
            re_model.save_pretrained(re_dir)
            re_tok.save_pretrained(re_dir)
            (re_dir / "schema_labels.json").write_text(json.dumps({"relation_labels": RELATION_LABELS}, indent=2), encoding="utf-8")

        pred_spans = predict_ner_spans(ner_model, ner_tok, test_texts, args.max_length, device, args.pred_batch_size)
        y_true = get_re_y(test_ex)
        y_prob = predict_re_probs(re_model, re_tok, test_texts, args.max_length, device, args.pred_batch_size)

        ner_per_label, ner_summary = exact_span_metrics(test_ex, pred_spans)
        ner_per_label["fold"] = fold
        ner_per_label.to_csv(fold_dir / "ner_per_label_metrics.csv", index=False)
        all_ner_per_label.append(ner_per_label)

        re_per_label, re_summary = relation_metrics(y_true, y_prob, threshold=args.re_threshold)
        re_per_label["fold"] = fold
        re_per_label.to_csv(fold_dir / "re_per_label_metrics.csv", index=False)
        all_re_per_label.append(re_per_label)

        # Store OOF predictions by post_id, because G -> S routing uses post_id.
        for ex, spans, probs in zip(test_ex, pred_spans, y_prob):
            post_id = safe_str(ex.get("post_id"))
            if not post_id:
                continue
            oof_by_post_id[post_id] = {
                "source": "oof_heldout_fold",
                "fold": fold,
                "text": ex["text"],
                "caption_fingerprint": ex.get("caption_fingerprint", ""),
                "spans": spans,
                "re_probs": probs.astype(float).tolist(),
            }

        fold_summaries.append(
            {
                "fold": fold,
                "train_size": len(train_ex),
                "heldout_size": len(test_ex),
                **{f"ner_{k}": v for k, v in ner_summary.items()},
                **{f"re_{k}": v for k, v in re_summary.items()},
            }
        )

        fold_model_dirs.append({"fold": fold, "ner_dir": ner_dir, "re_dir": re_dir})

        del ner_model, re_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pd.DataFrame(fold_summaries).to_csv(folds_dir / "fold_summary_metrics.csv", index=False)
        pd.DataFrame(fold_summaries).to_csv(cv_dir / "fold_summary_metrics.csv", index=False)

    # Save aggregate metrics.
    fold_summary_df = pd.DataFrame(fold_summaries)
    fold_summary_df.to_csv(folds_dir / "fold_summary_metrics.csv", index=False)
    fold_summary_df.to_csv(cv_dir / "fold_summary_metrics.csv", index=False)

    avg: Dict[str, Any] = {}
    for c in fold_summary_df.columns:
        if c != "fold" and pd.api.types.is_numeric_dtype(fold_summary_df[c]):
            avg[f"{c}_mean"] = float(fold_summary_df[c].mean())
            avg[f"{c}_std"] = float(fold_summary_df[c].std(ddof=0))
    (folds_dir / "cv_average_metrics.json").write_text(json.dumps(avg, indent=2), encoding="utf-8")
    (cv_dir / "cv_average_metrics.json").write_text(json.dumps(avg, indent=2), encoding="utf-8")

    if all_ner_per_label:
        ner_all = pd.concat(all_ner_per_label, ignore_index=True)
        ner_all.to_csv(folds_dir / "ner_per_label_all_folds.csv", index=False)
        ner_all.to_csv(cv_dir / "ner_per_label_all_folds.csv", index=False)
        ner_avg = ner_all.groupby("label", as_index=False).agg(
            support_mean=("support", "mean"),
            support_sum=("support", "sum"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        ner_avg.to_csv(cv_dir / "ner_per_label_average.csv", index=False)
        ner_avg.to_csv(folds_dir / "ner_per_label_average.csv", index=False)

    if all_re_per_label:
        re_all = pd.concat(all_re_per_label, ignore_index=True)
        re_all.to_csv(folds_dir / "re_per_label_all_folds.csv", index=False)
        re_all.to_csv(cv_dir / "re_per_label_all_folds.csv", index=False)
        re_avg = re_all.groupby("label", as_index=False).agg(
            support_mean=("support", "mean"),
            support_sum=("support", "sum"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        re_avg.to_csv(cv_dir / "re_per_label_average.csv", index=False)
        re_avg.to_csv(folds_dir / "re_per_label_average.csv", index=False)

    # Save OOF predictions in JSONL form.
    oof_path = folds_dir / "oof_predictions_by_post_id.jsonl"
    with oof_path.open("w", encoding="utf-8") as f:
        for post_id, item in oof_by_post_id.items():
            f.write(json.dumps({"post_id": post_id, **item}, ensure_ascii=False) + "\n")
    print(f"[{now()}] OOF predictions saved to {oof_path}")
    print(f"[{now()}] CV result files saved to {cv_dir}")
    return fold_model_dirs, oof_by_post_id, avg



def unique_preserve_order(values: Sequence[str]) -> List[str]:
    """Return values in their first-seen order while removing duplicates."""
    seen = set()
    out: List[str] = []
    for v in values:
        v = safe_str(v)
        if not v:
            continue
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out

def aggregate_spans_from_models(span_lists: List[List[Dict[str, Any]]], vote_threshold: int) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, float], Dict[str, float], Dict[str, List[str]]]:
    n_models = max(1, len(span_lists))
    has_by_label: Dict[str, int] = {}
    vote_rate_by_label: Dict[str, float] = {}
    count_mean_by_label: Dict[str, float] = {}
    text_by_label: Dict[str, List[str]] = {}

    for lab in NER_LABELS:
        per_model_texts = [[s["text"] for s in spans if s.get("label") == lab] for spans in span_lists]
        votes = sum(1 for vals in per_model_texts if vals)
        has_by_label[lab] = int(votes >= vote_threshold)
        vote_rate_by_label[lab] = float(votes / n_models)
        count_mean_by_label[lab] = float(np.mean([len(vals) for vals in per_model_texts]))
        if has_by_label[lab]:
            text_by_label[lab] = unique_preserve_order([v for vals in per_model_texts for v in vals])
        else:
            text_by_label[lab] = []

    # Keep span candidates that are supported by enough models using label+normalized text.
    candidate_votes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    candidate_counts: Counter = Counter()
    for spans in span_lists:
        seen_in_model = set()
        for s in spans:
            key = (s.get("label", ""), normalize_caption_for_matching(s.get("text", "")))
            if not key[0] or not key[1]:
                continue
            if key in seen_in_model:
                continue
            seen_in_model.add(key)
            candidate_counts[key] += 1
            if key not in candidate_votes:
                candidate_votes[key] = dict(s)
    agg_spans = []
    for key, count in candidate_counts.items():
        lab, _ = key
        if has_by_label.get(lab, 0) and count >= vote_threshold:
            s = candidate_votes[key]
            s = dict(s)
            s["vote_count"] = int(count)
            s["vote_rate"] = float(count / n_models)
            agg_spans.append(s)
    agg_spans.sort(key=lambda x: (x.get("start", 10**9), x.get("end", 10**9), x.get("label", "")))
    return agg_spans, has_by_label, vote_rate_by_label, count_mean_by_label, text_by_label


def feature_record_from_prediction(
    span_lists: List[List[Dict[str, Any]]],
    re_probs_list: List[np.ndarray],
    re_threshold: float,
    prediction_source: str,
    fold: Optional[int] = None,
    ner_vote_threshold: Optional[int] = None,
) -> Dict[str, Any]:
    n_models = max(1, len(span_lists))
    vote_threshold = ner_vote_threshold if ner_vote_threshold is not None else max(1, (n_models // 2) + 1)
    agg_spans, has_by_label, vote_rate_by_label, count_mean_by_label, text_by_label = aggregate_spans_from_models(span_lists, vote_threshold)

    re_probs = np.mean(np.vstack(re_probs_list), axis=0) if re_probs_list else np.zeros(len(RELATION_LABELS), dtype=float)
    re_pred_labels = [RELATION_LABELS[i] for i, p in enumerate(re_probs) if p >= re_threshold]

    row: Dict[str, Any] = {
        "ner_re_prediction_source": prediction_source,
        "ner_re_oof_fold": fold if fold is not None else "",
        "ner_re_num_models_aggregated": int(n_models),
        "ner_re_ner_vote_threshold": int(vote_threshold),
        "ner_entities": json.dumps(agg_spans, ensure_ascii=False),
        "has_NER": int(any(has_by_label.values())),
        "re_labels": json.dumps(re_pred_labels, ensure_ascii=False),
        "has_RE": int(bool(re_pred_labels)),
    }

    for lab in NER_LABELS:
        row[f"ner_{lab}"] = json.dumps(text_by_label[lab], ensure_ascii=False)
        row[f"has_{lab}"] = int(has_by_label[lab])
        row[f"ner_vote_rate_{lab}"] = float(vote_rate_by_label[lab])
        row[f"ner_count_{lab}_mean"] = float(count_mean_by_label[lab])

    for i, lab in enumerate(RELATION_LABELS):
        row[f"p_{lab}"] = float(re_probs[i])
        row[f"re_{lab}"] = int(re_probs[i] >= re_threshold)

    return row


def predict_ensemble_feature_records(
    texts: List[str],
    fold_model_dirs: List[Dict[str, Path]],
    args: argparse.Namespace,
    prediction_source: str,
) -> List[Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    n = len(texts)
    spans_by_model: List[List[List[Dict[str, Any]]]] = []
    re_probs_by_model: List[np.ndarray] = []

    for item in fold_model_dirs:
        fold = item["fold"]
        print(f"[{now()}] Ensemble prediction with fold {fold} models for {n} rows.")
        ner_tok = AutoTokenizer.from_pretrained(item["ner_dir"], use_fast=True)
        ner_model = AutoModelForTokenClassification.from_pretrained(item["ner_dir"])
        spans = predict_ner_spans(ner_model, ner_tok, texts, args.max_length, device, args.pred_batch_size)
        spans_by_model.append(spans)
        del ner_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        re_tok = AutoTokenizer.from_pretrained(item["re_dir"], use_fast=True)
        re_model = AutoModelForSequenceClassification.from_pretrained(item["re_dir"])
        probs = predict_re_probs(re_model, re_tok, texts, args.max_length, device, args.pred_batch_size)
        re_probs_by_model.append(probs)
        del re_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records: List[Dict[str, Any]] = []
    for i in range(n):
        span_lists = [model_spans[i] for model_spans in spans_by_model]
        probs_list = [model_probs[i] for model_probs in re_probs_by_model]
        records.append(
            feature_record_from_prediction(
                span_lists,
                probs_list,
                args.re_threshold,
                prediction_source=prediction_source,
                fold=None,
                ner_vote_threshold=args.ner_vote_threshold,
            )
        )
    return records


def feature_record_from_oof_item(item: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return feature_record_from_prediction(
        [item["spans"]],
        [np.array(item["re_probs"], dtype=float)],
        args.re_threshold,
        prediction_source="oof_heldout_fold",
        fold=int(item.get("fold")) if item.get("fold") is not None else None,
        ner_vote_threshold=1,
    )


def concat_features(base_df: pd.DataFrame, feature_records: List[Dict[str, Any]]) -> pd.DataFrame:
    feat_df = pd.DataFrame(feature_records)
    feat_df.index = base_df.index
    return pd.concat([base_df.copy(), feat_df], axis=1)



def enrich_silver_with_oof_and_ensemble(
    s_df: pd.DataFrame,
    g_post_ids: set,
    oof_by_post_id: Dict[str, Dict[str, Any]],
    fold_model_dirs: List[Dict[str, Path]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if "post_id" not in s_df.columns:
        raise ValueError("Silver CSV must contain post_id for post_id-based OOF routing.")

    s_post_ids = s_df["post_id"].fillna("").astype(str)
    is_g_post = s_post_ids.isin(g_post_ids)
    s_g = s_df[is_g_post].copy()
    s_non_g = s_df[~is_g_post].copy()

    print(f"[{now()}] Silver rows with post_id in G: {len(s_g)}. Using OOF where available.")
    print(f"[{now()}] Silver rows outside G by post_id: {len(s_non_g)}. Using 5-fold ensemble.")

    output_parts = []

    if len(s_g):
        records = []
        fallback_rows = []
        for idx, row in s_g.iterrows():
            post_id = safe_str(row.get("post_id"))
            if post_id in oof_by_post_id:
                records.append((idx, feature_record_from_oof_item(oof_by_post_id[post_id], args)))
            else:
                fallback_rows.append(idx)
        if fallback_rows:
            print(f"[{now()}] WARNING: {len(fallback_rows)} S rows matched G post_ids but had no OOF prediction; using ensemble fallback.")
            fallback_df = s_g.loc[fallback_rows]
            fallback_records = predict_ensemble_feature_records(
                fallback_df[args._silver_text_col].fillna("").astype(str).tolist(),
                fold_model_dirs,
                args,
                prediction_source="ensemble_fallback_for_missing_oof",
            )
            for idx, rec in zip(fallback_rows, fallback_records):
                records.append((idx, rec))
        rec_by_idx = {idx: rec for idx, rec in records}
        feature_records = [rec_by_idx[idx] for idx in s_g.index]
        output_parts.append(concat_features(s_g, feature_records))

    if len(s_non_g):
        records = predict_ensemble_feature_records(
            s_non_g[args._silver_text_col].fillna("").astype(str).tolist(),
            fold_model_dirs,
            args,
            prediction_source="post_id_5fold_ensemble",
        )
        output_parts.append(concat_features(s_non_g, records))

    out = pd.concat(output_parts, axis=0).sort_index() if output_parts else s_df.copy()
    return out


def enrich_binary_gold_with_ensemble(
    b_df: pd.DataFrame,
    fold_model_dirs: List[Dict[str, Path]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    print(f"[{now()}] Binary gold rows: {len(b_df)}. Using 5-fold ensemble prediction.")
    records = predict_ensemble_feature_records(
        b_df[args._binary_text_col].fillna("").astype(str).tolist(),
        fold_model_dirs,
        args,
        prediction_source="caption_5fold_ensemble",
    )
    return concat_features(b_df, records)


def save_prediction_stats(df: pd.DataFrame, path: Path, dataset_name: str, re_threshold: float) -> None:
    stats = {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "has_NER_count": int(df["has_NER"].sum()) if "has_NER" in df.columns else None,
        "has_RE_count": int(df["has_RE"].sum()) if "has_RE" in df.columns else None,
        "ner_label_counts": {lab: int(df[f"has_{lab}"].sum()) for lab in NER_LABELS if f"has_{lab}" in df.columns},
        "re_label_counts": {lab: int(df[f"re_{lab}"].sum()) for lab in RELATION_LABELS if f"re_{lab}" in df.columns},
        "re_threshold": re_threshold,
        "prediction_source_counts": df["ner_re_prediction_source"].value_counts().to_dict() if "ner_re_prediction_source" in df.columns else {},
    }
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------



def main() -> None:
    parser = argparse.ArgumentParser(description="Post-id-routed OOF/ensemble DistilBERT NER/RE extraction pipeline")
    parser.add_argument("--nerre-json", required=True, help="Label Studio JSON export with NER/RE annotations, i.e. G")
    parser.add_argument("--silver-csv", required=True, help="Silver / weak-labeling pool CSV, i.e. S")
    parser.add_argument("--binary-gold-csv", required=True, help="Manual binary gold CSV, i.e. B. B overlap is not checked in this script.")
    parser.add_argument("--out-dir", default=str(Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve() / "data" / "processed" / "ner_re"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--silver-text-col", default=None, help="Text column in silver for ensemble prediction. Default: clean_text > model_text > caption_text")
    parser.add_argument("--binary-text-col", default=None, help="Text column in binary gold for ensemble prediction. Default: clean_text > model_text > caption_text")
    parser.add_argument("--output-silver-name", default="silver_with_ner_re_oof_ensemble.csv")
    parser.add_argument("--output-binary-gold-name", default="binary_gold_with_ner_re_ensemble.csv")

    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs-ner", type=float, default=5)
    parser.add_argument("--epochs-re", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pred-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--re-threshold", type=float, default=0.5)
    parser.add_argument("--ner-vote-threshold", type=int, default=None, help="NER vote threshold for ensemble. Default: majority.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--keep-overlapping-spans", action="store_true")
    parser.add_argument("--reuse-existing-fold-models", action="store_true", help="Reuse fold models in out-dir if they already exist.")

    args = parser.parse_args()
    set_all_seeds(args.seed)

    nerre_json = Path(args.nerre_json)
    silver_csv = Path(args.silver_csv)
    binary_gold_csv = Path(args.binary_gold_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not nerre_json.exists():
        raise FileNotFoundError(f"NER/RE JSON not found: {nerre_json}")
    if not silver_csv.exists():
        raise FileNotFoundError(f"Silver CSV not found: {silver_csv}")
    if not binary_gold_csv.exists():
        raise FileNotFoundError(f"Binary gold CSV not found: {binary_gold_csv}")

    print(f"[{now()}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and not args.cpu:
        print(f"[{now()}] GPU: {torch.cuda.get_device_name(0)}")

    # Load G, S, B.
    examples = parse_labelstudio_export(nerre_json, keep_overlapping=args.keep_overlapping_spans)
    if len(examples) < args.num_folds:
        raise ValueError(f"Need at least {args.num_folds} annotated examples, found {len(examples)}")
    g_df = examples_to_df(examples)

    s_df_raw = pd.read_csv(silver_csv)
    b_df_raw = pd.read_csv(binary_gold_csv)
    silver_text_col = choose_text_column(s_df_raw, args.silver_text_col, "silver")
    binary_text_col = choose_text_column(b_df_raw, args.binary_text_col, "binary_gold")
    args._silver_text_col = silver_text_col
    args._binary_text_col = binary_text_col

    # Add caption fingerprints for fold grouping/debugging only. G -> S routing uses post_id.
    s_df = add_caption_fingerprints_to_df(s_df_raw, silver_text_col)
    b_df = add_caption_fingerprints_to_df(b_df_raw, binary_text_col)

    # Save parsed examples and run_config.
    g_df.to_csv(out_dir / "parsed_ner_re_examples_with_caption_fingerprint.csv", index=False, encoding="utf-8-sig")
    config = vars(args).copy()
    config.pop("_silver_text_col", None)
    config.pop("_binary_text_col", None)
    config.update(
        {
            "silver_text_col_used": silver_text_col,
            "binary_text_col_used": binary_text_col,
            "ner_labels": NER_LABELS,
            "relation_labels": RELATION_LABELS,
            "n_annotated_examples_G": len(examples),
            "n_silver_rows_S": len(s_df),
            "n_binary_gold_rows_B": len(b_df),
            "G_to_S_routing": "post_id",
            "binary_gold_overlap_check": "skipped_by_design",
            "fold_grouping": "caption_fingerprint_when_possible",
        }
    )
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # Check only G subset of S by post_id.
    post_id_report = audit_g_subset_s_by_post_id(g_df, s_df, out_dir)

    # Train one shared 5-fold system and get OOF predictions for G.
    fold_model_dirs, oof_by_post_id, cv_avg = train_folds_and_create_oof(args, examples, out_dir)

    predictions_dir = out_dir / "predictions"
    ensure_dir(predictions_dir)

    # Enrich S: OOF for post_id in G, ensemble for non-G.
    g_post_ids = set(g_df["post_id"].fillna("").astype(str))
    s_out = enrich_silver_with_oof_and_ensemble(s_df, g_post_ids, oof_by_post_id, fold_model_dirs, args)
    s_out_path = predictions_dir / args.output_silver_name
    s_out.to_csv(s_out_path, index=False, encoding="utf-8-sig")
    save_prediction_stats(s_out, predictions_dir / "silver_prediction_stats.json", "silver", args.re_threshold)
    print(f"[{now()}] Saved enriched silver to: {s_out_path}")

    # Enrich B with the same five fold models. No B overlap filtering in this script.
    b_out = enrich_binary_gold_with_ensemble(b_df, fold_model_dirs, args)
    b_out_path = predictions_dir / args.output_binary_gold_name
    b_out.to_csv(b_out_path, index=False, encoding="utf-8-sig")
    save_prediction_stats(b_out, predictions_dir / "binary_gold_prediction_stats.json", "binary_gold", args.re_threshold)
    print(f"[{now()}] Saved enriched binary gold to: {b_out_path}")

    final_summary = {
        "methodology": "single shared 5-fold NER/RE system with post_id-routed OOF for G-in-S",
        "G_to_S_routing": "post_id",
        "binary_gold_overlap_check": "skipped_by_design",
        "silver_output": str(s_out_path),
        "binary_gold_output": str(b_out_path),
        "cv_results_dir": str(out_dir / "cv_results"),
        "required_quality_files": {
            "ner_per_label_average": str(out_dir / "cv_results" / "ner_per_label_average.csv"),
            "re_per_label_average": str(out_dir / "cv_results" / "re_per_label_average.csv"),
        },
        "cv_average_metrics": cv_avg,
        "post_id_subset_report": post_id_report,
    }
    (out_dir / "final_extraction_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("DONE")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
