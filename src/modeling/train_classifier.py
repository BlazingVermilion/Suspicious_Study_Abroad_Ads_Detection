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
Classifier training, ablation, and gold-set evaluation pipeline for Suspicious Educational Ads Detection.

TRAINING
  Train clean classifier variants using silver labels:
    - weak supervision is used only as y_train in the silver set
    - high-level weak-labeling/rule aggregation outputs are excluded by default
    - main model: frozen DistilBERT embedding + low-level NER/RE features -> MLP

ABLATION
  Run baselines and ablations:
    - majority baseline
    - TF-IDF + Logistic Regression
    - frozen DistilBERT + Logistic Regression
    - low-level NER/RE + Logistic Regression
    - frozen DistilBERT + low-level NER/RE + Logistic Regression
    - frozen DistilBERT + MLP
    - low-level NER/RE + MLP
    - frozen DistilBERT + low-level NER/RE + MLP
    - optional weak-evidence feature ablations if weak-evidence columns exist

EVALUATION
  Evaluate all models on the human-verified binary gold set.

Design choices for class imbalance:
  - Stratified validation split by default.
  - Logistic Regression uses class_weight='balanced'.
  - MLP supports weighted BCE, focal loss, or weighted sampler.
  - Threshold is selected only from silver validation, never from gold.
  - Reports suspicious-class precision/recall/F1, macro F1, balanced accuracy,
    ROC-AUC, and PR-AUC.
  - Supports multi-seed runs and reports mean/std.

Expected input:
  --train data/processed/weak_labeling/silver_train.csv
  --gold  data/processed/ner_re/predictions/binary_gold_with_ner_re_ensemble.csv

Author: generated for Huy Vu's classifier evaluation pipeline.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

SCRIPT_VERSION = "2026-06-06-classifier-evaluation-refactor"

# -----------------------------------------------------------------------------
# Schema constants
# -----------------------------------------------------------------------------
TEXT_COLUMN_CANDIDATES = [
    "clean_text",
    "caption_text",
    "model_text",
    "core_caption",
    "text",
    "caption",
]

LABEL_COLUMN_CANDIDATES = [
    "seed_label",
    "final_train_label",
    "binary_label",
    "gold_label",
    "human_binary_label",
    "human_label",
    "manual_label",
    "final_label",
    "label",
    "framework_weak_label",
]

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

RE_LABELS = [
    "OUTCOME_GUARANTEED",
    "REQUIREMENT_WAIVED",
    "TESTIMONIAL_SUCCESS_CLAIM",
    "PROGRAM_OR_OUTCOME_HAS_VERIFIABLE_ORG",
]

# Low-level extractor features only. These are allowed in the clean classifier model.
LOW_LEVEL_NERRE_FEATURES: List[str] = ["has_NER", "has_RE"]
for _lab in NER_LABELS:
    LOW_LEVEL_NERRE_FEATURES.extend([
        f"has_{_lab}",
        f"ner_vote_rate_{_lab}",
        f"ner_count_{_lab}_mean",
    ])
for _lab in RE_LABELS:
    LOW_LEVEL_NERRE_FEATURES.extend([
        f"p_{_lab}",
        f"re_{_lab}",
    ])

# Optional weak-evidence ablation features. These are NOT used in the clean model.
# They are only used if both train and gold contain them and --include_weak_feature_ablations is passed.
WEAK_EVIDENCE_FEATURES = [
    "E_GUARANTEED_OUTCOME",
    "E_ELIGIBILITY_MISREPRESENTATION",
    "E_PRESSURE_TACTICS",
    "E_MISLEADING_TESTIMONIAL",
    "E_LACK_OF_TRANSPARENCY",
    "E_OMISSION",
    "E_IMPLICATIVE_LANGUAGE",
    "OUTCOME_GUARANTEED_EVIDENCE",
    "REQUIREMENT_WAIVED_EVIDENCE",
    "TESTIMONIAL_SUCCESS_EVIDENCE",
    "HAS_VERIFIABLE_ORG_LINK",
    "CLAIM_SCOPE",
    "GENERIC_ONLY_INSTITUTION",
    "BROAD_FREE_CLAIM",
    "MATERIAL_CAVEAT",
    "PRESSURE_INTENSITY",
    "IMPLICATIVE_PROMISE",
    "critical_risk_score",
    "omission_accumulation_score",
    "testimonial_accumulation_score",
    "transparency_persuasion_accumulation_score",
    "soft_accumulation_risk",
]

# Explicit leakage/high-level columns that must not enter the clean model.
FORBIDDEN_FEATURE_COLUMNS = set(TEXT_COLUMN_CANDIDATES + LABEL_COLUMN_CANDIDATES + [
    "index",
    "post_id",
    "post_url",
    "platform",
    "account_name",
    "source_file",
    "external_link",
    "screenshot_url",
    "posting_time",
    "language",
    "original_seed_label",
    "hashtags",
    "mentions",
    "ner_entities",
    "re_labels",
    "caption_match_text",
    "caption_fingerprint",
    "ner_re_prediction_source",
    "ner_re_oof_fold",
    "ner_re_num_models_aggregated",
    "ner_re_ner_vote_threshold",
    "triggered_deception_types",
    "top_deception_type",
    "top_risk_channel",
    "aggregation_method",
    "weak_labeling_score",
    "framework_weak_label",
    "top_deception_score",
    "critical_risk_score",
    "soft_accumulation_risk",
    "soft_deception_bundle_score",
    "presence_admission_program",
    "presence_international_recruitment_career",
    "presence_scholarship_funding",
    "presence_germany_destination",
    "presence_requirements_process",
    "presence_webinar_open_day_event",
    "content_group_presence_list",
    "audit_content_group",
    "audit_rationale",
    "scope_status",
])

CONTENT_GROUP_COLUMNS = [
    "presence_admission_program",
    "presence_international_recruitment_career",
    "presence_scholarship_funding",
    "presence_germany_destination",
    "presence_requirements_process",
    "presence_webinar_open_day_event",
]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class RunConfig:
    train_path: str
    gold_path: str
    output_dir: str
    bert_model: str = "distilbert-base-uncased"
    max_length: int = 256
    bert_batch_size: int = 16
    mlp_batch_size: int = 32
    hidden_dim: int = 256
    dropout: float = 0.35
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 8
    val_size: float = 0.20
    split_mode: str = "stratified"  # stratified, group, auto
    threshold_mode: str = "val_f1"  # fixed_05, val_f1, recall_at_precision, balanced_accuracy
    min_precision: float = 0.60
    imbalance_mode: str = "pos_weight"  # pos_weight, focal, sampler, none
    focal_gamma: float = 2.0
    seeds: str = "42"
    cache_embeddings: bool = True
    text_col: Optional[str] = None
    train_label_col: Optional[str] = None
    gold_label_col: Optional[str] = None
    include_weak_feature_ablations: bool = False
    save_models: bool = False
    device: str = "auto"  # auto, cpu, gpu


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_seeds(s: str) -> List[int]:
    vals = []
    for part in re.split(r"[,;\s]+", str(s).strip()):
        if part:
            vals.append(int(part))
    return vals or [42]


def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str], kind: str, override: Optional[str] = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"Requested {kind} column '{override}' does not exist. Available: {list(df.columns)[:80]}")
        return override
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find {kind} column. Tried {candidates}. Available: {list(df.columns)[:80]}")


def normalize_label_value(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    positive = {"1", "true", "suspicious", "weak_suspicious", "misleading", "risky"}
    negative = {"0", "false", "normal", "weak_normal", "legitimate", "legit", "official"}
    dropped = {"", "nan", "none", "unlabeled", "unknown", "uncertain", "skip", "out_scope", "out-of-scope", "review_required"}
    if s in positive:
        return 1
    if s in negative:
        return 0
    if s in dropped:
        return None
    if "suspicious" in s:
        return 1
    if "normal" in s or "legit" in s:
        return 0
    if "uncertain" in s or "unlabeled" in s:
        return None
    return None


def prepare_labels(df: pd.DataFrame, label_col: str, split_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[label_col].apply(normalize_label_value)
    keep = y.notna()
    dropped = int((~keep).sum())
    if dropped:
        print(f"[{split_name}] Dropped {dropped} rows with non-binary labels from '{label_col}'.")
    out = df.loc[keep].copy().reset_index(drop=True)
    y_arr = y.loc[keep].astype(int).to_numpy()
    print(f"[{split_name}] Label distribution: normal={int((y_arr == 0).sum())}, suspicious={int((y_arr == 1).sum())}")
    return out, y_arr


def remove_gold_overlap(train_df: pd.DataFrame, y_train: np.ndarray, gold_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, int]:
    if "post_id" not in train_df.columns or "post_id" not in gold_df.columns:
        return train_df, y_train, 0
    gold_ids = set(gold_df["post_id"].astype(str))
    keep = ~train_df["post_id"].astype(str).isin(gold_ids)
    removed = int((~keep).sum())
    if removed:
        print(f"[leakage-check] Removed {removed} train rows whose post_id appears in gold.")
    return train_df.loc[keep].copy().reset_index(drop=True), y_train[keep.to_numpy()], removed


def clean_numeric_series(s: pd.Series) -> pd.Series:
    mapped = s.replace({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0, "[]": 0, "": np.nan, "none": np.nan, "None": np.nan})
    return pd.to_numeric(mapped, errors="coerce")


def dataframe_from_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            out[col] = clean_numeric_series(df[col])
        else:
            out[col] = 0.0
    return out


def get_texts(df: pd.DataFrame, text_col: str) -> List[str]:
    return df[text_col].fillna("").astype(str).tolist()


def file_hash_short(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# -----------------------------------------------------------------------------
# Feature builders
# -----------------------------------------------------------------------------
class DensePreprocessor:
    """Median-impute and scale dense feature arrays without leakage."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.fit_transform(X)
        return self.scaler.fit_transform(X_imp).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        return self.scaler.transform(X_imp).astype(np.float32)


class NumericFeatureSet:
    def __init__(self, name: str, columns: Sequence[str]):
        self.name = name
        self.columns = list(columns)

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(df), 0), dtype=np.float32)
        raw_df = dataframe_from_columns(df, self.columns)
        return raw_df.to_numpy(dtype=np.float32)


def select_low_level_nerre_features(train_df: pd.DataFrame, gold_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    selected = [c for c in LOW_LEVEL_NERRE_FEATURES if c in train_df.columns or c in gold_df.columns]
    missing_train = [c for c in selected if c not in train_df.columns]
    missing_gold = [c for c in selected if c not in gold_df.columns]
    return selected, sorted(set(missing_train + missing_gold))


def select_weak_features(train_df: pd.DataFrame, gold_df: pd.DataFrame) -> List[str]:
    # Weak-evidence ablation is only valid if the same feature exists in both train and gold.
    return [c for c in WEAK_EVIDENCE_FEATURES if c in train_df.columns and c in gold_df.columns]


def embedding_cache_path(output_dir: Path, source_path: str, bert_model: str, max_length: int, text_col: str) -> Path:
    safe_model = bert_model.replace("/", "__")
    stem = Path(source_path).stem
    return output_dir / f"cache_embeddings_{stem}_{text_col}_{safe_model}_len{max_length}.npy"


@torch.no_grad()
def compute_bert_embeddings(
    texts: List[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    expected_rows = len(texts)
    if cache_path is not None and cache_path.exists():
        print(f"[embeddings] Loading cached embeddings: {cache_path}")
        cached = np.load(cache_path)
        if cached.ndim == 2 and cached.shape[0] == expected_rows:
            return cached.astype(np.float32, copy=False)
        print(
            "[embeddings] Stale cache ignored: "
            f"expected {expected_rows} rows, found shape {cached.shape}. Recomputing embeddings."
        )

    print(f"[embeddings] Loading {model_name} on {device}...")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("The transformers package is required to compute DistilBERT embeddings. Install it in your project environment with: pip install transformers") from e
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        hidden = outputs.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(pooled.detach().cpu().numpy())

    embs = np.vstack(all_embs).astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embs)
        print(f"[embeddings] Saved cache: {cache_path}")
    return embs


# -----------------------------------------------------------------------------
# Splits, thresholds, metrics
# -----------------------------------------------------------------------------
def make_validation_split(df: pd.DataFrame, y: np.ndarray, val_size: float, seed: int, split_mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
    split_mode = split_mode.lower()
    if split_mode in {"group", "auto"} and "account_name" in df.columns and df["account_name"].nunique() >= 5:
        groups = df["account_name"].fillna("MISSING").astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=30, test_size=val_size, random_state=seed)
        for tr_idx, va_idx in splitter.split(df, y, groups):
            if len(np.unique(y[tr_idx])) == 2 and len(np.unique(y[va_idx])) == 2:
                return tr_idx, va_idx, "group_by_account_name"
        if split_mode == "group":
            print("[split] Group split could not preserve both classes; falling back to stratified.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(splitter.split(df, y))
    return tr_idx, va_idx, "stratified"


def safe_auc(fn, y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(fn(y_true, scores))
    except Exception:
        return None


def metric_dict(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    out: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_suspicious": float(precision_score(y_true, pred, zero_division=0)),
        "recall_suspicious": float(recall_score(y_true, pred, zero_division=0)),
        "f1_suspicious": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(
            y_true,
            pred,
            target_names=["normal", "suspicious"],
            zero_division=0,
            output_dict=True,
        ),
        "roc_auc": safe_auc(roc_auc_score, y_true, scores),
        "pr_auc": safe_auc(average_precision_score, y_true, scores),
    }
    return out


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    vals = np.unique(np.clip(scores, 0.0, 1.0))
    grid = np.linspace(0.05, 0.95, 91)
    return np.unique(np.concatenate([vals, grid, np.array([0.5])]))


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, mode: str, min_precision: float = 0.60) -> Tuple[float, Dict[str, Any]]:
    mode = mode.lower()
    if mode == "fixed_05":
        th = 0.5
        return th, {"threshold_selection_mode": mode, **metric_dict(y_true, scores, th)}

    best_th = 0.5
    best_key: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    best_metrics: Dict[str, Any] = {}
    for th in candidate_thresholds(scores):
        m = metric_dict(y_true, scores, float(th))
        if mode == "val_f1":
            key = (m["f1_suspicious"], m["recall_suspicious"], m["precision_suspicious"])
        elif mode == "balanced_accuracy":
            key = (m["balanced_accuracy"], m["f1_suspicious"], m["recall_suspicious"])
        elif mode == "recall_at_precision":
            if m["precision_suspicious"] >= min_precision:
                key = (m["recall_suspicious"], m["f1_suspicious"], m["precision_suspicious"])
            else:
                key = (-1.0, m["f1_suspicious"], m["precision_suspicious"])
        else:
            raise ValueError(f"Unknown threshold_mode: {mode}")
        if key > best_key:
            best_key = key
            best_th = float(th)
            best_metrics = m
    best_metrics = {"threshold_selection_mode": mode, "min_precision": min_precision, **best_metrics}
    return best_th, best_metrics


def flatten_metrics(model_name: str, seed: int, phase: str, metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {
        "model": model_name,
        "seed": seed,
        "phase": phase,
        "threshold": metrics.get("threshold"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision_suspicious": metrics.get("precision_suspicious"),
        "recall_suspicious": metrics.get("recall_suspicious"),
        "f1_suspicious": metrics.get("f1_suspicious"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
    }
    cm = metrics.get("confusion_matrix")
    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
        row.update({"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]})
    if extra:
        row.update(extra)
    return row


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        h2 = max(32, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, h2),
            nn.ReLU(),
            nn.LayerNorm(h2),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WeightedFocalLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, weighted_sampler: bool = False) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    if weighted_sampler:
        counts = np.bincount(y.astype(int), minlength=2).astype(float)
        weights = np.where(y == 1, 1.0 / max(counts[1], 1.0), 1.0 / max(counts[0], 1.0))
        sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_proba_mlp(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    dummy_y = np.zeros(len(X), dtype=np.float32)
    loader = make_loader(X, dummy_y, batch_size=batch_size, shuffle=False)
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    cfg: RunConfig,
    seed: int,
    device: torch.device,
    fixed_epochs: Optional[int] = None,
) -> Tuple[FusionMLP, Dict[str, Any]]:
    set_seed(seed)
    model = FusionMLP(X_train.shape[1], cfg.hidden_dim, cfg.dropout).to(device)
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    pos_weight_value = n_neg / n_pos

    if cfg.imbalance_mode == "focal":
        criterion = WeightedFocalLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device), gamma=cfg.focal_gamma)
        use_sampler = False
    elif cfg.imbalance_mode == "sampler":
        criterion = nn.BCEWithLogitsLoss()
        use_sampler = True
    elif cfg.imbalance_mode == "none":
        criterion = nn.BCEWithLogitsLoss()
        use_sampler = False
    else:  # pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
        use_sampler = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = make_loader(X_train, y_train, cfg.mlp_batch_size, shuffle=True, weighted_sampler=use_sampler)

    best_state = None
    best_epoch = 0
    best_val_metric = -1.0
    best_val_threshold = 0.5
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    total_epochs = fixed_epochs if fixed_epochs is not None else cfg.epochs

    for epoch in range(1, total_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        row: Dict[str, Any] = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None}

        if X_val is not None and y_val is not None:
            val_scores = predict_proba_mlp(model, X_val, cfg.mlp_batch_size, device)
            val_th, val_sel = choose_threshold(y_val, val_scores, cfg.threshold_mode, cfg.min_precision)
            val_fixed = metric_dict(y_val, val_scores, 0.5)
            row.update({
                "val_f1_at_0.5": val_fixed["f1_suspicious"],
                "val_recall_at_0.5": val_fixed["recall_suspicious"],
                "val_precision_at_0.5": val_fixed["precision_suspicious"],
                "val_selected_threshold": val_th,
                "val_selected_f1": val_sel["f1_suspicious"],
                "val_selected_recall": val_sel["recall_suspicious"],
                "val_selected_precision": val_sel["precision_suspicious"],
            })
            # Early stop on selected validation suspicious F1, with recall as secondary.
            current = val_sel["f1_suspicious"] + 0.01 * val_sel["recall_suspicious"]
            if current > best_val_metric:
                best_val_metric = current
                best_epoch = epoch
                best_val_threshold = float(val_th)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cfg.patience:
                print(f"[MLP] Early stopping at epoch {epoch}. Best epoch={best_epoch}, val_objective={best_val_metric:.4f}")
                history.append(row)
                break
        else:
            best_epoch = epoch

        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_objective": float(best_val_metric) if best_val_metric >= 0 else None,
        "best_val_threshold": float(best_val_threshold),
        "pos_weight": float(pos_weight_value),
        "imbalance_mode": cfg.imbalance_mode,
    }


# -----------------------------------------------------------------------------
# Experiment runners
# -----------------------------------------------------------------------------
def prepare_dense_for_split(X_train: np.ndarray, X_gold: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray) -> Dict[str, Any]:
    dev_pre = DensePreprocessor()
    X_tr = dev_pre.fit_transform(X_train[tr_idx])
    X_va = dev_pre.transform(X_train[va_idx])

    final_pre = DensePreprocessor()
    X_train_final = final_pre.fit_transform(X_train)
    X_gold_final = final_pre.transform(X_gold)
    return {
        "X_tr": X_tr,
        "X_va": X_va,
        "X_train_final": X_train_final,
        "X_gold_final": X_gold_final,
        "dev_preprocessor": dev_pre,
        "final_preprocessor": final_pre,
    }


def fit_eval_logreg(
    model_name: str,
    X_train: np.ndarray,
    X_gold: np.ndarray,
    y_train: np.ndarray,
    y_gold: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    threshold_mode: str,
    min_precision: float,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    prep = prepare_dense_for_split(X_train, X_gold, tr_idx, va_idx)
    clf_dev = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
    clf_dev.fit(prep["X_tr"], y_train[tr_idx])
    val_scores = clf_dev.predict_proba(prep["X_va"])[:, 1]
    selected_threshold, val_sel_metrics = choose_threshold(y_train[va_idx], val_scores, threshold_mode, min_precision)
    val_fixed_metrics = metric_dict(y_train[va_idx], val_scores, 0.5)

    clf_final = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
    clf_final.fit(prep["X_train_final"], y_train)
    gold_scores = clf_final.predict_proba(prep["X_gold_final"])[:, 1]
    gold_selected = metric_dict(y_gold, gold_scores, selected_threshold)
    gold_fixed = metric_dict(y_gold, gold_scores, 0.5)

    details = {
        "model_type": "logistic_regression",
        "selected_threshold": selected_threshold,
        "validation_selected": val_sel_metrics,
        "validation_fixed_05": val_fixed_metrics,
        "gold_selected": gold_selected,
        "gold_fixed_05": gold_fixed,
    }
    pred_df = pd.DataFrame({
        "model": model_name,
        "seed": seed,
        "gold_score_suspicious": gold_scores,
        "gold_pred_selected": np.where(gold_scores >= selected_threshold, "suspicious", "normal"),
        "gold_pred_fixed_05": np.where(gold_scores >= 0.5, "suspicious", "normal"),
    })
    return details, pred_df


def fit_eval_mlp(
    model_name: str,
    X_train: np.ndarray,
    X_gold: np.ndarray,
    y_train: np.ndarray,
    y_gold: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    cfg: RunConfig,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], pd.DataFrame, FusionMLP, DensePreprocessor]:
    prep = prepare_dense_for_split(X_train, X_gold, tr_idx, va_idx)
    dev_model, dev_info = train_mlp(prep["X_tr"], y_train[tr_idx], prep["X_va"], y_train[va_idx], cfg, seed, device)
    val_scores = predict_proba_mlp(dev_model, prep["X_va"], cfg.mlp_batch_size, device)
    selected_threshold, val_sel_metrics = choose_threshold(y_train[va_idx], val_scores, cfg.threshold_mode, cfg.min_precision)
    val_fixed_metrics = metric_dict(y_train[va_idx], val_scores, 0.5)

    best_epoch = max(1, int(dev_info.get("best_epoch") or cfg.epochs))
    final_model, final_info = train_mlp(
        prep["X_train_final"], y_train, None, None, cfg, seed, device, fixed_epochs=best_epoch
    )
    gold_scores = predict_proba_mlp(final_model, prep["X_gold_final"], cfg.mlp_batch_size, device)
    gold_selected = metric_dict(y_gold, gold_scores, selected_threshold)
    gold_fixed = metric_dict(y_gold, gold_scores, 0.5)

    details = {
        "model_type": "mlp",
        "selected_threshold": selected_threshold,
        "dev_training": dev_info,
        "final_training": final_info,
        "validation_selected": val_sel_metrics,
        "validation_fixed_05": val_fixed_metrics,
        "gold_selected": gold_selected,
        "gold_fixed_05": gold_fixed,
    }
    pred_df = pd.DataFrame({
        "model": model_name,
        "seed": seed,
        "gold_score_suspicious": gold_scores,
        "gold_pred_selected": np.where(gold_scores >= selected_threshold, "suspicious", "normal"),
        "gold_pred_fixed_05": np.where(gold_scores >= 0.5, "suspicious", "normal"),
    })
    return details, pred_df, final_model, prep["final_preprocessor"]


def fit_eval_tfidf_logreg(
    train_texts: List[str],
    gold_texts: List[str],
    y_train: np.ndarray,
    y_gold: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    threshold_mode: str,
    min_precision: float,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    vectorizer_dev = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=30000)
    X_tr = vectorizer_dev.fit_transform([train_texts[i] for i in tr_idx])
    X_va = vectorizer_dev.transform([train_texts[i] for i in va_idx])
    clf_dev = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
    clf_dev.fit(X_tr, y_train[tr_idx])
    val_scores = clf_dev.predict_proba(X_va)[:, 1]
    selected_threshold, val_sel_metrics = choose_threshold(y_train[va_idx], val_scores, threshold_mode, min_precision)
    val_fixed_metrics = metric_dict(y_train[va_idx], val_scores, 0.5)

    vectorizer_final = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=30000)
    X_train_final = vectorizer_final.fit_transform(train_texts)
    X_gold_final = vectorizer_final.transform(gold_texts)
    clf_final = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
    clf_final.fit(X_train_final, y_train)
    gold_scores = clf_final.predict_proba(X_gold_final)[:, 1]
    gold_selected = metric_dict(y_gold, gold_scores, selected_threshold)
    gold_fixed = metric_dict(y_gold, gold_scores, 0.5)

    details = {
        "model_type": "tfidf_logistic_regression",
        "selected_threshold": selected_threshold,
        "validation_selected": val_sel_metrics,
        "validation_fixed_05": val_fixed_metrics,
        "gold_selected": gold_selected,
        "gold_fixed_05": gold_fixed,
        "tfidf_vocab_size": int(len(vectorizer_final.vocabulary_)),
    }
    pred_df = pd.DataFrame({
        "model": "tfidf_logreg",
        "seed": seed,
        "gold_score_suspicious": gold_scores,
        "gold_pred_selected": np.where(gold_scores >= selected_threshold, "suspicious", "normal"),
        "gold_pred_fixed_05": np.where(gold_scores >= 0.5, "suspicious", "normal"),
    })
    return details, pred_df


def majority_baseline(y_train: np.ndarray, y_gold: np.ndarray) -> Dict[str, Any]:
    # Always predict the majority class in training. Score is constant 0 or 1.
    maj = int(np.mean(y_train) >= 0.5)
    scores = np.ones_like(y_gold, dtype=float) if maj == 1 else np.zeros_like(y_gold, dtype=float)
    # threshold=0.5: if maj=0 and score=0 -> normal; if maj=1 and score=1 -> suspicious.
    return {
        "model_type": "majority_baseline",
        "majority_class": "suspicious" if maj else "normal",
        "selected_threshold": 0.5,
        "gold_selected": metric_dict(y_gold, scores, 0.5),
        "gold_fixed_05": metric_dict(y_gold, scores, 0.5),
    }


# -----------------------------------------------------------------------------
# Error analysis
# -----------------------------------------------------------------------------
def make_gold_prediction_frame(gold_df: pd.DataFrame, y_gold: np.ndarray, pred_parts: List[pd.DataFrame]) -> pd.DataFrame:
    base_cols = [c for c in ["post_id", "account_name", "post_url", "clean_text", "caption_text", "seed_label"] + CONTENT_GROUP_COLUMNS if c in gold_df.columns]
    base = gold_df[base_cols].copy().reset_index(drop=True)
    base["true_binary"] = y_gold
    base["true_label"] = np.where(y_gold == 1, "suspicious", "normal")
    frames = []
    for p in pred_parts:
        q = pd.concat([base, p.reset_index(drop=True)], axis=1)
        frames.append(q)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def group_error_analysis(pred_all: pd.DataFrame) -> pd.DataFrame:
    if pred_all.empty:
        return pd.DataFrame()
    rows = []
    groups = [c for c in CONTENT_GROUP_COLUMNS if c in pred_all.columns]
    for (model, seed), sub in pred_all.groupby(["model", "seed"]):
        y_true = sub["true_binary"].to_numpy(dtype=int)
        scores = sub["gold_score_suspicious"].to_numpy(dtype=float)
        pred = (sub["gold_pred_selected"].astype(str) == "suspicious").astype(int).to_numpy()
        rows.append({
            "model": model,
            "seed": seed,
            "group": "ALL",
            "support": len(sub),
            "normal_support": int((y_true == 0).sum()),
            "suspicious_support": int((y_true == 1).sum()),
            "precision_suspicious": precision_score(y_true, pred, zero_division=0),
            "recall_suspicious": recall_score(y_true, pred, zero_division=0),
            "f1_suspicious": f1_score(y_true, pred, zero_division=0),
            "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        })
        for g in groups:
            mask = pd.to_numeric(sub[g], errors="coerce").fillna(0).astype(int).eq(1)
            if not mask.any():
                continue
            ss = sub.loc[mask]
            yt = ss["true_binary"].to_numpy(dtype=int)
            pr = (ss["gold_pred_selected"].astype(str) == "suspicious").astype(int).to_numpy()
            rows.append({
                "model": model,
                "seed": seed,
                "group": g,
                "support": len(ss),
                "normal_support": int((yt == 0).sum()),
                "suspicious_support": int((yt == 1).sum()),
                "precision_suspicious": precision_score(yt, pr, zero_division=0),
                "recall_suspicious": recall_score(yt, pr, zero_division=0),
                "f1_suspicious": f1_score(yt, pr, zero_division=0),
                "macro_f1": f1_score(yt, pr, average="macro", zero_division=0),
            })
    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    value_cols = [
        "accuracy", "balanced_accuracy", "precision_suspicious", "recall_suspicious", "f1_suspicious",
        "macro_f1", "weighted_f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp",
    ]
    rows = []
    for (model, phase), sub in metrics_df.groupby(["model", "phase"]):
        row = {"model": model, "phase": phase, "n_runs": len(sub)}
        for c in value_cols:
            if c in sub.columns:
                row[f"{c}_mean"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
                row[f"{c}_std"] = float(pd.to_numeric(sub[c], errors="coerce").std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "f1_suspicious_mean" in out.columns:
        out = out.sort_values(["phase", "f1_suspicious_mean", "macro_f1_mean", "pr_auc_mean"], ascending=[True, False, False, False])
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifier training, baselines, ablations, and gold evaluation.")
    parser.add_argument("--train", required=True, help="Path to balance_silver_training_dataset.csv")
    parser.add_argument("--gold", required=True, help="Path to binary_gold_with_ner_re_ensemble.csv")
    parser.add_argument("--output_dir", default=str(Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve() / "outputs" / "classifier_evaluation"))
    parser.add_argument("--bert_model", default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--bert_batch_size", type=int, default=16)
    parser.add_argument("--mlp_batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--split_mode", choices=["stratified", "group", "auto"], default="stratified")
    parser.add_argument("--threshold_mode", choices=["fixed_05", "val_f1", "recall_at_precision", "balanced_accuracy"], default="val_f1")
    parser.add_argument("--min_precision", type=float, default=0.60)
    parser.add_argument("--imbalance_mode", choices=["pos_weight", "focal", "sampler", "none"], default="pos_weight")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--seeds", default="42", help="Comma/space separated seeds, e.g. '1,7,13,21,42'.")
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--train_label_col", default=None)
    parser.add_argument("--gold_label_col", default=None)
    parser.add_argument("--no_cache_embeddings", action="store_true")
    parser.add_argument("--include_weak_feature_ablations", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--dry_run_schema", action="store_true", help="Only inspect schema/features, do not train models.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device policy for embedding/MLP: auto uses CUDA when available; gpu fails if CUDA is unavailable; cpu forces CPU.")
    return parser.parse_args()


def select_torch_device(policy: str) -> torch.device:
    cuda_ok = torch.cuda.is_available()
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "gpu":
        if not cuda_ok:
            raise RuntimeError(
                "--device gpu was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA PyTorch build with scripts/setup_environment.py first."
            )
        return torch.device("cuda")
    return torch.device("cuda" if cuda_ok else "cpu")


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        train_path=args.train,
        gold_path=args.gold,
        output_dir=args.output_dir,
        bert_model=args.bert_model,
        max_length=args.max_length,
        bert_batch_size=args.bert_batch_size,
        mlp_batch_size=args.mlp_batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        val_size=args.val_size,
        split_mode=args.split_mode,
        threshold_mode=args.threshold_mode,
        min_precision=args.min_precision,
        imbalance_mode=args.imbalance_mode,
        focal_gamma=args.focal_gamma,
        seeds=args.seeds,
        cache_embeddings=not args.no_cache_embeddings,
        text_col=args.text_col,
        train_label_col=args.train_label_col,
        gold_label_col=args.gold_label_col,
        include_weak_feature_ablations=args.include_weak_feature_ablations,
        save_models=args.save_models,
        device=args.device,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    if cfg.save_models:
        model_dir.mkdir(exist_ok=True)

    print(f"[script] {SCRIPT_VERSION}")
    print("[1/8] Loading input files...")
    train_raw = read_csv_robust(cfg.train_path)
    gold_raw = read_csv_robust(cfg.gold_path)

    train_text_col = find_first_existing_column(train_raw, TEXT_COLUMN_CANDIDATES, "train text", cfg.text_col)
    gold_text_col = find_first_existing_column(gold_raw, TEXT_COLUMN_CANDIDATES, "gold text", cfg.text_col)
    train_label_col = find_first_existing_column(train_raw, LABEL_COLUMN_CANDIDATES, "train label", cfg.train_label_col)
    gold_label_col = find_first_existing_column(gold_raw, LABEL_COLUMN_CANDIDATES, "gold label", cfg.gold_label_col)

    print(f"[columns] train text='{train_text_col}', train label='{train_label_col}'")
    print(f"[columns] gold  text='{gold_text_col}', gold label='{gold_label_col}'")

    train_df, y_train = prepare_labels(train_raw, train_label_col, "silver_train")
    gold_df, y_gold = prepare_labels(gold_raw, gold_label_col, "binary_gold")
    train_df, y_train, overlap_removed = remove_gold_overlap(train_df, y_train, gold_df)

    nerre_cols, missing_nerre = select_low_level_nerre_features(train_df, gold_df)
    weak_cols = select_weak_features(train_df, gold_df) if cfg.include_weak_feature_ablations else []

    schema_audit = {
        "script_version": SCRIPT_VERSION,
        "config": asdict(cfg),
        "input_hashes": {
            "train_sha256_16": file_hash_short(cfg.train_path),
            "gold_sha256_16": file_hash_short(cfg.gold_path),
        },
        "columns": {
            "train_text_col": train_text_col,
            "gold_text_col": gold_text_col,
            "train_label_col": train_label_col,
            "gold_label_col": gold_label_col,
        },
        "dataset_sizes": {
            "train_rows_raw": int(len(train_raw)),
            "gold_rows_raw": int(len(gold_raw)),
            "train_rows_binary_after_overlap_check": int(len(train_df)),
            "gold_rows_binary": int(len(gold_df)),
            "train_normal": int((y_train == 0).sum()),
            "train_suspicious": int((y_train == 1).sum()),
            "gold_normal": int((y_gold == 0).sum()),
            "gold_suspicious": int((y_gold == 1).sum()),
            "overlap_removed_from_train": int(overlap_removed),
        },
        "feature_policy": {
            "clean_main_features": "DistilBERT clean_text embedding + low-level NER/RE only",
            "weak_supervision_role": "silver training labels only; no high-level weak-labeling features in clean model",
            "low_level_nerre_feature_count": len(nerre_cols),
            "low_level_nerre_features": nerre_cols,
            "missing_nerre_features": missing_nerre,
            "weak_feature_ablations_requested": cfg.include_weak_feature_ablations,
            "weak_feature_count": len(weak_cols),
            "weak_features": weak_cols,
        },
    }
    (output_dir / "classifier_schema_audit.json").write_text(json.dumps(schema_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[features] low-level NER/RE feature count: {len(nerre_cols)}")
    if missing_nerre:
        print(f"[features] Warning: missing NER/RE columns will be zero-filled: {missing_nerre}")
    if cfg.include_weak_feature_ablations:
        print(f"[features] weak-evidence ablation feature count: {len(weak_cols)}")

    if args.dry_run_schema:
        print("[dry-run] Schema audit saved. No training performed.")
        print(output_dir / "classifier_schema_audit.json")
        return

    device = select_torch_device(cfg.device)
    print(f"[device] requested={cfg.device} effective={device} cuda_available={torch.cuda.is_available()}")

    print("[2/8] Building text and tabular matrices...")
    train_texts = get_texts(train_df, train_text_col)
    gold_texts = get_texts(gold_df, gold_text_col)
    nerre_set = NumericFeatureSet("nerre", nerre_cols)
    X_train_nerre = nerre_set.raw(train_df)
    X_gold_nerre = nerre_set.raw(gold_df)

    if weak_cols:
        weak_set = NumericFeatureSet("weak", weak_cols)
        X_train_weak = weak_set.raw(train_df)
        X_gold_weak = weak_set.raw(gold_df)
    else:
        X_train_weak = np.zeros((len(train_df), 0), dtype=np.float32)
        X_gold_weak = np.zeros((len(gold_df), 0), dtype=np.float32)

    print("[3/8] Computing or loading frozen DistilBERT embeddings...")
    train_cache = embedding_cache_path(output_dir, cfg.train_path, cfg.bert_model, cfg.max_length, train_text_col) if cfg.cache_embeddings else None
    gold_cache = embedding_cache_path(output_dir, cfg.gold_path, cfg.bert_model, cfg.max_length, gold_text_col) if cfg.cache_embeddings else None
    X_train_bert = compute_bert_embeddings(train_texts, cfg.bert_model, cfg.max_length, cfg.bert_batch_size, device, train_cache)
    X_gold_bert = compute_bert_embeddings(gold_texts, cfg.bert_model, cfg.max_length, cfg.bert_batch_size, device, gold_cache)

    feature_mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "bert": (X_train_bert, X_gold_bert),
        "nerre": (X_train_nerre, X_gold_nerre),
        "bert_nerre": (np.hstack([X_train_bert, X_train_nerre]), np.hstack([X_gold_bert, X_gold_nerre])),
    }
    if weak_cols:
        feature_mats.update({
            "weak": (X_train_weak, X_gold_weak),
            "bert_weak": (np.hstack([X_train_bert, X_train_weak]), np.hstack([X_gold_bert, X_gold_weak])),
            "bert_nerre_weak": (np.hstack([X_train_bert, X_train_nerre, X_train_weak]), np.hstack([X_gold_bert, X_gold_nerre, X_gold_weak])),
        })

    seeds = parse_seeds(cfg.seeds)
    print(f"[4/8] Running experiments for seeds: {seeds}")

    all_details: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "schema_audit": schema_audit,
        "runs": {},
    }
    metric_rows: List[Dict[str, Any]] = []
    pred_parts: List[pd.DataFrame] = []

    # Majority baseline is seed-independent, but repeated per seed for summary consistency.
    for seed in seeds:
        set_seed(seed)
        tr_idx, va_idx, split_used = make_validation_split(train_df, y_train, cfg.val_size, seed, cfg.split_mode)
        print(f"\n[seed={seed}] split={split_used}, train_dev={len(tr_idx)}, val={len(va_idx)}")
        all_details["runs"][str(seed)] = {"split_used": split_used, "models": {}}

        # Majority baseline
        maj = majority_baseline(y_train, y_gold)
        all_details["runs"][str(seed)]["models"]["majority_normal"] = maj
        metric_rows.append(flatten_metrics("majority_normal", seed, "gold_selected", maj["gold_selected"], {"split_used": split_used}))
        metric_rows.append(flatten_metrics("majority_normal", seed, "gold_fixed_05", maj["gold_fixed_05"], {"split_used": split_used}))
        maj_score = np.zeros(len(y_gold), dtype=float) if maj["majority_class"] == "normal" else np.ones(len(y_gold), dtype=float)
        pred_parts.append(pd.DataFrame({
            "model": "majority_normal",
            "seed": seed,
            "gold_score_suspicious": maj_score,
            "gold_pred_selected": maj["majority_class"],
            "gold_pred_fixed_05": maj["majority_class"],
        }))

        # TF-IDF baseline
        print(f"[seed={seed}] TF-IDF + Logistic Regression...")
        tfidf_details, tfidf_pred = fit_eval_tfidf_logreg(train_texts, gold_texts, y_train, y_gold, tr_idx, va_idx, cfg.threshold_mode, cfg.min_precision, seed)
        all_details["runs"][str(seed)]["models"]["tfidf_logreg"] = tfidf_details
        metric_rows.append(flatten_metrics("tfidf_logreg", seed, "validation_selected", tfidf_details["validation_selected"], {"split_used": split_used}))
        metric_rows.append(flatten_metrics("tfidf_logreg", seed, "gold_selected", tfidf_details["gold_selected"], {"split_used": split_used}))
        metric_rows.append(flatten_metrics("tfidf_logreg", seed, "gold_fixed_05", tfidf_details["gold_fixed_05"], {"split_used": split_used}))
        pred_parts.append(tfidf_pred)

        # Logistic Regression ablations
        for feat_name in ["bert", "nerre", "bert_nerre"] + (["weak", "bert_weak", "bert_nerre_weak"] if weak_cols else []):
            Xtr, Xg = feature_mats[feat_name]
            model_name = f"{feat_name}_logreg"
            print(f"[seed={seed}] {model_name}...")
            details, pred = fit_eval_logreg(model_name, Xtr, Xg, y_train, y_gold, tr_idx, va_idx, cfg.threshold_mode, cfg.min_precision, seed)
            all_details["runs"][str(seed)]["models"][model_name] = details
            metric_rows.append(flatten_metrics(model_name, seed, "validation_selected", details["validation_selected"], {"split_used": split_used}))
            metric_rows.append(flatten_metrics(model_name, seed, "gold_selected", details["gold_selected"], {"split_used": split_used}))
            metric_rows.append(flatten_metrics(model_name, seed, "gold_fixed_05", details["gold_fixed_05"], {"split_used": split_used}))
            pred_parts.append(pred)

        # MLP ablations. Main model is bert_nerre_mlp.
        for feat_name in ["bert", "nerre", "bert_nerre"] + (["weak", "bert_weak", "bert_nerre_weak"] if weak_cols else []):
            Xtr, Xg = feature_mats[feat_name]
            if Xtr.shape[1] == 0:
                continue
            model_name = f"{feat_name}_mlp"
            print(f"[seed={seed}] {model_name}...")
            details, pred, model, preproc = fit_eval_mlp(model_name, Xtr, Xg, y_train, y_gold, tr_idx, va_idx, cfg, seed, device)
            all_details["runs"][str(seed)]["models"][model_name] = details
            metric_rows.append(flatten_metrics(model_name, seed, "validation_selected", details["validation_selected"], {"split_used": split_used}))
            metric_rows.append(flatten_metrics(model_name, seed, "gold_selected", details["gold_selected"], {"split_used": split_used}))
            metric_rows.append(flatten_metrics(model_name, seed, "gold_fixed_05", details["gold_fixed_05"], {"split_used": split_used}))
            pred_parts.append(pred)

            if cfg.save_models and model_name == "bert_nerre_mlp":
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_dim": Xtr.shape[1],
                    "hidden_dim": cfg.hidden_dim,
                    "dropout": cfg.dropout,
                    "selected_threshold": details["selected_threshold"],
                    "config": asdict(cfg),
                    "feature_set": feat_name,
                    "nerre_cols": nerre_cols,
                    "weak_cols": weak_cols,
                }, model_dir / f"{model_name}_seed{seed}.pt")
                joblib.dump(preproc, model_dir / f"{model_name}_preprocessor_seed{seed}.joblib")

    print("[5/8] Saving metric tables...")
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "metrics_by_run.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    print("[6/8] Saving gold predictions and error analysis...")
    pred_all = make_gold_prediction_frame(gold_df, y_gold, pred_parts)
    pred_all.to_csv(output_dir / "gold_predictions_all_models.csv", index=False, encoding="utf-8-sig")
    group_df = group_error_analysis(pred_all)
    group_df.to_csv(output_dir / "error_analysis_by_content_group.csv", index=False, encoding="utf-8-sig")

    print("[7/8] Saving full JSON results...")
    with open(output_dir / "classifier_results.json", "w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=2, ensure_ascii=False)

    # Save concise best model ranking on gold_selected.
    gold_summary = summary_df[summary_df["phase"].eq("gold_selected")].copy() if not summary_df.empty else pd.DataFrame()
    if not gold_summary.empty:
        gold_summary = gold_summary.sort_values(["f1_suspicious_mean", "macro_f1_mean", "pr_auc_mean"], ascending=False)
        gold_summary.to_csv(output_dir / "gold_model_ranking.csv", index=False, encoding="utf-8-sig")
        print("\n[gold selected threshold ranking]")
        show_cols = [c for c in ["model", "n_runs", "f1_suspicious_mean", "recall_suspicious_mean", "precision_suspicious_mean", "macro_f1_mean", "roc_auc_mean", "pr_auc_mean"] if c in gold_summary.columns]
        print(gold_summary[show_cols].head(12).to_string(index=False))

    print("\n[8/8] Done.")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print(" - classifier_schema_audit.json")
    print(" - metrics_by_run.csv")
    print(" - metrics_summary.csv")
    print(" - gold_model_ranking.csv")
    print(" - gold_predictions_all_models.csv")
    print(" - error_analysis_by_content_group.csv")
    print(" - classifier_results.json")


if __name__ == "__main__":
    main()
