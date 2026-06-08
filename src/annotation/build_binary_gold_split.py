#!/usr/bin/env python3
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

"""
build_binary_gold_split.py

Interactive terminal tool to build a human-verified binary gold subset from a cleaned
Instagram dataset while excluding an existing NER/RE annotation subset.

Main behavior:
1. Read normalized_posts CSV.
2. Read annotation_subset CSV and exclude those post_ids from gold candidates.
3. Write binary_gold_candidates.csv from the non-NER/RE candidate pool.
4. Interactively review `none` posts using account/source round-robin sampling.
   Commands:
       1 / n  -> label current post as normal
       2 / s  -> label current post as suspicious
       3 / k  -> skip current post
       4 / b  -> back / undo previous action
       q      -> save progress and quit
5. Stop manual review after reaching target quotas, by default:
       50 normal from `none`
       100 suspicious from `none`
6. Automatically add trusted legitimate normal examples from the candidate pool:
       20 official/public/portal-like posts, preferring non-DAAD first, then DAAD
       30 private/commercial university posts via round-robin
   If official/public posts are insufficient, the default behavior fills the shortage
   from private/commercial posts and prints a warning. Use --strict-legit-quota to fail instead.
7. Write:
       binary_gold_eval.csv   -> same schema as normalize_dataset, but seed_label is normal/suspicious
       silver_pool.csv -> normalize_dataset minus gold_subset post_ids
       binary_gold_selection_audit.csv -> extra audit metadata for transparency

Example:
    python build_binary_gold_split.py \
      --normalize "normalize_dataset(11).csv" \
      --annotation-subset "annotation_subset_200_new_ner_multilabel_re(1).csv"

Note:
- The filename intentionally preserves the user's requested spelling: build_binary_gold_split.py
- `none` is never treated as suspicious automatically; every selected normal/suspicious
  example from `none` requires manual terminal input.
"""

from __future__ import annotations

import os

import argparse
import json
import math
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_NORMALIZE_PATH = str(PROJECT_ROOT / "data" / "processed" / "normalized" / "normalized_posts.csv")
DEFAULT_ANNOTATION_PATH = str(PROJECT_ROOT / "data" / "processed" / "annotations" / "ner_re_gold_annotated_subset.json")

# You should edit these lists if your source registry changes.
# Important: DAAD is treated as official/public/portal-like, not as a private account.
DEFAULT_OFFICIAL_PUBLIC_ACCOUNTS = [
    "daad_worldwide",
    "studyingermany",
    "study.in.germany",
    "study_in_germany",
    "tu.muenchen",
    "tumofficial",
    "rwthaachenuniversity",
    "fu_berlin",
    "unistuttgart",
    "tu_dortmund",
    "uniassist_ev",
    "makeitingermany",
    "deutschland_de",
    "goetheinstitut",
]

DEFAULT_PRIVATE_UNIVERSITY_ACCOUNTS = [
    "iu.international",
    "gisma.university",
    "gisma_university",
    "constructor_university",
    "srh_university",
    "srh_university_international",
    "berlinsbi",
    "eu_business_school",
    "esmtberlin",
]


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_source_name(x: object) -> str:
    if pd.isna(x):
        return "UNKNOWN_SOURCE"
    return str(x).strip()


def read_csv_safely(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # dtype=str preserves IDs, URLs, and label values exactly.
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def read_annotation_keys(path: str, key_col: str) -> pd.DataFrame:
    """Read NER/RE annotation-subset keys from CSV or Label Studio-style JSON.

    The binary gold split must exclude the NER/RE annotated posts. Older versions
    expected a CSV. This refactor also accepts a JSON export and extracts
    `data.post_id`, top-level `post_id`, or `post_url` when available.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False)

    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = []
    records = obj if isinstance(obj, list) else obj.get("data", obj.get("tasks", obj.get("records", []))) if isinstance(obj, dict) else []
    for item in records:
        if not isinstance(item, dict):
            continue
        data = item.get("data", {}) if isinstance(item.get("data", {}), dict) else {}
        key = item.get(key_col) or data.get(key_col)
        if key is None and key_col != "post_url":
            key = item.get("post_url") or data.get("post_url")
        if key is not None and str(key).strip():
            rows.append({key_col: str(key).strip()})
    if not rows:
        raise ValueError(f"Could not extract '{key_col}' values from annotation JSON: {path}")
    return pd.DataFrame(rows).drop_duplicates(subset=[key_col])


def require_columns(df: pd.DataFrame, cols: Sequence[str], file_label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}. Available columns: {df.columns.tolist()}")


def choose_key_col(norm: pd.DataFrame, ann: pd.DataFrame, preferred: str) -> str:
    if preferred in norm.columns and preferred in ann.columns:
        return preferred
    for candidate in ["post_id", "post_url"]:
        if candidate in norm.columns and candidate in ann.columns:
            print(f"[WARN] Preferred key column '{preferred}' unavailable in both files. Falling back to '{candidate}'.")
            return candidate
    raise ValueError("Could not find a shared key column. Expected post_id or post_url in both files.")


def make_round_robin_order(df: pd.DataFrame, source_col: str, sort_sources: bool = True) -> List[int]:
    """Return row indices in source/account round-robin order."""
    if df.empty:
        return []

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, source in zip(df.index.tolist(), df[source_col].map(normalize_source_name).tolist()):
        groups[source].append(idx)

    sources = list(groups.keys())
    if sort_sources:
        sources = sorted(sources)

    max_len = max(len(v) for v in groups.values())
    order: List[int] = []
    for i in range(max_len):
        for source in sources:
            rows = groups[source]
            if i < len(rows):
                order.append(rows[i])
    return order


def wrap_text(text: str, width: int = 100) -> str:
    text = " ".join(str(text).split())
    if not text:
        return "[EMPTY TEXT]"
    return textwrap.fill(text, width=width)


def label_counts_from_actions(actions: List[dict]) -> Tuple[int, int]:
    normal = sum(1 for a in actions if a.get("stage") == "none_manual" and a.get("action") == "label" and a.get("new_label") == "normal")
    suspicious = sum(1 for a in actions if a.get("stage") == "none_manual" and a.get("action") == "label" and a.get("new_label") == "suspicious")
    return normal, suspicious


def load_state(path: Path, reset: bool = False) -> dict:
    if reset and path.exists():
        path.unlink()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "cursor": 0,
        "actions": [],
        "completed_manual_review": False,
        "finalized": False,
    }


def save_state(path: Path, state: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def print_progress(state: dict, normal_quota: int, suspicious_quota: int) -> None:
    normal, suspicious = label_counts_from_actions(state["actions"])
    skipped = sum(1 for a in state["actions"] if a.get("stage") == "none_manual" and a.get("action") == "skip")
    print("\n" + "=" * 110)
    print(f"Progress: normal from none = {normal}/{normal_quota} | suspicious from none = {suspicious}/{suspicious_quota} | skipped = {skipped}")
    print("Commands: [1/n] normal | [2/s] suspicious | [3/k] skip | [4/b] back | [q] save & quit")
    print("=" * 110)


def interactive_review_none(
    candidates: pd.DataFrame,
    key_col: str,
    source_col: str,
    text_col: str,
    label_col: str,
    normal_quota: int,
    suspicious_quota: int,
    progress_path: Path,
    reset: bool,
) -> List[dict]:
    none_df = candidates[candidates[label_col].str.lower().eq("none")].copy()
    order = make_round_robin_order(none_df, source_col=source_col)
    index_to_pos = {idx: pos for pos, idx in enumerate(order)}

    state = load_state(progress_path, reset=reset)

    # Validate cursor.
    state["cursor"] = int(state.get("cursor", 0))
    state.setdefault("actions", [])

    print(f"\nLoaded candidate `none` pool: {len(none_df)} posts across {none_df[source_col].nunique()} sources/accounts.")
    print(f"Progress file: {progress_path}")

    while True:
        normal, suspicious = label_counts_from_actions(state["actions"])
        if normal >= normal_quota and suspicious >= suspicious_quota:
            state["completed_manual_review"] = True
            save_state(progress_path, state)
            print("\nManual quota reached. Proceeding to automatic legitimate-normal selection.")
            return state["actions"]

        if state["cursor"] >= len(order):
            save_state(progress_path, state)
            raise RuntimeError(
                "Ran out of `none` candidate posts before reaching quotas. "
                f"Current: normal={normal}/{normal_quota}, suspicious={suspicious}/{suspicious_quota}."
            )

        idx = order[state["cursor"]]
        row = none_df.loc[idx]
        source = normalize_source_name(row[source_col])
        key = str(row[key_col])

        print_progress(state, normal_quota, suspicious_quota)
        print(f"Post {state['cursor'] + 1}/{len(order)} | key={key}")
        print(f"Source/account: {source}")
        print("-" * 110)
        print(wrap_text(row.get(text_col, ""), width=110))
        print("-" * 110)

        choice = input("Your choice: ").strip().lower()

        if choice in {"1", "n", "normal"}:
            state["actions"].append({
                "stage": "none_manual",
                "action": "label",
                "key": key,
                "cursor": state["cursor"],
                "source": source,
                "old_seed_label": str(row[label_col]),
                "new_label": "normal",
            })
            state["cursor"] += 1
            save_state(progress_path, state)

        elif choice in {"2", "s", "suspicious", "sus"}:
            state["actions"].append({
                "stage": "none_manual",
                "action": "label",
                "key": key,
                "cursor": state["cursor"],
                "source": source,
                "old_seed_label": str(row[label_col]),
                "new_label": "suspicious",
            })
            state["cursor"] += 1
            save_state(progress_path, state)

        elif choice in {"3", "k", "skip"}:
            state["actions"].append({
                "stage": "none_manual",
                "action": "skip",
                "key": key,
                "cursor": state["cursor"],
                "source": source,
                "old_seed_label": str(row[label_col]),
                "new_label": "skipped",
            })
            state["cursor"] += 1
            save_state(progress_path, state)

        elif choice in {"4", "b", "back", "undo"}:
            if not state["actions"]:
                print("Nothing to undo.")
                continue
            last = state["actions"].pop()
            state["cursor"] = int(last.get("cursor", max(0, state["cursor"] - 1)))
            save_state(progress_path, state)
            print(f"Undone previous action: key={last.get('key')} action={last.get('action')} label={last.get('new_label')}")

        elif choice in {"q", "quit", "exit"}:
            save_state(progress_path, state)
            print(f"Progress saved to {progress_path}. Re-run the script to resume.")
            sys.exit(0)

        else:
            print("Invalid command. Please enter 1, 2, 3, 4, or q.")


def select_first_n_by_order(df: pd.DataFrame, n: int, source_col: str) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    order = make_round_robin_order(df, source_col=source_col)
    selected_idx = order[: min(n, len(order))]
    return df.loc[selected_idx].copy()


def classify_legit_group(account: str, official_accounts: Sequence[str], private_accounts: Sequence[str]) -> str:
    a = normalize_source_name(account).lower()
    official_set = {x.lower() for x in official_accounts}
    private_set = {x.lower() for x in private_accounts}

    if a in official_set:
        return "official_public"
    if a in private_set:
        return "private_university"

    # Conservative fallbacks.
    if "daad" in a or "uniassist" in a or "goethe" in a or "makeitingermany" in a or "deutschland" in a:
        return "official_public"

    # If a legitimate account is not recognized, treat it as private/commercial for quota-filling,
    # but it will be reported in warnings.
    return "private_university_unmapped"


def auto_select_legit_normals(
    candidates: pd.DataFrame,
    already_selected_keys: set,
    key_col: str,
    source_col: str,
    label_col: str,
    official_quota: int,
    private_quota: int,
    official_accounts: Sequence[str],
    private_accounts: Sequence[str],
    strict_legit_quota: bool,
) -> Tuple[List[dict], List[str]]:
    warnings: List[str] = []

    legit = candidates[candidates[label_col].str.lower().eq("legitimate")].copy()
    legit = legit[~legit[key_col].astype(str).isin(already_selected_keys)].copy()
    if legit.empty:
        raise RuntimeError("No legitimate posts available in candidate pool for automatic normal selection.")

    legit["_legit_group"] = legit[source_col].apply(lambda x: classify_legit_group(x, official_accounts, private_accounts))
    unmapped = sorted(set(legit.loc[legit["_legit_group"].eq("private_university_unmapped"), source_col].astype(str)))
    if unmapped:
        warnings.append(
            "Unmapped legitimate accounts were treated as private/commercial for quota filling: " + ", ".join(unmapped)
        )

    official = legit[legit["_legit_group"].eq("official_public")].copy()
    private = legit[legit["_legit_group"].isin(["private_university", "private_university_unmapped"])].copy()

    # Official/public: non-DAAD first, then DAAD.
    non_daad = official[~official[source_col].str.lower().str.contains("daad", na=False)].copy()
    daad = official[official[source_col].str.lower().str.contains("daad", na=False)].copy()

    selected_off_non_daad = select_first_n_by_order(non_daad, official_quota, source_col)
    remaining_official_needed = official_quota - len(selected_off_non_daad)
    selected_daad = select_first_n_by_order(daad, remaining_official_needed, source_col)
    selected_official = pd.concat([selected_off_non_daad, selected_daad], ignore_index=False)

    official_shortage = official_quota - len(selected_official)
    if official_shortage > 0:
        msg = (
            f"Official/public legitimate quota shortage: needed {official_quota}, "
            f"available/selected {len(selected_official)}. Shortage={official_shortage}."
        )
        if strict_legit_quota:
            raise RuntimeError(msg + " Disable --strict-legit-quota to fill shortage from private/commercial posts.")
        warnings.append(msg + " Filling the shortage from private/commercial legitimate posts.")

    private_needed = private_quota + max(0, official_shortage)
    selected_private = select_first_n_by_order(private, private_needed, source_col)
    private_shortage = private_needed - len(selected_private)
    if private_shortage > 0:
        raise RuntimeError(
            f"Private/commercial legitimate quota shortage: needed {private_needed}, selected {len(selected_private)}. "
            "Cannot create enough legitimate normal examples."
        )

    actions: List[dict] = []
    for _, row in selected_official.iterrows():
        actions.append({
            "stage": "legit_auto",
            "action": "label",
            "key": str(row[key_col]),
            "source": normalize_source_name(row[source_col]),
            "old_seed_label": str(row[label_col]),
            "new_label": "normal",
            "legit_group": "official_public",
        })
    for _, row in selected_private.iterrows():
        actions.append({
            "stage": "legit_auto",
            "action": "label",
            "key": str(row[key_col]),
            "source": normalize_source_name(row[source_col]),
            "old_seed_label": str(row[label_col]),
            "new_label": "normal",
            "legit_group": "private_university" if row.get("_legit_group") == "private_university" else "private_university_unmapped",
        })

    return actions, warnings


def build_outputs(
    normalize_df: pd.DataFrame,
    candidates: pd.DataFrame,
    key_col: str,
    label_col: str,
    source_col: str,
    manual_actions: List[dict],
    legit_actions: List[dict],
    gold_out: Path,
    silver_out: Path,
    audit_out: Path,
) -> None:
    # Keep only label actions for gold. Skips are not included.
    all_label_actions = [a for a in manual_actions + legit_actions if a.get("action") == "label"]
    selected_keys = [str(a["key"]) for a in all_label_actions]

    if len(selected_keys) != len(set(selected_keys)):
        duplicates = [k for k in selected_keys if selected_keys.count(k) > 1]
        raise RuntimeError(f"Duplicate selected keys detected: {sorted(set(duplicates))[:10]}")

    label_map = {str(a["key"]): a["new_label"] for a in all_label_actions}
    stage_map = {str(a["key"]): a.get("stage", "unknown") for a in all_label_actions}
    group_map = {str(a["key"]): a.get("legit_group", "none_manual") for a in all_label_actions}

    # Gold output should preserve normalize_dataset schema.
    gold = normalize_df[normalize_df[key_col].astype(str).isin(selected_keys)].copy()
    # Preserve selected order.
    order_map = {k: i for i, k in enumerate(selected_keys)}
    gold["_selection_order"] = gold[key_col].astype(str).map(order_map)
    gold = gold.sort_values("_selection_order")
    gold[label_col] = gold[key_col].astype(str).map(label_map)
    gold = gold.drop(columns=["_selection_order"])

    # Silver is full normalize minus gold subset, as requested.
    silver = normalize_df[~normalize_df[key_col].astype(str).isin(set(selected_keys))].copy()

    gold.to_csv(gold_out, index=False)
    silver.to_csv(silver_out, index=False)

    audit_rows = []
    for i, a in enumerate(all_label_actions, start=1):
        audit_rows.append({
            "selection_order": i,
            "key": a["key"],
            "stage": a.get("stage"),
            "source": a.get("source"),
            "old_seed_label": a.get("old_seed_label"),
            "new_binary_label": a.get("new_label"),
            "legit_group": a.get("legit_group", ""),
        })
    pd.DataFrame(audit_rows).to_csv(audit_out, index=False)

    print("\nFinal outputs written:")
    print(f"  Gold subset:  {gold_out}  rows={len(gold)}")
    print(f"  Silver subset: {silver_out} rows={len(silver)}")
    print(f"  Audit file:    {audit_out} rows={len(audit_rows)}")
    print("\nGold label counts:")
    print(gold[label_col].value_counts(dropna=False).to_string())
    print("\nGold source counts top 20:")
    print(gold[source_col].value_counts(dropna=False).head(20).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build binary gold subset interactively with source round-robin sampling.")
    parser.add_argument("--normalize", default=DEFAULT_NORMALIZE_PATH, help="Path to normalized_posts CSV.")
    parser.add_argument("--annotation-subset", default=DEFAULT_ANNOTATION_PATH, help="Path to NER/RE annotation subset CSV to exclude from gold candidates.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "processed" / "splits"), help="Output directory.")
    parser.add_argument("--key-col", default="post_id", help="Unique key column, usually post_id.")
    parser.add_argument("--source-col", default="account_name", help="Source/account column.")
    parser.add_argument("--text-col", default="clean_text", help="Text column shown in terminal.")
    parser.add_argument("--label-col", default="seed_label", help="Seed label column to replace in gold output.")
    parser.add_argument("--candidate-out", default="binary_gold_candidates.csv", help="Candidate output CSV filename.")
    parser.add_argument("--gold-out", default="binary_gold_eval.csv", help="Gold subset output CSV filename.")
    parser.add_argument("--silver-out", default="silver_pool.csv", help="Silver subset output CSV filename.")
    parser.add_argument("--audit-out", default="binary_gold_selection_audit.csv", help="Audit output CSV filename.")
    parser.add_argument("--progress-file", default="binary_gold_review_progress.json", help="Progress JSON filename.")
    parser.add_argument("--normal-none-quota", type=int, default=50, help="Manual normal quota from none/unlabeled pool.")
    parser.add_argument("--suspicious-quota", type=int, default=100, help="Manual suspicious quota from none/unlabeled pool.")
    parser.add_argument("--official-legit-quota", type=int, default=20, help="Auto-selected official/public legitimate normal quota.")
    parser.add_argument("--private-legit-quota", type=int, default=30, help="Auto-selected private/commercial legitimate normal quota.")
    parser.add_argument("--official-accounts", default=",".join(DEFAULT_OFFICIAL_PUBLIC_ACCOUNTS), help="Comma-separated official/public account names.")
    parser.add_argument("--private-accounts", default=",".join(DEFAULT_PRIVATE_UNIVERSITY_ACCOUNTS), help="Comma-separated private/commercial university account names.")
    parser.add_argument("--strict-legit-quota", action="store_true", help="Fail if official/private legitimate quotas cannot be met exactly.")
    parser.add_argument("--reset", action="store_true", help="Reset saved interactive progress.")
    parser.add_argument("--dry-run", action="store_true", help="Only create candidate file and print availability stats; do not start interactive review.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_out = out_dir / args.candidate_out
    gold_out = out_dir / args.gold_out
    silver_out = out_dir / args.silver_out
    audit_out = out_dir / args.audit_out
    progress_path = out_dir / args.progress_file

    normalize_df = read_csv_safely(args.normalize)
    annotation_df = read_annotation_keys(args.annotation_subset, args.key_col)

    key_col = choose_key_col(normalize_df, annotation_df, args.key_col)
    require_columns(normalize_df, [key_col, args.source_col, args.text_col, args.label_col], "normalize_dataset")
    require_columns(annotation_df, [key_col], "annotation_subset")

    annotation_keys = set(annotation_df[key_col].astype(str))
    candidates = normalize_df[~normalize_df[key_col].astype(str).isin(annotation_keys)].copy()
    candidates.to_csv(candidate_out, index=False)

    print("\nPrepared candidate file:")
    print(f"  normalize rows:          {len(normalize_df)}")
    print(f"  annotation excluded rows:{normalize_df[key_col].astype(str).isin(annotation_keys).sum()}")
    print(f"  candidate rows:          {len(candidates)}")
    print(f"  candidate file:          {candidate_out}")
    print("\nCandidate seed_label counts:")
    print(candidates[args.label_col].value_counts(dropna=False).to_string())

    official_accounts = parse_csv_list(args.official_accounts)
    private_accounts = parse_csv_list(args.private_accounts)

    # Availability stats for legit groups.
    legit_tmp = candidates[candidates[args.label_col].str.lower().eq("legitimate")].copy()
    if not legit_tmp.empty:
        legit_tmp["_legit_group"] = legit_tmp[args.source_col].apply(lambda x: classify_legit_group(x, official_accounts, private_accounts))
        print("\nLegitimate candidate availability by mapped group:")
        print(legit_tmp["_legit_group"].value_counts(dropna=False).to_string())
        print("\nLegitimate candidate accounts:")
        print(legit_tmp[args.source_col].value_counts(dropna=False).to_string())

    if args.dry_run:
        print("\nDry run complete. Interactive review was not started.")
        return

    manual_actions = interactive_review_none(
        candidates=candidates,
        key_col=key_col,
        source_col=args.source_col,
        text_col=args.text_col,
        label_col=args.label_col,
        normal_quota=args.normal_none_quota,
        suspicious_quota=args.suspicious_quota,
        progress_path=progress_path,
        reset=args.reset,
    )

    selected_manual_keys = {str(a["key"]) for a in manual_actions if a.get("action") == "label"}
    legit_actions, warnings = auto_select_legit_normals(
        candidates=candidates,
        already_selected_keys=selected_manual_keys,
        key_col=key_col,
        source_col=args.source_col,
        label_col=args.label_col,
        official_quota=args.official_legit_quota,
        private_quota=args.private_legit_quota,
        official_accounts=official_accounts,
        private_accounts=private_accounts,
        strict_legit_quota=args.strict_legit_quota,
    )

    if warnings:
        print("\nWarnings during legitimate auto-selection:")
        for w in warnings:
            print(f"  [WARN] {w}")

    build_outputs(
        normalize_df=normalize_df,
        candidates=candidates,
        key_col=key_col,
        label_col=args.label_col,
        source_col=args.source_col,
        manual_actions=manual_actions,
        legit_actions=legit_actions,
        gold_out=gold_out,
        silver_out=silver_out,
        audit_out=audit_out,
    )

    # Mark finalized in progress file without deleting it, so your decision trail remains recoverable.
    state = load_state(progress_path, reset=False)
    state["finalized"] = True
    save_state(progress_path, state)


if __name__ == "__main__":
    main()
