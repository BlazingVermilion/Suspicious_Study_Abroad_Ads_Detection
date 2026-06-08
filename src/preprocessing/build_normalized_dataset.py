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
Phase 2-3 processing for the Instagram education-ad dataset.

Goal
----
Create `normalized_posts.csv` with a minimal Phase-4 input schema only.

Output schema
-------------
post_id, post_url, platform, account_name, source_file,
caption_text, clean_text, model_text,
hashtags, mentions, url_count, emoji_count, external_link,
screenshot_url, posting_time, language, seed_label

Important design choices
------------------------
1. Keep `account_name` only as metadata. It is NOT injected into `model_text`.
2. `seed_label` is kept only as one of: `none`, `legitimate`.
3. Rows with `seed_label=none` remain unlabeled and should go to weak labeling.
4. `clean_text` is core-caption only for NER/RE: Instagram UI header, account/time prefix, URLs, mentions, emojis, and hashtags are removed.
5. URLs, mentions, and emojis are abstracted in `model_text` as [URL], [MENTION], [EMOJI].
6. Hashtags are normalized and appended to `model_text` as a separate [HASHTAGS] section for the later BERT/MLP branch.

Example usage
-------------
From existing clean/normalized CSV:
    python process_phase2_phase3_normalize_minimal.py \
        --input-csv data/processed/clean_dataset.csv \
        --output data/processed/normalized/normalized_posts.csv

From raw JSON metadata directory:
    python process_phase2_phase3_normalize_minimal.py \
        --input-dir data/raw/metadata \
        --output data/processed/normalized/normalized_posts.csv
"""

from __future__ import annotations

import os

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


# Raw fields expected from JSON/CSV input. Missing columns are filled with empty strings.
CANONICAL_COLUMNS = [
    "post_id",
    "post_url",
    "platform",
    "account_name",
    "caption_text",
    "hashtags",
    "screenshot_url",
    "posting_time",
    "external_link",
    "language",
    "seed_label",
]

# Final output schema requested for normalize_dataset.csv.
OUTPUT_COLUMNS = [
    "post_id",
    "post_url",
    "platform",
    "account_name",
    "source_file",

    # Text fields
    "caption_text",
    "clean_text",
    "model_text",

    # Social-media signals extracted before weak labeling
    "hashtags",
    "mentions",
    "url_count",
    "emoji_count",
    "external_link",

    # Time and scope fields
    "screenshot_url",
    "posting_time",
    "language",

    # Initial seed label only: none or legitimate
    "seed_label",
]

REQUIRED_POST_FIELDS = {"post_id", "post_url", "account_name", "caption_text"}

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@([A-Za-z0-9_.]+)")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")

# Emoji / pictograph ranges. This avoids treating ordinary non-ASCII letters
# or punctuation as emojis.
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"          # misc symbols
    "\u2700-\u27BF"          # dingbats
    "]+",
    flags=re.UNICODE,
)

INSTAGRAM_PREFIX_RE = re.compile(
    r"^\s*\d[\d,\.]*\s+likes?\s*,\s*\d[\d,\.]*\s+comments?\s*-\s*"
    r"[^:]{1,120}?\s+on\s+[^:]{1,80}:\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

# Instagram copy/scrape prefixes often appear as:
#   account_name 15w Caption...
#   account_name Edited • 157w Caption...
#   @account_name 2 days ago Caption...
# These are metadata, not caption content, and should not be learned by NER/RE.
TIME_PREFIX_RE = re.compile(
    r"^\s*@?[A-Za-z0-9_.]{2,40}\s+"
    r"(?:Edited\s*[•·\-]\s*)?"
    r"(?:"
    r"\d+\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks|mo|mon|month|months|y|yr|yrs|year|years)\b"
    r"|\d+w\b"
    r"|\d+d\b"
    r"|\d+h\b"
    r"|\d+m\b"
    r"|\d+s\b"
    r")"
    r"\s*[•·\-:–—]?\s*",
    flags=re.IGNORECASE,
)

# Optional stricter version using the known account_name metadata.
def build_account_time_prefix_re(account_name: str) -> re.Pattern[str] | None:
    account = str(account_name or "").strip().lstrip("@").lower()
    if not account:
        return None
    escaped = re.escape(account)
    return re.compile(
        rf"^\s*@?{escaped}\s+"
        r"(?:Edited\s*[•·\-]\s*)?"
        r"(?:"
        r"\d+\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks|mo|mon|month|months|y|yr|yrs|year|years)\b"
        r"|\d+w\b|\d+d\b|\d+h\b|\d+m\b|\d+s\b"
        r")"
        r"\s*[•·\-:–—]?\s*",
        flags=re.IGNORECASE,
    )


def build_account_on_date_prefix_re(account_name: str) -> re.Pattern[str] | None:
    """Remove prefixes like 'account on October 5, 2023:'."""
    account = str(account_name or "").strip().lstrip("@").lower()
    if not account:
        return None
    escaped = re.escape(account)
    return re.compile(
        rf"^\s*@?{escaped}\s+on\s+[^:]{{3,80}}:\s*",
        flags=re.IGNORECASE,
    )

UI_NOISE_RE = re.compile(
    r"\b(?:view all \d+ comments?|view comments?|add a comment|see translation|translate|edited)\b",
    flags=re.IGNORECASE,
)


def is_missing(value: Any) -> bool:
    """Return True for None/NaN-like scalar values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def find_project_root() -> Path:
    """Best-effort project root finder for scripts under src/<module>."""
    if os.getenv("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()
    try:
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return Path.cwd()


def load_json_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in ("data", "posts", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    raise ValueError(f"Unsupported JSON structure in {path}")


def looks_like_post_dataset(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False

    sample = records[: min(len(records), 5)]
    return any(len(REQUIRED_POST_FIELDS.intersection(record.keys())) >= 3 for record in sample)


def normalize_unicode_text(value: Any) -> str:
    """Normalize punctuation while preserving meaningful non-ASCII letters."""
    if is_missing(value):
        return ""

    text = str(value).replace("\u00a0", " ")
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def strip_instagram_metadata(caption: Any) -> str:
    """Remove UI prefix such as '47 likes, 1 comments - account on date:'."""
    text = normalize_unicode_text(caption)
    text = INSTAGRAM_PREFIX_RE.sub("", text).strip()

    # Remove quote artifacts left by Instagram-scraped captions.
    text = text.strip()
    if text.startswith('"'):
        text = text[1:].strip()
    if text.endswith('".'):
        text = text[:-2].rstrip() + "."
    elif text.endswith('"'):
        text = text[:-1].strip()

    return text


def extract_urls(text: str) -> list[str]:
    """Return URLs found directly in caption text."""
    return URL_RE.findall(text or "")


def extract_mentions(text: str) -> list[str]:
    mentions = [m.lower().strip(".") for m in MENTION_RE.findall(text or "")]
    return list(dict.fromkeys([m for m in mentions if m]))


def normalize_hashtags(value: Any, caption_text: str = "") -> list[str]:
    """Normalize hashtags from both the hashtags field and caption text."""
    raw_tags: list[Any] = []

    if is_missing(value):
        raw_tags = []
    elif isinstance(value, list):
        raw_tags = value
    elif isinstance(value, str):
        # Handles strings such as "daad|scholarship" or "['DAAD', 'Scholarship']".
        raw_tags = re.split(r"[|,\s\[\]'\"]+", value)
    else:
        raw_tags = []

    raw_tags.extend(HASHTAG_RE.findall(caption_text or ""))

    cleaned: list[str] = []
    for tag in raw_tags:
        tag = str(tag).strip().lstrip("#")
        tag = re.sub(r"[^A-Za-z0-9_]", "", tag).lower()
        if tag:
            cleaned.append(tag)

    return list(dict.fromkeys(cleaned))


def count_emojis(text: str) -> int:
    matches = EMOJI_RE.findall(text or "")
    return len(matches)


def strip_account_time_prefix(text: str, account_name: str = "") -> str:
    """Remove leading 'account_name 15w' / 'account_name Edited • 157w' metadata."""
    text = normalize_unicode_text(text).strip()

    # Prefer exact account-name prefix removal when metadata is available.
    account_date_re = build_account_on_date_prefix_re(account_name)
    if account_date_re is not None:
        text = account_date_re.sub("", text).strip()

    account_re = build_account_time_prefix_re(account_name)
    if account_re is not None:
        text = account_re.sub("", text).strip()

    # Fallback for rows where account_name is missing/inconsistent.
    text = TIME_PREFIX_RE.sub("", text).strip()
    return text


def remove_trailing_hashtag_block(text: str) -> str:
    """
    Remove a trailing block made mostly of hashtags.

    This keeps the caption core close to the Label Studio NER/RE text, while
    hashtags remain preserved in the separate `hashtags` field and in `model_text`.
    """
    if not text:
        return ""

    # Remove consecutive hashtag tokens at the end, including punctuation/spaces.
    text = re.sub(r"(?:\s*[#＃][A-Za-z0-9_]+[\.,;:!?]*)+\s*$", "", text).strip()
    return text


def normalize_for_clean_text(text: Any, account_name: str = "") -> str:
    """
    Core-caption text for NER/RE and rule scoring.

    Removes Instagram UI metadata, account/time prefix, URLs, mentions, emojis,
    and hashtags. Unlike the older version, hashtag words are NOT injected into
    clean_text, because the NER/RE gold annotations were made on caption core.
    """
    text = strip_instagram_metadata(text)
    text = strip_account_time_prefix(text, account_name)
    text = remove_trailing_hashtag_block(text)

    # Remove social-media artifacts from the NER/RE input.
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)  # Remove hashtags entirely, not as words.
    text = text.replace("#", " ")  # Remove leftover hashtag markers such as "# StudyInGermany".
    text = EMOJI_RE.sub(" ", text)
    text = UI_NOISE_RE.sub(" ", text)

    # Clean remaining separators and whitespace.
    text = re.sub(r"\s*[|•·]+\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip(" \t\n\r\"'.,;:-–—")


def normalize_for_model_text(caption_core: str, hashtags_text: str, external_link: str) -> str:
    """
    Create model text while avoiding account/source leakage.

    Exact URLs and exact mentions are not preserved in model_text. They are
    replaced with abstract markers.
    """
    text = normalize_unicode_text(caption_core)

    text = URL_RE.sub(" [URL] ", text)
    text = MENTION_RE.sub(" [MENTION] ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = EMOJI_RE.sub(" [EMOJI] ", text)

    for marker in ("[URL]", "[MENTION]", "[EMOJI]"):
        text = re.sub(rf"(?:{re.escape(marker)}\s*){{2,}}", f"{marker} ", text)

    text = re.sub(r"\s+", " ", text).strip()

    parts: list[str] = []
    if text:
        parts.append(f"[CAPTION] {text}")
    if hashtags_text:
        parts.append(f"[HASHTAGS] {hashtags_text}")

    # If external_link was stored outside the caption, keep only generic URL signal.
    if str(external_link or "").strip() and "[URL]" not in " ".join(parts):
        parts.append("[EXTERNAL_LINK] [URL]")

    return " ".join(parts).strip()


def normalize_path(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).replace("\\", "/").strip()


def normalize_seed_label(value: Any, source_file: str, missing_label: str) -> str:
    """
    Return only the allowed initial seed labels: `none` or `legitimate`.

    This script intentionally does not preserve weak/final labels such as
    suspicious/normal because this file is the Phase-4 input before weak labeling.
    """
    if is_missing(value) or str(value).strip() == "":
        raw = missing_label
    else:
        raw = str(value).strip().lower()

    label_aliases = {
        "": "none",
        "null": "none",
        "nan": "none",
        "none": "none",
        "unknown": "none",
        "unlabeled": "none",
        "label_none": "none",

        "legit": "legitimate",
        "legitimate": "legitimate",
        "official": "legitimate",

        # If an older intermediate file already stored normal seed rows,
        # treat them as legitimate seed examples for this minimal input schema.
        "normal": "legitimate",
        "safe": "legitimate",
        "non_suspicious": "legitimate",
        "non-suspicious": "legitimate",
        "non suspicious": "legitimate",

        # Suspicious should not exist as an initial seed label in this pipeline.
        # Map it to none so it must be re-derived by weak supervision later.
        "suspicious": "none",
        "sus": "none",
        "misleading": "none",
        "deceptive": "none",
        "fake": "none",
    }

    canonical = label_aliases.get(raw, "none")

    # If a legitimate seed file has missing labels, preserve it as legitimate.
    if canonical == "none" and "legitimate" in source_file.lower():
        canonical = "legitimate"

    return canonical if canonical in {"none", "legitimate"} else "none"


def normalize_record(record: dict[str, Any], source_file: str, missing_label: str) -> dict[str, Any]:
    row = {col: record.get(col, "") for col in CANONICAL_COLUMNS}

    row["post_id"] = str(row.get("post_id", "")).strip()
    row["post_url"] = str(row.get("post_url", "")).strip()
    row["platform"] = str(row.get("platform") or "instagram").strip().lower()
    row["account_name"] = str(row.get("account_name", "")).strip().lstrip("@").lower()
    row["source_file"] = source_file

    raw_caption = "" if is_missing(row.get("caption_text")) else str(row.get("caption_text"))
    caption_core = strip_instagram_metadata(raw_caption)

    hashtags = normalize_hashtags(row.get("hashtags"), caption_core)
    hashtags_text = " ".join(hashtags)
    mentions = extract_mentions(caption_core)
    urls = extract_urls(caption_core)
    emoji_count = count_emojis(caption_core)

    external_link_value = row.get("external_link")
    external_link = "" if is_missing(external_link_value) else str(external_link_value).strip()

    parsed_time = pd.to_datetime(row.get("posting_time"), errors="coerce", utc=True)

    return {
        "post_id": row["post_id"],
        "post_url": row["post_url"],
        "platform": row["platform"],
        "account_name": row["account_name"],
        "source_file": row["source_file"],

        "caption_text": normalize_unicode_text(raw_caption),
        "clean_text": normalize_for_clean_text(caption_core, row["account_name"]),
        "model_text": normalize_for_model_text(caption_core, hashtags_text, external_link),

        "hashtags": "|".join(hashtags),
        "mentions": "|".join(mentions),
        "url_count": len(urls),
        "emoji_count": emoji_count,
        "external_link": external_link,

        "screenshot_url": normalize_path(row.get("screenshot_url")),
        "posting_time": parsed_time.isoformat() if pd.notna(parsed_time) else "",
        "language": str(row.get("language") or "en").strip().lower(),

        "seed_label": normalize_seed_label(row.get("seed_label"), source_file, missing_label),
    }


def load_records_from_json_dir(input_dir: Path, missing_label: str) -> pd.DataFrame:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    rows: list[dict[str, Any]] = []
    skipped_files: list[str] = []

    for json_file in json_files:
        try:
            records = load_json_file(json_file)
        except ValueError:
            skipped_files.append(json_file.name)
            continue

        if not looks_like_post_dataset(records):
            skipped_files.append(json_file.name)
            continue

        for record in records:
            rows.append(normalize_record(record, json_file.name, missing_label))

    if skipped_files:
        print("Skipped non-post JSON files:")
        for file_name in skipped_files:
            print(f"- {file_name}")
        print()

    if not rows:
        raise ValueError(f"No valid Instagram post records found in: {input_dir}")

    return pd.DataFrame(rows)


def load_records_from_csv(input_csv: Path, missing_label: str) -> pd.DataFrame:
    df_in = pd.read_csv(input_csv)
    rows: list[dict[str, Any]] = []

    for _, record in df_in.iterrows():
        record_dict = record.to_dict()
        source_value = record_dict.get("source_file")
        source_file = input_csv.name if is_missing(source_value) or str(source_value).strip() == "" else str(source_value)
        rows.append(normalize_record(record_dict, source_file, missing_label))

    return pd.DataFrame(rows)


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Basic quality filters for the current research scope.
    df = df[df["post_id"].astype(str).str.len() > 0]
    df = df[df["clean_text"].astype(str).str.len() > 0]
    df = df[df["model_text"].astype(str).str.len() > 0]
    df = df[df["language"].eq("en")]
    df = df[df["seed_label"].isin(["none", "legitimate"])]

    # Remove duplicates after normalization.
    df = df.drop_duplicates(subset=["post_id"], keep="first")

    # Stable output schema.
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def build_dataset(input_dir: Path | None, input_csv: Path | None, missing_label: str) -> pd.DataFrame:
    if input_csv is not None:
        df = load_records_from_csv(input_csv, missing_label)
    elif input_dir is not None:
        df = load_records_from_json_dir(input_dir, missing_label)
    else:
        raise ValueError("Either --input-csv or --input-dir must be provided.")

    return finalize_dataframe(df)


def main() -> None:
    project_root = find_project_root()

    parser = argparse.ArgumentParser(
        description="Create normalized_posts.csv for downstream NER/RE and weak labeling."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing raw JSON metadata files.",
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Existing clean/normalized CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "processed" / "normalized" / "normalized_posts.csv",
        help="Output normalized CSV path.",
    )

    parser.add_argument(
        "--missing-label",
        choices=["none", "legitimate"],
        default="none",
        help="Label assigned when seed_label is missing/null.",
    )

    args = parser.parse_args()

    if args.input_csv is None and args.input_dir is None:
        args.input_dir = project_root / "data" / "raw" / "instagram" / "metadata"

    df = build_dataset(args.input_dir, args.input_csv, args.missing_label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")

    print("\nOutput columns:")
    print(", ".join(df.columns))

    print("\nSeed-label distribution:")
    print(df["seed_label"].value_counts(dropna=False).to_string())

    print("\nText policy:")
    print("clean_text = core-caption only for NER/RE; hashtags/URLs/mentions/emojis/account-time prefix removed")
    print("model_text = caption + normalized hashtags for later BERT/MLP branch")


if __name__ == "__main__":
    main()
