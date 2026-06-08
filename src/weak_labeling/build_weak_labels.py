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

# -*- coding: utf-8 -*-
"""
Weak-labeling pipeline for Suspicious Educational Ads Detection.

Critical Risk + Co-occurrence-based Deception Accumulation version.

Inputs:
  1) silver_subset_with_ner_re.csv
  2) Goldsubset_NER_RE_annotated_data.json
  3) ner_per_label_average.csv
  4) re_per_label_average.csv

Outputs:
  1) silver_training_dataset.csv
     - modeling-ready schema.
     - Keeps the input silver/NER-RE columns that should also exist in the
       binary-gold NER/RE extraction output, with seed_label overwritten as
       the final weak training label: normal / suspicious.
     - Removes rows with final seed_label == uncertain.
     - Does NOT include weak-labeling evidence/debug columns such as E_*, S_*,
       A_*, weak_labeling_score, top_deception_type, accumulation scores, etc.
       Those are moved to statistics/debug outputs so the classifier cannot accidentally
       use high-level weak-labeling decision features as model inputs.
  2) uncertain.csv
     - all posts whose framework_weak_label == uncertain, including legitimate
       and none/unlabeled posts
     - full scored columns preserved for review.
  3) weak_labeling_statistics.csv
     - summary counts
     - lexical/rule precision estimates from the annotated gold JSON
     - schema audit: columns kept/removed from silver_training_dataset.csv
     - per-row weak-labeling decision audit with score JSON
     - legitimate posts flagged by the framework as uncertain/suspicious
  4) all_scored_dataset_debug.csv
     - full scored dataset with all methodology/debug columns

Methodological design:
  E_FEATURE = max(R_NER * NER_feature, R_RULE * RULE_feature)
  E_RE      = max(R_RE  * q_RE,       R_RULE * RULE_RE)

  Critical risk is handled directly:
    CriticalRiskScore = max(E_GUARANTEED_OUTCOME, E_ELIGIBILITY_MISREPRESENTATION)

  Non-critical soft deception is not aggregated with additive expert weights.
  Instead, it is modeled as co-occurrence-based deception accumulation:
    OmissionAccumulation = min(A_OMISSION, max(A_TRANSPARENCY, A_IMPLICATIVE, A_PRESSURE))
    TestimonialAccumulation = min(A_TESTIMONIAL, max(A_TRANSPARENCY, A_IMPLICATIVE, A_PRESSURE))
    TransparencyPersuasionAccumulation = min(A_TRANSPARENCY, A_IMPLICATIVE, A_PRESSURE)
    SoftAccumulationRisk = max(OmissionAccumulation, TestimonialAccumulation, TransparencyPersuasionAccumulation)

  A_TYPE = active_evidence(E_TYPE): below SOFT_ACTIVE_THRESHOLD the signal is treated as inactive;
  above the threshold the original confidence score is retained.

  FinalScore = max(CriticalRiskScore, SoftAccumulationRisk)
"""

from __future__ import annotations

import os

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default project-relative paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_SILVER_CSV = str(PROJECT_ROOT / "data" / "processed" / "ner_re" / "predictions" / "silver_with_ner_re_oof_ensemble.csv")
DEFAULT_GOLD_JSON = str(PROJECT_ROOT / "data" / "processed" / "annotations" / "ner_re_gold_annotated_subset.json")
DEFAULT_NER_METRICS = str(PROJECT_ROOT / "data" / "processed" / "ner_re" / "cv_results" / "ner_per_label_average.csv")
DEFAULT_RE_METRICS = str(PROJECT_ROOT / "data" / "processed" / "ner_re" / "cv_results" / "re_per_label_average.csv")


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

# Critical-risk types can directly determine suspiciousness.
CRITICAL_DECEPTION_TYPES = {
    "GUARANTEED_OUTCOME",
    "ELIGIBILITY_MISREPRESENTATION",
}

# Non-critical soft types participate in co-occurrence patterns instead of
# additive per-type expert weights.
SUBSTANTIVE_SOFT_TYPES = {
    "OMISSION",
    "MISLEADING_TESTIMONIAL",
    "LACK_OF_TRANSPARENCY",
}

RHETORICAL_AMPLIFIER_TYPES = {
    "PRESSURE_TACTICS",
    "IMPLICATIVE_LANGUAGE",
}

# Evidence below this value is treated as too weak/noisy to activate a
# co-occurrence pattern. Above the threshold the continuous confidence is retained.
SOFT_ACTIVE_THRESHOLD = 0.35

TRIGGER_THRESHOLD = SOFT_ACTIVE_THRESHOLD
NORMAL_THRESHOLD = 0.40
SUSPICIOUS_THRESHOLD = 0.70
SUPPORT_FACTOR = 0.30  # legacy CLI compatibility; not used by co-occurrence aggregation
BOOST_VALUE = 0.10  # legacy CLI compatibility; not used by co-occurrence aggregation
SOFT_SATURATION_THRESHOLD = 1.50  # legacy output compatibility only; not used for FinalScore
ACCUMULATION_TRIGGER_THRESHOLD = SUSPICIOUS_THRESHOLD
BOOST_MIN_TRIGGERED = 3
BOOST_MIN_MAX_SCORE = 0.40

# Cumulative-risk mechanism: count distinct deception types, not repeated cues.
CUMULATIVE_ENABLED = True
CUMULATIVE_BASE_THRESHOLD = 0.60
CUMULATIVE_MIN_TRIGGERED_TYPES = 4
CUMULATIVE_MIN_MEDIUM_RISK_TYPES = 2
CUMULATIVE_REQUIRE_SUPPORTING_TYPE = False

HIGH_RISK_TYPES = {
    "GUARANTEED_OUTCOME",
    "ELIGIBILITY_MISREPRESENTATION",
}

MEDIUM_RISK_TYPES = {
    "PRESSURE_TACTICS",
    "MISLEADING_TESTIMONIAL",
    "LACK_OF_TRANSPARENCY",
}

SUPPORTING_TYPES = {
    "OMISSION",
    "IMPLICATIVE_LANGUAGE",
}


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)


def clean_text_value(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text_value(text)).strip()


def has(pattern: re.Pattern, text: str) -> int:
    return 1 if pattern.search(text or "") else 0


def cap01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))


def fuzzy_or(*vals: Any) -> float:
    return max([cap01(v) for v in vals] + [0.0])


def fuzzy_and(*vals: Any) -> float:
    if not vals:
        return 0.0
    return min(cap01(v) for v in vals)


def fuzzy_not(v: Any) -> float:
    return 1.0 - cap01(v)


def active_evidence(v: Any, threshold: float = SOFT_ACTIVE_THRESHOLD) -> float:
    """Return continuous confidence only when the signal is strong enough.

    This avoids treating weak lexical/model noise as a real co-occurrence signal,
    while avoiding additive per-type expert weights.
    """
    v = cap01(v)
    return v if v >= threshold else 0.0


# ---------------------------------------------------------------------------
# Lexical rules and relation fallback rules
# ---------------------------------------------------------------------------
PATTERNS: Dict[str, re.Pattern] = {
    # NER-like lexical fallbacks. These are intentionally text-only and do not use account_name.
    "COST_CLAIM": rx(
        r"\b(study(?:ing)?\s+(?:in\s+germany\s+)?for\s+free|free\s+tuition|tuition[-\s]?free|no\s+tuition\s+fee|no\s+tuition|without\s+tuition\s+fee|free\s+education|low[-\s]?cost\s+education|low\s+tuition\s+fee)\b"
    ),
    "COST_DETAIL": rx(
        r"\b(semester\s+(?:fee|fees|contribution)|living\s+(?:cost|costs|expenses)|cost\s+of\s+living|rent|accommodation\s+(?:cost|costs|fee|fees)|health\s+insurance|application\s+fee|administrative\s+fee|monthly\s+expenses?|tuition\s+fees?)\b"
    ),
    "REQUIREMENT": rx(
        r"\b(ielts|aps|gpa|blocked\s+account|account\s+money|proof\s+of\s+funds|financial\s+proof|bank\s+statement|documents?|academic\s+transcripts?|sop|lor|cv|resume|passport|bachelor(?:'s)?\s+degree|degree\s+certificate|english\s+proficiency|german\s+language|german\s+(?:a1|a2|b1|b2|c1|c2)|language\s+certificate|visa\s+(?:requirements?|interview)|moi|medium\s+of\s+instruction|toefl|pte|duolingo|\bgre\b|\bgmat\b)\b"
    ),
    "NEGATION_CUE": rx(
        r"\b(no|without|not\s+required|not\s+mandatory|not\s+needed|not\s+necessary|waived|exempted|no\s+need|isn'?t\s+required|are\s+not\s+required)\b"
    ),
    "PROGRAM_OR_INTAKE": rx(
        r"\b(bachelor|master|m\.?\s?(?:sc|a)\.?|mba|ph\.?d\.?|mbbs|diploma|degree|course|programme|program|study\s+program|foundation\s+course|summer\s+\d{4}\s+intake|winter\s+\d{4}\s+intake|summer\s+intake|winter\s+intake|march\s+'?\d{2}\s+intake|\d{4}\s+intake)\b"
    ),
    "FIELD_OF_STUDY": rx(
        r"\b(computer\s+science|data\s+science|artificial\s+intelligence|\bAI\b|\bIT\b|engineering|nursing|hospitality|business|management|cyber\s*security|chemical\s+and\s+energy\s+engineering|medicine|healthcare|robotics|automotive|mechanical|electrical|civil\s+engineering|finance|logistics|public\s+health)\b"
    ),
    "GENERIC_INSTITUTION": rx(
        r"\b(top\s+(?:public\s+)?universit(?:y|ies)|public\s+universit(?:y|ies)|private\s+universit(?:y|ies)|german\s+universit(?:y|ies)|universit(?:y|ies)\s+in\s+germany|university\s+in\s+germany|leading\s+universit(?:y|ies)|prestigious\s+(?:german\s+)?universit(?:y|ies)|selected\s+institutions?|higher\s+education\s+institutions?|colleges?|public\s+institutions?)\b"
    ),
    "SUPPORT_SERVICE": rx(
        r"\b(admission\s+support|visa\s+assistance|application\s+support|application\s+assistance|counselling|counseling|consultation|guidance|documentation|document\s+support|complete\s+assistance|assistance|coaching|guide\s+you\s+through|end[-\s]?to[-\s]?end\s+support|profile\s+evaluation)\b"
    ),
    "OUTCOME": rx(
        r"\b(admission|admissions|admit|visa\s+approval|visa\s+success|student\s+visa|study\s+visa|scholarship|job\s+offer|employment\s+contract|placement|seat|spot|offer\s+letter|acceptance\s+letter|admission\s+letter|enrollment|enrolment)\b"
    ),
    "GUARANTEE_CUE": rx(
        r"\b(guaranteed|guarantee|assured|confirmed|sure[-\s]?shot|100\s*%\s+(?:visa|admission|success|placement|scholarship|guarantee))\b"
    ),
    "VAGUE_BENEFIT": rx(
        r"\b(dream\s+job|dream\s+career|job\s+opportunit(?:y|ies)|career\s+opportunit(?:y|ies)|global\s+(?:career|opportunit(?:y|ies))|bright\s+future|secure\s+your\s+future|life[-\s]?changing|world[-\s]?class\s+education|incredible\s+opportunities|future[-\s]?ready|academic\s+excellence|successful\s+academic\s+journey|rewarding\s+journey|transformative\s+journey|future\s+filled\s+with\s+possibilities|make\s+your\s+dreams\s+come\s+true|turn\s+your\s+dreams\s+into\s+reality|unlock\s+your\s+future|international\s+career|global\s+success)\b"
    ),
    "TESTIMONIAL_ACTOR": rx(
        r"\b(congratulations(?:\s+to)?\s+[A-Z][A-Za-z]+|big\s+congratulations\s+(?:to\s+)?[A-Z][A-Za-z]+|our\s+student(?:s)?|our\s+client(?:s)?|successful\s+applicant|candidate|applicant|student\s+success\s+stor(?:y|ies))\b"
    ),
    "DESTINATION": rx(r"\b(germany|deutschland|europe|berlin|munich|hamburg|magdeburg|stuttgart)\b"),

    # Specific lexical rules / intermediate rule atoms.
    "PRESSURE_STRONG": rx(
        r"\b(limited\s+(?:seats|slots|places)|few\s+(?:seats|slots|places)\s+left|last\s+chance|hurry\s+up|deadline\s+today|today\s+only|don'?t\s+miss|closing\s+soon|final\s+call|only\s+\d+\s+(?:seats|slots|places))\b"
    ),
    "PRESSURE_MEDIUM": rx(
        r"\b(apply\s+now|book\s+now|register\s+now|contact\s+now|enroll\s+now|enrol\s+now|start\s+today|reach\s+out\s+today|schedule\s+(?:a\s+)?(?:free\s+)?consultation|book\s+(?:a\s+)?(?:free\s+)?consultation)\b"
    ),
    "PRESSURE_WEAK": rx(r"\b(dm\s+us|dm\s+now|contact\s+us|link\s+in\s+bio|reach\s+out|call\s+us|message\s+us|inbox\s+us|comment\s+\"?germany\"?)\b"),

    # Material-caveat atoms for OMISSION.
    "STUDY_FREE": rx(r"\b(study(?:ing)?\s+(?:in\s+germany\s+)?for\s+free|free\s+study\s+in\s+germany|study\s+free\s+in\s+germany)\b"),
    "NO_TUITION": rx(r"\b(no\s+tuition\s+fee|no\s+tuition|free\s+tuition|tuition[-\s]?free|without\s+tuition\s+fee)\b"),
    "SEMESTER_FEE": rx(r"\b(semester\s+(?:fee|fees|contribution))\b"),
    "LIVING_COST": rx(r"\b(living\s+(?:cost|costs|expenses)|cost\s+of\s+living|monthly\s+expenses?|rent|accommodation\s+(?:cost|costs|fee|fees)|health\s+insurance)\b"),
    "BLOCKED_ACCOUNT": rx(r"\b(blocked\s+account|account\s+money)\b"),
    "PROOF_OF_FUNDS": rx(r"\b(proof\s+of\s+funds|financial\s+proof|bank\s+statement)\b"),
    "PUBLIC_PRIVATE_QUALIFIER": rx(r"\b(public\s+universit(?:y|ies)|private\s+universit(?:y|ies)|state\s+universit(?:y|ies)|public\s+institutions?|private\s+institutions?)\b"),
    "MATERIAL_CAVEAT": rx(
        r"\b(semester\s+(?:fee|fees|contribution)|living\s+(?:cost|costs|expenses)|cost\s+of\s+living|rent|health\s+insurance|blocked\s+account|account\s+money|proof\s+of\s+funds|financial\s+proof|bank\s+statement|public\s+universit(?:y|ies)|private\s+universit(?:y|ies)|public\s+institutions?|private\s+institutions?)\b"
    ),

    # Implicative-language atoms.
    "FUTURE_PROMISE": rx(r"\b(unlock\s+your\s+future|start\s+your\s+journey|transform\s+your\s+career|future\s+filled\s+with\s+possibilities|secure\s+your\s+future)\b"),
    "DREAM_JOB": rx(r"\b(dream\s+job|dream\s+career|dream\s+university|dream\s+destination)\b"),
    "GLOBAL_CAREER": rx(r"\b(global\s+career|international\s+career|global\s+opportunit(?:y|ies)|career\s+opportunit(?:y|ies))\b"),
    "LIFE_CHANGING": rx(r"\b(life[-\s]?changing|transformative\s+journey|turn\s+your\s+dreams\s+into\s+reality|make\s+your\s+dreams\s+come\s+true)\b"),
    "BRIGHT_FUTURE": rx(r"\b(bright\s+future|successful\s+future|future[-\s]?ready|future\s+filled\s+with\s+possibilities)\b"),
    "VAGUE_SUCCESS_PHRASE": rx(r"\b(success\s+stor(?:y|ies)|successful\s+journey|achievement|milestone|academic\s+excellence)\b"),
}

# Named education organization rule. It intentionally focuses on named orgs in the text,
# not account_name, to avoid metadata leakage.
NAMED_EDU_ORG = rx(
    r"\b(DAAD|TUM|TUHH|IU\s+International\s+University|SRH\s+University|BSBI|Uni[-\s]?Assist|"
    r"University\s+of\s+[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,5}|"
    r"[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,6}\s+(?:University|Universit(?:y|ät)|Institute|Hochschule|School))\b"
)
PROGRAM_OUTCOME_TERMS = rx(r"\b(admission|admissions|admit|scholarship|master|bachelor|m\.?sc|mba|ph\.?d|course|programme|program|degree|visa\s+approval|student\s+visa)\b")

RULE_REQUIREMENT_WAIVED = rx(
    r"\b(?:no|without|not\s+required|not\s+mandatory|not\s+needed|not\s+necessary|waived|exempted|no\s+need)\b.{0,60}\b(?:ielts|aps|blocked\s+account|account\s+money|proof\s+of\s+funds|financial\s+proof|german\s+language|gpa|documents?|toefl|pte|duolingo|moi|medium\s+of\s+instruction)\b|"
    r"\b(?:ielts|aps|blocked\s+account|account\s+money|proof\s+of\s+funds|financial\s+proof|german\s+language|gpa|documents?|toefl|pte|duolingo|moi|medium\s+of\s+instruction)\b.{0,60}\b(?:not\s+required|not\s+mandatory|not\s+needed|not\s+necessary|waived|exempted)\b|"
    r"\b(?:low\s+gpa|less\s+than\s+60\s*%|below\s+60\s*%|backlogs?)\b.{0,60}\b(?:accepted|eligible|allowed)\b"
)
RULE_OUTCOME_GUARANTEED = rx(
    r"\b(?:guaranteed|guarantee|assured|confirmed|sure[-\s]?shot)\b.{0,60}\b(?:admission|admissions|visa\s+approval|visa\s+success|student\s+visa|study\s+visa|scholarship|job|placement|seat|spot|offer\s+letter|acceptance\s+letter)\b|"
    r"\b(?:admission|admissions|visa\s+approval|visa\s+success|student\s+visa|study\s+visa|scholarship|job|placement|seat|spot|offer\s+letter|acceptance\s+letter)\b.{0,60}\b(?:guaranteed|guarantee|assured|confirmed|sure[-\s]?shot)\b|"
    r"\b100\s*%\b.{0,50}\b(?:visa|admission|success|placement|scholarship|guarantee)\b|"
    r"\b(?:visa|admission|success|placement|scholarship)\b.{0,50}\b100\s*%\b"
)
RULE_DIRECT_OUTCOME_GUARANTEE = RULE_OUTCOME_GUARANTEED
RULE_TESTIMONIAL_SUCCESS = rx(
    r"\b(?:congratulations(?:\s+to)?|big\s+congratulations(?:\s+to)?|our\s+student(?:s)?|our\s+client(?:s)?|student\s+success\s+stor(?:y|ies))\b.{0,140}\b(?:secured|securing|received|receiving|got|achieved|obtained|has\s+secured|have\s+secured|was\s+admitted|were\s+admitted)\b.{0,100}\b(?:admission|admissions|visa\s+approval|student\s+visa|study\s+visa|scholarship|admit|offer\s+letter|acceptance\s+letter)\b|"
    r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b.{0,100}\b(?:secured|received|got|achieved|obtained)\b.{0,100}\b(?:admission|visa\s+approval|student\s+visa|study\s+visa|scholarship|admit|offer\s+letter|acceptance\s+letter)\b"
)

def rule_org_link(text: str) -> int:
    """Return 1 if named edu org appears near a program/outcome term."""
    t = norm_space(text)
    if not t:
        return 0
    for m in NAMED_EDU_ORG.finditer(t):
        start = max(0, m.start() - 120)
        end = min(len(t), m.end() + 120)
        window = t[start:end]
        if PROGRAM_OUTCOME_TERMS.search(window):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_metric_f1(path: str, labels: Iterable[str]) -> Dict[str, float]:
    df = read_csv_robust(path)
    required = {"label", "f1_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metric file {path} is missing columns: {missing}")
    f1 = {str(row["label"]): cap01(row["f1_mean"]) for _, row in df.iterrows()}
    return {lab: cap01(f1.get(lab, 0.0)) for lab in labels}


def extract_gold_records(gold_json_path: str) -> List[Dict[str, Any]]:
    with open(gold_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, Any]] = []
    for task in data:
        text = task.get("data", {}).get("text") or task.get("data", {}).get("clean_text") or ""
        post_id = task.get("data", {}).get("post_id", "")
        account_name = task.get("data", {}).get("account_name", "")

        # Prefer human/guideline annotations. Fall back to predictions if no annotations exist.
        result = []
        anns = task.get("annotations") or []
        preds = task.get("predictions") or []
        if anns and anns[0].get("result") is not None:
            result = anns[0].get("result") or []
        elif preds and preds[0].get("result") is not None:
            result = preds[0].get("result") or []

        ner_labels: Set[str] = set()
        rel_labels: Set[str] = set()
        for item in result:
            if item.get("type") == "labels":
                for lab in item.get("value", {}).get("labels", []) or []:
                    ner_labels.add(str(lab))
            elif item.get("from_name") == "relation_signals" or item.get("type") == "choices":
                for ch in item.get("value", {}).get("choices", []) or []:
                    rel_labels.add(str(ch))

        records.append({
            "post_id": post_id,
            "account_name": account_name,
            "text": norm_space(text),
            "ner_labels": ner_labels,
            "relation_labels": rel_labels,
        })
    return records


def safe_parse_list_or_json(x: Any) -> Any:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (list, dict)):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return s


def raw_ner_binary(row: pd.Series, label: str) -> int:
    col = f"has_{label}"
    if col in row.index:
        return int(cap01(row[col]) >= 0.5)

    # Fallback: parse ner_entities if available.
    ents = safe_parse_list_or_json(row.get("ner_entities", None))
    if isinstance(ents, list):
        for e in ents:
            labs = e.get("labels") or e.get("label") or [] if isinstance(e, dict) else []
            if isinstance(labs, str):
                labs = [labs]
            if label in labs:
                return 1
    return 0


def q_re(row: pd.Series, label: str) -> float:
    p_col = f"p_{label}"
    if p_col in row.index:
        return cap01(row[p_col])
    b_col = f"re_{label}"
    if b_col in row.index:
        return 1.0 if cap01(row[b_col]) >= 0.5 else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Lexical rule precision estimation on gold
# ---------------------------------------------------------------------------
RuleFn = Callable[[str], int]


def rule_any(pattern_name: str) -> RuleFn:
    return lambda text: has(PATTERNS[pattern_name], text)


def rule_pressure_any(text: str) -> int:
    return int(any(has(PATTERNS[k], text) for k in ["PRESSURE_STRONG", "PRESSURE_MEDIUM", "PRESSURE_WEAK"]))


def rule_outcome_guaranteed(text: str) -> int:
    return has(RULE_OUTCOME_GUARANTEED, text)


def rule_requirement_waived(text: str) -> int:
    return has(RULE_REQUIREMENT_WAIVED, text)


def rule_direct_outcome_guarantee(text: str) -> int:
    return has(RULE_DIRECT_OUTCOME_GUARANTEE, text)


def rule_testimonial_success(text: str) -> int:
    return has(RULE_TESTIMONIAL_SUCCESS, text)


RULE_SPECS: Dict[str, Dict[str, Any]] = {
    # NER-like lexical fallbacks
    "LEX_COST_CLAIM": {"fn": rule_any("COST_CLAIM"), "target_type": "ner", "target": "COST_CLAIM", "default": 0.80},
    "LEX_COST_DETAIL": {"fn": rule_any("COST_DETAIL"), "target_type": "ner", "target": "COST_DETAIL", "default": 0.80},
    "LEX_REQUIREMENT": {"fn": rule_any("REQUIREMENT"), "target_type": "ner", "target": "REQUIREMENT", "default": 0.75},
    "LEX_NEGATION_CUE": {"fn": rule_any("NEGATION_CUE"), "target_type": "ner", "target": "NEGATION_CUE", "default": 0.90},
    "LEX_PROGRAM_OR_INTAKE": {"fn": rule_any("PROGRAM_OR_INTAKE"), "target_type": "ner", "target": "PROGRAM_OR_INTAKE", "default": 0.75},
    "LEX_FIELD_OF_STUDY": {"fn": rule_any("FIELD_OF_STUDY"), "target_type": "ner", "target": "FIELD_OF_STUDY", "default": 0.70},
    "LEX_GENERIC_INSTITUTION": {"fn": rule_any("GENERIC_INSTITUTION"), "target_type": "ner", "target": "GENERIC_INSTITUTION", "default": 0.75},
    "LEX_SUPPORT_SERVICE": {"fn": rule_any("SUPPORT_SERVICE"), "target_type": "ner", "target": "SUPPORT_SERVICE", "default": 0.70},
    "LEX_OUTCOME": {"fn": rule_any("OUTCOME"), "target_type": "ner", "target": "OUTCOME", "default": 0.75},
    "LEX_GUARANTEE_CUE": {"fn": rule_any("GUARANTEE_CUE"), "target_type": "ner", "target": "GUARANTEE_CUE", "default": 0.90},
    "LEX_VAGUE_BENEFIT": {"fn": rule_any("VAGUE_BENEFIT"), "target_type": "ner", "target": "VAGUE_BENEFIT", "default": 0.65},
    "LEX_PRESSURE_CUE": {"fn": rule_pressure_any, "target_type": "ner", "target": "PRESSURE_CUE", "default": 0.80},
    "LEX_TESTIMONIAL_ACTOR": {"fn": rule_any("TESTIMONIAL_ACTOR"), "target_type": "ner", "target": "TESTIMONIAL_ACTOR", "default": 0.75},
    "LEX_DESTINATION": {"fn": rule_any("DESTINATION"), "target_type": "ner", "target": "DESTINATION", "default": 0.90},

    # Formula-specific lexical atoms for intermediate/deception logic.
    "LEX_STUDY_FREE": {"fn": rule_any("STUDY_FREE"), "target_type": "ner", "target": "COST_CLAIM", "default": 0.85},
    "LEX_NO_TUITION": {"fn": rule_any("NO_TUITION"), "target_type": "ner", "target": "COST_CLAIM", "default": 0.90},
    "RULE_SEMESTER_FEE": {"fn": rule_any("SEMESTER_FEE"), "target_type": "ner", "target": "COST_DETAIL", "default": 0.90},
    "RULE_LIVING_COST": {"fn": rule_any("LIVING_COST"), "target_type": "ner", "target": "COST_DETAIL", "default": 0.85},
    "RULE_BLOCKED_ACCOUNT": {"fn": rule_any("BLOCKED_ACCOUNT"), "target_type": "ner", "target": "REQUIREMENT", "default": 0.90},
    "RULE_PROOF_OF_FUNDS": {"fn": rule_any("PROOF_OF_FUNDS"), "target_type": "ner", "target": "REQUIREMENT", "default": 0.90},
    "RULE_PUBLIC_PRIVATE_QUALIFIER": {"fn": rule_any("PUBLIC_PRIVATE_QUALIFIER"), "target_type": "any_ner", "target": {"GENERIC_INSTITUTION", "COST_DETAIL"}, "default": 0.85},
    "RULE_DIRECT_OUTCOME_GUARANTEE": {"fn": rule_direct_outcome_guarantee, "target_type": "relation", "target": "OUTCOME_GUARANTEED", "default": 0.90},
    "RULE_FUTURE_PROMISE": {"fn": rule_any("FUTURE_PROMISE"), "target_type": "ner", "target": "VAGUE_BENEFIT", "default": 0.60},
    "RULE_DREAM_JOB": {"fn": rule_any("DREAM_JOB"), "target_type": "ner", "target": "VAGUE_BENEFIT", "default": 0.65},
    "RULE_GLOBAL_CAREER": {"fn": rule_any("GLOBAL_CAREER"), "target_type": "ner", "target": "VAGUE_BENEFIT", "default": 0.65},
    "RULE_LIFE_CHANGING": {"fn": rule_any("LIFE_CHANGING"), "target_type": "ner", "target": "VAGUE_BENEFIT", "default": 0.65},
    "RULE_BRIGHT_FUTURE": {"fn": rule_any("BRIGHT_FUTURE"), "target_type": "ner", "target": "VAGUE_BENEFIT", "default": 0.65},

    # Relation fallback rules
    "RULE_REQUIREMENT_WAIVED": {"fn": rule_requirement_waived, "target_type": "relation", "target": "REQUIREMENT_WAIVED", "default": 0.90},
    "RULE_OUTCOME_GUARANTEED": {"fn": rule_outcome_guaranteed, "target_type": "relation", "target": "OUTCOME_GUARANTEED", "default": 0.90},
    "RULE_TESTIMONIAL_SUCCESS": {"fn": rule_testimonial_success, "target_type": "relation", "target": "TESTIMONIAL_SUCCESS_CLAIM", "default": 0.80},
    "RULE_ORG_LINK": {"fn": rule_org_link, "target_type": "relation", "target": "PROGRAM_OR_OUTCOME_HAS_VERIFIABLE_ORG", "default": 0.70},
    "RULE_MATERIAL_CAVEAT": {"fn": rule_any("MATERIAL_CAVEAT"), "target_type": "any_ner", "target": {"COST_DETAIL", "REQUIREMENT", "GENERIC_INSTITUTION"}, "default": 0.80},
    "RULE_VAGUE_SUCCESS_PHRASE": {"fn": rule_any("VAGUE_SUCCESS_PHRASE"), "target_type": "ner", "target": "VAGUE_BENEFIT", "default": 0.60},
    "RULE_PRESSURE_STRONG": {"fn": rule_any("PRESSURE_STRONG"), "target_type": "ner", "target": "PRESSURE_CUE", "default": 0.90},
    "RULE_PRESSURE_MEDIUM": {"fn": rule_any("PRESSURE_MEDIUM"), "target_type": "ner", "target": "PRESSURE_CUE", "default": 0.80},
    "RULE_PRESSURE_WEAK": {"fn": rule_any("PRESSURE_WEAK"), "target_type": "ner", "target": "PRESSURE_CUE", "default": 0.65},
}


def gold_target_positive(rec: Dict[str, Any], target_type: str, target: Any) -> bool:
    if target_type == "ner":
        return str(target) in rec["ner_labels"]
    if target_type == "relation":
        return str(target) in rec["relation_labels"]
    if target_type == "any_ner":
        return bool(set(target) & set(rec["ner_labels"]))
    raise ValueError(f"Unknown target_type: {target_type}")


def estimate_rule_precision(gold_records: List[Dict[str, Any]]) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    reliabilities: Dict[str, float] = {}

    for name, spec in RULE_SPECS.items():
        fn: RuleFn = spec["fn"]
        tp = fp = fn_matches = gold_pos = 0
        examples_fp = []
        examples_tp = []
        for rec in gold_records:
            match = bool(fn(rec["text"]))
            pos = gold_target_positive(rec, spec["target_type"], spec["target"])
            if pos:
                gold_pos += 1
            if match:
                fn_matches += 1
                if pos:
                    tp += 1
                    if len(examples_tp) < 3:
                        examples_tp.append(rec.get("post_id", ""))
                else:
                    fp += 1
                    if len(examples_fp) < 3:
                        examples_fp.append(rec.get("post_id", ""))

        if fn_matches > 0:
            precision = tp / fn_matches
            used_default = False
        else:
            precision = float(spec.get("default", 0.50))
            used_default = True

        reliabilities[name] = cap01(precision)
        rows.append({
            "section": "lexical_rule_precision",
            "metric": name,
            "value": cap01(precision),
            "rule_target": spec["target"],
            "target_type": spec["target_type"],
            "tp": tp,
            "fp": fp,
            "rule_matches": fn_matches,
            "gold_positive": gold_pos,
            "used_default_because_no_matches": used_default,
            "example_tp_post_ids": "|".join(examples_tp),
            "example_fp_post_ids": "|".join(examples_fp),
        })

    return reliabilities, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def get_text_for_row(row: pd.Series) -> str:
    for c in ["clean_text", "core_caption", "caption_text", "model_text", "text"]:
        if c in row.index and clean_text_value(row[c]).strip():
            return norm_space(row[c])
    return ""


def compute_row_scores(row: pd.Series, r_ner: Dict[str, float], r_re: Dict[str, float], r_rule: Dict[str, float]) -> Dict[str, Any]:
    text = get_text_for_row(row)
    out: Dict[str, Any] = {}

    # Raw lexical matches for audit/intermediate calculations.
    lex_matches = {name: spec["fn"](text) for name, spec in RULE_SPECS.items()}

    # Base E_NER/lexical evidence features.
    # For each NER label: E_LABEL = max(R_NER * NER_pred, R_LEX * lexical_fallback).
    for label in NER_LABELS:
        ner_raw = raw_ner_binary(row, label)
        ner_part = cap01(r_ner.get(label, 0.0)) * ner_raw
        rule_name = f"LEX_{label}"
        if rule_name in r_rule:
            lex_part = cap01(r_rule[rule_name]) * int(lex_matches.get(rule_name, 0))
        elif label == "PRESSURE_CUE":
            lex_part = cap01(r_rule.get("LEX_PRESSURE_CUE", 0.0)) * int(lex_matches.get("LEX_PRESSURE_CUE", 0))
        else:
            lex_part = 0.0
        out[f"E_{label}"] = cap01(max(ner_part, lex_part))

    # These two labels are open-ended; avoid hallucinating them with broad lexical rules.
    for label in ["SPECIFIC_EDU_ORG", "SERVICE_PROVIDER"]:
        out[f"E_{label}"] = cap01(r_ner.get(label, 0.0)) * raw_ner_binary(row, label)

    # Relation evidence from RE probabilities plus rule fallbacks.
    q_outcome_guar = q_re(row, "OUTCOME_GUARANTEED")
    q_req_waived = q_re(row, "REQUIREMENT_WAIVED")
    q_test = q_re(row, "TESTIMONIAL_SUCCESS_CLAIM")
    q_org = q_re(row, "PROGRAM_OR_OUTCOME_HAS_VERIFIABLE_ORG")

    out["OUTCOME_GUARANTEED_EVIDENCE"] = cap01(max(
        cap01(r_re.get("OUTCOME_GUARANTEED", 0.0)) * q_outcome_guar,
        cap01(r_rule.get("RULE_OUTCOME_GUARANTEED", 0.0)) * int(lex_matches.get("RULE_OUTCOME_GUARANTEED", 0)),
    ))
    out["REQUIREMENT_WAIVED_EVIDENCE"] = cap01(max(
        cap01(r_re.get("REQUIREMENT_WAIVED", 0.0)) * q_req_waived,
        cap01(r_rule.get("RULE_REQUIREMENT_WAIVED", 0.0)) * int(lex_matches.get("RULE_REQUIREMENT_WAIVED", 0)),
    ))
    out["TESTIMONIAL_SUCCESS_EVIDENCE"] = cap01(max(
        cap01(r_re.get("TESTIMONIAL_SUCCESS_CLAIM", 0.0)) * q_test,
        cap01(r_rule.get("RULE_TESTIMONIAL_SUCCESS", 0.0)) * int(lex_matches.get("RULE_TESTIMONIAL_SUCCESS", 0)),
    ))
    out["HAS_VERIFIABLE_ORG_LINK"] = cap01(max(
        cap01(r_re.get("PROGRAM_OR_OUTCOME_HAS_VERIFIABLE_ORG", 0.0)) * q_org,
        cap01(r_rule.get("RULE_ORG_LINK", 0.0)) * int(lex_matches.get("RULE_ORG_LINK", 0)),
    ))

    # -------------------------
    # Intermediate components matching the user's methodology table.
    # -------------------------
    out["CLAIM_SCOPE"] = fuzzy_or(
        out["E_PROGRAM_OR_INTAKE"],
        out["E_OUTCOME"],
        out["E_COST_CLAIM"],
        out["E_GENERIC_INSTITUTION"],
        out["E_REQUIREMENT"],
    )
    out["GENERIC_ONLY_INSTITUTION"] = fuzzy_and(
        out["E_GENERIC_INSTITUTION"],
        fuzzy_not(out["HAS_VERIFIABLE_ORG_LINK"]),
    )
    out["BROAD_FREE_CLAIM"] = fuzzy_or(
        out["E_COST_CLAIM"],
        cap01(r_rule.get("LEX_STUDY_FREE", 0.0)) * int(lex_matches.get("LEX_STUDY_FREE", 0)),
        cap01(r_rule.get("LEX_NO_TUITION", 0.0)) * int(lex_matches.get("LEX_NO_TUITION", 0)),
    )
    out["MATERIAL_CAVEAT"] = fuzzy_or(
        out["E_COST_DETAIL"],
        cap01(r_rule.get("RULE_SEMESTER_FEE", 0.0)) * int(lex_matches.get("RULE_SEMESTER_FEE", 0)),
        cap01(r_rule.get("RULE_LIVING_COST", 0.0)) * int(lex_matches.get("RULE_LIVING_COST", 0)),
        cap01(r_rule.get("RULE_BLOCKED_ACCOUNT", 0.0)) * int(lex_matches.get("RULE_BLOCKED_ACCOUNT", 0)),
        cap01(r_rule.get("RULE_PROOF_OF_FUNDS", 0.0)) * int(lex_matches.get("RULE_PROOF_OF_FUNDS", 0)),
        cap01(r_rule.get("RULE_PUBLIC_PRIVATE_QUALIFIER", 0.0)) * int(lex_matches.get("RULE_PUBLIC_PRIVATE_QUALIFIER", 0)),
    )

    # Pressure and implicative atoms.
    out["PRESSURE_INTENSITY"] = fuzzy_or(
        cap01(r_rule.get("RULE_PRESSURE_STRONG", 0.0)) * 1.00 * int(lex_matches.get("RULE_PRESSURE_STRONG", 0)),
        cap01(r_rule.get("RULE_PRESSURE_MEDIUM", 0.0)) * 0.75 * int(lex_matches.get("RULE_PRESSURE_MEDIUM", 0)),
        cap01(r_rule.get("RULE_PRESSURE_WEAK", 0.0)) * 0.45 * int(lex_matches.get("RULE_PRESSURE_WEAK", 0)),
        cap01(r_ner.get("PRESSURE_CUE", 0.0)) * raw_ner_binary(row, "PRESSURE_CUE"),
    )
    out["IMPLICATIVE_PROMISE"] = fuzzy_or(
        out["E_VAGUE_BENEFIT"],
        cap01(r_rule.get("RULE_FUTURE_PROMISE", 0.0)) * int(lex_matches.get("RULE_FUTURE_PROMISE", 0)),
        cap01(r_rule.get("RULE_DREAM_JOB", 0.0)) * int(lex_matches.get("RULE_DREAM_JOB", 0)),
        cap01(r_rule.get("RULE_GLOBAL_CAREER", 0.0)) * int(lex_matches.get("RULE_GLOBAL_CAREER", 0)),
        cap01(r_rule.get("RULE_LIFE_CHANGING", 0.0)) * int(lex_matches.get("RULE_LIFE_CHANGING", 0)),
        cap01(r_rule.get("RULE_BRIGHT_FUTURE", 0.0)) * int(lex_matches.get("RULE_BRIGHT_FUTURE", 0)),
    )

    # -------------------------
    # Deception type evidence formulas matching the user's methodology table.
    # -------------------------
    out["E_GUARANTEED_OUTCOME"] = fuzzy_or(
        out["OUTCOME_GUARANTEED_EVIDENCE"],
        cap01(r_rule.get("RULE_DIRECT_OUTCOME_GUARANTEE", 0.0)) * int(lex_matches.get("RULE_DIRECT_OUTCOME_GUARANTEE", 0)),
    )
    out["E_ELIGIBILITY_MISREPRESENTATION"] = out["REQUIREMENT_WAIVED_EVIDENCE"]
    out["E_PRESSURE_TACTICS"] = out["PRESSURE_INTENSITY"]
    out["E_MISLEADING_TESTIMONIAL"] = fuzzy_and(
        out["TESTIMONIAL_SUCCESS_EVIDENCE"],
        fuzzy_not(out["HAS_VERIFIABLE_ORG_LINK"]),
    )
    out["E_LACK_OF_TRANSPARENCY"] = fuzzy_and(
        out["CLAIM_SCOPE"],
        fuzzy_not(out["HAS_VERIFIABLE_ORG_LINK"]),
    )
    out["E_OMISSION"] = fuzzy_and(
        out["BROAD_FREE_CLAIM"],
        fuzzy_not(out["MATERIAL_CAVEAT"]),
    )
    out["E_IMPLICATIVE_LANGUAGE"] = out["IMPLICATIVE_PROMISE"]

    # ------------------------------------------------------------------
    # Co-occurrence-based accumulation framework.
    # ------------------------------------------------------------------
    # No per-type expert weights are used for soft deception. S_* columns are
    # retained as unweighted evidence scores for backward-compatible auditing.
    score_map = {
        "OMISSION": out["E_OMISSION"],
        "IMPLICATIVE_LANGUAGE": out["E_IMPLICATIVE_LANGUAGE"],
        "LACK_OF_TRANSPARENCY": out["E_LACK_OF_TRANSPARENCY"],
        "GUARANTEED_OUTCOME": out["E_GUARANTEED_OUTCOME"],
        "PRESSURE_TACTICS": out["E_PRESSURE_TACTICS"],
        "MISLEADING_TESTIMONIAL": out["E_MISLEADING_TESTIMONIAL"],
        "ELIGIBILITY_MISREPRESENTATION": out["E_ELIGIBILITY_MISREPRESENTATION"],
    }
    for k, v in score_map.items():
        out[f"S_{k}"] = cap01(v)

    # Critical risk: these two types can independently create suspiciousness.
    critical_risk_score = fuzzy_or(
        out["E_GUARANTEED_OUTCOME"],
        out["E_ELIGIBILITY_MISREPRESENTATION"],
    )

    # Activated soft evidence: below threshold = inactive; above threshold retains confidence.
    a_omission = active_evidence(out["E_OMISSION"])
    a_testimonial = active_evidence(out["E_MISLEADING_TESTIMONIAL"])
    a_transparency = active_evidence(out["E_LACK_OF_TRANSPARENCY"])
    a_implicative = active_evidence(out["E_IMPLICATIVE_LANGUAGE"])
    a_pressure = active_evidence(out["E_PRESSURE_TACTICS"])

    out["A_OMISSION"] = a_omission
    out["A_MISLEADING_TESTIMONIAL"] = a_testimonial
    out["A_LACK_OF_TRANSPARENCY"] = a_transparency
    out["A_IMPLICATIVE_LANGUAGE"] = a_implicative
    out["A_PRESSURE_TACTICS"] = a_pressure
    out["soft_active_threshold"] = SOFT_ACTIVE_THRESHOLD

    # Deception-by-accumulation patterns.
    # Pattern 1: broad free/no-tuition omission becomes suspicious only when it
    # co-occurs with another opacity/persuasion signal.
    omission_accumulation_score = fuzzy_and(
        a_omission,
        fuzzy_or(a_transparency, a_implicative, a_pressure),
    )

    # Pattern 2: unverifiable testimonial becomes suspicious when it is combined
    # with lack of transparency, implicative language, or pressure.
    testimonial_accumulation_score = fuzzy_and(
        a_testimonial,
        fuzzy_or(a_transparency, a_implicative, a_pressure),
    )

    # Pattern 3: transparency gap + attractive implication + pressure creates a
    # persuasion-based suspicious pattern even if each signal alone is non-critical.
    transparency_persuasion_accumulation_score = fuzzy_and(
        a_transparency,
        a_implicative,
        a_pressure,
    )

    soft_accumulation_risk = fuzzy_or(
        omission_accumulation_score,
        testimonial_accumulation_score,
        transparency_persuasion_accumulation_score,
    )

    final_score = cap01(max(critical_risk_score, soft_accumulation_risk))

    triggered = [(name, score) for name, score in score_map.items() if score >= TRIGGER_THRESHOLD]
    triggered_sorted = sorted(triggered, key=lambda kv: kv[1], reverse=True)
    if triggered_sorted:
        top_name, top_score = triggered_sorted[0]
    else:
        top_name, top_score = "NONE", 0.0

    triggered_type_names = [name for name, _ in triggered_sorted]
    triggered_type_count = len(triggered_type_names)
    high_risk_type_count = sum(1 for name in triggered_type_names if name in HIGH_RISK_TYPES)
    medium_risk_type_count = sum(1 for name in triggered_type_names if name in MEDIUM_RISK_TYPES)
    supporting_type_count = sum(1 for name in triggered_type_names if name in SUPPORTING_TYPES)

    channel_scores = {
        "critical_risk": critical_risk_score,
        "soft_accumulation": soft_accumulation_risk,
    }
    top_channel = max(channel_scores.items(), key=lambda kv: kv[1])[0]

    if final_score < NORMAL_THRESHOLD:
        fw_label = "normal"
    elif final_score >= SUSPICIOUS_THRESHOLD:
        fw_label = "suspicious"
    else:
        fw_label = "uncertain"

    out["weak_labeling_score"] = final_score
    out["framework_weak_label"] = fw_label
    out["top_deception_type"] = top_name
    out["top_deception_score"] = cap01(top_score)

    # Backward-compatible debug/statistics columns.
    out["supporting_score_component"] = 0.0
    out["boost_component"] = 0.0
    out["triggered_deception_types"] = "|".join(triggered_type_names)
    out["triggered_deception_count"] = triggered_type_count
    out["triggered_type_count"] = triggered_type_count
    out["high_risk_type_count"] = high_risk_type_count
    out["medium_risk_type_count"] = medium_risk_type_count
    out["supporting_type_count"] = supporting_type_count
    out["cumulative_rule_matched"] = int(soft_accumulation_risk >= NORMAL_THRESHOLD)
    out["cumulative_suspicious_override"] = int(top_channel == "soft_accumulation" and final_score >= SUSPICIOUS_THRESHOLD)
    out["cumulative_base_threshold"] = "cooccurrence_accumulation"
    out["boost_value_used"] = 0.0
    out["support_factor_used"] = "not_used_cooccurrence_accumulation"

    # Main methodology columns.
    out["critical_risk_score"] = critical_risk_score
    out["omission_accumulation_score"] = omission_accumulation_score
    out["testimonial_accumulation_score"] = testimonial_accumulation_score
    out["transparency_persuasion_accumulation_score"] = transparency_persuasion_accumulation_score
    out["soft_accumulation_risk"] = soft_accumulation_risk
    out["top_risk_channel"] = top_channel
    out["aggregation_method"] = "critical_risk_plus_cooccurrence_deception_accumulation"

    # Legacy column names are kept only to avoid breaking old analysis notebooks.
    # They are not part of the current methodology and are not used in FinalScore.
    out["soft_evidence_mass"] = 0.0
    out["soft_risk_score"] = 0.0
    out["soft_saturation_threshold"] = "not_used_cooccurrence_accumulation"
    out["omission_bundle_score"] = omission_accumulation_score
    out["testimonial_bundle_score"] = testimonial_accumulation_score
    out["transparency_implicative_pressure_bundle_score"] = transparency_persuasion_accumulation_score
    out["soft_deception_bundle_score"] = soft_accumulation_risk

    return out

def build_scored_dataframe(df: pd.DataFrame, r_ner: Dict[str, float], r_re: Dict[str, float], r_rule: Dict[str, float]) -> pd.DataFrame:
    scored_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        scored_rows.append(compute_row_scores(row, r_ner, r_re, r_rule))
    score_df = pd.DataFrame(scored_rows)

    out_df = df.copy()
    out_df.insert(len(out_df.columns), "original_seed_label", out_df["seed_label"].astype(str) if "seed_label" in out_df.columns else "")
    out_df = pd.concat([out_df, score_df], axis=1)

    def final_label(row: pd.Series) -> str:
        original = str(row.get("original_seed_label", "")).strip().lower()
        if original == "legitimate":
            return "normal"
        if original in {"normal", "weak_normal"}:
            return "normal"
        if original in {"suspicious", "weak_suspicious"}:
            return "suspicious"
        # For none/unlabeled: use framework output.
        return str(row.get("framework_weak_label", "normal"))

    out_df["seed_label"] = out_df.apply(final_label, axis=1)
    return out_df


# ---------------------------------------------------------------------------
# Classifier-training output schema helpers
# ---------------------------------------------------------------------------
# These columns are produced by weak labeling and should not be present in the
# classifier-training CSV, because the classifier should learn from text embeddings plus
# low-level NER/RE features, not from the weak-labeling decision logic itself.
WEAK_LABELING_EXACT_COLUMNS = {
    "original_seed_label",
    "weak_labeling_score",
    "framework_weak_label",
    "top_deception_type",
    "top_deception_score",
    "supporting_score_component",
    "boost_component",
    "triggered_deception_types",
    "triggered_deception_count",
    "triggered_type_count",
    "high_risk_type_count",
    "medium_risk_type_count",
    "supporting_type_count",
    "cumulative_rule_matched",
    "cumulative_suspicious_override",
    "cumulative_base_threshold",
    "boost_value_used",
    "support_factor_used",
    "critical_risk_score",
    "omission_accumulation_score",
    "testimonial_accumulation_score",
    "transparency_persuasion_accumulation_score",
    "soft_accumulation_risk",
    "top_risk_channel",
    "aggregation_method",
    "soft_evidence_mass",
    "soft_risk_score",
    "soft_saturation_threshold",
    "omission_bundle_score",
    "testimonial_bundle_score",
    "transparency_implicative_pressure_bundle_score",
    "soft_deception_bundle_score",
    "CLAIM_SCOPE",
    "GENERIC_ONLY_INSTITUTION",
    "BROAD_FREE_CLAIM",
    "MATERIAL_CAVEAT",
    "PRESSURE_INTENSITY",
    "IMPLICATIVE_PROMISE",
    "OUTCOME_GUARANTEED_EVIDENCE",
    "REQUIREMENT_WAIVED_EVIDENCE",
    "TESTIMONIAL_SUCCESS_EVIDENCE",
    "HAS_VERIFIABLE_ORG_LINK",
    "soft_active_threshold",
}

WEAK_LABELING_PREFIXES = ("E_", "S_", "A_")

# These columns are not decision features. They can remain in the classifier-training CSV
# because the binary-gold NER/RE extraction output also contains them; the classifier
# code should still explicitly select usable feature columns rather than blindly
# training on every numeric column.
LOW_LEVEL_NER_RE_PREFIXES = ("has_", "ner_", "p_", "re_")
LOW_LEVEL_NER_RE_EXACT_COLUMNS = {
    "ner_entities",
    "has_NER",
    "re_labels",
    "has_RE",
    "ner_re_prediction_source",
    "ner_re_oof_fold",
    "ner_re_num_models_aggregated",
    "ner_re_ner_vote_threshold",
}


def is_weak_labeling_methodology_column(col: str) -> bool:
    if col == "seed_label":
        return False
    if col in WEAK_LABELING_EXACT_COLUMNS:
        return True
    if any(col.startswith(prefix) for prefix in WEAK_LABELING_PREFIXES):
        return True
    return False


def modeling_training_columns(scored_df: pd.DataFrame) -> List[str]:
    """Return a Phase-9-ready column list.

    The output intentionally keeps original silver columns and NER/RE extraction
    columns, but removes all weak-labeling decision/evidence/debug columns. The
    only weak-labeling result retained is seed_label, overwritten as the final
    training target normal/suspicious.
    """
    cols: List[str] = []
    for c in scored_df.columns:
        if is_weak_labeling_methodology_column(c):
            continue
        if c not in cols:
            cols.append(c)
    return cols


def build_modeling_training_dataframe(scored_training_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    keep_cols = modeling_training_columns(scored_training_df)
    removed_cols = [c for c in scored_training_df.columns if c not in keep_cols]
    out = scored_training_df.loc[:, keep_cols].copy()

    # Keep the final target explicit and clean. classifier training should normally use
    # seed_label as y_train, matching binary gold where seed_label is manual.
    if "seed_label" not in out.columns:
        raise ValueError("classifier-training output must contain seed_label as the final target column.")
    out["seed_label"] = out["seed_label"].astype(str).str.lower().str.strip()
    bad_labels = sorted(set(out["seed_label"]) - {"normal", "suspicious"})
    if bad_labels:
        raise ValueError(f"classifier-training output contains non-binary seed_label values: {bad_labels}")
    return out, keep_cols, removed_cols


def missing_modeling_feature_columns(df: pd.DataFrame) -> List[str]:
    """Audit low-level NER/RE columns expected from the extraction output."""
    expected = []
    for lab in NER_LABELS:
        expected.extend([f"has_{lab}", f"ner_{lab}", f"ner_vote_rate_{lab}", f"ner_count_{lab}_mean"])
    for lab in RE_LABELS:
        expected.extend([f"p_{lab}", f"re_{lab}"])
    expected.extend(["ner_entities", "has_NER", "re_labels", "has_RE"])
    return [c for c in expected if c not in df.columns]


def row_methodology_payload(row: pd.Series) -> str:
    """Compact row-level audit payload for weak_labeling_statistics.csv."""
    keys = [
        "original_seed_label",
        "seed_label",
        "framework_weak_label",
        "weak_labeling_score",
        "top_deception_type",
        "top_deception_score",
        "triggered_deception_types",
        "critical_risk_score",
        "soft_accumulation_risk",
        "omission_accumulation_score",
        "testimonial_accumulation_score",
        "transparency_persuasion_accumulation_score",
        "E_GUARANTEED_OUTCOME",
        "E_ELIGIBILITY_MISREPRESENTATION",
        "E_OMISSION",
        "E_MISLEADING_TESTIMONIAL",
        "E_LACK_OF_TRANSPARENCY",
        "E_IMPLICATIVE_LANGUAGE",
        "E_PRESSURE_TACTICS",
        "OUTCOME_GUARANTEED_EVIDENCE",
        "REQUIREMENT_WAIVED_EVIDENCE",
        "TESTIMONIAL_SUCCESS_EVIDENCE",
        "HAS_VERIFIABLE_ORG_LINK",
    ]
    payload: Dict[str, Any] = {}
    for k in keys:
        if k in row.index:
            v = row.get(k)
            if isinstance(v, (np.integer, np.floating)):
                v = v.item()
            payload[k] = v
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def make_statistics(
    scored_df: pd.DataFrame,
    training_df: pd.DataFrame,
    rule_precision_df: pd.DataFrame,
    modeling_kept_columns: Optional[List[str]] = None,
    modeling_removed_columns: Optional[List[str]] = None,
    modeling_missing_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add_summary(metric: str, value: Any):
        rows.append({
            "section": "summary",
            "metric": metric,
            "value": value,
            "post_id": "",
            "account_name": "",
            "post_url": "",
            "framework_weak_label": "",
            "weak_labeling_score": "",
            "top_deception_type": "",
            "clean_text_preview": "",
            "rule_target": "",
            "target_type": "",
            "tp": "",
            "fp": "",
            "rule_matches": "",
            "gold_positive": "",
            "used_default_because_no_matches": "",
        })

    original = scored_df["original_seed_label"].astype(str).str.lower() if "original_seed_label" in scored_df.columns else pd.Series([""] * len(scored_df))
    framework = scored_df["framework_weak_label"].astype(str).str.lower()
    final_seed = scored_df["seed_label"].astype(str).str.lower()

    legit_mask = original.eq("legitimate")
    none_mask = original.eq("none") | original.eq("unlabeled") | original.eq("")

    add_summary("input_total_posts", len(scored_df))
    add_summary("legitimate_total", int(legit_mask.sum()))
    add_summary("legitimate_framework_normal", int((legit_mask & framework.eq("normal")).sum()))
    add_summary("legitimate_framework_uncertain", int((legit_mask & framework.eq("uncertain")).sum()))
    add_summary("legitimate_framework_suspicious", int((legit_mask & framework.eq("suspicious")).sum()))
    add_summary("none_total", int(none_mask.sum()))
    add_summary("none_framework_normal", int((none_mask & framework.eq("normal")).sum()))
    add_summary("none_framework_uncertain", int((none_mask & framework.eq("uncertain")).sum()))
    add_summary("none_framework_suspicious", int((none_mask & framework.eq("suspicious")).sum()))
    add_summary("silver_training_dataset_total", len(training_df))
    add_summary("silver_training_normal_total", int(training_df["seed_label"].astype(str).str.lower().eq("normal").sum()))
    add_summary("silver_training_suspicious_total", int(training_df["seed_label"].astype(str).str.lower().eq("suspicious").sum()))
    add_summary("removed_uncertain_total", int(final_seed.eq("uncertain").sum()))

    # Phase-9 schema audit. Methodology/debug columns removed from the training
    # CSV are recorded here rather than kept as classifier input features.
    modeling_kept_columns = modeling_kept_columns or []
    modeling_removed_columns = modeling_removed_columns or []
    modeling_missing_columns = modeling_missing_columns or []
    rows.append({
        "section": "modeling_schema",
        "metric": "training_schema_mode",
        "value": "modeling_minimal_binary_label_plus_low_level_ner_re",
        "post_id": "",
        "account_name": "",
        "post_url": "",
        "framework_weak_label": "",
        "weak_labeling_score": "",
        "top_deception_type": "",
        "clean_text_preview": "",
        "schema_column": "",
        "schema_action": "",
        "methodology_json": "",
    })
    rows.append({
        "section": "modeling_schema",
        "metric": "training_columns_kept_count",
        "value": len(modeling_kept_columns),
        "schema_column": "",
        "schema_action": "kept_count",
    })
    rows.append({
        "section": "modeling_schema",
        "metric": "methodology_columns_removed_from_training_count",
        "value": len(modeling_removed_columns),
        "schema_column": "",
        "schema_action": "removed_count",
    })
    for c in modeling_kept_columns:
        rows.append({
            "section": "modeling_schema",
            "metric": "kept_in_silver_training_dataset",
            "value": "",
            "schema_column": c,
            "schema_action": "keep_modeling_input_schema",
        })
    for c in modeling_removed_columns:
        rows.append({
            "section": "modeling_schema",
            "metric": "removed_from_silver_training_dataset",
            "value": "",
            "schema_column": c,
            "schema_action": "moved_to_statistics_and_debug_not_modeling_input",
        })
    for c in modeling_missing_columns:
        rows.append({
            "section": "modeling_schema",
            "metric": "missing_expected_ner_re_column",
            "value": "",
            "schema_column": c,
            "schema_action": "check_ner_re_extraction_output",
        })

    # Row-level weak-labeling decision audit. This is where the high-level
    # weak-labeling methodology columns live after being removed from
    # silver_training_dataset.csv.
    audit_cols = [c for c in [
        "post_id", "account_name", "post_url", "clean_text", "caption_text",
        "original_seed_label", "seed_label", "framework_weak_label",
        "weak_labeling_score", "top_deception_type", "triggered_deception_types",
        "critical_risk_score", "soft_accumulation_risk",
    ] if c in scored_df.columns]
    for _, r in scored_df.iterrows():
        text = clean_text_value(r.get("clean_text", r.get("caption_text", "")))
        rows.append({
            "section": "weak_labeling_decision_audit",
            "metric": "row_decision",
            "value": r.get("seed_label", ""),
            "post_id": r.get("post_id", ""),
            "account_name": r.get("account_name", ""),
            "post_url": r.get("post_url", ""),
            "original_seed_label": r.get("original_seed_label", ""),
            "final_seed_label": r.get("seed_label", ""),
            "framework_weak_label": r.get("framework_weak_label", ""),
            "weak_labeling_score": r.get("weak_labeling_score", ""),
            "top_deception_type": r.get("top_deception_type", ""),
            "triggered_deception_types": r.get("triggered_deception_types", ""),
            "critical_risk_score": r.get("critical_risk_score", ""),
            "soft_accumulation_risk": r.get("soft_accumulation_risk", ""),
            "clean_text_preview": norm_space(text)[:350],
            "methodology_json": row_methodology_payload(r),
        })

    # Add lexical/rule precision section.
    for _, r in rule_precision_df.iterrows():
        rows.append({
            "section": "lexical_rule_precision",
            "metric": r.get("metric", ""),
            "value": r.get("value", ""),
            "post_id": "",
            "account_name": "",
            "post_url": "",
            "framework_weak_label": "",
            "weak_labeling_score": "",
            "top_deception_type": "",
            "clean_text_preview": "",
            "rule_target": r.get("rule_target", ""),
            "target_type": r.get("target_type", ""),
            "tp": r.get("tp", ""),
            "fp": r.get("fp", ""),
            "rule_matches": r.get("rule_matches", ""),
            "gold_positive": r.get("gold_positive", ""),
            "used_default_because_no_matches": r.get("used_default_because_no_matches", ""),
        })

    # Add legitimate false-positive review rows.
    fp_mask = legit_mask & framework.isin(["uncertain", "suspicious"])
    fp_cols = [c for c in ["post_id", "account_name", "post_url", "clean_text", "caption_text"] if c in scored_df.columns]
    for _, r in scored_df.loc[fp_mask, fp_cols + ["framework_weak_label", "weak_labeling_score", "top_deception_type"]].iterrows():
        text = clean_text_value(r.get("clean_text", r.get("caption_text", "")))
        rows.append({
            "section": "legitimate_false_positive_review",
            "metric": "legitimate_flagged_by_framework",
            "value": "",
            "post_id": r.get("post_id", ""),
            "account_name": r.get("account_name", ""),
            "post_url": r.get("post_url", ""),
            "framework_weak_label": r.get("framework_weak_label", ""),
            "weak_labeling_score": r.get("weak_labeling_score", ""),
            "top_deception_type": r.get("top_deception_type", ""),
            "clean_text_preview": norm_space(text)[:350],
            "rule_target": "",
            "target_type": "",
            "tp": "",
            "fp": "",
            "rule_matches": "",
            "gold_positive": "",
            "used_default_because_no_matches": "",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weak-label silver subset using Critical Risk + Co-occurrence-based Deception Accumulation.")
    parser.add_argument("--silver-csv", default=DEFAULT_SILVER_CSV, help="Path to silver_subset_with_ner_re.csv")
    parser.add_argument("--gold-json", default=DEFAULT_GOLD_JSON, help="Path to Goldsubset_NER_RE_annotated_data.json")
    parser.add_argument("--ner-metrics", default=DEFAULT_NER_METRICS, help="Path to ner_per_label_average.csv")
    parser.add_argument("--re-metrics", default=DEFAULT_RE_METRICS, help="Path to re_per_label_average.csv")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "processed" / "weak_labeling"), help="Output directory for weak-labeling artifacts.")
    parser.add_argument("--training-output-name", default="silver_train.csv")
    parser.add_argument("--statistics-output-name", default="weak_labeling_report.csv")
    parser.add_argument("--uncertain-output-name", default="uncertain_posts.csv", help="Output CSV containing all rows with framework_weak_label == uncertain, including legitimate and none/unlabeled posts.")
    parser.add_argument("--debug-output-name", default="weak_labeling_scored_posts_debug.csv")
    parser.add_argument("--boost-value", type=float, default=BOOST_VALUE, help="Boost added when enough deception types are triggered.")
    parser.add_argument("--support-factor", type=float, default=SUPPORT_FACTOR, help="Alpha multiplier for average supporting triggered scores.")
    parser.add_argument("--disable-cumulative", action="store_true", help="Disable cumulative minor/medium-risk override.")
    parser.add_argument("--cumulative-base-threshold", type=float, default=CUMULATIVE_BASE_THRESHOLD)
    parser.add_argument("--cumulative-min-triggered-types", type=int, default=CUMULATIVE_MIN_TRIGGERED_TYPES)
    parser.add_argument("--cumulative-min-medium-risk-types", type=int, default=CUMULATIVE_MIN_MEDIUM_RISK_TYPES)
    parser.add_argument("--cumulative-require-supporting-type", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override methodological hyperparameters from CLI.
    global BOOST_VALUE, SUPPORT_FACTOR, CUMULATIVE_ENABLED, CUMULATIVE_BASE_THRESHOLD
    global CUMULATIVE_MIN_TRIGGERED_TYPES, CUMULATIVE_MIN_MEDIUM_RISK_TYPES
    global CUMULATIVE_REQUIRE_SUPPORTING_TYPE
    BOOST_VALUE = float(args.boost_value)
    SUPPORT_FACTOR = float(args.support_factor)
    CUMULATIVE_ENABLED = not bool(args.disable_cumulative)
    CUMULATIVE_BASE_THRESHOLD = float(args.cumulative_base_threshold)
    CUMULATIVE_MIN_TRIGGERED_TYPES = int(args.cumulative_min_triggered_types)
    CUMULATIVE_MIN_MEDIUM_RISK_TYPES = int(args.cumulative_min_medium_risk_types)
    CUMULATIVE_REQUIRE_SUPPORTING_TYPE = bool(args.cumulative_require_supporting_type)

    silver_path = Path(args.silver_csv)
    gold_path = Path(args.gold_json)
    ner_metrics_path = Path(args.ner_metrics)
    re_metrics_path = Path(args.re_metrics)

    for p in [silver_path, gold_path, ner_metrics_path, re_metrics_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "data" / "processed" / "weak_labeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading files...")
    silver_df = read_csv_robust(str(silver_path))
    gold_records = extract_gold_records(str(gold_path))
    r_ner = load_metric_f1(str(ner_metrics_path), NER_LABELS)
    r_re = load_metric_f1(str(re_metrics_path), RE_LABELS)

    print("[2/6] Estimating lexical/rule precision from annotated gold JSON...")
    r_rule, rule_precision_df = estimate_rule_precision(gold_records)

    print("[3/6] Computing E_FEATURE, co-occurrence accumulation patterns, and FinalScore...")
    scored_df = build_scored_dataframe(silver_df, r_ner, r_re, r_rule)

    print("[4/6] Building modeling-ready training dataset: keep final seed_label normal/suspicious, remove uncertain, drop weak-labeling decision columns...")
    training_full_df = scored_df[scored_df["seed_label"].astype(str).str.lower().isin(["normal", "suspicious"])].copy()
    training_df, modeling_kept_columns, modeling_removed_columns = build_modeling_training_dataframe(training_full_df)
    missing_modeling_cols = missing_modeling_feature_columns(training_df)
    if missing_modeling_cols:
        print("WARNING: missing expected low-level NER/RE columns in classifier-training output:")
        print("  " + ", ".join(missing_modeling_cols[:30]) + (" ..." if len(missing_modeling_cols) > 30 else ""))

    # Save every post that the framework itself marked as uncertain.
    # Important: this uses framework_weak_label, not final seed_label, because
    # legitimate posts are forced to final seed_label == normal for training.
    # Therefore, this file includes both legitimate and none/unlabeled uncertain cases.
    print("[4b/6] Building uncertain dataset: keep all framework_weak_label == uncertain rows...")
    uncertain_df = scored_df[scored_df["framework_weak_label"].astype(str).str.lower().eq("uncertain")].copy()

    print("[5/6] Building statistics file with schema audit and row-level decision audit...")
    stats_df = make_statistics(
        scored_df,
        training_df,
        rule_precision_df,
        modeling_kept_columns=modeling_kept_columns,
        modeling_removed_columns=modeling_removed_columns,
        modeling_missing_columns=missing_modeling_cols,
    )

    training_out = out_dir / args.training_output_name
    uncertain_out = out_dir / args.uncertain_output_name
    stats_out = out_dir / args.statistics_output_name
    debug_out = out_dir / args.debug_output_name

    print("[6/6] Saving outputs...")
    training_df.to_csv(training_out, index=False, encoding="utf-8-sig")
    uncertain_df.to_csv(uncertain_out, index=False, encoding="utf-8-sig")
    stats_df.to_csv(stats_out, index=False, encoding="utf-8-sig")
    scored_df.to_csv(debug_out, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Training dataset:  {training_out}")
    print(f"Uncertain dataset: {uncertain_out}")
    print(f"Statistics file:   {stats_out}")
    print(f"Debug scored file: {debug_out}")
    print("\nUncertain file summary:")
    if "original_seed_label" in uncertain_df.columns:
        print(uncertain_df["original_seed_label"].astype(str).str.lower().value_counts(dropna=False).to_string())
    else:
        print(f"uncertain_total: {len(uncertain_df)}")

    print("\nSummary:")
    summary_rows = stats_df[stats_df["section"].eq("summary")][["metric", "value"]]
    print(summary_rows.to_string(index=False))


if __name__ == "__main__":
    main()
