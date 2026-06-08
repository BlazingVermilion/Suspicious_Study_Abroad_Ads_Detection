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

# =========================================================
# suspicious_post_crawler_no_group_per_account_caps.py
# =========================================================
# Purpose:
#   Crawl Instagram posts from a flat no-group account_registry.json list where
#   posts_crawled == 0, with a strict suspicious-focused filter.
#
# Pipeline:
#   1) Run discover_suspicious_accounts.py first to add accounts.
#   2) Run this script to scan accounts with posts_crawled == 0.
#   3) Relevant posts are appended to data/raw/metadata/suspicious_candidate_posts.json.
#
# Output schema in suspicious_candidate_posts.json stays unchanged:
#   post_id, post_url, platform, account_name, caption_text, hashtags,
#   screenshot_url, posting_time, external_link, language, seed_label
#
# Notes:
#   - Active mode is suspicious-only to collect more suspicious silver candidates.
#   - A broader research-scope pattern block is kept as comments for later.
# =========================================================

import asyncio
import json
import os
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# =========================================================
# PATHS
# =========================================================

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()

DATA_DIR = ROOT / "data" / "raw" / "instagram"
METADATA_DIR = DATA_DIR / "metadata"
SCREENSHOT_DIR = DATA_DIR / "screenshots"
SESSION_FILE = ROOT / "secrets" / "instagram_session.json"
if not SESSION_FILE.exists():
    SESSION_FILE = ROOT / "instagram_session.json"

PRE_LABELED_JSON = METADATA_DIR / "suspicious_candidate_posts.json"
REGISTRY_JSON = METADATA_DIR / "account_registry.json"
CRAWL_STATE_JSON = METADATA_DIR / "crawl_state_suspicious_posts.json"
CRAWL_AUDIT_JSONL = METADATA_DIR / "crawl_audit_suspicious_posts.jsonl"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("suspicious_post_crawler")

# =========================================================
# CONFIG
# =========================================================

VIEWPORT = {"width": 1920, "height": 1080}

# Account-level limits to avoid source imbalance.
MAX_ACCOUNTS_PER_RUN = 120
MAX_POSTS_TO_SCAN_PER_ACCOUNT = 120
MAX_SAVED_POSTS_PER_ACCOUNT = 10

# Feed scrolling limits. More scrolls = more loaded post URLs, but higher blocking risk.
MAX_SCROLLS_PER_ACCOUNT = 24
NO_NEW_POSTS_LIMIT = 4

DELAY_POST = (2.0, 4.5)
DELAY_ACCOUNT = (5.0, 10.0)

# Suspicious-only mode: save a post if it is in target research scope and either
# a high-risk trigger is found or enough suspicious signals accumulate.
MIN_SUSPICIOUS_SCORE = 2.5
MIN_SUSPICIOUS_GROUPS = 2

# If True, accounts with no relevant saved posts are marked so they are not rescanned forever.
MARK_EMPTY_ACCOUNTS_AS_SCANNED = True

# =========================================================
# CONTENT FILTERS
# =========================================================

NORMALIZATION_MAP = {
    "m.sc": "msc", "m.sc.": "msc", "m sc": "msc",
    "b.sc": "bsc", "b.sc.": "bsc", "b sc": "bsc",
    "m.a": "ma", "b.a": "ba", "m.eng": "meng", "b.eng": "beng",
    "ph.d": "phd", "ph.d.": "phd",
}

# -----------------------------
# ACTIVE MODE: suspicious-trigger post filter
# -----------------------------
SUSPICIOUS_TRIGGER_PATTERNS = {
    # Direct high-risk deception
    "guaranteed_outcome": {
        "weight": 3.5,
        "patterns": [
            r"\b100\s*%\s*(visa|admission|success|approval)\b",
            r"\bguaranteed\s+(admission|visa|job|scholarship|seat|success)\b",
            r"\b(admission|visa|job|scholarship|seat)\s+guaranteed\b",
            r"\bconfirmed\s+admission\b",
            r"\bassured\s+(admission|visa|scholarship|job)\b",
            r"\bsure[-\s]?shot\s+visa\b",
            r"\bvisa\s+success\b",
            r"\bvisa\s+approval\s+guarantee\b",
            r"\binstant\s+admission\b",
        ],
    },

    # Eligibility misrepresentation / waived requirements
    "requirement_waived": {
        "weight": 3.5,
        "patterns": [
            r"\bno\s+ielts\b",
            r"\bwithout\s+ielts\b",
            r"\bielts\s+(not\s+required|waived|not\s+needed)\b",
            r"\bno\s+aps\b",
            r"\bwithout\s+aps\b",
            r"\baps\s+(not\s+required|waived|not\s+needed)\b",
            r"\bno\s+blocked\s+account\b",
            r"\bwithout\s+blocked\s+account\b",
            r"\bno\s+account\s+money\b",
            r"\bwithout\s+account\s+money\b",
            r"\b(any|low)\s+(gpa|cgpa)\s+(accepted|acceptable)\b",
            r"\bno\s+german\s+language\s+(required|needed)\b",
        ],
    },

    # Omission / broad free-tuition claim
    "free_or_no_tuition_claim": {
        "weight": 2.0,
        "patterns": [
            r"\bstudy\s+in\s+germany\s+for\s+free\b",
            r"\bstudy\s+for\s+free\s+in\s+germany\b",
            r"\bfree\s+education\s+in\s+germany\b",
            r"\bfree\s+tuition\b",
            r"\bno\s+tuition\s+fee\b",
            r"\btuition[-\s]?free\b",
            r"\blow\s+or\s+no\s+tuition\b",
            r"\blittle\s+to\s+no\s+tuition\b",
        ],
    },

    # Testimonial success story
    "testimonial_success": {
        "weight": 2.0,
        "patterns": [
            r"\bcongratulations?\b.{0,120}\b(admission|visa approval|student visa|offer letter|scholarship)\b",
            r"\bour\s+(student|client|candidate)\b.{0,120}\b(admission|visa approval|student visa|offer letter|scholarship)\b",
            r"\bsuccess\s+stor(y|ies)\b.{0,120}\b(admission|visa|scholarship|offer)\b",
            r"\bsecured\s+(admission|student visa|visa approval|scholarship|offer letter)\b",
            r"\breceived\s+(admission|student visa|visa approval|scholarship|offer letter)\b",
        ],
    },

    # Lack of transparency / generic high-value claims
    "generic_institution_claim": {
        "weight": 1.2,
        "patterns": [
            r"\btop\s+(public\s+)?universit(y|ies)\b",
            r"\bleading\s+(public\s+)?universit(y|ies)\b",
            r"\bprestigious\s+(german\s+)?universit(y|ies)\b",
            r"\bgerman\s+public\s+universit(y|ies)\b",
            r"\bgateway\s+to\s+german\s+public\s+universit(y|ies)\b",
        ],
    },

    # Pressure tactics
    "pressure_tactics": {
        "weight": 1.5,
        "patterns": [
            r"\blimited\s+(seats|slots)\b",
            r"\bfew\s+(seats|slots)\s+left\b",
            r"\blast\s+chance\b",
            r"\bhurry\s+up\b",
            r"\bdeadline\s+(today|soon|approaching)\b",
            r"\bapply\s+now\b",
            r"\bcontact\s+now\b",
            r"\bdm\s+us\b",
            r"\bregister\s+now\b",
        ],
    },

    # Implicative / vague benefit, low weight; never enough alone
    "implicative_language": {
        "weight": 0.7,
        "patterns": [
            r"\bdream\s+(job|career|future)\b",
            r"\bglobal\s+career\b",
            r"\bbright\s+future\b",
            r"\blife[-\s]?changing\s+opportunit(y|ies)\b",
            r"\bmake\s+your\s+dreams?\s+come\s+true\b",
            r"\bturn\s+your\s+dreams?\s+into\s+reality\b",
            r"\bcareer\s+opportunit(y|ies)\b",
        ],
    },

    # Service/agency context; not suspicious alone, but helps target consulting ads
    "service_context": {
        "weight": 0.8,
        "patterns": [
            r"\bconsultant(s)?\b",
            r"\bconsultancy\b",
            r"\bagency\b",
            r"\badmission\s+support\b",
            r"\bvisa\s+assistance\b",
            r"\bapplication\s+support\b",
            r"\bfree\s+consultation\b",
            r"\bexpert\s+guidance\b",
            r"\bcounselling\b",
            r"\bcounseling\b",
        ],
    },
}

# -----------------------------
# BROADER RESEARCH-SCOPE FILTER: keep for later, currently commented
# -----------------------------
# Use this later if you want general educational-recruitment posts, not only
# suspicious-trigger posts.
#
# RESEARCH_SCOPE_PATTERNS = {
#     "program_promotion": [
#         r"\bstudy in germany\b", r"\bapplications? open\b", r"\bmaster program(me)?\b",
#         r"\bbachelor program(me)?\b", r"\bmba\b", r"\bmsc\b", r"\binternational students?\b",
#         r"\badmission\b", r"\benroll\b",
#     ],
#     "scholarship_funding": [
#         r"\bscholarship\b", r"\bfunding\b", r"\btuition\b", r"\bstipend\b",
#     ],
#     "requirements": [
#         r"\bielts\b", r"\bvisa\b", r"\baps\b", r"\brequirements?\b", r"\bblocked account\b",
#     ],
#     "timeline_event": [
#         r"\bdeadline\b", r"\bwinter semester\b", r"\bsummer semester\b", r"\bwebinar\b", r"\bopen day\b",
#     ],
#     "recruitment": [
#         r"\bcareer opportunities\b", r"\bfuture in germany\b", r"\bstudy and work\b",
#     ],
# }

TARGET_SCOPE_PATTERNS = [
    r"\bstudy\s+in\s+germany\b",
    r"\bmasters?\s+in\s+germany\b",
    r"\bms\s+in\s+germany\b",
    r"\bgermany\s+student\s+visa\b",
    r"\bgerman\s+public\s+universit(y|ies)\b",
    r"\bpublic\s+universit(y|ies)\s+in\s+germany\b",
    r"\bstudy\s+abroad\b.{0,80}\bgermany\b",
    r"\bgermany\b.{0,80}\b(admission|visa|ielts|aps|blocked account|tuition|university|masters?|bachelor|mba|msc)\b",
]

BAD_CONTENT_PATTERNS = [
    r"\bgiveaway\b",
    r"\bfashion\b",
    r"\bmakeup\b",
    r"\bfood\b",
    r"\btravel\s+blog\b",
    r"\bfootball\b",
]

# =========================================================
# HELPERS
# =========================================================

def normalize_text(text: str) -> str:
    txt = (text or "").lower()
    for k, v in NORMALIZATION_MAP.items():
        txt = txt.replace(k, v)
    txt = txt.replace("\u00a0", " ")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def rdelay(low, high):
    time.sleep(random.uniform(low, high))


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
            return default
    return default


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_registry(registry):
    """Return a flat account list.

    Preferred registry format is now:
      [
        {"handle": "account1", "posts_crawled": 0, ...},
        {"handle": "account2", "posts_crawled": 0, ...}
      ]

    The old {"A": [...], "B": [...], "C": [...]} format is still
    accepted only as a backward-compatible fallback and is flattened here.
    No account group is required or used by the crawler.
    """
    if isinstance(registry, list):
        return registry
    if isinstance(registry, dict):
        flat = []
        for value in registry.values():
            if isinstance(value, list):
                flat.extend(value)
        return flat
    return []

def extract_post_id(url: str):
    m = re.search(r"/(?:p|reel)/([A-Za-z0-9_-]+)", url or "")
    return m.group(1) if m else None


def extract_hashtags(text: str):
    return re.findall(r"#([A-Za-z0-9_]+)", text or "")


def extract_external_link(text: str):
    urls = re.findall(r"https?://[^\s]+", text or "")
    for u in urls:
        if "instagram.com" not in u:
            return u.rstrip(".,);]")
    return None


def looks_english(text: str) -> bool:
    txt = text or ""
    if len(txt.strip()) < 40:
        return False
    letters = [c for c in txt if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if ord(c) < 128]
    ascii_ratio = len(ascii_letters) / max(1, len(letters))
    common = re.search(r"\b(the|and|for|with|study|germany|admission|visa|university|students?)\b", txt.lower())
    return ascii_ratio >= 0.75 and bool(common)


def is_target_scope(text: str) -> bool:
    txt = normalize_text(text)
    if any(re.search(p, txt) for p in BAD_CONTENT_PATTERNS):
        return False
    return any(re.search(p, txt) for p in TARGET_SCOPE_PATTERNS)


def score_suspicious_text(text: str) -> dict:
    txt = normalize_text(text)
    matched_groups = []
    matched_patterns = []
    score = 0.0

    for group, cfg in SUSPICIOUS_TRIGGER_PATTERNS.items():
        group_hit = False
        for pat in cfg["patterns"]:
            if re.search(pat, txt, flags=re.I):
                group_hit = True
                matched_patterns.append({"group": group, "pattern": pat})
        if group_hit:
            matched_groups.append(group)
            score += float(cfg["weight"])

    high_risk = any(g in matched_groups for g in ["guaranteed_outcome", "requirement_waived"])

    save_candidate = (
        is_target_scope(text)
        and looks_english(text)
        and (
            high_risk
            or (score >= MIN_SUSPICIOUS_SCORE and len(matched_groups) >= MIN_SUSPICIOUS_GROUPS)
        )
    )

    return {
        "save_candidate": save_candidate,
        "score": round(score, 3),
        "matched_groups": matched_groups,
        "matched_patterns": matched_patterns,
        "high_risk": high_risk,
        "target_scope": is_target_scope(text),
        "english_like": looks_english(text),
    }


def valid_handle(handle: str) -> bool:
    h = (handle or "").strip().lower()
    if not h:
        return False
    if h in {"p", "reel", "reels", "explore", "accounts", "tags", "stories", "popular"}:
        return False
    return bool(re.match(r"^[a-z0-9._]{2,30}$", h))

def account_int_cap(account_obj: dict, *keys: str, default: int) -> int:
    """Read per-account cap from registry, falling back to global default."""
    for key in keys:
        try:
            value = account_obj.get(key, None)
            if value is not None and str(value).strip() != "":
                return max(0, int(float(value)))
        except Exception:
            continue
    return int(default)


def account_scan_cap(account_obj: dict) -> int:
    return account_int_cap(
        account_obj,
        "max_posts_to_scan_per_account",
        "max_posts_to_load_cap",
        default=MAX_POSTS_TO_SCAN_PER_ACCOUNT,
    )


def account_saved_cap(account_obj: dict) -> int:
    return account_int_cap(
        account_obj,
        "max_saved_posts_per_account",
        "target_new_saved_posts_cap",
        default=MAX_SAVED_POSTS_PER_ACCOUNT,
    )


def account_scroll_cap(account_obj: dict) -> int:
    return account_int_cap(
        account_obj,
        "max_scrolls_per_account",
        default=MAX_SCROLLS_PER_ACCOUNT,
    )


# =========================================================
# LOAD STATE
# =========================================================

records = load_json(PRE_LABELED_JSON, [])
registry = normalize_registry(load_json(REGISTRY_JSON, []))
crawl_state = load_json(CRAWL_STATE_JSON, {"runs": []})

SEEN_POST_IDS = {r.get("post_id") for r in records if r.get("post_id")}
SEEN_POST_URLS = {r.get("post_url") for r in records if r.get("post_url")}

# =========================================================
# SESSION
# =========================================================

async def load_session(context):
    if not SESSION_FILE.exists():
        log.warning("Session file not found: %s", SESSION_FILE)
        return
    data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cookies" in data:
        await context.add_cookies(data["cookies"])
    elif isinstance(data, list):
        await context.add_cookies(data)


async def verify_login(page) -> bool:
    await page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(5000)
    return "accounts/login" not in page.url

# =========================================================
# SCROLL / EXTRACT
# =========================================================

async def scroll_account_posts(page, max_scrolls: int = None):
    max_scrolls = int(max_scrolls or MAX_SCROLLS_PER_ACCOUNT)
    previous_count = 0
    stagnant_rounds = 0
    for i in range(max_scrolls):
        try:
            anchors = await page.query_selector_all("a[href*='/p/']")
            current_count = len(anchors)
            log.info("[SCROLL] round=%d loaded_posts=%d", i + 1, current_count)

            if current_count == previous_count:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            if stagnant_rounds >= NO_NEW_POSTS_LIMIT:
                break

            previous_count = current_count
            await page.mouse.wheel(0, random.randint(10000, 16000))
            await page.wait_for_timeout(random.randint(1200, 2400))
        except Exception as e:
            log.warning("[SCROLL] Failed: %s", e)
            break


async def extract_caption(page) -> str:
    # Try the most useful sources first. Instagram DOM changes often, so keep fallbacks.
    selectors = [
        "meta[property='og:description']",
        "h1",
        "article h1",
        "article span",
        "main span",
    ]

    # meta og:description
    try:
        meta = await page.query_selector("meta[property='og:description']")
        if meta:
            content = await meta.get_attribute("content")
            if content and len(content.strip()) > 40:
                return content.strip()
    except Exception:
        pass

    best = ""
    for sel in selectors[1:]:
        try:
            els = await page.query_selector_all(sel)
            for el in els:
                txt = (await el.inner_text()).strip()
                if len(txt) > len(best):
                    best = txt
            if len(best) > 80:
                return best
        except Exception:
            pass
    return best.strip()


async def extract_post(page, post_url: str, account_name: str):
    post_id = extract_post_id(post_url)
    if not post_id or post_id in SEEN_POST_IDS or post_url in SEEN_POST_URLS:
        return False

    try:
        await page.goto(post_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(random.randint(2500, 5000))
    except PWTimeout:
        return False

    caption = await extract_caption(page)
    decision = score_suspicious_text(caption)

    append_jsonl(CRAWL_AUDIT_JSONL, {
        "event": "post_scored",
        "post_id": post_id,
        "post_url": post_url,
        "account_name": account_name,
        "score": decision["score"],
        "matched_groups": decision["matched_groups"],  # suspicious trigger groups, not account groups
        "save_candidate": decision["save_candidate"],
        "target_scope": decision["target_scope"],
        "english_like": decision["english_like"],
        "created_at": datetime.utcnow().isoformat(),
    })

    if not decision["save_candidate"]:
        log.info("[DROP] @%-25s %s score=%.2f groups=%s", account_name, post_id, decision["score"], decision["matched_groups"])
        return False

    posting_time = ""
    try:
        t = await page.query_selector("time[datetime]")
        if t:
            posting_time = await t.get_attribute("datetime") or ""
    except Exception:
        pass

    screenshot_path = SCREENSHOT_DIR / f"{post_id}.png"
    try:
        await page.screenshot(path=str(screenshot_path), full_page=False)
        screenshot_url = str(screenshot_path.relative_to(ROOT))
    except Exception:
        screenshot_url = ""

    record = {
        "post_id": post_id,
        "post_url": post_url,
        "platform": "instagram",
        "account_name": account_name,
        "caption_text": caption,
        "hashtags": extract_hashtags(caption),
        "screenshot_url": screenshot_url,
        "posting_time": posting_time,
        "external_link": extract_external_link(caption),
        "language": "en",
        "seed_label": "none",
    }

    records.append(record)
    SEEN_POST_IDS.add(post_id)
    SEEN_POST_URLS.add(post_url)
    save_json(PRE_LABELED_JSON, records)

    append_jsonl(CRAWL_AUDIT_JSONL, {
        "event": "post_saved",
        "post_id": post_id,
        "post_url": post_url,
        "account_name": account_name,
        "score": decision["score"],
        "matched_groups": decision["matched_groups"],  # suspicious trigger groups, not account groups
        "matched_patterns": decision["matched_patterns"],
        "created_at": datetime.utcnow().isoformat(),
    })

    log.info("[SAVED] @%-25s %s score=%.2f groups=%s", account_name, post_id, decision["score"], decision["matched_groups"])
    return True

# =========================================================
# CRAWL ACCOUNT
# =========================================================

async def crawl_account(page, account_obj: dict):
    handle = str(account_obj.get("handle", "")).strip().lower()
    scan_cap = account_scan_cap(account_obj)
    saved_cap = account_saved_cap(account_obj)
    scroll_cap = account_scroll_cap(account_obj)

    if not valid_handle(handle):
        account_obj["status"] = "invalid_handle"
        save_json(REGISTRY_JSON, registry)
        return 0, 0

    if account_obj.get("status", "active") != "active":
        return 0, 0

    if int(account_obj.get("posts_crawled", 0) or 0) != 0:
        return 0, 0

    log.info("[ACCOUNT] @%s", handle)

    try:
        await page.goto(f"https://www.instagram.com/{handle}/", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(random.randint(4500, 7500))
    except PWTimeout:
        log.warning("[TIMEOUT] opening @%s", handle)
        account_obj["last_scan_error"] = "timeout_opening_account"
        save_json(REGISTRY_JSON, registry)
        return 0, 0

    await scroll_account_posts(page, scroll_cap)

    post_urls = []
    try:
        anchors = await page.query_selector_all("a[href*='/p/']")
        for a in anchors:
            href = await a.get_attribute("href") or ""
            if "/p/" not in href:
                continue
            full = "https://www.instagram.com" + href if href.startswith("/") else href
            if full not in post_urls:
                post_urls.append(full)
    except Exception as e:
        log.warning("[ERROR] loading post URLs @%s: %s", handle, e)

    post_urls = post_urls[:scan_cap]
    log.info("[ACCOUNT] @%s loaded=%d scan_cap=%d saved_cap=%d scroll_cap=%d", handle, len(post_urls), scan_cap, saved_cap, scroll_cap)

    saved = 0
    scanned = 0
    for post_url in post_urls:
        if saved >= saved_cap:
            log.info("[CAP] @%s reached saved cap=%d", handle, saved_cap)
            break

        scanned += 1
        ok = await extract_post(page, post_url, handle)
        if ok:
            saved += 1
            account_obj["posts_crawled"] = int(account_obj.get("posts_crawled", 0) or 0) + 1
            save_json(REGISTRY_JSON, registry)

        rdelay(*DELAY_POST)

    account_obj["last_suspicious_scan_at"] = datetime.utcnow().isoformat()
    account_obj["last_suspicious_scan_scanned_posts"] = scanned
    account_obj["last_suspicious_scan_saved_posts"] = saved
    account_obj["last_suspicious_scan_used_scan_cap"] = scan_cap
    account_obj["last_suspicious_scan_used_saved_cap"] = saved_cap
    account_obj["last_suspicious_scan_used_scroll_cap"] = scroll_cap

    if saved == 0 and MARK_EMPTY_ACCOUNTS_AS_SCANNED:
        account_obj["status"] = "scanned_no_relevant_posts"

    save_json(REGISTRY_JSON, registry)
    log.info("[ACCOUNT DONE] @%s scanned=%d saved=%d", handle, scanned, saved)
    rdelay(*DELAY_ACCOUNT)
    return scanned, saved

# =========================================================
# MAIN
# =========================================================

async def main():
    run_started = datetime.utcnow().isoformat()
    accounts_seen = 0
    total_scanned = 0
    total_saved = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,
            slow_mo=150,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(viewport=VIEWPORT, locale="en-US")
        await load_session(context)
        page = await context.new_page()

        ok = await verify_login(page)
        if not ok:
            log.error("Instagram session invalid. Please refresh instagram_session.json")
            await browser.close()
            return

        for account_obj in registry:
            if accounts_seen >= MAX_ACCOUNTS_PER_RUN:
                break
            if account_obj.get("status", "active") != "active":
                continue
            if int(account_obj.get("posts_crawled", 0) or 0) != 0:
                continue

            accounts_seen += 1
            scanned, saved = await crawl_account(page, account_obj)
            total_scanned += scanned
            total_saved += saved

        await browser.close()

    crawl_state.setdefault("runs", []).append({
        "started_at": run_started,
        "finished_at": datetime.utcnow().isoformat(),
        "accounts_processed": accounts_seen,
        "posts_scanned": total_scanned,
        "posts_saved": total_saved,
        "max_accounts_per_run": MAX_ACCOUNTS_PER_RUN,
        "max_posts_to_scan_per_account": MAX_POSTS_TO_SCAN_PER_ACCOUNT,
        "max_saved_posts_per_account": MAX_SAVED_POSTS_PER_ACCOUNT,
        "mode": "suspicious_trigger_only_per_account_caps",
    })
    save_json(CRAWL_STATE_JSON, crawl_state)

    log.info("=" * 60)
    log.info("RUN DONE accounts=%d scanned=%d saved=%d total_records=%d", accounts_seen, total_scanned, total_saved, len(records))

if __name__ == "__main__":
    asyncio.run(main())
