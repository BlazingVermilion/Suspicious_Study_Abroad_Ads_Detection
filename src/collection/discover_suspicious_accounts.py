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
# discover_suspicious_accounts.py
# =========================================================
# Purpose:
#   Discover Instagram accounts that are likely to contain suspicious
#   study-in-Germany educational advertising posts.
#
# Pipeline position:
#   1) Run this script first.
#   2) It appends newly discovered accounts to data/raw/metadata/account_registry.json.
#   3) Then run crawl_suspicious_posts.py, which crawls accounts with posts_crawled == 0.
#
# Notes:
#   - The active query list below is intentionally suspicious-trigger focused.
#   - A broader research-scope query block is kept as comments so you can switch back later.
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
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# =========================================================
# PATHS
# =========================================================

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()

DATA_DIR = ROOT / "data" / "raw" / "instagram"
METADATA_DIR = DATA_DIR / "metadata"
REGISTRY_JSON = METADATA_DIR / "account_registry.json"
DISCOVERY_LOG_JSONL = METADATA_DIR / "suspicious_account_discovery_log.jsonl"

METADATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("discover_suspicious_accounts")

# =========================================================
# CONFIG
# =========================================================

VIEWPORT = {"width": 1920, "height": 1080}

# Stop conditions. The script stops when either condition is reached.
TARGET_NEW_ACCOUNTS_TOTAL = 80
MAX_SEARCH_MINUTES = 60
MAX_QUERIES_PER_RUN = 120

# New accounts discovered by suspicious-trigger queries are stored in group B.
# Keep A/C only if you later want broader/ambiguous grouping again.
DEFAULT_NEW_ACCOUNT_GROUP = "B"

DELAY_DISCOVERY = (10.0, 22.0)
GOOGLE_WAIT_MS = (9000, 16000)

SEARCH_SUFFIXES = [
    "site:instagram.com",
    "site:instagram.com/p",
]

# =========================================================
# QUERY DESIGN
# =========================================================


RESEARCH_SCOPE_QUERY_SEEDS = [
    '"study in Germany" "admission" consultant',
    '"study in Germany" "international students"',
    '"study in Germany" "master" "consultancy"',
    '"study in Germany" "student visa" consultant',
    '"German public university" "masters" consultant',
    '"Germany education consultant" "Instagram"',
    '"study abroad Germany" "agency"',
    '"Germany university admission" consultant',
    '"Germany application support" "students"',
 ]

# =========================================================
# EXCLUSIONS / VALIDATION
# =========================================================

EXCLUDED_ACCOUNTS = {
    # official/legitimate seed accounts already used elsewhere
    "daad_worldwide",
    "studyingermany",
    "makeitingermany",
    "deutschland_de",
    "goetheinstitut",
    "uniassist_ev",
    "expatrio",
    "fintiba",
    "tu.muenchen",
    "rwthaachenuniversity",
    "fu_berlin",
    "unistuttgart",
    "tu_dortmund",
}

BAD_HANDLE_PATTERNS = [
    r"^explore$",
    r"^accounts$",
    r"^reel$",
    r"^reels$",
    r"^p$",
    r"^tv$",
    r"^stories$",
    r"^tags$",
    r"^directory$",
    r"^about$",
    r"^privacy$",
    r"^developer$",
    r"^instagram$",
]

# Instagram usernames are usually 1-30 chars, but we use >=2 to avoid junk.
HANDLE_RE = re.compile(r"^[a-z0-9._]{2,30}$")

# =========================================================
# HELPERS
# =========================================================

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


def rdelay(low, high):
    time.sleep(random.uniform(low, high))


def normalize_registry(registry):
    if isinstance(registry, list):
        # tolerate older flat-list format
        return {"A": [], "B": registry, "C": []}
    if not isinstance(registry, dict):
        return {"A": [], "B": [], "C": []}
    for g in ["A", "B", "C"]:
        registry.setdefault(g, [])
    return registry


def all_existing_handles(registry):
    handles = set()
    for items in registry.values():
        for x in items:
            h = str(x.get("handle", "")).lower().strip()
            if h:
                handles.add(h)
    return handles


def should_exclude_account(handle: str) -> bool:
    h = (handle or "").lower().strip().strip("/.")

    if not h or h in EXCLUDED_ACCOUNTS:
        return True

    if not HANDLE_RE.match(h):
        return True

    if h.endswith(".") or h.startswith(".") or ".." in h:
        return True

    for pat in BAD_HANDLE_PATTERNS:
        if re.match(pat, h):
            return True

    return False


def unwrap_google_url(href: str) -> str:
    """Convert Google redirect URLs into direct URLs when possible."""
    if not href:
        return ""
    if href.startswith("/url?"):
        qs = parse_qs(urlparse(href).query)
        if "q" in qs and qs["q"]:
            return qs["q"][0]
    if "google.com/url" in href:
        qs = parse_qs(urlparse(href).query)
        if "q" in qs and qs["q"]:
            return qs["q"][0]
    return unquote(href)


def extract_instagram_handle(url: str):
    url = unwrap_google_url(url)
    m = re.search(
        r"instagram\.com/(?!p/|reel/|reels/|explore/|stories/|tv/|tags/)([A-Za-z0-9._]{2,30})(?:[/?#]|$)",
        url,
    )
    if not m:
        return None
    handle = m.group(1).lower().strip().strip("/.")
    return None if should_exclude_account(handle) else handle


def build_search_queries(max_queries=MAX_QUERIES_PER_RUN):
    queries = []
    for seed in RESEARCH_SCOPE_QUERY_SEEDS:
        for suffix in SEARCH_SUFFIXES:
            q = f"{seed} {suffix}"
            queries.append(q)

    # Add a few randomized combinations to avoid only exact repeated templates.
    high_risk_terms = [
        '"no IELTS"', '"no APS"', '"no blocked account"', '"100% visa"',
        '"guaranteed admission"', '"confirmed admission"', '"visa success"',
        '"study in Germany for free"', '"no tuition fee"', '"limited seats"',
    ]
    scope_terms = [
        '"study in Germany"', '"Masters in Germany"', '"MS in Germany"',
        '"Germany public university"', '"Germany student visa"',
    ]
    service_terms = [
        "consultant", "consultancy", "agency", '"education consultant"', '"visa assistance"',
    ]
    for _ in range(80):
        q = " ".join([
            random.choice(scope_terms),
            random.choice(high_risk_terms),
            random.choice(service_terms),
            random.choice(SEARCH_SUFFIXES),
        ])
        queries.append(q)

    # Preserve order but remove duplicates.
    seen = set()
    deduped = []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            deduped.append(q)

    return deduped[:max_queries]

# =========================================================
# LOAD REGISTRY
# =========================================================

registry = normalize_registry(load_json(REGISTRY_JSON, {"A": [], "B": [], "C": []}))

# =========================================================
# DISCOVERY
# =========================================================

async def discover_accounts(page):
    start_time = time.time()
    existing = all_existing_handles(registry)
    new_count = 0

    queries = build_search_queries(MAX_QUERIES_PER_RUN)

    log.info("[START] target_new_accounts=%d max_minutes=%d max_queries=%d",
             TARGET_NEW_ACCOUNTS_TOTAL, MAX_SEARCH_MINUTES, len(queries))

    for i, query in enumerate(queries, start=1):
        elapsed_min = (time.time() - start_time) / 60
        if new_count >= TARGET_NEW_ACCOUNTS_TOTAL:
            log.info("[DONE] Reached target new accounts: %d", new_count)
            break
        if elapsed_min >= MAX_SEARCH_MINUTES:
            log.info("[DONE] Reached max search time: %.1f minutes", elapsed_min)
            break

        google_url = "https://www.google.com/search?q=" + quote_plus(query)
        log.info("[SEARCH %d/%d] %s", i, len(queries), query)

        try:
            await page.goto(google_url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(random.randint(*GOOGLE_WAIT_MS))

            anchors = await page.query_selector_all("a")
            urls = []
            for a in anchors:
                href = await a.get_attribute("href")
                if href:
                    urls.append(href)
            random.shuffle(urls)

            for href in urls:
                handle = extract_instagram_handle(href)
                if not handle or handle in existing:
                    continue

                obj = {
                    "handle": handle,
                    "group": DEFAULT_NEW_ACCOUNT_GROUP,
                    "posts_crawled": 0,
                    "status": "active",
                    "discovered_query": query,
                    "discovered_mode": "suspicious_trigger_search",
                    "discovered_at": datetime.utcnow().isoformat(),
                }

                registry.setdefault(DEFAULT_NEW_ACCOUNT_GROUP, []).append(obj)
                existing.add(handle)
                new_count += 1
                save_json(REGISTRY_JSON, registry)

                append_jsonl(DISCOVERY_LOG_JSONL, {
                    "event": "new_account",
                    "handle": handle,
                    "group": DEFAULT_NEW_ACCOUNT_GROUP,
                    "query": query,
                    "source_href": href,
                    "created_at": datetime.utcnow().isoformat(),
                })

                log.info("[NEW] @%s (%d/%d)", handle, new_count, TARGET_NEW_ACCOUNTS_TOTAL)

                if new_count >= TARGET_NEW_ACCOUNTS_TOTAL:
                    break

        except PWTimeout:
            log.warning("[TIMEOUT] Google query: %s", query)
        except Exception as e:
            log.warning("[ERROR] Discovery failed for query=%s error=%s", query, e)

        rdelay(*DELAY_DISCOVERY)

    log.info("[FINISH] new_accounts=%d elapsed_min=%.1f", new_count, (time.time() - start_time) / 60)

# =========================================================
# MAIN
# =========================================================

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,
            slow_mo=600,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(viewport=VIEWPORT, locale="en-US")
        page = await context.new_page()
        await discover_accounts(page)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
