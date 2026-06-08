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
# FILE: src/crawler/crawl_legitimate_posts.py
# =========================================================

import json
import os
import re
import time
import random

from pathlib import Path
from urllib.parse import urljoin
from collections import Counter

from playwright.sync_api import sync_playwright
from langdetect import detect, LangDetectException


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()

RAW_DIR = BASE_DIR / "data" / "raw" / "instagram"
METADATA_DIR = RAW_DIR / "metadata"
SCREENSHOT_DIR = RAW_DIR / "screenshots"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Existing file used only for duplicate checking
EXISTING_FILE = METADATA_DIR / "legitimate_seed_posts.json"

# New output file for rerun/additional clean candidates
JSON_OUTPUT = METADATA_DIR / "legitimate_seed_posts.json"

SESSION_FILE = BASE_DIR / "secrets" / "instagram_session.json"
if not SESSION_FILE.exists():
    SESSION_FILE = BASE_DIR / "instagram_session.json"


# =========================================================
# CRAWL CONFIG
# =========================================================

MAX_LINKS_PER_ACCOUNT = 250
MAX_POSTS_TO_COLLECT_PER_ACCOUNT = 50
MAX_SCROLL_ATTEMPTS = 200
MAX_NO_NEW_LINK_ROUNDS = 8


# =========================================================
# ACCOUNT CONFIG
# =========================================================

ACCOUNT_CONFIG = {
    # =====================================================
    # OFFICIAL
    # =====================================================

    "daad_worldwide": {
        "category": "official"
    },

    # =====================================================
    # PUBLIC UNIVERSITIES
    # =====================================================

    "tu.muenchen": {
        "category": "public_university"
    },

    "huaheidelberg": {
        "category": "public_university"
    },

    "unibonn.international.students": {
        "category": "public_university"
    },

    "rwthinternationaloffice": {
        "category": "public_university"
    },

    "tuberlin_international": {
        "category": "public_university"
    },

    # =====================================================
    # PRIVATE UNIVERSITIES
    # =====================================================

    "iu.international": {
        "category": "private_university"
    },

    "gisma.university": {
        "category": "private_university"
    },

    "esmtberlin": {
        "category": "private_university"
    },

    "srh_university_international": {
        "category": "private_university"
    },

    "berlinsbi": {
        "category": "private_university"
    },

    "eu_business_school": {
        "category": "private_university"
    },
}


# =========================================================
# CONTENT KEYWORDS
# =========================================================

CONTENT_KEYWORDS = {
    # =====================================================
    # DEGREE / PROGRAM
    # =====================================================

    "degree_programs": [
        # ---------------------------------------------
        # Generic study promotion
        # ---------------------------------------------

        r"\bstudy in germany\b",
        r"\bstudy abroad\b",
        r"\bstudy with us\b",
        r"\bstudy at\b",
        r"\bjoin our\b",
        r"\bjoin us\b",
        r"\bapply now\b",
        r"\bapply today\b",
        r"\bapplications? open\b",
        r"\badmissions?\b",
        r"\benrol(l)?ment\b",
        r"\benrol(l)? now\b",

        # ---------------------------------------------
        # Program wording
        # ---------------------------------------------

        r"\bprogram(me)?\b",
        r"\bdegree\b",
        r"\bcourse\b",
        r"\bcurriculum\b",
        r"\bstudy track\b",
        r"\bstudy option\b",
        r"\bacademic path\b",

        # ---------------------------------------------
        # Bachelor
        # ---------------------------------------------

        r"\bbachelor('?s)?\b",
        r"\bbachelors\b",
        r"\bbachelor program(me)?\b",
        r"\bbachelor degree\b",

        r"\bbsc\b",
        r"\bb\.sc\b",
        r"\bb\.sc\.\b",

        r"\bba\b",
        r"\bb\.a\b",

        r"\bbeng\b",
        r"\bb\.eng\b",

        r"\bllb\b",

        r"\bundergraduate\b",
        r"\bundergraduate program(me)?\b",

        # ---------------------------------------------
        # Master
        # ---------------------------------------------

        r"\bmaster('?s)?\b",
        r"\bmasters\b",
        r"\bmaster degree\b",
        r"\bmaster program(me)?\b",

        r"\bmsc\b",
        r"\bm\.sc\b",
        r"\bm\.sc\.\b",

        r"\bma\b",
        r"\bm\.a\b",

        r"\bmeng\b",
        r"\bm\.eng\b",

        r"\bmba\b",
        r"\bemba\b",

        r"\bgraduate program(me)?\b",
        r"\bpostgraduate\b",

        # ---------------------------------------------
        # PhD / Doctoral
        # ---------------------------------------------

        r"\bphd\b",
        r"\bph\.d\b",
        r"\bdoctorate\b",
        r"\bdoctoral\b",
        r"\bdoctoral program(me)?\b",
        r"\bdoctoral research\b",

        # ---------------------------------------------
        # Diploma / certificate
        # ---------------------------------------------

        r"\bcertificate\b",
        r"\bdiploma\b",
        r"\bexecutive education\b",

        # ---------------------------------------------
        # Program descriptors
        # ---------------------------------------------

        r"\bon-campus\b",
        r"\bon campus\b",
        r"\bonline\b",
        r"\bhybrid learning\b",
        r"\bfull[- ]time\b",
        r"\bpart[- ]time\b",
        r"\benglish[- ]taught\b",
        r"\benglish speaking\b",

        # ---------------------------------------------
        # Academic / career wording
        # ---------------------------------------------

        r"\bcareer(s)?\b",
        r"\bglobal career(s)?\b",
        r"\bcareer opportunities\b",
        r"\bcareer path\b",

        r"\bskills?\b",
        r"\bpractical knowledge\b",
        r"\bindustry skills\b",
        r"\breal[- ]world\b",
        r"\bhands[- ]on\b",

        r"\btech industry\b",
        r"\bfuture career\b",
        r"\bjob market\b",

        r"\bbuild your future\b",
        r"\bprepare(s|d)? you for\b",
        r"\bcareer success\b",

        r"\binnovation\b",
        r"\bdigital future\b",
        r"\bnext generation\b",

        # ---------------------------------------------
        # International recruitment
        # ---------------------------------------------

        r"\binternational student(s)?\b",
        r"\bstudents from around the world\b",
    ],

    # =====================================================
    # SCHOLARSHIP / FUNDING
    # =====================================================

    "scholarship": [
        r"\bscholarship(s)?\b",
        r"\bfunding\b",
        r"\bstipend\b",
        r"\bgrant\b",
        r"\bfinancial support\b",
        r"\bfinancial aid\b",
        r"\btuition\b",
        r"\btuition fee\b",
        r"\bfully funded\b",
        r"\bpartially funded\b",
        r"\bfee waiver\b",
    ],

    # =====================================================
    # APPLICATION TIMELINE
    # =====================================================

    "timeline": [
        r"\bwinter semester\b",
        r"\bsummer semester\b",

        r"\bfall intake\b",
        r"\bspring intake\b",
        r"\bintake\b",
    ],

    # =====================================================
    # WORK / MIGRATION / RECRUITMENT
    # =====================================================

    "recruitment": [
        r"\bwork in germany\b",
        r"\bstudy and work\b",
        r"\bskilled workers?\b",
        r"\binternational talents?\b",

        r"\bjob opportunities\b",
        r"\bcareer opportunities\b",

        r"\bfuture in germany\b",
        r"\bmove to germany\b",

        r"\bstart your career\b",
        r"\bbuild your career\b",
    ],
}


# =========================================================
# EXCLUSION KEYWORDS
# =========================================================

EXCLUSION_KEYWORDS = [
    r"\bcampus life\b",
    r"\bparty\b",
    r"\bfood\b",
    r"\bsports day\b",
    r"\bholiday greetings?\b",
    r"\bresearch paper\b",
    r"\bfaculty award\b",
    r"\bvacation\b",

    r"\balumni reunion\b",
    r"\bgraduation ceremony\b",

    r"\bmerry christmas\b",
    r"\bhappy easter\b",
    r"\bnew year wishes\b",

    r"\bthrowback\b",
    r"\bbehind the scenes\b",

    r"\bstudent club\b",
    r"\bfootball\b",
    r"\bconcert\b",
]


# =========================================================
# HELPERS
# =========================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"[‐-–—]", "-", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[!?.]{2,}", " ", text)

    replacements = {
        "m.sc": "msc",
        "m.sc.": "msc",
        "m sc": "msc",

        "b.sc": "bsc",
        "b.sc.": "bsc",
        "b sc": "bsc",

        "ph.d": "phd",
        "ph.d.": "phd",

        "m.a": "ma",
        "b.a": "ba",

        "m.eng": "meng",
        "b.eng": "beng",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def contains_target_keywords(text):
    text = normalize_text(text)

    for category, patterns in CONTENT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return True

    return False


def contains_exclusion_keywords(text):
    text = normalize_text(text)

    for pattern in EXCLUSION_KEYWORDS:
        if re.search(pattern, text):
            return True

    return False


def extract_hashtags(text):
    if not text:
        return []

    return re.findall(r"#(\w+)", text)


def extract_external_link(text):
    if not text:
        return ""

    urls = re.findall(r"https?://\S+", text)

    return urls[0] if urls else ""


def is_english(text):
    if not text:
        return False

    try:
        return detect(text) == "en"

    except LangDetectException:
        return False


def extract_caption(page):
    selectors = [
        "h1",
        "meta[property='og:description']",
        "div._a9zs",
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector)

            if locator.count() > 0:
                if selector.startswith("meta"):
                    content = locator.first.get_attribute("content")

                    if content:
                        return content

                else:
                    text = locator.first.inner_text()

                    if text:
                        return text

        except Exception:
            continue

    return ""


def random_sleep(a=2, b=5):
    time.sleep(random.uniform(a, b))


def load_json_list(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return []


def save_json_list(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )




def main() -> None:
    # =========================================================
    # LOAD EXISTING DATA
    # =========================================================

    existing_original_posts = load_json_list(EXISTING_FILE)
    all_posts = load_json_list(JSON_OUTPUT)

    existing_post_ids = {
        post.get("post_id")
        for post in existing_original_posts + all_posts
        if post.get("post_id")
    }

    account_post_counter = Counter(
        post.get("account_name")
        for post in all_posts
        if post.get("account_name")
    )

    print("\nExisting original posts loaded:", len(existing_original_posts))
    print("Existing rerun posts loaded:", len(all_posts))
    print("Existing unique post IDs:", len(existing_post_ids))


    # =========================================================
    # MAIN
    # =========================================================

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=500
        )

        context = browser.new_context(
            storage_state=str(SESSION_FILE),

            viewport={
                "width": 1920,
                "height": 1080
            },

            device_scale_factor=1,

            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        page = context.new_page()

        # =====================================================
        # ACCOUNT LOOP
        # =====================================================

        for account, config in ACCOUNT_CONFIG.items():
            print("\n" + "=" * 60)
            print(f"ACCOUNT: @{account}")

            existing_count = account_post_counter.get(
                account,
                0
            )

            print(
                f"Existing collected from @{account}: "
                f"{existing_count}"
            )

            profile_url = (
                f"https://www.instagram.com/{account}/"
            )

            print(f"Opening profile: {profile_url}")

            try:
                page.goto(profile_url, timeout=60000)
                page.wait_for_timeout(5000)

            except Exception as e:
                print(f"Profile open error @{account}: {e}")
                continue

            # =================================================
            # COLLECT POST LINKS
            # Max: 500 post links per account
            # =================================================

            collected_links = set()

            scroll_attempts = 0
            no_new_links_rounds = 0
            previous_link_count = 0

            while (
                len(collected_links) < MAX_LINKS_PER_ACCOUNT
                and scroll_attempts < MAX_SCROLL_ATTEMPTS
                and no_new_links_rounds < MAX_NO_NEW_LINK_ROUNDS
            ):
                try:
                    links = page.eval_on_selector_all(
                        "a",
                        """
                        elements => elements
                            .map(e => e.href)
                            .filter(h => h.includes('/p/'))
                        """
                    )

                    for link in links:
                        clean_link = link.split("?")[0].rstrip("/")
                        collected_links.add(clean_link)

                except Exception as e:
                    print(
                        f"Link extraction error: {e}"
                    )

                current_link_count = len(collected_links)

                print(
                    f"Collected links: "
                    f"{current_link_count}/"
                    f"{MAX_LINKS_PER_ACCOUNT}"
                )

                if current_link_count == previous_link_count:
                    no_new_links_rounds += 1
                else:
                    no_new_links_rounds = 0

                previous_link_count = current_link_count

                if current_link_count >= MAX_LINKS_PER_ACCOUNT:
                    print(
                        f"Reached max "
                        f"{MAX_LINKS_PER_ACCOUNT} links "
                        f"for @{account}"
                    )
                    break

                if no_new_links_rounds >= MAX_NO_NEW_LINK_ROUNDS:
                    print(
                        "Stopping scroll: "
                        "no new post links found."
                    )
                    break

                page.mouse.wheel(0, 7000)

                page.wait_for_timeout(3000)

                scroll_attempts += 1

            post_links = list(collected_links)[
                :MAX_LINKS_PER_ACCOUNT
            ]

            print(
                f"Scanning {len(post_links)} posts "
                f"from @{account}..."
            )

            collected_posts_this_account = 0
            skipped_existing = 0
            skipped_no_caption = 0
            skipped_non_english = 0
            skipped_irrelevant = 0
            skipped_excluded = 0
            errors = 0

            # =================================================
            # POST LOOP
            # =================================================

            for post_url in post_links:

                if collected_posts_this_account >= MAX_POSTS_TO_COLLECT_PER_ACCOUNT:
                    print(
                        f"Reached max collected posts "
                        f"{MAX_POSTS_TO_COLLECT_PER_ACCOUNT} "
                        f"for @{account}"
                    )
                    break

                try:
                    print("\nOpening post:")
                    print(post_url)

                    page.goto(post_url, timeout=60000)

                    page.wait_for_timeout(4000)

                    # =========================================
                    # SHORTCODE
                    # =========================================

                    shortcode = (
                        post_url.rstrip("/")
                        .split("/")[-1]
                    )

                    if shortcode in existing_post_ids:
                        print(
                            "Skipped: already crawled"
                        )

                        skipped_existing += 1

                        continue

                    # =========================================
                    # CAPTION
                    # =========================================

                    caption = extract_caption(page)

                    if not caption:
                        print(
                            "Skipped: no caption"
                        )

                        skipped_no_caption += 1

                        continue

                    # =========================================
                    # LANGUAGE
                    # =========================================

                    if not is_english(caption):
                        print(
                            "Skipped: non-English"
                        )

                        skipped_non_english += 1

                        continue

                    # =========================================
                    # CONTENT FILTER
                    # =========================================

                    if not contains_target_keywords(
                        caption
                    ):
                        print(
                            "Skipped: irrelevant content"
                        )

                        skipped_irrelevant += 1

                        continue

                    if contains_exclusion_keywords(
                        caption
                    ):
                        print(
                            "Skipped: excluded content"
                        )

                        skipped_excluded += 1

                        continue

                    # =========================================
                    # SCREENSHOT
                    # =========================================

                    screenshot_relative_path = (
                        f"data/raw/screenshots/"
                        f"{shortcode}.png"
                    )

                    screenshot_absolute_path = (
                        BASE_DIR
                        / screenshot_relative_path
                    )

                    page.evaluate(
                        "window.scrollTo(0, 0)"
                    )

                    page.wait_for_timeout(1000)

                    page.screenshot(
                        path=str(
                            screenshot_absolute_path
                        ),
                        full_page=False
                    )

                    # =========================================
                    # TIMESTAMP
                    # =========================================

                    posting_time = ""

                    try:
                        time_element = (
                            page.locator("time")
                            .first
                        )

                        if time_element.count() > 0:
                            posting_time = (
                                time_element
                                .get_attribute(
                                    "datetime"
                                )
                            )

                    except Exception:
                        pass

                    # =========================================
                    # METADATA
                    # =========================================

                    hashtags = (
                        extract_hashtags(caption)
                    )

                    external_link = (
                        extract_external_link(caption)
                    )

                    post_data = {
                        "post_id": shortcode,

                        "post_url": post_url,

                        "platform": "instagram",

                        "account_name": account,

                        "account_category":
                            config["category"],

                        "caption_text": caption,

                        "hashtags": hashtags,

                        "screenshot_url":
                            screenshot_relative_path,

                        "posting_time":
                            posting_time,

                        "external_link":
                            external_link,

                        "language": "en",

                        "seed_label":
                            "legitimate"
                    }

                    all_posts.append(post_data)

                    existing_post_ids.add(shortcode)

                    account_post_counter[
                        account
                    ] += 1

                    collected_posts_this_account += 1

                    # =========================================
                    # SAVE IMMEDIATELY
                    # =========================================

                    save_json_list(
                        JSON_OUTPUT,
                        all_posts
                    )

                    print(
                        f"Collected new post from @{account}: "
                        f"{collected_posts_this_account}"
                    )

                    print(
                        f"Total collected posts: "
                        f"{len(all_posts)}"
                    )

                    random_sleep()

                except Exception as e:
                    errors += 1

                    print(f"ERROR: {e}")

                    continue

            # =================================================
            # ACCOUNT SUMMARY
            # =================================================

            print("\n" + "-" * 60)
            print(f"SUMMARY @{account}")
            print("-" * 60)
            print(f"Loaded links: {len(post_links)}")
            print(f"New posts collected: {collected_posts_this_account}")
            print(f"Skipped existing: {skipped_existing}")
            print(f"Skipped no caption: {skipped_no_caption}")
            print(f"Skipped non-English: {skipped_non_english}")
            print(f"Skipped irrelevant: {skipped_irrelevant}")
            print(f"Skipped excluded: {skipped_excluded}")
            print(f"Errors: {errors}")

        browser.close()


    # =========================================================
    # DONE
    # =========================================================

    print("\nDONE")
    print(f"Collected posts: {len(all_posts)}")
    print(f"JSON saved to: {JSON_OUTPUT}")

    print("\nPosts per account in rerun output:")
    for account, count in account_post_counter.most_common():
        print(f"- {account}: {count}")


if __name__ == "__main__":
    main()
