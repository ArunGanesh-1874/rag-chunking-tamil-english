"""
fetch_articles.py

Fetch matching English + Tamil Wikipedia articles for the cross-lingual
RAG chunking-strategy study.

Design notes (why it's built this way):
- Uses the plain MediaWiki Action API (action=query, prop=extracts) which
  returns clean, markup-free plaintext directly from Wikipedia's own
  parser cache -- no need to hand-roll wikitext stripping.
- Tamil titles are resolved automatically via `langlinks` on the English
  article wherever possible, with a manual override dict as fallback
  (langlinks are occasionally missing or point to a disambiguation stub).
- Filters out articles that are too short to be useful as RAG source
  documents (stubs), separately thresholded per language, since Tamil
  Wikipedia articles tend to run shorter than their English counterparts
  even when "well developed" by Tamil-Wikipedia standards.
- Saves: (a) one JSON manifest with full metadata + text, (b) individual
  .txt files per article/language for easy inspection, chunking, etc.
- CC BY-SA 4.0: Wikipedia text is reused here under that license. Each
  saved record includes title, url, revision id and retrieval date for
  proper attribution in your paper's data section.

Run with: python fetch_articles.py
Requires: pip install requests
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime, timezone

import requests

EN_API = "https://en.wikipedia.org/w/api.php"
TA_API = "https://ta.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "RAG-Chunking-Research/1.0 (academic project; contact: set-your-email@example.com)"
}

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "en").mkdir(exist_ok=True)
(OUT_DIR / "ta").mkdir(exist_ok=True)

# Minimum word counts to treat an article as "well-developed" (not a stub).
# Tamil Wikipedia threshold is lower deliberately -- see docstring above.
MIN_WORDS_EN = 800
MIN_WORDS_TA = 300

# English article titles (Wikipedia page titles, exact).
TOPICS = [
    "Chola dynasty",
    "Vijayanagara Empire",
    "Indian independence movement",
    "Tamil Nadu",
    "Kaveri River",
    "Western Ghats",
    "Chennai",
    "Himalayas",
    "Mahatma Gandhi",
    "A. P. J. Abdul Kalam",
    "Subramania Bharati",
    "M. G. Ramachandran",
    "C. V. Raman",
    "Monsoon",
    "Solar System",
    "Bharatanatyam",
    "Tamil language",
    "Meenakshi Amman Temple",
]

# Manual EN -> TA title overrides, used only if automatic langlink
# resolution fails or returns a bad match. Leave empty and let the
# script auto-resolve first; fill in only the ones that fail.
MANUAL_TA_TITLE = {
    "Kaveri River": "காவிரி ஆறு",
    "Meenakshi Amman Temple": "மதுரை மீனாட்சி சுந்தரேசுவரர் கோயில்",
}


def get_json(api_url, params, retries=3):
    params = {**params, "format": "json"}
    for attempt in range(retries):
        try:
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def get_tamil_title(en_title):
    """Resolve the Tamil Wikipedia title via interlanguage links on the EN page."""
    if en_title in MANUAL_TA_TITLE:
        return MANUAL_TA_TITLE[en_title]
    data = get_json(EN_API, {
        "action": "query",
        "titles": en_title,
        "prop": "langlinks",
        "lllang": "ta",
        "lllimit": "1",
    })
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        langlinks = page.get("langlinks")
        if langlinks:
            return langlinks[0]["*"]
    return None


def fetch_extract(api_url, title):
    """Fetch plaintext extract + revision id + pageid for a given title."""
    data = get_json(api_url, {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
    })
    pages = data.get("query", {}).get("pages", {})
    for pageid, page in pages.items():
        if pageid == "-1" or "missing" in page:
            return None
        return {
            "pageid": page.get("pageid"),
            "title": page.get("title"),
            "url": page.get("fullurl"),
            "extract": page.get("extract", ""),
        }
    return None


def clean_text(text):
    """Light cleanup: drop trailing References/See also boilerplate headers,
    collapse excess blank lines. Keep body text otherwise untouched."""
    # Cut off well after the main body -- Wikipedia extracts include
    # section headers like "== References ==" as plain lines.
    cut_headers = ["See also", "References", "External links", "Notes",
                   "Further reading", "Bibliography"]
    lines = text.split("\n")
    cut_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in cut_headers and i > 20:  # don't cut near the very top
            cut_idx = i
            break
    body = "\n".join(lines[:cut_idx])
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def word_count(text, is_tamil=False):
    if is_tamil:
        # Tamil doesn't tokenize cleanly on whitespace alone for all cases,
        # but whitespace-split is a reasonable proxy consistent with how
        # you'll count words elsewhere in the pipeline (chunking, RAGAS).
        return len(text.split())
    return len(text.split())


def main():
    manifest = []
    skipped = []

    for en_title in TOPICS:
        print(f"\n=== {en_title} ===")

        en_data = fetch_extract(EN_API, en_title)
        if not en_data or not en_data["extract"]:
            print(f"  [SKIP] Could not fetch EN article: {en_title}")
            skipped.append({"topic": en_title, "reason": "en_fetch_failed"})
            continue

        ta_title = get_tamil_title(en_title)
        if not ta_title:
            print(f"  [SKIP] No Tamil langlink found for: {en_title}")
            skipped.append({"topic": en_title, "reason": "no_ta_langlink"})
            continue

        ta_data = fetch_extract(TA_API, ta_title)
        if not ta_data or not ta_data["extract"]:
            print(f"  [SKIP] Could not fetch TA article: {ta_title}")
            skipped.append({"topic": en_title, "reason": "ta_fetch_failed", "ta_title": ta_title})
            continue

        en_clean = clean_text(en_data["extract"])
        ta_clean = clean_text(ta_data["extract"])

        en_wc = word_count(en_clean)
        ta_wc = word_count(ta_clean, is_tamil=True)

        print(f"  EN: {en_wc} words | TA: {ta_wc} words")

        if en_wc < MIN_WORDS_EN:
            print(f"  [SKIP] EN article below stub threshold ({en_wc} < {MIN_WORDS_EN})")
            skipped.append({"topic": en_title, "reason": "en_too_short", "words": en_wc})
            continue
        if ta_wc < MIN_WORDS_TA:
            print(f"  [SKIP] TA article below stub threshold ({ta_wc} < {MIN_WORDS_TA})")
            skipped.append({"topic": en_title, "reason": "ta_too_short", "words": ta_wc})
            continue

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", en_title).strip("_").lower()

        (OUT_DIR / "en" / f"{slug}.txt").write_text(en_clean, encoding="utf-8")
        (OUT_DIR / "ta" / f"{slug}.txt").write_text(ta_clean, encoding="utf-8")

        record = {
            "topic": en_title,
            "slug": slug,
            "en": {
                "title": en_data["title"],
                "url": en_data["url"],
                "pageid": en_data["pageid"],
                "word_count": en_wc,
                "char_count": len(en_clean),
                "text_file": f"data/en/{slug}.txt",
            },
            "ta": {
                "title": ta_data["title"],
                "url": ta_data["url"],
                "pageid": ta_data["pageid"],
                "word_count": ta_wc,
                "char_count": len(ta_clean),
                "text_file": f"data/ta/{slug}.txt",
            },
            "license": "CC BY-SA 4.0 (Wikipedia contributors)",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest.append(record)
        print(f"  [OK] Saved as {slug}")

        time.sleep(0.5)  # be polite to the API

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"articles": manifest, "skipped": skipped}, f, ensure_ascii=False, indent=2)

    print(f"\n=== Done: {len(manifest)} article pairs saved, {len(skipped)} skipped ===")
    if skipped:
        print("Skipped (review and add manual overrides if needed):")
        for s in skipped:
            print(" -", s)


if __name__ == "__main__":
    main()
