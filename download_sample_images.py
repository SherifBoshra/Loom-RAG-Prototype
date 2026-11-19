"""Download ~30 varied, freely licensed clothing photos from Wikimedia Commons
into images/, and write the credits to images/CREDITS.md.

Run: python download_sample_images.py
Only photos licensed CC0, CC BY, CC BY-SA or public domain are kept.
"""

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
API = "https://commons.wikimedia.org/w/api.php?"
HEADERS = {"User-Agent": "stylist-rag-sample-downloader/0.1 (educational prototype)"}

# (Commons category, how many photos to take from it) -> 30 photos in total.
CATEGORIES = [
    ("Evening dresses", 3),
    ("Wedding dresses", 2),
    ("Cocktail dresses", 2),
    ("Summer dresses", 2),
    ("Men's suits", 3),
    ("Blazers", 2),
    ("Jeans", 2),
    ("T-shirts", 2),
    ("Leather jackets", 2),
    ("Trench coats", 2),
    ("Hoodies", 2),
    ("Skirts", 2),
    ("Sweaters", 2),
    ("Kimono", 2),
]
LICENSE_OK = re.compile(r"CC0|CC BY|Public domain", re.I)


def fetch(url, retries=6):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s...")
            time.sleep(wait)


def candidates(category):
    params = dict(
        action="query", format="json", generator="categorymembers",
        gcmtitle="Category:" + category, gcmtype="file", gcmlimit=50,
        prop="imageinfo", iiprop="url|extmetadata|mime|size", iiurlwidth=640,
    )
    data = json.loads(fetch(API + urllib.parse.urlencode(params)))
    for page in data.get("query", {}).get("pages", {}).values():
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if info["mime"] != "image/jpeg" or info.get("width", 0) < 400:
            continue
        if not LICENSE_OK.search(license_name):
            continue
        artist = re.sub(r"<[^>]+>", "", html.unescape(meta.get("Artist", {}).get("value", ""))).strip()
        yield dict(title=page["title"], thumb=info["thumburl"], page=info["descriptionurl"],
                   license=license_name, artist=artist or "Unknown")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main():
    IMAGE_DIR.mkdir(exist_ok=True)
    credits = []
    for category, count in CATEGORIES:
        print(f"{category}:")
        try:
            picks = list(candidates(category))[:count]
        except Exception as e:  # keep going if one category fails
            print(f"  skipped ({e})")
            continue
        for n, item in enumerate(picks, 1):
            name = f"{slug(category)}_{n}.jpg"
            try:
                (IMAGE_DIR / name).write_bytes(fetch(item["thumb"]))
            except Exception as e:
                print(f"  failed {name} ({e})")
                continue
            credits.append(f"| {name} | [{item['title']}]({item['page']}) | {item['artist']} | {item['license']} |")
            print(f"  saved {name}")
            time.sleep(1)
        time.sleep(2)

    (IMAGE_DIR / "CREDITS.md").write_text(
        "# Photo credits\n\nAll photos are from Wikimedia Commons.\n\n"
        "| File | Source | Author | License |\n|---|---|---|---|\n" + "\n".join(credits) + "\n",
        encoding="utf-8",
    )
    print(f"\nDone: {len(credits)} photos saved to images/ (credits in images/CREDITS.md).")


if __name__ == "__main__":
    main()
