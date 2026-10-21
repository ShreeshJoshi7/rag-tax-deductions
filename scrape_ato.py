"""
ATO Deductions Document Scraper (auto-crawl version)
------------------------------------------------------
Starts from the ATO "Deductions you can claim" hub page and automatically
follows internal links within that section, up to MAX_DEPTH levels deep,
saving each page's clean text as a .txt file ready for the RAG ingestion
pipeline (load_documents() in ingest.py).

Note: ato.gov.au blocks requests from some cloud/datacenter IPs (you may
see 403 errors if running this from a hosted sandbox/VM). Run it from a
normal home/university connection and it will very likely work fine.

Usage:
    python scrape_ato.py

Output:
    docs/<slug>.txt          -- one file per page
    docs/_sources.csv        -- mapping of filename -> source URL (for citations)
"""

import os
import re
import csv
import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

START_URL = "https://www.ato.gov.au/individuals-and-families/income-deductions-offsets-and-records/deductions-you-can-claim"
ALLOWED_PREFIX = "https://www.ato.gov.au/individuals-and-families/income-deductions-offsets-and-records/deductions-you-can-claim"
MAX_DEPTH = 4          # how many link-hops deep to crawl from START_URL
MAX_PAGES = 80         # safety cap so a crawl can't run away
REQUEST_DELAY = 1.0    # seconds between requests -- be polite to ATO's servers

OUTPUT_DIR = "docs"
SOURCES_CSV = os.path.join(OUTPUT_DIR, "_sources.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}


def slugify(url: str) -> str:
    path = url.rstrip("/").split("ato.gov.au")[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower()
    return slug[-90:] or "page"


def clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "svg", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("body") or soup
    text = main.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    deduped = []
    for ln in lines:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)
    return "\n".join(deduped)


def extract_internal_links(soup: BeautifulSoup, current_url: str) -> list:
    """Return links that stay within the deductions section of the site."""
    links = []
    main = soup.find("main") or soup
    for a in main.find_all("a", href=True):
        href = a["href"]
        full = urljoin(current_url, href)
        full = full.split("#")[0].split("?")[0].rstrip("/")
        if full.startswith(ALLOWED_PREFIX) and urlparse(full).netloc == "www.ato.gov.au":
            links.append(full)
    return links


def fetch(url: str, retries: int = 2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)
    raise last_err


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visited = set()
    queue = [(START_URL, 0)]
    rows = []

    while queue and len(visited) < MAX_PAGES:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[depth {depth}] Fetching: {url}")
        try:
            soup = fetch(url)
        except Exception as e:
            print(f"  FAILED: {e}")
            if "403" in str(e):
                print("  (403 = ATO's bot protection. Try running this from a normal home/uni")
                print("   connection rather than a cloud VM or sandbox.)")
            continue

        text = clean_text(soup)
        slug = slugify(url)
        filename = f"{slug}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        rows.append({"filename": filename, "url": url, "depth": depth})
        print(f"  Saved -> {filepath} ({len(text)} chars)")

        if depth < MAX_DEPTH:
            for link in extract_internal_links(soup, url):
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(REQUEST_DELAY)

    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "url", "depth"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} pages saved to {OUTPUT_DIR}/")
    print(f"Source mapping saved to {SOURCES_CSV}")


if __name__ == "__main__":
    main()
