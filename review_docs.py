"""
Review scraped ATO documents before trimming.
------------------------------------------------
Reads docs/_sources.csv (produced by scrape_ato.py) plus the actual .txt
files, and prints a sorted summary table so you can decide what to keep.

Usage:
    python review_docs.py

Also writes docs/_review.csv -- open this in Excel/Sheets to sort, filter,
and mark a "keep" column if that's easier than reading the console output.
"""

import os
import csv

DOCS_DIR = "docs"
SOURCES_CSV = os.path.join(DOCS_DIR, "_sources.csv")
REVIEW_CSV = os.path.join(DOCS_DIR, "_review.csv")


def main():
    if not os.path.exists(SOURCES_CSV):
        print(f"Couldn't find {SOURCES_CSV}. Run scrape_ato.py first.")
        return

    rows = []
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(DOCS_DIR, row["filename"])
            char_count = 0
            if os.path.exists(filepath):
                with open(filepath, encoding="utf-8") as tf:
                    char_count = len(tf.read())
            rows.append({
                "filename": row["filename"],
                "depth": row.get("depth", ""),
                "chars": char_count,
                "url": row["url"],
                "keep": "y",  # default -- edit this column, then run trim_docs.py
            })

    # Sort: shallow hub-ish pages first, then by size (tiny pages are often
    # just navigation stubs worth dropping)
    rows.sort(key=lambda r: (int(r["depth"] or 0), r["chars"]))

    print(f"{'filename':45} {'depth':6} {'chars':7} url")
    print("-" * 120)
    for r in rows:
        flag = "  <- very short, check this one" if r["chars"] < 400 else ""
        print(f"{r['filename']:45} {r['depth']:<6} {r['chars']:<7} {r['url']}{flag}")

    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "depth", "chars", "url", "keep"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} files listed. Also saved to {REVIEW_CSV}")
    print("Open that CSV, set the 'keep' column to 'n' for anything you don't want,")
    print("save it, then run: python trim_docs.py")


if __name__ == "__main__":
    main()
