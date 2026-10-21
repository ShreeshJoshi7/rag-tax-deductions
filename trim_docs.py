"""
Trim scraped ATO documents based on your reviewed _review.csv.
------------------------------------------------------------------
Run review_docs.py first, open docs/_review.csv, set the "keep" column to
"n" for any row you don't want, save the file, then run this script.

Files marked keep=n are MOVED (not deleted) into docs_excluded/, so
nothing is lost -- you can always move something back if you change your
mind.

Usage:
    python trim_docs.py
"""

import os
import csv
import shutil

DOCS_DIR = "docs"
EXCLUDED_DIR = "docs_excluded"
REVIEW_CSV = os.path.join(DOCS_DIR, "_review.csv")
SOURCES_CSV = os.path.join(DOCS_DIR, "_sources.csv")


def main():
    if not os.path.exists(REVIEW_CSV):
        print(f"Couldn't find {REVIEW_CSV}. Run review_docs.py first, edit the")
        print("'keep' column, then run this script.")
        return

    os.makedirs(EXCLUDED_DIR, exist_ok=True)

    kept_rows = []
    excluded_count = 0

    with open(REVIEW_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            keep = row.get("keep", "y").strip().lower()
            src = os.path.join(DOCS_DIR, filename)

            if keep in ("n", "no", "0", "false"):
                if os.path.exists(src):
                    shutil.move(src, os.path.join(EXCLUDED_DIR, filename))
                    excluded_count += 1
                    print(f"Excluded: {filename}")
            else:
                kept_rows.append({"filename": filename, "url": row["url"]})

    # Rewrite _sources.csv so it only reflects what's actually still in docs/
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "url"])
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"\nDone. {excluded_count} files moved to {EXCLUDED_DIR}/")
    print(f"{len(kept_rows)} files remain in {DOCS_DIR}/")
    print(f"{SOURCES_CSV} updated to match.")


if __name__ == "__main__":
    main()
