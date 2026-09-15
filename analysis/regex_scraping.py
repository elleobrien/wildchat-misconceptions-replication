#!/usr/bin/env python3
"""Lightweight regex probe: what share of the "Internet Access" conversations
are explicitly about web scraping?

Denominator: in-corpus conversations (data/conversations.csv, in_corpus == 1)
whose labels_final contains "Internet Access".  A conversation counts as a hit
if any raw user turn matches the scraping family below (no turn cleaning --
this is a quick lexical probe, not the cleaned-turn pipeline of
regex_homework.py).

Writes data/scraping_regex.csv (one row per Internet Access conversation:
hash, scraping_hit, first matched term) and prints the summary.

Usage:  python analysis/regex_scraping.py
"""
import csv
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CONVERSATIONS_CSV = ROOT / "data" / "conversations.csv"
INPUT_PARQUET = REPO_ROOT / "data" / "wildchat_1m_english_python.parquet"
OUT_CSV = ROOT / "data" / "scraping_regex.csv"

# scrape/scrapes/scraped/scraping/scraper + crawl forms + common tool names
SCRAPING = re.compile(
    r"\b(scrap(?:e[srd]?|ing)|web[- ]?scraping|crawl(?:er|ing|s)?|"
    r"beautifulsoup|bs4|scrapy)\b",
    re.I,
)


def main():
    with open(CONVERSATIONS_CSV) as f:
        rows = list(csv.DictReader(f))
    ia_hashes = [
        r["conversation_hash"]
        for r in rows
        if r["in_corpus"] == "1" and "Internet Access" in r["labels_final"]
    ]

    t = pq.read_table(INPUT_PARQUET, columns=["conversation_hash", "conversation"])
    convs = dict(
        zip(t.column("conversation_hash").to_pylist(), t.column("conversation").to_pylist())
    )

    out_rows = []
    for h in ia_hashes:
        first_term = ""
        for turn in convs[h]:
            if turn["role"] != "user":
                continue
            m = SCRAPING.search(turn["content"])
            if m:
                first_term = m.group(0).lower()
                break
        out_rows.append(
            {"conversation_hash": h, "scraping_hit": int(bool(first_term)), "first_term": first_term}
        )

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conversation_hash", "scraping_hit", "first_term"])
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    hits = sum(r["scraping_hit"] for r in out_rows)
    print(f"Internet Access conversations (in corpus): {n}")
    print(f"Explicit scraping (regex hit in a user turn): {hits} ({100 * hits / n:.1f}%)")
    print("First-match terms:", Counter(r["first_term"] for r in out_rows if r["first_term"]).most_common())
    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
