#!/usr/bin/env python3
"""Regex probe: in the "Algebra" conversations, which side of the conversation
introduces Python?

Denominator: in-corpus conversations (data/conversations.csv, in_corpus == 1)
whose labels_final contains "Algebra".  Per conversation and per role we test
two signal tiers against the raw turn text (no cleaning; same quick-probe
style as regex_scraping.py):

  MENTION  the word "python" or a fenced code block (```)
  CODEISH  MENTION plus code-ish tokens a user might paste without saying
           "python": numpy / np. / pandas / pd. / scipy / sympy / matplotlib,
           `import` and `def` as standalone words, and `print(`.
           (Every alternative is word-bounded on both sides where applicable;
           an earlier ad-hoc version matched "important" via a one-sided
           \bimport, which inflated user hits by 2.)

A conversation is "model-introduced" under a tier when NO user turn matches
the tier's pattern but at least one assistant turn does.

Writes data/python_intro_regex.csv (one row per Algebra conversation with the
four role x tier flags) and prints the summary counts used in the qualitative
results ("in N of M conversations the user never ...").

Usage:  python analysis/regex_python_intro.py
"""
import csv
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CONVERSATIONS_CSV = ROOT / "data" / "conversations.csv"
INPUT_PARQUET = REPO_ROOT / "data" / "wildchat_1m_english_python.parquet"
OUT_CSV = ROOT / "data" / "python_intro_regex.csv"

MENTION = re.compile(r"\bpython\b|```", re.I)
CODEISH = re.compile(
    r"\bpython\b|```|"
    r"\b(?:numpy|pandas|scipy|sympy|matplotlib|import|def)\b|"
    r"\b(?:np|pd)\.|"
    r"\bprint\s*\(",
    re.I,
)


def role_hit(conv, role, rx):
    return any(rx.search(t["content"] or "") for t in conv if t["role"] == role)


def main():
    with open(CONVERSATIONS_CSV) as f:
        rows = list(csv.DictReader(f))
    alg = [
        (r["conv_id"], r["conversation_hash"])
        for r in rows
        if r["in_corpus"] == "1"
        and "Algebra" in [x.strip() for x in r["labels_final"].split(";")]
    ]

    t = pq.read_table(INPUT_PARQUET, columns=["conversation_hash", "conversation"])
    convs = dict(
        zip(t.column("conversation_hash").to_pylist(), t.column("conversation").to_pylist())
    )

    out_rows = []
    for conv_id, h in alg:
        conv = convs[h]
        out_rows.append(
            {
                "conv_id": conv_id,
                "conversation_hash": h,
                "user_mention": int(role_hit(conv, "user", MENTION)),
                "asst_mention": int(role_hit(conv, "assistant", MENTION)),
                "user_codeish": int(role_hit(conv, "user", CODEISH)),
                "asst_codeish": int(role_hit(conv, "assistant", CODEISH)),
            }
        )

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    for tier in ("mention", "codeish"):
        intro = [r for r in out_rows if not r[f"user_{tier}"] and r[f"asst_{tier}"]]
        neither = [r for r in out_rows if not r[f"user_{tier}"] and not r[f"asst_{tier}"]]
        print(
            f"{tier}: model-introduced {len(intro)}/{n} ({100 * len(intro) / n:.1f}%)"
            f"  [user never hits, assistant does; {len(neither)} hit on neither side]"
        )
    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
