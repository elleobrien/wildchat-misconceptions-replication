#!/usr/bin/env python3
"""Does the assistant push back with a first-person capability negation
("I cannot / I'm unable to / I don't have access ...")?

One regex, applied sentence by sentence to every assistant turn. Runs over
every in-corpus conversation with a final human label and writes
analysis/data/corrections.csv, one row per conversation:
  any_correction          1 if any assistant turn has a sentence matching NEG
  first_correction_turn   1-based index of the first assistant turn that does ('' if none)
  n_assistant_turns
  as_an_ai                1 if that first hit sentence also carries an "as an AI ..." preface (flavor only)
  snippet                 the first matching sentence
The "as an AI" preface is not a signal (every "As an AI ... I cannot" already
matches NEG); what was declined is read off the misconception label.
Consumed by correction_omnibus_tests.R and sensitivity_clustering.R.

    python analysis/regex_corrections.py        # needs data/wildchat_1m_english_python.parquet
"""
import re
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
INPUT_PARQUET = ROOT / "data" / "wildchat_1m_english_python.parquet"

NEG = re.compile(
    r"\bI(?:'m| am)?\s+(?:currently )?(?:cannot|can'?t|can not|(?:not able|unable) to|do(?:n't| not) have (?:(?:direct )?access|the (?:ability|capability)|(?:a )?(?:\w+ )?(?:means|interface))|have no (?:ability|means))\b",
    re.I)
AS_AN_AI = re.compile(r"\bas an? (?:ai|artificial intelligence|language model|chatbot|text-based)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")


def classify(conv):
    ai = 0
    for m in conv:
        if m.get("role") != "assistant":
            continue
        ai += 1
        for sent in SENT.split(m.get("content") or ""):
            if NEG.search(sent):
                return {"any_correction": 1, "first_correction_turn": ai,
                        "as_an_ai": int(bool(AS_AN_AI.search(sent))),
                        "snippet": " ".join(sent.split())[:230]}
    return {"any_correction": 0, "first_correction_turn": "", "as_an_ai": 0, "snippet": ""}


if __name__ == "__main__":
    conv = pd.read_csv(ROOT/"analysis"/"data"/"conversations.csv", dtype=str).fillna("")
    conv = conv[(conv.in_corpus == "1") & (conv.labels_final != "")]
    t = pq.read_table(INPUT_PARQUET, columns=["conversation_hash", "conversation"])
    convs = dict(zip(t.column("conversation_hash").to_pylist(), t.column("conversation").to_pylist()))
    rows = []
    for h in conv.conversation_hash:
        r = classify(convs[h]); r["n_assistant_turns"] = sum(m.get("role") == "assistant" for m in convs[h])
        rows.append({"conversation_hash": h, **r})
    out = pd.DataFrame(rows)[["conversation_hash","any_correction","first_correction_turn","n_assistant_turns","as_an_ai","snippet"]]
    out.to_csv(ROOT/"analysis"/"data"/"corrections.csv", index=False)
    print(f"{len(out)} conversations, {out.any_correction.sum()} with a correction ({100*out.any_correction.mean():.0f}%)")
