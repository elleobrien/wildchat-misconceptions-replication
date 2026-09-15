#!/usr/bin/env python3
"""Lexical (regex) classifier for the pasted-assignment question ("is this
conversation possibly pasted coursework?"). Writes data/homework_regex.csv,
one row per labeled conversation in data/conversations.csv (in_corpus==1,
labels_final nonempty), or per hash in --input if given.

Method
  * user turns only; each turn is cleaned before matching: fenced/inline code,
    HTML tags, URLs, and traceback/log-looking lines are removed (those are
    where "submit", "task 1.0", "student" false positives come from);
  * signal families, each a regex, grouped by strength:
      STRONG_A  one hit -> A   (mark values, explicit homework self-report,
                               course codes, submission logistics, teaching-
                               staff words, mark scheme / rubric ...)
      MEDIUM_A  two distinct families -> A  (numbered questions/tasks,
                               lettered sub-parts, multiple-choice options,
                               "Required:/Compute:" prompts, instructor->
                               student persona, "you are required to" ...)
    otherwise C.  Every alternative in every family was checked against the
    cleaned user turns of all 953 parquet conversations on 2026-08-19 and
    alternatives that never fire were removed (pruning preserves the hit set
    exactly); re-expand if the corpus changes.  No B or D: contest/practice-problem shape (Sample Input/
    Output, LeetCode...) was dropped 2026-08-19 for parsimony -- the flag is
    binary "possibly pasted coursework"; a regex cannot express "unclear".
  * pasted_turns = user turns where any signal fired; evidence = matched
    snippets; reasoning = which families fired.
  * the reported flag (hw_flag) is the union: regex category A OR an
    annotator mark -- a human verification note (source == "verify" in
    data/labels_long.csv) mentioning homework/HW, minus hand-reviewed
    negations (see ANNOTATOR_HW_NEGATED). annotator_hw records the mark
    on its own.

Usage
  python regex_homework.py                # classify the labeled corpus, write data/homework_regex.csv
  python regex_homework.py --show A       # print snippets for every A (or C) for eyeballing
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_PARQUET = REPO_ROOT / "data" / "wildchat_1m_english_python.parquet"

RESULTS_DIR = SCRIPT_DIR / "data"
# Default input: the current labeled corpus (in_corpus==1, labels_final nonempty).
# Pass --input <csv with a conversation_hash column> to classify another set.
CONVERSATIONS_CSV = SCRIPT_DIR / "data" / "conversations.csv"
LABELS_LONG_CSV = SCRIPT_DIR / "data" / "labels_long.csv"
OUT_CSV = RESULTS_DIR / "homework_regex.csv"
OUT_HEADER = ["conversation_hash", "category", "annotator_hw", "hw_flag",
              "pasted_turns", "evidence",
              "reasoning", "raw_last_line", "prompt_tokens", "completion_tokens",
              "cached_tokens"]

# The reported flag is regex OR annotator mark: during label verification the
# human annotators sometimes noted a conversation as likely coursework
# ("possibly HW", "looks like homework", ...). Any human verification note
# (source == "verify" in labels_long.csv) mentioning homework/HW marks the
# conversation, except hashes listed in ANNOTATOR_HW_NEGATED, whose notes
# mention HW only to say the conversation does NOT look like it. All matching
# notes were reviewed by hand on 2026-08-25; re-review if notes change.
ANNOTATOR_HW_NOTE = re.compile(r"homework|\bhw\b", re.I)
ANNOTATOR_HW_NEGATED = {
    # C164: "it doesn't look like the note with the dataset link is HW paste"
    "2afdf133085e2eea993c780a0d82e8ee",
}


def annotator_hw_hashes() -> set[str]:
    marked = set()
    with open(LABELS_LONG_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("source") == "verify" and ANNOTATOR_HW_NOTE.search(r.get("note") or ""):
                marked.add(r["conversation_hash"])
    return marked - ANNOTATOR_HW_NEGATED

# ---------------------------------------------------------------- cleaning
FENCE = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`[^`\n]{1,200}`")
HTML = re.compile(r"</?[a-zA-Z][^>]{0,120}>")
URL = re.compile(r"https?://\S+")
LOGLINE = re.compile(
    r"^\s*(?:Traceback|File \"|\s+at |ERROR|WARN(?:ING)?|INFO|DEBUG|Exception|"
    r"\d{2,4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}|>>>|\$ |In \[\d+\]|Out\[\d+\])",
    re.M,
)
CODEISH = re.compile(  # lines that are overwhelmingly code even without fences
    r"^\s*(?:def |class |import |from \S+ import|return |print\(|if .*:$|for .+ in .+:$|"
    r"while .+:$|try:|except|#include|public |private |function\s*\(|var |const |let |"
    r"\}|\{|\)\s*$|[A-Za-z_][\w\.\[\]]*\s*=\s*\S+\s*$|[A-Za-z_][\w\.\[\]]*\s*=.*[\)\]\}\"']\s*$)",
    re.M,
)


def clean(text: str) -> str:
    t = FENCE.sub(" ", text or "")
    t = INLINE_CODE.sub(" ", t)
    t = HTML.sub(" ", t)
    t = URL.sub(" ", t)
    lines = [ln for ln in t.splitlines()
             if not LOGLINE.match(ln) and not CODEISH.match(ln)]
    return "\n".join(lines)


# ---------------------------------------------------------------- signals
COURSE_CODE_STOP = {"GPT", "HTTP", "HTTPS", "UTF", "ISO", "RGB", "GPU", "CPU", "MD",
                    "SHA", "AES", "RSA", "IPV", "TCP", "UDP", "PEP", "RFC", "SQL", "AMD",
                    "USB", "PDF", "PNG", "JPG", "MP", "COVID", "ID", "IP", "AI", "API",
                    "URL", "CSV", "XML", "MB", "GB", "KB", "TB", "MHZ", "GHZ", "PM", "AM",
                    "BERT", "YOLO", "RTX", "GTX", "DDR", "ARM", "X", "R",
                    "HKD", "USD", "EUR", "GBP", "CNY", "JPY", "INR", "AUD", "CAD", "SGD", "INV",
                    "REF", "PO", "SKU", "ORD", "ISBN", "DOI", "VAT", "PIN", "ZIP", "SS", "SSD", "HDD",
                    "RAM", "ROM", "LED", "LCD", "IMG", "DSC", "IMG", "WIN", "MAC", "IOS", "TLS", "SSL"}
COURSE_CODE = re.compile(r"\b([A-Z]{2,4})[ -]?(\d{3,4})[A-Z]?\b")


def course_code_hits(t: str) -> list[str]:
    out = []
    for m in COURSE_CODE.finditer(t):
        if m.group(1) in COURSE_CODE_STOP:
            continue
        # years and plain numbers glued to caps words are not course codes
        if 1900 <= int(m.group(2)) <= 2099:
            continue
        if re.fullmatch(r"[0-9A-Fa-f]{6}", m.group(0)):  # hex colour, e.g. FFA500
            continue
        out.append(m.group(0))
    return out


# checked on the RAW turn (clean() strips URLs). Moodle quiz URL; other LMS hosts (Canvas,
# Gradescope, ...) were in the list but never occur in the corpus, so pruned 2026-08-19.
LMS_URL = re.compile(r"https?://\S*/mod/quiz", re.I)

STRONG_A = {
    "LMS / quiz artifacts": re.compile(
        r"\bgroup of answer choices\b|\b\d+ pts\b|\bflag question\b|\bnot yet answered\b|"
        r"\bmarked out of\b|\bselect one:|\bquestion text\b|\benter your response here\b|"
        r"\bquestion content area\b|\btime left \d|\bwithin \d+ minutes?\b|\bcorrect answer\(s\)|"
        r"(?:^|\n)\s*[○O•\-\*]?\s*[a-eA-E][\).:]\s+\S[^\n]{2,}\n\s*[○O•\-\*]?\s*[a-eA-E][\).:]\s+\S[^\n]{2,}\n\s*[○O•\-\*]?\s*[a-eA-E][\).:]\s+\S",
        re.I | re.M),
    "starter code / courseware": re.compile(
        r"#\s*(?:write|add) your code here|\byour code here\b|\btodo:? implement\b|"
        r"\bfill in the (?:blanks?|missing|code)\b|\bcomplete the code (?:below|above)\b|"
        r"\bdo not (?:modify|change|edit) (?:the|this)\b|\byou may not use\b|"
        r"\bwithout using (?:any )?(?:library|libraries|import)\b",
        re.I),
    "student persona": re.compile(
        r"\bstudent (?:id|number)\b",
        re.I),
    "marks/points": re.compile(
        r"\(\s*\d+(?:\.\d+)?\s*(?:marks?|points?|pts|%)\s*\)|\b\d+\s*marks?\b|\bmark(?:ing)? scheme\b|"
        r"\bworth \d+\s*(?:points|%)",
        re.I),
    "homework self-report": re.compile(
        r"\bhomework\b|\bhw\b|\bcoursework\b|\bmy (?:assignment|teacher)\b|"
        r"\bfor (?:this|the) (?:class|course|assignment|lab)\b|\b(?:this|the|an?) assignment\b|"
        r"\bpset\b|\b(?:professor|teacher)\s+(?:wants|gave)\b|\bai detect",
        re.I),
    "submission logistics": re.compile(
        r"\bdue (?:date|\w+day)\b|\bsubmit(?:ted|ting)? (?:it|this|the|your|them|by|on|to)\b|"
        r"\bsubmission\s+(?:date|instructions|format)\b|\bdeliverables?\b|\bautograder\b|"
        r"\bstarter code\b|\bskeleton code\b|\bwill be (?:graded|marked)\b|\bword count\s*of|"
        r"\b(?:at least|about|approximately|no more than|in|around|of)\s+\d{3,4}\s+words\b|"
        r"\(\s*\d{3,4}\s+words\s*\)|\bplagiaris",
        re.I),
    "teaching context": re.compile(
        r"\bteaching staff\b|\bmodule tutors?\b|\byour marker\b|\blab (?:report|exercise)s?\b|"
        r"\blecture (?:notes|\d+)\b|\bweek \d+\b|\bexam(?:ination)? questions?\b|\bquiz\b|\bmidterm\b|"
        r"\blearning outcomes?\b|\bmodule code\b",
        re.I),
}

MEDIUM_A = {
    "numbered question/task": re.compile(
        r"\b(?:question|task|exercise|problem|part|step|section)\s*(?:no\.?\s*)?(?:\d{1,2}|[ivx]{1,4})\b[\s:.)-]|"
        r"\bq\.?\s?\d{1,2}\b",
        re.I),
    "lettered sub-parts": re.compile(
        r"(?:^|\n)\s*\(?[a-e]\)\s+\S.{10,}\n\s*\(?[b-f]\)\s+\S|\bpart\s*\(?[a-e]\)",
        re.I | re.M),
    "multiple choice": re.compile(
        r"(?:^|\n)\s*(?:[A-Da-d][\).:]|\([A-Da-d]\))\s+\S[^\n]{2,}\n\s*(?:[A-Da-d][\).:]|\([A-Da-d]\))\s+\S|"
        r"\bgroup of answer choices\b|\bselect (?:one|all that apply|the correct)\b|"
        r"\bwhich of the following\b|\btrue or false\b|\bcorrect answer\b|\bchoose the correct\b",
        re.I),
    "required/compute prompt": re.compile(
        r"(?:^|\n)\s*(?:required|instructions?|deliverables?|questions?|tasks?)\s*:|"
        r"\b(?:compute|calculate|find|state|explain why)\b[^\n]{0,80}\.\s*(?:\(?[a-e]\)|\d\))",
        re.I | re.M),
    "instructor persona / directive": re.compile(
        r"\byou are (?:required|expected|asked) to\b|"
        r"\byou (?:must|should|will|need to) (?:implement|write|create|develop|submit)\b[^\n]{0,60}\b(?:program|function|script|report|solution|code)\b|"
        r"\bin this (?:task|exercise|assignment|project)\b|"
        r"\bthe (?:goal|purpose) of this (?:lab|assignment|project) is\b",
        re.I),
    "pdf / latex paste artifacts": re.compile(
        r"[\U0001D400-\U0001D7FF]{2}|\$[^$\n]{2,80}\$|\\\\frac\b| • [^\n]{5,} • |"
        r"\bfig(?:ure)?\.? \d+\b|\btable \d+\b|\beq(?:uation)?\.? ?\(?\d+\)?\b",
        re.I),
    "given-data problem": re.compile(
        r"\b(?:the following|given|consider the) (?:data|table|dataset|information|scenario|function|program|code|list|array)\b[^\n]{0,120}(?:\n[^\n]*){0,6}\b(?:compute|calculate|determine|find|what is|how many|which)\b",
        re.I),
}



def user_turns(conv: list[dict]) -> list[str]:
    return [(m.get("content") or "") for m in conv if m.get("role") == "user"]


def classify(conv: list[dict]) -> dict:
    fired: dict[str, list[tuple[int, str]]] = defaultdict(list)  # family -> [(turn, snippet)]
    for i, raw in enumerate(user_turns(conv), start=1):
        t = clean(raw)
        m = LMS_URL.search(raw)
        if m:
            fired["LMS URL"].append((i, m.group(0)[:80]))
        for fam, rx in {**STRONG_A, **MEDIUM_A}.items():
            m = rx.search(t)
            if m:
                s = t[max(0, m.start() - 30): m.end() + 40].replace("\n", " ")
                fired[fam].append((i, s.strip()))
        for cc in course_code_hits(t):
            fired["course code"].append((i, cc))
    strong = [f for f in fired if f in STRONG_A or f == "LMS URL"]
    medium = [f for f in fired if f in MEDIUM_A or f == "course code"]
    cat = "A" if (strong or len(medium) >= 2) else "C"
    turns = sorted({t for f in fired for t, _ in fired[f]}) if cat != "C" else []
    evidence = " | ".join(f"{f}: “{fired[f][0][1][:70]}”" for f in strong + medium)
    reasoning = (f"strong={strong} medium={medium}" if fired else "no signals")
    return {"category": cat, "pasted_turns": turns, "evidence": evidence[:600],
            "reasoning": reasoning, "families": dict(fired)}


def load_hashes(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(dict.fromkeys(r["conversation_hash"].strip() for r in csv.DictReader(f)
                                  if r.get("conversation_hash", "").strip()))


def labeled_corpus_hashes() -> list[str]:
    with open(CONVERSATIONS_CSV, newline="", encoding="utf-8") as f:
        return [r["conversation_hash"] for r in csv.DictReader(f)
                if r["in_corpus"] == "1" and r["labels_final"].strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None,
                    help="CSV with a conversation_hash column (default: labeled corpus from data/conversations.csv)")
    ap.add_argument("--show", choices=["A", "C"], default=None, help="print snippets for a category")
    args = ap.parse_args()

    import json
    import pyarrow.parquet as pq
    hashes = load_hashes(args.input) if args.input else labeled_corpus_hashes()
    t = pq.read_table(INPUT_PARQUET, columns=["conversation_hash", "conversation"])
    convs = dict(zip(t.column("conversation_hash").to_pylist(), t.column("conversation").to_pylist()))
    RESULTS_DIR.mkdir(exist_ok=True)
    ann_hw = annotator_hw_hashes()
    rows, results = [], {}
    for h in hashes:
        r = classify(convs[h])
        results[h] = r
        a = int(h in ann_hw)
        flag = int(r["category"] == "A" or a)
        rows.append([h, r["category"], a, flag, json.dumps(r["pasted_turns"]), r["evidence"], r["reasoning"], "", 0, 0, 0])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(OUT_HEADER); w.writerows(rows)
    cats = Counter(r["category"] for r in results.values())
    fam = Counter(f for r in results.values() for f in r["families"])
    n_flag = sum(1 for h, r in results.items() if r["category"] == "A" or h in ann_hw)
    n_ann_only = sum(1 for h, r in results.items() if r["category"] != "A" and h in ann_hw)
    print(f"{len(results)} conversations -> {dict(cats)}   wrote {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"annotator HW marks: {len(ann_hw & set(hashes))} in input ({n_ann_only} not caught by regex); hw_flag (regex A or annotator): {n_flag}")
    print("signal families firing (conversations):")
    for k, v in fam.most_common():
        print(f"  {k:<30}{v}")

    if args.show:
        print(f"\n=== category {args.show} ===")
        for h, r in results.items():
            if r["category"] == args.show:
                print(f"{h[:8]} turns={r['pasted_turns']}  {r['reasoning']}\n    {r['evidence'][:400]}")


if __name__ == "__main__":
    main()
