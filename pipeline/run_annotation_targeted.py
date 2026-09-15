#!/usr/bin/env python3
"""Annotation stage: re-judge Sonnet's filter positives with a stronger model.

For each (conversation, label) pair in label_results-claude-sonnet-4-6-
targeted-full.csv with verdict == 1 (the pairs that survived both filter
stages), call the chosen model with the ANNOTATION-track prompts
(prompts_annotation/ + system_prompt_annotation.txt) and sandwich wrapping.

Output is long-format (one row per call):
  conversation_hash, label, verdict, reasoning,
  prompt_tokens, completion_tokens, cached_tokens

Resumable: pre-existing (conversation_hash, label) rows in the output CSV are
skipped on re-run. --retry-errors additionally drops rows whose verdict is
"ERR" (rewriting the CSV without them) so those pairs are re-queried.

--concurrency N runs up to N calls in parallel; rows are committed under a
write lock so the CSV stays consistent on crashes.

--dry-run prints the selection and exits before any API call.
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from openai import OpenAI
from tqdm import tqdm

from _labeling_core import (
    REPO_ROOT,
    RESULTS_DIR,
    SCRIPT_DIR,
    INPUT_PARQUET,
    build_user_msg,
    file_sha256,
    get_prompt_paths,
    parse_answer,
    read_prompt_body,
    render_conversation,
)

SONNET_CSV = RESULTS_DIR / "label_results-claude-sonnet-4-6-targeted-full.csv"
PROMPTS_DIR = SCRIPT_DIR / "prompts_annotation"
SYSTEM_PROMPT_FILE = SCRIPT_DIR / "system_prompt_annotation.txt"
KEYS_FILE = SCRIPT_DIR / "keys.json"
BASE_URL = "https://api.toolkit.umgpt.umich.edu/v1"
DEFAULT_MODEL = "claude-opus-4-7"
SCRIPT_PATH = Path(__file__)

OUT_HEADER = ["conversation_hash", "label", "verdict", "reasoning",
              "prompt_tokens", "completion_tokens", "cached_tokens"]


def load_api_key() -> str:
    return json.loads(KEYS_FILE.read_text(encoding="utf-8"))["umgpt_api_key"]


def load_positive_pairs() -> list[tuple[str, str]]:
    """(conversation_hash, label) pairs with Sonnet verdict == 1, in file order."""
    pairs = []
    with open(SONNET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["verdict"].strip() == "1":
                pairs.append((row["conversation_hash"], row["label"]))
    return pairs


def load_done_pairs(csv_path: Path, retry_errors: bool) -> set[tuple[str, str]]:
    """Pairs already present in the output CSV.

    With retry_errors, ERR rows are removed from the file (rewritten in place)
    and not counted as done, so they get re-queried and re-appended.
    """
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if retry_errors:
        keep = [r for r in rows if r["verdict"] in ("0", "1")]
        dropped = len(rows) - len(keep)
        if dropped:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=OUT_HEADER)
                w.writeheader()
                w.writerows(keep)
            print(f"--retry-errors: dropped {dropped} non-0/1 row(s) for re-query")
        rows = keep
    return {(r["conversation_hash"], r["label"]) for r in rows}


def cumulative_rollup(csv_path: Path, label_names: list[str]) -> tuple[dict, dict]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
              "calls": 0, "errors": 0}
    per_label = {L: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
                 for L in label_names}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pt = int(row["prompt_tokens"] or 0)
            ct = int(row["completion_tokens"] or 0)
            totals["prompt_tokens"] += pt
            totals["completion_tokens"] += ct
            totals["cached_tokens"] += int(row["cached_tokens"] or 0)
            totals["calls"] += 1
            if row["verdict"] == "ERR":
                totals["errors"] += 1
                continue
            if row["label"] in per_label:
                per_label[row["label"]]["prompt_tokens"] += pt
                per_label[row["label"]]["completion_tokens"] += ct
                per_label[row["label"]]["calls"] += 1
    return totals, per_label


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tag", default="anno-targeted-full",
                        help="Output filename suffix: label_results-<model>-<tag>.csv")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent UM-GPT requests via ThreadPoolExecutor.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of (conversation, label) pairs (for probes).")
    parser.add_argument("--only", type=Path, default=None, metavar="CSV",
                        help="CSV with a conversation_hash column — restrict the run to "
                             "those conversations (smoke tests against already-adjudicated "
                             "items).")
    parser.add_argument("--label", action="append", default=None, metavar="LABEL",
                        help="Restrict the run to this label (repeatable). Use when a "
                             "codebook revision only touches some codes, so the rest of "
                             "the run is not needlessly re-queried.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Drop existing ERR rows from the output CSV and re-query them.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selection and exit before calling UM-GPT.")
    args = parser.parse_args()

    prompt_pairs = get_prompt_paths(PROMPTS_DIR)
    if not prompt_pairs:
        sys.exit(f"No prompt_*.txt in {PROMPTS_DIR}.")
    label_names = [lab for lab, _ in prompt_pairs]
    label_prompt_bodies = {lab: read_prompt_body(p) for lab, p in prompt_pairs}
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    pairs = load_positive_pairs()
    unknown = sorted({lab for _, lab in pairs} - set(label_names))
    if unknown:
        sys.exit(f"Labels in {SONNET_CSV.name} without a prompt: {unknown}")
    if args.label:
        wanted = set(args.label)
        bad = sorted(wanted - set(label_names))
        if bad:
            sys.exit(f"--label {bad} not in the codebook. Known labels: {label_names}")
        pairs = [(h, lab) for h, lab in pairs if lab in wanted]
        if not pairs:
            sys.exit(f"No Sonnet-positive pairs for --label {sorted(wanted)}")
        print(f"--label: restricted to {sorted(wanted)}")
    if args.only:
        with open(args.only, newline="", encoding="utf-8") as f:
            wanted_h = {r["conversation_hash"].strip() for r in csv.DictReader(f)
                        if r.get("conversation_hash", "").strip()}
        if not wanted_h:
            sys.exit(f"--only {args.only} has no conversation_hash rows")
        pairs = [(h, lab) for h, lab in pairs if h in wanted_h]
        print(f"--only: restricted to {len(wanted_h)} conversations from {args.only.name}")
    if args.limit is not None:
        pairs = pairs[: args.limit]

    convs = {h for h, _ in pairs}
    print(f"Sonnet-positive pairs: {len(pairs)} across {len(convs)} conversations")
    print(f"Per-label: {dict(Counter(lab for _, lab in pairs).most_common())}")

    print("Loading parquet...")
    table = pq.read_table(INPUT_PARQUET, columns=["conversation_hash", "conversation"])
    hash_to_conv = dict(zip(table.column("conversation_hash").to_pylist(),
                            table.column("conversation").to_pylist()))
    missing = sorted(h for h in convs if h not in hash_to_conv)
    if missing:
        sys.exit(f"{len(missing)} conversation(s) not in {INPUT_PARQUET.name}: "
                 f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    if args.dry_run:
        print("\n--dry-run: stopping before any UM-GPT calls.")
        return

    output_csv = RESULTS_DIR / f"label_results-{args.model.replace(':', '-')}-{args.tag}.csv"
    meta_path = output_csv.with_suffix(".meta.json")

    done_pairs = load_done_pairs(output_csv, args.retry_errors)
    write_header = not output_csv.exists()
    if done_pairs:
        print(f"Resuming: {len(done_pairs)} pairs already in {output_csv.name}")

    jobs = [(h, lab) for h, lab in pairs if (h, lab) not in done_pairs]
    print(f"Calls to make this run: {len(jobs)}")
    if not jobs:
        print("Nothing to do.")
        return

    transcripts = {h: render_conversation(hash_to_conv[h]) for h in {h for h, _ in jobs}}
    client = OpenAI(api_key=load_api_key(), base_url=BASE_URL)
    write_lock = threading.Lock()

    def call_one(conv_hash: str, lab: str):
        user_msg = build_user_msg(
            label_prompt_bodies[lab], transcripts[conv_hash], label=lab, sandwich=True,
        )
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            verdict, reasoning = parse_answer(raw)
            u = resp.usage.model_dump() if resp.usage else {}
            pt = int(u.get("prompt_tokens") or 0)
            ct = int(u.get("completion_tokens") or 0)
            cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
            return conv_hash, lab, verdict, reasoning, pt, ct, cached
        except Exception as e:
            return conv_hash, lab, "ERR", str(e), 0, 0, 0

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(OUT_HEADER)
            f.flush()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(call_one, h, lab): (h, lab) for h, lab in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Calls"):
                row = fut.result()
                with write_lock:
                    writer.writerow(row)
                    f.flush()

    totals, per_label_tokens = cumulative_rollup(output_csv, label_names)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=5,
        ).stdout.strip() or None
    except Exception:
        git_commit = None
    meta = {
        "output_csv": output_csv.name,
        "model": args.model,
        "endpoint": BASE_URL,
        "stage": "annotation",
        "sandwich": True,
        "concurrency": args.concurrency,
        "selection_source": str(SONNET_CSV.relative_to(REPO_ROOT)),
        "selection_criteria": "verdict == 1 in Sonnet targeted-full (both filter stages)"
                              + (f"; restricted to label(s) {sorted(set(args.label))}"
                                 if args.label else ""),
        "label_filter": sorted(set(args.label)) if args.label else None,
        "conversation_filter": str(args.only) if args.only else None,
        "n_pairs": len(pairs),
        "n_conversations": len(convs),
        "input_parquet": str(INPUT_PARQUET.relative_to(REPO_ROOT)),
        "system_prompt_file": SYSTEM_PROMPT_FILE.name,
        "system_prompt_sha256": file_sha256(SYSTEM_PROMPT_FILE),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "prompts": {p.name: file_sha256(p) for _, p in prompt_pairs},
        "totals": totals,
        "per_label_tokens": per_label_tokens,
        "git_commit": git_commit,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {totals['calls']} total rows to {output_csv}")
    print(f"prompt_tokens (in):  {totals['prompt_tokens']:,}")
    print(f"completion_tokens:   {totals['completion_tokens']:,}")
    if totals["errors"]:
        print(f"errors: {totals['errors']} (re-run with --retry-errors)")


if __name__ == "__main__":
    main()
