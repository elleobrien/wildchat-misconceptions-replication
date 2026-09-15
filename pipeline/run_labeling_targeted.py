#!/usr/bin/env python3
"""Label gpt-oss-120b positives with Sonnet (or any UM-GPT model).

For each conversation in label_results-gpt-oss-120b.csv where (a) all 8 label
cells are in {0,1} and (b) at least one cell == 1, call the chosen model ONLY
for the labels gpt-oss flagged as 1. Uses build_user_msg(sandwich=True) so
the prompt matches run_labeling_umgpt.py --sandwich.

Output is long-format (one row per call):
  conversation_hash, label, verdict, reasoning,
  prompt_tokens, completion_tokens, cached_tokens

Resumable: pre-existing (conversation_hash, label) rows in the output CSV are
read at start and skipped, so re-running picks up where the previous run died.

--concurrency N runs up to N calls in parallel via ThreadPoolExecutor. The
OpenAI SDK client is thread-safe; rows are committed under a single write
lock so the CSV stays consistent on crashes.

--n int caps the number of sampled conversations (sampled via random.Random
with --seed for repeatability). --n all uses every eligible conversation.

--exclude-meta points at a previous run's .meta.json whose sampled_hashes
should be removed from the candidate pool — useful for disjoint test sets.
"""
import argparse
import csv
import json
import random
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
    SYSTEM_PROMPT_FILE,
    INPUT_PARQUET,
    build_user_msg,
    file_sha256,
    get_prompt_paths,
    parse_answer,
    read_prompt_body,
    render_conversation,
)

GPTOSS_CSV = RESULTS_DIR / "label_results-gpt-oss-120b.csv"
KEYS_FILE = SCRIPT_DIR / "keys.json"
BASE_URL = "https://api.toolkit.umgpt.umich.edu/v1"
DEFAULT_MODEL = "claude-sonnet-4-6"
SCRIPT_PATH = Path(__file__)


def load_api_key() -> str:
    return json.loads(KEYS_FILE.read_text(encoding="utf-8"))["umgpt_api_key"]


def select_sample(label_names: list[str], n: str, seed: int, exclude: set[str]) -> list[str]:
    candidates: list[str] = []
    with open(GPTOSS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = row["conversation_hash"]
            if h in exclude:
                continue
            vals = [row[L].strip() for L in label_names]
            if all(v in ("0", "1") for v in vals) and any(v == "1" for v in vals):
                candidates.append(h)
    if n == "all":
        return sorted(candidates)
    k = int(n)
    if len(candidates) < k:
        sys.exit(f"Only {len(candidates)} eligible rows; need {k}")
    return random.Random(seed).sample(candidates, k)


def load_positive_labels(hashes: list[str], label_names: list[str]) -> dict[str, list[str]]:
    wanted = set(hashes)
    out: dict[str, list[str]] = {}
    with open(GPTOSS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["conversation_hash"] in wanted:
                out[row["conversation_hash"]] = [L for L in label_names if row[L].strip() == "1"]
    return out


def load_done_pairs(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "conversation_hash" in reader.fieldnames:
            for row in reader:
                done.add((row["conversation_hash"], row["label"]))
    return done


def cumulative_rollup(csv_path: Path, label_names: list[str]) -> tuple[dict, dict]:
    """Re-derive totals and per-label rollups from the final CSV.

    Run-loop counters reflect *this* invocation only; meta should reflect
    the entire CSV (including any resumed rows from prior runs).
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
              "calls": 0, "errors": 0}
    per_label = {L: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
                 for L in label_names}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pt = int(row["prompt_tokens"] or 0)
            ct = int(row["completion_tokens"] or 0)
            cached = int(row["cached_tokens"] or 0)
            totals["prompt_tokens"] += pt
            totals["completion_tokens"] += ct
            totals["cached_tokens"] += cached
            totals["calls"] += 1
            if row["verdict"] == "ERR":
                totals["errors"] += 1
                continue
            lab = row["label"]
            if lab in per_label:
                per_label[lab]["prompt_tokens"] += pt
                per_label[lab]["completion_tokens"] += ct
                per_label[lab]["calls"] += 1
    return totals, per_label


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", default="all",
                        help='Cap sample size (int) or "all" (default).')
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed when --n is an int.")
    parser.add_argument("--tag", required=True,
                        help="Output filename suffix: label_results-<model>-<tag>.csv")
    parser.add_argument("--exclude-meta", type=Path, default=None,
                        help="Previous .meta.json; its sampled_hashes are excluded.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent UM-GPT requests via ThreadPoolExecutor.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selection and exit before calling UM-GPT.")
    args = parser.parse_args()

    prompt_pairs = get_prompt_paths()
    label_names = [lab for lab, _ in prompt_pairs]
    label_prompt_bodies = {lab: read_prompt_body(p) for lab, p in prompt_pairs}
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    exclude: set[str] = set()
    if args.exclude_meta:
        m = json.loads(args.exclude_meta.read_text(encoding="utf-8"))
        exclude = set(m.get("sampled_hashes") or [])
        if not exclude:
            sys.exit(f"{args.exclude_meta} has no sampled_hashes")
        print(f"Excluding {len(exclude)} hashes from {args.exclude_meta.name}")

    hashes = select_sample(label_names, args.n, args.seed, exclude)
    print(f"Selected {len(hashes)} conversations (n={args.n}, seed={args.seed})")

    positives = load_positive_labels(hashes, label_names)
    total_calls = sum(len(positives[h]) for h in hashes)
    label_call_counts = Counter(L for h in hashes for L in positives[h])
    calls_per_conv = Counter(len(positives[h]) for h in hashes)
    print(f"Calls-per-conv distribution: {dict(sorted(calls_per_conv.items()))}")
    print(f"Per-label call counts: {dict(label_call_counts.most_common())}")
    print(f"Total calls if no resume: {total_calls}")

    print("\nLoading parquet...")
    table = pq.read_table(INPUT_PARQUET)
    all_h = table.column("conversation_hash").to_pylist()
    all_c = table.column("conversation").to_pylist()
    hash_to_conv = {h: c for h, c in zip(all_h, all_c)}
    missing = [h for h in hashes if h not in hash_to_conv]
    if missing:
        sys.exit(f"{len(missing)} sampled hashes not in parquet")

    if args.dry_run:
        print("\n--dry-run: stopping before any UM-GPT calls.")
        return

    output_csv = RESULTS_DIR / f"label_results-{args.model.replace(':','-')}-{args.tag}.csv"
    meta_path = output_csv.with_suffix(".meta.json")

    done_pairs = load_done_pairs(output_csv)
    write_header = len(done_pairs) == 0
    mode = "w" if write_header else "a"
    if done_pairs:
        print(f"Resuming: {len(done_pairs)} (hash, label) pairs already in {output_csv.name}")

    jobs = []
    for h in hashes:
        for lab in positives[h]:
            if (h, lab) not in done_pairs:
                jobs.append((h, lab))
    print(f"Calls to make this run: {len(jobs)}")
    if not jobs:
        print("Nothing to do.")
        return

    transcripts = {h: render_conversation(hash_to_conv[h]) for h in {h for h, _ in jobs}}
    client = OpenAI(api_key=load_api_key(), base_url=BASE_URL)
    write_lock = threading.Lock()

    def call_one(conv_hash: str, lab: str):
        transcript = transcripts[conv_hash]
        user_msg = build_user_msg(label_prompt_bodies[lab], transcript, label=lab, sandwich=True)
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

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["conversation_hash", "label", "verdict", "reasoning",
                             "prompt_tokens", "completion_tokens", "cached_tokens"])
            f.flush()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(call_one, h, lab): (h, lab) for h, lab in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Calls"):
                row = fut.result()
                with write_lock:
                    writer.writerow(row)
                    f.flush()

    # Sidecar metadata: re-derive totals from final CSV so they reflect the
    # entire run (including any rows kept across a resume).
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
        "sandwich": True,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "n_arg": args.n,
        "n_conversations": len(hashes),
        "selection_source": str(GPTOSS_CSV.relative_to(REPO_ROOT)),
        "selection_criteria": "all 8 label cells in {0,1} AND >=1 label == 1"
                              + (f"; excluding {args.exclude_meta.name}" if args.exclude_meta else ""),
        "excluded_meta": args.exclude_meta.name if args.exclude_meta else None,
        "input_parquet": str(INPUT_PARQUET.relative_to(REPO_ROOT)),
        "system_prompt_sha256": file_sha256(SYSTEM_PROMPT_FILE),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "prompts": {p.name: file_sha256(p) for _, p in prompt_pairs},
        "sampled_hashes": hashes,
        "totals": totals,
        "per_label_tokens": per_label_tokens,
        "git_commit": git_commit,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {totals['calls']} total rows to {output_csv}")
    print(f"prompt_tokens (in):  {totals['prompt_tokens']:,}")
    print(f"completion_tokens:   {totals['completion_tokens']:,}")
    if totals["errors"]:
        print(f"errors: {totals['errors']}")


if __name__ == "__main__":
    main()
