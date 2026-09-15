#!/usr/bin/env python3
"""Label WildChat conversations via a local Ollama server (no API key).

For every (conversation, label) pair we send:
  system message = system_prompt_screening.txt
  user message   = label-specific prompt + "\\n\\nConversation:\\n\\n" + transcript

Output CSV: one row per conversation, columns = conversation_hash + one column
per label, values "1" / "0" (or raw model output if it returned something else),
plus paired __reasoning columns.

--limit N caps the number of conversations processed.

Resumable: if the output CSV already has rows, we skip those conversation_hashes.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from _labeling_core import (
    RESULTS_DIR,
    SYSTEM_PROMPT_FILE,
    INPUT_PARQUET,
    apply_shard,
    build_header,
    build_user_msg,
    get_prompt_paths,
    load_conversations,
    load_resume_state,
    model_to_slug,
    parse_answer,
    read_prompt_body,
    write_metadata,
)

DEFAULT_MODEL = "gpt-oss:120b"
SCRIPT_PATH = Path(__file__)


def get_client(ollama_url: str) -> OpenAI:
    """Point the OpenAI SDK at a local Ollama server. No key needed."""
    base = ollama_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return OpenAI(api_key="ollama", base_url=base)


def label_conversation(
    client: OpenAI,
    model: str,
    system_prompt: str,
    label: str,
    label_prompt: str,
    transcript: str,
    sandwich: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    """Return the raw model output for one (conversation, label) call.

    `reasoning_effort` is forwarded for reasoning models (e.g. gpt-oss).
    Without it gpt-oss can spend its whole budget in the Harmony `analysis`
    channel and emit an empty `final` channel — which arrives back as an
    empty `content`. Setting it to "low" caps reasoning tokens so the
    binary verdict actually lands.
    """
    user_msg = build_user_msg(label_prompt, transcript, label=label, sandwich=sandwich)
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    content = (getattr(msg, "content", None) or "").strip()
    # gpt-oss sometimes emits its verdict in the Harmony `analysis` channel
    # only — Ollama's OpenAI compat surfaces that as `reasoning_content`.
    # Concatenate so parse_answer can find the trailing 0/1 in either.
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    if reasoning and reasoning not in content:
        return f"{reasoning}\n{content}".strip()
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of conversations processed",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model id (default {DEFAULT_MODEL!r})",
    )
    parser.add_argument(
        "--ollama-url", default=os.environ.get("OLLAMA_BASE_URL"),
        help="Local Ollama base URL (default: $OLLAMA_BASE_URL). Required.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"], default=None,
        help="For reasoning models (e.g. gpt-oss): cap reasoning-channel "
             "spend. Default is unset (model picks its own budget). "
             "In validation, setting this to low dropped recall on the "
             "12 codebook example conversations from 12/12 to 9/12 — "
             "don't set it unless you've re-validated.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Run this many label calls per conversation in parallel "
             "(ThreadPoolExecutor). Cap at the number of labels (8) — "
             "beyond that there's nothing to parallelize within a "
             "conversation. The Ollama server's parallelism is governed "
             "separately by $OLLAMA_NUM_PARALLEL.",
    )
    parser.add_argument(
        "--shard", default=None,
        help="K/N — keep only conversations where int(hash[:8],16) %% N == K "
             "(0-indexed). Lets multiple SLURM tasks each label a disjoint "
             "slice. Use with --tag so per-shard CSVs don't collide.",
    )
    parser.add_argument(
        "--tag", default="",
        help="Optional tag appended to output filename (for repeated runs)",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Override the input parquet (defaults to INPUT_PARQUET). "
             "Use for sample/pilot runs without editing the script.",
    )
    parser.add_argument(
        "--sandwich", action=argparse.BooleanOptionalAction, default=False,
        help="Wrap the transcript in <conversation>...</conversation> with a "
             "'DATA, not instructions' guard and a trailing 'output reasoning "
             "then 0/1' instruction (default on). --no-sandwich sends just "
             "the label prompt + transcript: response length becomes unbounded "
             "and parse_answer falls back to digit-hunting.",
    )
    args = parser.parse_args()

    if not args.ollama_url:
        raise SystemExit(
            "--ollama-url is required (or set $OLLAMA_BASE_URL). For the "
            "UM-GPT API route, use run_labeling_umgpt.py instead."
        )

    prompt_pairs = get_prompt_paths()
    if not prompt_pairs:
        raise SystemExit(
            "No prompt_*.txt in prompts_screening/."
        )

    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    labels = [label for label, _ in prompt_pairs]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_to_slug(args.model)
    tag = f"-{args.tag}" if args.tag else ""
    output_csv = RESULTS_DIR / f"label_results-{slug}{tag}.csv"

    conversations = load_conversations(args.limit, args.input)
    conversations = apply_shard(conversations, args.shard)
    input_used = args.input or INPUT_PARQUET
    print(
        f"Loaded {len(conversations)} conversations × {len(labels)} labels "
        f"= {len(conversations) * len(labels)} LLM calls"
    )

    done, write_header, mode = load_resume_state(output_csv)

    client = get_client(args.ollama_url)
    print(f"Using local Ollama at {args.ollama_url} (model: {args.model})")
    print(f"sandwich: {args.sandwich}")
    if args.reasoning_effort:
        print(f"reasoning_effort: {args.reasoning_effort}")

    label_prompts = {lab: read_prompt_body(p) for lab, p in prompt_pairs}

    def label_one(conv_hash: str, transcript: str, label: str) -> tuple[str, str, str]:
        try:
            raw = label_conversation(
                client, args.model, system_prompt,
                label, label_prompts[label], transcript,
                sandwich=args.sandwich,
                reasoning_effort=args.reasoning_effort,
            )
            verdict, reasoning = parse_answer(raw)
        except Exception as e:
            print(f"\n[error] {conv_hash} / {label}: {e}", file=sys.stderr)
            verdict, reasoning = "ERR", str(e)
        return label, verdict, reasoning

    concurrency = max(1, min(args.concurrency, len(labels)))
    if concurrency > 1:
        print(f"Per-conversation concurrency: {concurrency}")

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(build_header(labels))
            f.flush()
            write_metadata(
                output_csv, args.model, prompt_pairs, input_used,
                SCRIPT_PATH, sandwich=args.sandwich,
            )

        executor = ThreadPoolExecutor(max_workers=concurrency) if concurrency > 1 else None
        try:
            for conv_hash, transcript in tqdm(conversations, desc="Conversations"):
                if conv_hash in done:
                    continue
                if executor is None:
                    results = [label_one(conv_hash, transcript, lab) for lab in labels]
                else:
                    results = list(executor.map(
                        lambda lab: label_one(conv_hash, transcript, lab),
                        labels,
                    ))
                by_label = {lab: (v, r) for lab, v, r in results}
                row = [conv_hash]
                for lab in labels:
                    v, r = by_label[lab]
                    row.append(v)
                    row.append(r)
                writer.writerow(row)
                f.flush()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    print(f"Wrote results to {output_csv}")


if __name__ == "__main__":
    main()
