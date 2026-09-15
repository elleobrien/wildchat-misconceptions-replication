#!/usr/bin/env python3
"""Dump the exact assembled messages a labeling run sends to the model.

Uses the same code paths as run_labeling_umgpt.py (read_prompt_body,
build_user_msg), so what you see here is byte-for-byte what the API receives:
one file per label containing the system message and the user message
(label prompt + sandwiched transcript).

Examples:
  # Annotation-stage prompts for one conversation, all 8 labels:
  python preview_prompts.py --input ../data/wildchat_round2_test.parquet \
      --prompts-dir prompts_annotation \
      --system-prompt system_prompt_annotation.txt \
      --sandwich --hash 5d578f69

  # Just one label, printed to stdout:
  python preview_prompts.py ... --label Algebra --stdout
"""
import argparse
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from _labeling_core import (
    PROMPTS_DIR,
    SCRIPT_DIR,
    SYSTEM_PROMPT_FILE,
    INPUT_PARQUET,
    build_user_msg,
    get_prompt_paths,
    read_prompt_body,
    render_conversation,
)


def slugify(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", label.strip())
    return s.strip("_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=INPUT_PARQUET,
        help="Parquet to pull the conversation from.",
    )
    parser.add_argument(
        "--hash", default=None,
        help="conversation_hash (or unique prefix) to preview. Defaults to "
             "the first conversation in the parquet.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Preview only this label (default: all labels).",
    )
    parser.add_argument(
        "--prompts-dir", type=Path, default=PROMPTS_DIR,
        help="Per-label prompt directory (default prompts_screening/).",
    )
    parser.add_argument(
        "--system-prompt", type=Path, default=SYSTEM_PROMPT_FILE,
        help="System prompt file (default system_prompt_screening.txt).",
    )
    parser.add_argument(
        "--sandwich", action=argparse.BooleanOptionalAction, default=False,
        help="Match the runner's --sandwich setting.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=SCRIPT_DIR / "prompt_previews",
        help="Directory for assembled_<Label>.txt files.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print to stdout instead of writing files.",
    )
    args = parser.parse_args()

    table = pq.read_table(args.input, columns=["conversation_hash", "conversation"])
    hashes = table.column("conversation_hash").to_pylist()
    convs = table.column("conversation").to_pylist()

    if args.hash:
        idx = [i for i, h in enumerate(hashes) if h.startswith(args.hash)]
        if not idx:
            raise SystemExit(f"No conversation_hash starting with {args.hash!r}")
        if len(idx) > 1:
            raise SystemExit(f"Prefix {args.hash!r} matches {len(idx)} hashes")
        i = idx[0]
    else:
        i = 0
    conv_hash, transcript = hashes[i], render_conversation(convs[i])

    prompt_pairs = get_prompt_paths(args.prompts_dir)
    if args.label:
        prompt_pairs = [(lab, p) for lab, p in prompt_pairs if lab == args.label]
        if not prompt_pairs:
            raise SystemExit(f"No prompt for label {args.label!r} in {args.prompts_dir}")

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()

    if not args.stdout:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for label, path in prompt_pairs:
        user_msg = build_user_msg(
            read_prompt_body(path), transcript, label=label, sandwich=args.sandwich,
        )
        text = (
            f"# conversation_hash: {conv_hash}\n"
            f"# prompts_dir: {args.prompts_dir.name}  "
            f"system_prompt: {args.system_prompt.name}  "
            f"sandwich: {args.sandwich}\n\n"
            f"{'=' * 20} SYSTEM MESSAGE {'=' * 20}\n\n{system_prompt}\n\n"
            f"{'=' * 20} USER MESSAGE {'=' * 20}\n\n{user_msg}\n"
        )
        if args.stdout:
            sys.stdout.write(text + "\n")
        else:
            out = args.out_dir / f"assembled_{slugify(label)}.txt"
            out.write_text(text, encoding="utf-8")
            print(f"Wrote {out.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
