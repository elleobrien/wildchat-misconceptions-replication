"""Filter WildChat-1M for English conversations that contain Python code snippets.

Criteria:
  - conversation-level `language` == "English"
  - at least one message whose `content` contains "```python" (case-insensitive)

Writes the matching rows (full original schema) to a single parquet file.
"""
import gc
import glob
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

HERE = Path(__file__).parent
RAW_GLOB = str(HERE / "wildchat_1m_raw" / "data" / "*.parquet")
OUT = HERE / "wildchat_1m_english_python.parquet"
PATTERN = "```python"  # matched case-insensitively


def has_python(conversation):
    """True if any message content contains the python code-fence marker."""
    if not conversation:
        return False
    for msg in conversation:
        content = msg.get("content")
        if content and PATTERN in content.lower():
            return True
    return False


def main():
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise SystemExit(f"No parquet files found at {RAW_GLOB}")

    writer = None
    total_in = total_eng = total_out = 0
    BATCH_ROWS = 2000  # cap peak memory per batch
    try:
        for f in files:
            pf = pq.ParquetFile(f)
            if writer is None:
                writer = pq.ParquetWriter(OUT, pf.schema_arrow)

            file_in = file_eng = file_out = 0
            for batch in pf.iter_batches(batch_size=BATCH_ROWS):
                table = pa.Table.from_batches([batch])
                file_in += table.num_rows

                # 1) conversation-level English filter (cheap, at arrow level)
                eng = table.filter(pc.equal(table["language"], "English"))
                file_eng += eng.num_rows

                # 2) python-code filter: build a boolean mask from the conversation
                #    column only, then filter the pristine arrow table (no lossy
                #    pandas round-trip of the nested structs).
                conversations = eng.column("conversation").to_pylist()
                mask = pa.array([has_python(c) for c in conversations], type=pa.bool_())
                kept = eng.filter(mask)
                file_out += kept.num_rows

                if kept.num_rows:
                    writer.write_table(kept)

                del table, eng, conversations, mask, kept
                gc.collect()

            total_in += file_in
            total_eng += file_eng
            total_out += file_out
            print(
                f"{Path(f).name}: rows={file_in} "
                f"english={file_eng} kept={file_out}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    print("---")
    print(f"Total rows scanned:        {total_in}")
    print(f"English conversations:     {total_eng}")
    print(f"English + Python (output): {total_out}")
    print(f"Written to: {OUT}")


if __name__ == "__main__":
    main()
