"""Download all train parquet files of allenai/WildChat-1M into a local raw dir."""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "allenai/WildChat-1M"
N_FILES = 14
RAW_DIR = Path(__file__).parent / "wildchat_1m_raw"
RAW_DIR.mkdir(exist_ok=True)

for i in range(N_FILES):
    fname = f"data/train-{i:05d}-of-{N_FILES:05d}.parquet"
    print(f"Downloading {fname} ...", flush=True)
    hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=fname,
        local_dir=str(RAW_DIR),
    )
print("DONE")
