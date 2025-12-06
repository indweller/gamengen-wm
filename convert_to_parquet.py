#!/usr/bin/env python

import os
import glob
import argparse
from typing import List

import numpy as np
from PIL import Image

from datasets import Dataset, Features, Array3D, Value
from huggingface_hub import HfApi, create_repo

from tqdm import tqdm


# ----------------------------
# Config
# ----------------------------

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 240
DEFAULT_SHARD_SIZE = 10_000
REPO_ID = "Flaaaande/mario-png"  # your HF repo


# ----------------------------
# Image loading
# ----------------------------

def load_image(path: str) -> np.ndarray:
    """Load PNG, convert to RGB, resize, return uint8 array (H, W, 3)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BICUBIC)
        arr = np.asarray(img, dtype="uint8")
    return arr


FEATURES = Features({
    "image": Array3D(
        dtype="uint8",
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    ),
    "path": Value("string"),
})


# ----------------------------
# Build parquet shards (bounded memory)
# ----------------------------

def build_parquet_shards(
    img_root: str,
    out_dir: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> List[str]:
    """
    Scan img_root for PNGs, build parquet shards with at most `shard_size`
    images each. Returns the list of parquet file paths.
    """
    image_paths = sorted(
        glob.glob(os.path.join(img_root, "**", "*.png"), recursive=True)
    )
    print(f"Found {len(image_paths)} PNG images under {img_root}")

    os.makedirs(out_dir, exist_ok=True)

    shard_idx = 0
    records = []
    parquet_files: List[str] = []

    for i, p in tqdm(enumerate(image_paths, start=1), total=len(image_paths), desc="Processing images"):
        arr = load_image(p)
        records.append({"image": arr, "path": p})

        # when we hit shard_size, flush to parquet
        if len(records) >= shard_size:
            shard_path = os.path.join(
                out_dir, f"mario_png_shard_{shard_idx:05d}.parquet"
            )
            ds = Dataset.from_list(records, features=FEATURES)
            ds.to_parquet(shard_path)
            parquet_files.append(shard_path)

            print(f"[Shard {shard_idx}] Saved {len(records)} images -> {shard_path}")
            records = []
            shard_idx += 1

    # final partial shard
    if records:
        shard_path = os.path.join(
            out_dir, f"mario_png_shard_{shard_idx:05d}.parquet"
        )
        ds = Dataset.from_list(records, features=FEATURES)
        ds.to_parquet(shard_path)
        parquet_files.append(shard_path)
        print(f"[Shard {shard_idx}] Saved FINAL {len(records)} images -> {shard_path}")

    print(f"Done. Wrote {len(parquet_files)} parquet shards to {out_dir}")
    return parquet_files


# ----------------------------
# Upload all shards to HF Hub
# ----------------------------

def upload_parquet_folder_to_hub(
    local_dir: str,
    repo_id: str = REPO_ID,
    repo_type: str = "dataset",
):
    """
    Upload the entire local_dir (all parquet files, plus any metadata) to HF Hub.
    HF does NOT care that shards are 10k each; this is just a flat folder upload.
    """
    api = HfApi()

    # Assumes you already did `huggingface-cli login` or have HF_TOKEN set.
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    print(f"Uploading folder {local_dir} to HF Hub repo {repo_id}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Upload complete.")


# ----------------------------
# CLI entrypoint
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Shard PNG images into parquet files and upload to HF Hub."
    )
    parser.add_argument(
        "--img_root",
        type=str,
        required=True,
        help="Root directory containing PNG images (searched recursively).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="png_dataset_parquet_shards",
        help="Directory to store parquet shards.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of images per local parquet shard (for memory control).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="If set, upload the whole out_dir folder to HF Hub.",
    )
    args = parser.parse_args()

    parquet_files = build_parquet_shards(
        img_root=args.img_root,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
    )
    print(f"Generated {len(parquet_files)} parquet shard(s).")

    if args.push:
        upload_parquet_folder_to_hub(
            local_dir=args.out_dir,
            repo_id=REPO_ID,
        )


if __name__ == "__main__":
    main()
