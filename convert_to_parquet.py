#!/usr/bin/env python

import os
import glob
import argparse
from typing import List
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

import numpy as np
from PIL import Image

from datasets import Dataset, Features, Array3D, Value
from huggingface_hub import HfApi, create_repo

from tqdm import tqdm
import re
from collections import defaultdict

from datasets import load_dataset


# ----------------------------
# Config
# ----------------------------

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 240
DEFAULT_SHARD_SIZE = 10_000
REPO_ID = "Flaaaande/mario-png-actions"  # your HF repo


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

pattern = re.compile(
    r"^(?P<user>[^_]+)_"                # user
    r"(?P<sessid>[^_]+)_"              # sessid
    r"e(?P<episode>\d+)_"              # episode
    r"(?P<world>\d+)-(?P<level>\d+)_"  # world-level
    r"f(?P<frame>\d+)_"                # frame index
    r"a(?P<action>\d+)_"               # action
    r"(?P<datetime>[^.]+)\."           # datetime
    r"(?P<outcome>[^.]+)"              # outcome
    r"\.png$"
)

def save_episode_to_parquet(episode_data: dict, output_dir: str) -> None:
    # frames are numpy arrays -> convert to nested lists so Arrow can store them
    frames_as_lists = [f.tolist() for f in episode_data["frames"]]

    processed = {
        "episode_id": episode_data["episode_id"],            # global int ID
        "session_id": episode_data["session_id"],            # original session
        "episode_in_session": episode_data["episode_in_session"],
        "frames": frames_as_lists,                           # (H, W, 3) lists
        "actions": [int(a) for a in episode_data["actions"]],
        "step_id": [int(s) for s in episode_data["step_id"]],
    }

    df = pd.DataFrame.from_dict(processed)
    table = pa.Table.from_pandas(df)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"episode_{processed['episode_id'][0]}.parquet")
    pq.write_table(table, filename, compression="zstd")

def create_hf_dataset_from_parquets(parquet_dir: str, repo_id: str) -> None:
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        print(f"No parquet files found in {parquet_dir}")
        return
    os.makedirs("/scratch/ps5392/cache", exist_ok=True)
    dsd = load_dataset("parquet", data_files={"train": files}, cache_dir="/scratch/ps5392/cache")
    dsd.push_to_hub(repo_id, private=False)

# ----------------------------
# Build parquet shards (bounded memory)
# ----------------------------

def parse_filename(path: str):
    fname = os.path.basename(path)
    m = pattern.match(fname)
    if not m:
        return None
    g = m.groupdict()
    return {
        "sessid": g["sessid"],
        "episode": int(g["episode"]),
        "frame": int(g["frame"]),
        "action": int(g["action"]),
        "outcome": g["outcome"],
        "path": path,
    }

def build_and_save_episodes(root_dir: str, output_dir: str):
    # episodes[(session_id, episode_within_session)] = list(...)
    episodes = defaultdict(list)

    all_pngs = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    print(f"Found {len(all_pngs)} PNG files")

    for path in tqdm(all_pngs, desc="Scanning filenames"):
        info = parse_filename(path)
        if info is None:
            continue

        key = (info["sessid"], info["episode"])
        reward = 1.0 if info["outcome"] == "success" else 0.0

        episodes[key].append((
            info["frame"],     # sort key
            info["path"], 
            info["action"], 
            reward
        ))

    # Assign global running counter
    global_episode_counter = 0

    for (sessid, ep_session), events in tqdm(episodes.items(), desc="Saving episodes"):
        events.sort(key=lambda x: x[0])  # sort by frame index

        frames = [load_image(e[1]) for e in events]
        actions = [e[2] for e in events]
        steps = list(range(len(events)))

        # Global unique ID
        global_id = global_episode_counter
        global_episode_counter += 1

        episode_dict = {
            "episode_id": [global_id] * len(steps),  # global counter
            "session_id": [sessid] * len(steps),     # original session id
            "episode_in_session": [ep_session] * len(steps),
            "frames": frames,
            "actions": actions,
            "step_id": steps,
        }

        save_episode_to_parquet(episode_dict, output_dir)

    print(f"Finished! Created {global_episode_counter} episode parquet files.")


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
        "--push",
        action="store_true",
        help="If set, upload the whole out_dir folder to HF Hub.",
    )
    args = parser.parse_args()


    if args.push:
        create_hf_dataset_from_parquets(
            local_dir=args.out_dir,
            repo_id=REPO_ID,
        )


if __name__ == "__main__":
    main()
