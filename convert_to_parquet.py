#!/usr/bin/env python

import os
import glob
import argparse
from typing import List
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

import io
from PIL import Image

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
# Image loading (PNG bytes)
# ----------------------------

def load_image(path: str) -> bytes:
    """
    Load PNG, convert to RGB, resize, and return PNG-encoded bytes.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


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
    """
    Save one episode as a parquet file.

    frames: list of PNG bytes
    actions: list[int]
    step_id: list[int]
    """
    processed = {
        "episode_id": episode_data["episode_id"],            # global int ID
        "session_id": episode_data["session_id"],            # original session
        "episode_in_session": episode_data["episode_in_session"],
        "frames": episode_data["frames"],                    # list[bytes] (PNG data)
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
# NES action mapping
# ----------------------------

A = 128
UP = 64
LEFT = 32
B = 16
RIGHT = 4
DOWN = 2

COMPLEX_MOVEMENT = [
    ['NOOP'],           # 0
    ['right'],          # 1
    ['right', 'A'],     # 2
    ['right', 'B'],     # 3
    ['right', 'A', 'B'],# 4
    ['A'],              # 5
    ['left'],           # 6
    ['left', 'A'],      # 7
    ['left', 'B'],      # 8
    ['left', 'A', 'B'], # 9
    ['down'],           # 10
    ['up'],             # 11
]


def nes_byte_to_complex_action(action_byte: int) -> int:
    """
    Convert 8-bit NES action to COMPLEX_MOVEMENT index.

    Returns:
        int: Index into COMPLEX_MOVEMENT (0-11), or -1 if unmappable
    """
    a_pressed = bool(action_byte & A)
    b_pressed = bool(action_byte & B)
    up_pressed = bool(action_byte & UP)
    down_pressed = bool(action_byte & DOWN)
    left_pressed = bool(action_byte & LEFT)
    right_pressed = bool(action_byte & RIGHT)

    if right_pressed:
        if a_pressed and b_pressed:
            return 4  # right + A + B
        elif a_pressed:
            return 2  # right + A
        elif b_pressed:
            return 3  # right + B
        else:
            return 1  # right only
    elif left_pressed:
        if a_pressed and b_pressed:
            return 9  # left + A + B
        elif a_pressed:
            return 7  # left + A
        elif b_pressed:
            return 8  # left + B
        else:
            return 6  # left only
    elif down_pressed:
        return 10  # down
    elif up_pressed:
        return 11  # up
    elif a_pressed:
        return 5  # A (jump in place)
    else:
        return 0  # NOOP


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

    for path in tqdm(sorted(all_pngs), desc="Scanning filenames", total=len(all_pngs)):
        info = parse_filename(path)
        if info is None:
            continue

        key = (info["sessid"], info["episode"])

        episodes[key].append(
            (
                info["frame"],  # sort key
                info["path"],
                info["action"],
            )
        )

    # Assign global running counter
    global_episode_counter = 0

    for (sessid, ep_session), events in tqdm(episodes.items(), desc="Saving episodes", total=len(episodes)):

        frames = []
        actions = []

        # sort by frame index to preserve temporal order
        events_sorted = sorted(events, key=lambda e: e[0])

        for e in tqdm(events_sorted, total=len(events_sorted), desc="Loading frames", leave=False):
            png_bytes = load_image(e[1])           # <-- returns bytes
            frames.append(png_bytes)
            actions.append(nes_byte_to_complex_action(e[2]))

        steps = list(range(len(events_sorted)))

        # Global unique ID
        global_id = global_episode_counter
        global_episode_counter += 1

        episode_dict = {
            "episode_id": [global_id] * len(steps),      # global counter
            "session_id": [sessid] * len(steps),         # original session id
            "episode_in_session": [ep_session] * len(steps),
            "frames": frames,                            # list[bytes]
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

    build_and_save_episodes(
        root_dir=args.img_root, output_dir=args.out_dir
    )

    if args.push:
        create_hf_dataset_from_parquets(
            parquet_dir=args.out_dir,
            repo_id=REPO_ID,
        )


if __name__ == "__main__":
    main()
