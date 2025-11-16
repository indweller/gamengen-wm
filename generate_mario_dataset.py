"""
Run a trained PPO Mario agent to generate a dataset (GIF or Parquet).

Saves RGB frames from the env (render_mode=rgb_array) alongside actions and rewards.
"""

import argparse
import glob
import io
import os
import warnings

import imageio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
import torch

from datasets import load_dataset
from huggingface_hub import HfApi

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO

# Reuse env builders and model loader from training script for consistency
from train_ppo_mario import make_eval_env_mario, DEFAULT_CONFIG

warnings.filterwarnings("ignore")

ACTION_REPEAT = 1  # optional action repeat when capturing


def _render_rgb_frame(vec_env: VecEnv):
    """Unwrap VecEnv to the base Gym env and call render() to get RGB frame."""
    e = vec_env
    while hasattr(e, "venv"):
        e = e.venv
    # If it's a VecFrameStack wrapper, unwrap to the original env
    while hasattr(e, "envs") and hasattr(e.envs[0], "env"):
        e = e.envs[0].env
    return e.render()


def compress_image(
    image_array: np.ndarray, format: str = "JPEG", quality: int = 85
) -> bytes:
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def save_episode_to_parquet(episode_data: dict, output_dir: str) -> None:
    processed = {
        "episode_id": episode_data["episode_id"],
        "frames": episode_data["frames"],
        "actions": [int(a) for a in episode_data["actions"]],
        "rewards": [float(r) for r in episode_data["rewards"]],
        "step_id": [int(s) for s in episode_data["step_id"]],
    }
    df = pd.DataFrame.from_dict(processed)
    table = pa.Table.from_pandas(df)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"episode_{processed['episode_id'][0]}.parquet")
    pq.write_table(table, filename, compression="zstd")


def make_gif(agent, file_path: str, env_kwargs: dict, num_episodes: int = 1) -> None:
    env = make_eval_env_mario(env_kwargs)
    images = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = [False]
        step_i = 0
        action = None
        while not done[0]:
            if action is None or step_i % ACTION_REPEAT == 0:
                action, _ = agent.predict(obs)
            obs, _, done, _ = env.step(action)
            frame = _render_rgb_frame(env)
            if frame is not None:
                images.append(np.array(frame).copy())
            step_i += 1
    print(f"Saving GIF to {file_path} with {len(images)} frames")
    imageio.mimsave(file_path, images[:1000], fps=20)
    env.close()


def make_parquet_dataset(
    agent, output_dir: str, env_kwargs: dict, num_episodes: int = 1
) -> None:
    env = make_eval_env_mario(env_kwargs)

    for ep in tqdm(range(num_episodes), desc="Episodes"):
        obs = env.reset()
        done = False
        frames, actions, rewards, steps = [], [], [], []
        step_id = 0
        action = None

        while not bool(done):
            if action is None or step_id % ACTION_REPEAT == 0:
                action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            frame = _render_rgb_frame(env)
            if frame is not None:
                frames.append(compress_image(frame))
            actions.append(int(np.array(action).item()))
            rewards.append(float(np.array(reward).item()))
            steps.append(step_id)
            step_id += 1

        episode = {
            "frames": frames,
            "actions": actions,
            "rewards": rewards,
            "step_id": steps,
            "episode_id": [ep] * len(steps),
        }
        save_episode_to_parquet(episode, output_dir)

    env.close()


def upload_to_hf(local_path: str, repo_id: str):
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {local_path} to dataset {repo_id}")
    except Exception as e:
        print(f"Error uploading file: {e}")


def create_hf_dataset_from_parquets(parquet_dir: str, repo_id: str) -> None:
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        print(f"No parquet files found in {parquet_dir}")
        return
    dsd = load_dataset("parquet", data_files={"train": files})
    dsd.push_to_hub(repo_id, private=False)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Mario dataset or GIF from a trained PPO model"
    )
    p.add_argument(
        "--env_id", default=DEFAULT_CONFIG["env"]["env_id"], help="Mario env id"
    )
    p.add_argument("--model_path", default=None, help="Path to SB3 .zip model")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--output", choices=["gif", "parquet"], required=True)
    p.add_argument(
        "--out", default=None, help="Output file (gif) or directory (parquet)"
    )
    p.add_argument(
        "--upload", action="store_true", help="Upload to Hugging Face dataset hub"
    )
    p.add_argument("--hf_repo", default=None, help="HF dataset repo id (if uploading)")
    return p.parse_args()


def load_model(load_path: str, env: VecEnv, device: str = "cpu") -> PPO:
    """Load a trained model and move it to the specified device."""
    agent = PPO.load(load_path, env=env)
    agent.policy.to(device)
    print(f"Model loaded from {load_path} onto {device}")
    return agent


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scenario = f"mario_{args.env_id.replace('-', '_')}"

    # Environment kwargs from CONFIG
    env_kwargs = {
        "env_id": args.env_id,
        "frame_size": DEFAULT_CONFIG["env"]["frame_size"],
        "grayscale": DEFAULT_CONFIG["env"]["grayscale"],
        "frame_stack": DEFAULT_CONFIG["env"]["frame_stack"],
        "render_mode": "rgb_array",
    }

    # Load evaluation environment
    load_env = make_eval_env_mario(env_kwargs)

    # Load trained model
    model_path = args.model_path or os.path.join(
        "logs", "models", scenario, "final_model"
    )
    agent = load_model(model_path, load_env, device=str(device))
    agent.policy.to(device)

    # Run episodes
    if args.output == "gif":
        out_file = args.out or "mario_rollout.gif"
        make_gif(agent, out_file, env_kwargs, num_episodes=args.episodes)
        if args.upload and args.hf_repo:
            upload_to_hf(out_file, args.hf_repo)
    else:
        out_dir = args.out or "mario_dataset"
        make_parquet_dataset(agent, out_dir, env_kwargs, num_episodes=args.episodes)
        if args.upload and args.hf_repo:
            create_hf_dataset_from_parquets(out_dir, repo_id=args.hf_repo)


if __name__ == "__main__":
    main()
