import os
import time
import glob
import re
import warnings
import argparse
import multiprocessing
import wandb
import torch
import gym
from tqdm import tqdm
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import make as make_mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
)
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
    CheckpointCallback,
)
from wandb.integration.sb3 import WandbCallback

warnings.filterwarnings("ignore")


DEFAULT_CONFIG = {
    "env": {
        "env_id": "SuperMarioBros-1-1-v0",
        "frame_size": 84,
        "grayscale": True,
        "frame_stack": 4,
        "skip": 4,
        "n_envs": multiprocessing.cpu_count() // 2,
    },
    "ppo": {
        "n_steps": 256,
        "learning_rate": 2.5e-4,
        "batch_size": 256,
        "clip_range": 0.1,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "n_epochs": 4,
        "gamma": 0.99,
        "policy_kwargs": {},
    },
    "train": {
        "total_timesteps": 10_000_000,
        "eval_freq": 100_000,
        "eval_episodes": 5,
        "checkpoint_freq": 100_000,
        "seed": 0,
    },
    "wandb": {
        "project": "mario-ppo",
        "group": "experiment-1",
        "name": "ppo-mario-run",
        "log_interval": 10,
    }
}


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def build_mario_env(env_id, frame_size, grayscale, skip=4, render_mode=None):
    env = make_mario(env_id, render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = FrameSkip(env, skip=skip)
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=frame_size)

    # Final wrapper: adapt legacy gym reset signature to the (obs, info) tuple
    # expected by SB3's subprocess worker when it passes seed/options. We place
    # it last so earlier observation-transforming wrappers receive a raw array.
    class ResetCompatibilityWrapper(gym.Wrapper):
        def reset(self, seed=None, options=None, **kwargs):
            if seed is not None:
                _seed_fn = getattr(self.env, "seed", None)
                if callable(_seed_fn):
                    try:
                        _seed_fn(seed)
                    except Exception:
                        pass
            result = self.env.reset()
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return result, {}

    return ResetCompatibilityWrapper(env)


def make_vec_env_mario(cfg) -> VecMonitor:
    """Vectorized env from config."""

    def _factory():
        return lambda: build_mario_env(
            env_id=cfg["env_id"],
            frame_size=cfg["frame_size"],
            grayscale=cfg["grayscale"],
            skip=cfg.get("skip", 4),
            render_mode=cfg.get("render_mode", None),
        )

    vec = SubprocVecEnv([_factory() for _ in range(cfg["n_envs"])])
    vec = VecMonitor(vec)
    vec = VecTransposeImage(vec)
    vec = VecFrameStack(vec, n_stack=cfg["frame_stack"], channels_order="first")
    return vec


def make_eval_env_mario(cfg) -> VecMonitor:
    """Single env for evaluation."""
    env = DummyVecEnv(
        [
            lambda: build_mario_env(
                env_id=cfg["env_id"],
                frame_size=cfg["frame_size"],
                grayscale=cfg["grayscale"],
                render_mode=cfg.get("render_mode", None),
            )
        ]
    )
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=cfg["frame_stack"], channels_order="first")
    return env


class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.last_time = time.time()
        self.last_timesteps = 0

    def _on_step(self):
        current = self.num_timesteps - self.last_timesteps
        self.pbar.update(current)
        self.last_timesteps = self.num_timesteps

        now = time.time()
        steps_per_second = current / (now - self.last_time)
        self.pbar.set_postfix({"steps/s": f"{steps_per_second:.2f}"})
        self.last_time = now
        return True

    def on_training_end(self):
        self.pbar.close()


def solve_env(env, eval_env, scenario, config, resume=False, load_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_args = {**config["ppo"], "device": device}

    wandb_id = None
    if resume:
        ckpt_wandb_id = getattr(load_path, "wandb_id", None)
        wandb_id = ckpt_wandb_id
    
    run = wandb.init(
        project=config["wandb"]["project"],
        group=config["wandb"]["group"],
        name=config["wandb"]["name"],
        id=wandb_id,
        resume="must" if resume else None,
        config=config,
        sync_tensorboard=True,
    )

    # Load or create PPO
    if resume and load_path:
        agent = PPO.load(
            load_path, env=env, tensorboard_log=f"logs/tensorboard/{run.id}", **ppo_args
        )
        print(f"Resumed training from {load_path}")
    else:
        agent = PPO(
            "CnnPolicy",
            env,
            tensorboard_log=f"logs/tensorboard/{run.id}",
            seed=config["train"]["seed"],
            **ppo_args,
        )

    agent.policy.to(device)

    # Callbacks from config
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=config["train"]["eval_episodes"],
        eval_freq=config["train"]["eval_freq"],
        log_path=f"logs/evaluations/{scenario}",
        best_model_save_path=f"logs/models/{scenario}",
    )

    os.makedirs(f"logs/checkpoints/{scenario}", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=config["train"]["checkpoint_freq"],
        save_path=f"logs/checkpoints/{scenario}",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    pbar = tqdm(total=config["train"]["total_timesteps"], desc="Training Progress")
    progress_callback = ProgressBarCallback(pbar)

    callbacks = [eval_callback, checkpoint_callback, progress_callback]
    
    wandb_callback = WandbCallback(
        model_save_path=f"logs/checkpoints/{scenario}/wandb_models",
        verbose=2,
        gradient_save_freq=0,
        model_save_freq=config["train"]["checkpoint_freq"],
        log="all",
    )
    callbacks.append(wandb_callback)

    total_ts = config["train"]["total_timesteps"]
    remaining = total_ts - agent.num_timesteps if resume else total_ts

    try:
        agent.learn(
            total_timesteps=remaining,
            tb_log_name=scenario,
            callback=callbacks,
            reset_num_timesteps=not resume,
        )
    finally:
        pbar.close()
        env.close()
        eval_env.close()
        wandb.finish()

    return agent


def _find_latest_checkpoint(scenario):
    ckpt_dir = f"logs/checkpoints/{scenario}"
    pattern = re.compile(r"rl_model_(\d+)_steps\.zip$")
    candidates = []

    if os.path.isdir(ckpt_dir):
        for path in glob.glob(os.path.join(ckpt_dir, "*.zip")):
            m = pattern.search(os.path.basename(path))
            if m:
                candidates.append((int(m.group(1)), path))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    best = f"logs/models/{scenario}/best_model.zip"
    final = f"logs/models/{scenario}/final_model.zip"
    return best if os.path.exists(best) else final if os.path.exists(final) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- env arguments
    parser.add_argument("--env_id", type=str, default=DEFAULT_CONFIG["env"]["env_id"])
    parser.add_argument(
        "--frame_size", type=int, default=DEFAULT_CONFIG["env"]["frame_size"]
    )
    parser.add_argument(
        "--grayscale", type=int, default=int(DEFAULT_CONFIG["env"]["grayscale"])
    )
    parser.add_argument(
        "--frame_stack", type=int, default=DEFAULT_CONFIG["env"]["frame_stack"]
    )
    parser.add_argument("--n_envs", type=int, default=DEFAULT_CONFIG["env"]["n_envs"])

    # --- train arguments
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=DEFAULT_CONFIG["train"]["total_timesteps"],
    )
    parser.add_argument("--n_steps", type=int, default=DEFAULT_CONFIG["ppo"]["n_steps"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["ppo"]["batch_size"])
    parser.add_argument(
        "--eval_freq", type=int, default=DEFAULT_CONFIG["train"]["eval_freq"]
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=DEFAULT_CONFIG["train"]["eval_episodes"]
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=DEFAULT_CONFIG["train"]["checkpoint_freq"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["train"]["seed"])
    parser.add_argument("--force_fresh", action="store_true")

    args = parser.parse_args()

    # Merge config
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["env"] = {
        "env_id": args.env_id,
        "frame_size": args.frame_size,
        "grayscale": bool(args.grayscale),
        "frame_stack": args.frame_stack,
        "n_envs": args.n_envs,
    }
    CONFIG["train"]["total_timesteps"] = args.total_timesteps
    CONFIG["ppo"]["n_steps"] = args.n_steps
    CONFIG["ppo"]["batch_size"] = args.batch_size
    CONFIG["train"]["eval_freq"] = args.eval_freq
    CONFIG["train"]["eval_episodes"] = args.eval_episodes
    CONFIG["train"]["checkpoint_freq"] = args.checkpoint_freq
    CONFIG["train"]["seed"] = args.seed
    print("Configuration:")
    for section, params in CONFIG.items():
        print(f"  {section}:")
        for k, v in params.items():
            print(f"    {k}: {v}")
    env_cfg = CONFIG["env"]
    scenario = f"mario_{env_cfg['env_id'].replace('-', '_')}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build envs
    train_env = make_vec_env_mario(env_cfg)
    eval_env = make_eval_env_mario(env_cfg)

    resume_path = None if args.force_fresh else _find_latest_checkpoint(scenario)
    resume_flag = resume_path is not None

    if resume_flag:
        print(f"Resuming from checkpoint: {resume_path}")

    agent = solve_env(
        train_env,
        eval_env,
        scenario,
        CONFIG,
        resume=resume_flag,
        load_path=resume_path,
    )

    os.makedirs(f"logs/models/{scenario}", exist_ok=True)
    agent.save(f"logs/models/{scenario}/final_model")
    print(f"Model saved to logs/models/{scenario}/final_model")
