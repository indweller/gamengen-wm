import os
import time
import glob
import re
import warnings
import argparse
import multiprocessing
import datetime
import wandb
import torch
import gym
import numpy as np
from tqdm import tqdm
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import make as make_mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from custom_extractor import MarioCNN
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
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
        "n_envs": multiprocessing.cpu_count() - 2,
    },
    "ppo": {
        "n_steps": 2048,
        "batch_size": 64,
        "learning_rate": 2.5e-4,
        "ent_coef": 0.01,
        "clip_range": 0.1,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    },
    "train": {
        "total_timesteps": 10_000_000,
        "eval_freq": 50_000,
        "eval_episodes": 5,
        "save_freq": 100_000,
        "seed": 0,
    },
    "wandb": {
        "project": "mario-ppo",
        "group": "experiment-1",
        "name": "ppo-mario-run",
        "log_interval": 1,
        "enable": True,
    },
}


# class FrameSkip(gym.Wrapper):
#     def __init__(self, env, skip=4):
#         super().__init__(env)
#         self.skip = skip

#     def step(self, action):
#         total_reward = 0.0
#         terminated = False
#         truncated = False
#         info = {}
#         for _ in range(self.skip):
#             obs, reward, terminated, truncated, info = self.env.step(action)
#             total_reward += reward
#             if terminated or truncated:
#                 break
#         return obs, total_reward, terminated, truncated, info


#### FROM STABLE BASELINES3 (library's wrapper doesn't work with old gym) ####
class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame (frameskipping)
    and return the max between the two last frames (to deal with flickering).
    Compatible with the legacy 'gym' library used by Super Mario Bros.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)

            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs

            total_reward += reward
            if terminated or truncated:
                break

        # Note: If the episode ends in the first step, the buffer might not be fully filled.
        # In that case, we just return the last valid observation we saw.
        if terminated or truncated:
            max_frame = obs
        else:
            max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def build_mario_env(env_id, frame_size, grayscale, skip=4, render_mode=None):
    env = make_mario(env_id, render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=skip)
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
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0)
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
                skip=cfg.get("skip", 4),
                render_mode=cfg.get("render_mode", None),
            )
        ]
    )
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
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


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.verbose > 0:
            print(f"Saving VecNormalize stats to {self.save_path}")
        self.training_env.save(self.save_path)
        return True


def solve_env(env, eval_env, config, resume=False, load_path=None, wandb_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_args = {**config["ppo"], "device": device}

    run = None
    if config["wandb"]["enable"]:
        run = wandb.init(
            project=config["wandb"]["project"],
            group=config["wandb"]["group"],
            name=config["wandb"]["name"],
            id=wandb_id,  # Use passed wandb_id for resume
            resume="allow" if wandb_id else None,
            config=config,
            sync_tensorboard=True,
        )
        print(f"W&B run initialized: {run.url}")
        print(f"W&B run ID: {run.id}")

    if run:
        scenario = f"mario_{run.id}"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = f"mario_{timestamp}"

    print(f"Scenario directory: {scenario}")

    # Load or create PPO
    if resume and load_path:
        agent = PPO.load(
            load_path, env=env, tensorboard_log="logs/tensorboard", **ppo_args
        )
        print(f"Resumed training from {load_path}")
    else:
        policy_kwargs = dict(
            features_extractor_class=MarioCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
        agent = PPO(
            "CnnPolicy",
            env,
            tensorboard_log="logs/tensorboard",
            seed=config["train"]["seed"],
            policy_kwargs=policy_kwargs,
            **ppo_args,
        )

    agent.policy.to(device)

    # Callbacks from config
    stats_path = os.path.join(f"logs/models/{scenario}", "vec_normalize.pkl")
    save_stats_cb = SaveVecNormalizeCallback(save_path=stats_path)
    eval_freq = max(config["train"]["eval_freq"] // config["env"]["n_envs"], 1)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=config["train"]["eval_episodes"],
        eval_freq=eval_freq,
        log_path=f"logs/evaluations/{scenario}",
        best_model_save_path=f"logs/models/{scenario}",
        callback_on_new_best=save_stats_cb,
    )

    os.makedirs(f"logs/checkpoints/{scenario}", exist_ok=True)
    save_freq = max(config["train"]["save_freq"] // config["env"]["n_envs"], 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"logs/checkpoints/{scenario}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    pbar = tqdm(total=config["train"]["total_timesteps"], desc="Training Progress")
    progress_callback = ProgressBarCallback(pbar)

    callbacks = [eval_callback, checkpoint_callback, progress_callback]

    if config["wandb"]["enable"]:
        wandb_callback = WandbCallback(
            model_save_path=None,
            verbose=2,
            gradient_save_freq=0,
            model_save_freq=0,
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

        # Save final model
        final_path = f"logs/models/{scenario}/final_model"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        agent.save(final_path)
        stats_path = os.path.join(os.path.dirname(final_path), "vec_normalize.pkl")
        train_env.save(stats_path)
        print(f"Final model saved to {final_path} and stats to {stats_path}")

        if config["wandb"]["enable"]:
            wandb.finish()

    return agent


def _find_latest_checkpoint(wandb_id=None):
    """Find latest checkpoint for a given W&B run ID."""
    if not wandb_id:
        return None, None

    scenario = f"mario_{wandb_id}"
    ckpt_dir = f"logs/checkpoints/{scenario}"
    pattern = re.compile(r"rl_model_(\d+)_steps\.zip$")
    candidates = []

    if os.path.isdir(ckpt_dir):
        for path in glob.glob(os.path.join(ckpt_dir, "*.zip")):
            m = pattern.search(os.path.basename(path))
            if m:
                candidates.append((int(m.group(1)), path))

    checkpoint_path = None
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        checkpoint_path = candidates[0][1]
    else:
        best = f"logs/models/{scenario}/best_model.zip"
        final = f"logs/models/{scenario}/final_model.zip"
        if os.path.exists(best):
            checkpoint_path = best
        elif os.path.exists(final):
            checkpoint_path = final

    return checkpoint_path, wandb_id


def _find_all_wandb_runs():
    """Find all W&B run IDs that have been used."""
    runs = []
    checkpoint_dir = "logs/checkpoints"

    if os.path.isdir(checkpoint_dir):
        for entry in os.listdir(checkpoint_dir):
            if entry.startswith("mario_"):
                wandb_id = entry.replace("mario_", "")
                runs.append(wandb_id)

    return runs


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
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_CONFIG["ppo"]["batch_size"]
    )
    parser.add_argument(
        "--eval_freq", type=int, default=DEFAULT_CONFIG["train"]["eval_freq"]
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=DEFAULT_CONFIG["train"]["eval_episodes"]
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=DEFAULT_CONFIG["train"]["save_freq"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["train"]["seed"])
    parser.add_argument("--force_fresh", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--resume_id", type=str, default=None, help="W&B run ID to resume from"
    )

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
    CONFIG["train"]["save_freq"] = args.save_freq
    CONFIG["train"]["seed"] = args.seed

    if args.no_wandb:
        CONFIG["wandb"]["enable"] = False

    env_cfg = CONFIG["env"]
    train_cfg = CONFIG["train"]
    ppo_cfg = CONFIG["ppo"]

    print("Configuration:")
    for section, params in CONFIG.items():
        print(f"  {section}:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_env = make_vec_env_mario(env_cfg)
    eval_env = make_eval_env_mario(env_cfg)

    wandb_id = args.resume_id
    resume_path = None

    if not args.force_fresh:
        if wandb_id:
            resume_path, _ = _find_latest_checkpoint(wandb_id)
        elif CONFIG["wandb"]["enable"]:
            available_runs = _find_all_wandb_runs()
            if available_runs:
                print(
                    f"\nFound {len(available_runs)} previous run(s). Use --resume_id <ID> to resume."
                )
                print("Available run IDs:")
                for run_id in available_runs[:10]:
                    print(f"  - {run_id}")
            exit(0)

    resume_flag = resume_path is not None

    if resume_flag:
        print(f"\nResuming from checkpoint: {resume_path}")
        print(f"Resuming W&B run: {wandb_id}")
    else:
        print("\nStarting fresh training run...")

    agent = solve_env(
        train_env,
        eval_env,
        CONFIG,
        resume=resume_flag,
        load_path=resume_path,
        wandb_id=wandb_id,
    )

    print(f"\nTraining complete!")
