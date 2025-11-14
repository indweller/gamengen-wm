import os
import sys
import time
from tqdm import tqdm
import multiprocessing
from typing import Callable, Dict, Optional
import torch
import gym
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
    VecEnv,
    VecMonitor
)
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


def build_mario_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    frame_size: int = 84,
    grayscale: bool = True,
    render_mode: Optional[str] = None,
):
    """Create a single Mario environment with common wrappers (without frame stacking).
    """
    env = make_mario(env_id, render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # Remove per-env Monitor (expects gymnasium env); we'll use VecMonitor instead on vector env.
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)  # type: ignore[arg-type]
    env = ResizeObservation(env, shape=frame_size)  # type: ignore[arg-type]

    # Final wrapper: adapt legacy gym reset signature to the (obs, info) tuple
    # expected by SB3's subprocess worker when it passes seed/options. We place
    # it last so earlier observation-transforming wrappers receive a raw array.
    class ResetCompatibilityWrapper(gym.Wrapper):
        def reset(self, seed=None, options=None, **kwargs):  # type: ignore[override]
            # Best-effort seeding for legacy envs
            if seed is not None:
                _seed_fn = getattr(self.env, 'seed', None)
                if callable(_seed_fn):
                    try:
                        _seed_fn(seed)
                    except Exception:
                        pass
            result = self.env.reset()
            # If underlying env already gives (obs, info) forward directly
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs, info = result, {}
            return obs, info
    env = ResetCompatibilityWrapper(env)
    return env


def make_vec_env_mario(
    n_envs: int,
    env_id: str = "SuperMarioBros-1-1-v0",
    frame_size: int = 84,
    grayscale: bool = True,
    frame_stack: int = 4,
    render_mode: Optional[str] = None,
) -> VecEnv:
    """Create a vectorized Mario environment suitable for SB3 CNN policies.

    Order: SubprocVecEnv -> VecTransposeImage (channel-last to channel-first) -> VecFrameStack
    """

    def _factory() -> Callable[[], gym.Env]:
        return lambda: build_mario_env(
            env_id=env_id,
            frame_size=frame_size,
            grayscale=grayscale,
            render_mode=render_mode,
        )

    vec_env = SubprocVecEnv([_factory() for _ in range(n_envs)])  # type: ignore[arg-type]
    vec_env = VecMonitor(vec_env)
    vec_env = VecTransposeImage(vec_env)  # now channels-first single frame
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")
    return vec_env


def make_eval_env_mario(
    env_id: str = "SuperMarioBros-1-1-v0",
    frame_size: int = 84,
    grayscale: bool = True,
    frame_stack: int = 4,
    render_mode: Optional[str] = None,
) -> VecEnv:
    """Single-process evaluation environment mirroring training preprocessing."""
    eval_env = DummyVecEnv([
        lambda: build_mario_env(
            env_id=env_id,
            frame_size=frame_size,
            grayscale=grayscale,
            render_mode=render_mode,
        )  # type: ignore[arg-type]
    ])  # type: ignore[arg-type]
    eval_env = VecMonitor(eval_env)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack, channels_order="first")
    return eval_env

class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.last_time = time.time()
        self.last_timesteps = 0

    def _on_step(self):
        current_timesteps = self.num_timesteps - self.last_timesteps
        self.pbar.update(current_timesteps)
        self.last_timesteps = self.num_timesteps

        current_time = time.time()
        steps_per_second = current_timesteps / (current_time - self.last_time)
        self.pbar.set_postfix({"steps/s": f"{steps_per_second:.2f}"})
        self.last_time = current_time

        return True

    def on_training_end(self):
        self.pbar.close()

def solve_env(
    env: VecEnv,
    eval_env: VecEnv,
    scenario: str,
    agent_args: Dict,
    resume: bool = False,
    load_path: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if resume:
        # Load the existing model
        if load_path:
            agent = PPO.load(
                load_path, env=env, tensorboard_log="logs/tensorboard", **agent_args
            )
            print(f"Resumed training from {load_path}")
        else:
            print("Resume selected but no valid path provided")
            sys.exit()
    else:
        # Create a new agent
        agent = PPO(
            "CnnPolicy",
            env,
            tensorboard_log="logs/tensorboard",
            seed=0,
            **agent_args,
        )
        # init_model(agent)

    agent.policy.to(device)

    # Create callbacks.
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=10_000,
        log_path=f"logs/evaluations/{scenario}",
        best_model_save_path=f"logs/models/{scenario}",
    )

    # Set up progress bar
    total_timesteps = 10_000_000
    pbar = tqdm(total=total_timesteps, desc="Training Progress")
    progress_callback = ProgressBarCallback(pbar)

    # Start the training process.
    try:
        agent.learn(
            total_timesteps=10_000_000,
            tb_log_name=scenario,
            callback=[eval_callback, progress_callback],
            reset_num_timesteps=not resume,  # Don't reset timesteps if resuming
        )
    finally:
        pbar.close()
        env.close()
        eval_env.close()

    return agent


def save_model(agent: PPO, scenario: str):
    """Save the trained model."""
    save_path = f"logs/models/{scenario}/final_model"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    agent.save(save_path)
    print(f"Model saved to {save_path}")


def load_model(load_path: str, env: VecEnv) -> PPO:
    """Load a trained model."""
    if not os.path.exists(os.path.dirname(load_path)):
        os.makedirs(os.path.dirname(load_path))
    agent = PPO.load(load_path, env=env)
    print(f"Model loaded from {load_path}")
    return agent


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment configuration
    env_id = os.environ.get("MARIO_ENV_ID", "SuperMarioBros-1-1-v0")
    frame_size = int(os.environ.get("MARIO_FRAME_SIZE", 84))
    grayscale = bool(int(os.environ.get("MARIO_GRAYSCALE", 1)))
    frame_stack = int(os.environ.get("MARIO_FRAME_STACK", 4))

    # Scenario name for logs and checkpoints
    scenario = f"mario_{env_id.replace('-', '_')}"
    
    n_envs = 4 # max(1, multiprocessing.cpu_count() - 1)

    train_env = make_vec_env_mario(
        n_envs=n_envs,
        env_id=env_id,
        frame_size=frame_size,
        grayscale=grayscale,
        frame_stack=frame_stack,
        render_mode=None,
    )
    eval_env = make_eval_env_mario(
        env_id=env_id,
        frame_size=frame_size,
        grayscale=grayscale,
        frame_stack=frame_stack,
        render_mode=None,
    )

    # PPO hyperparameters
    agent_args = {
        "n_steps": 2048,
        "learning_rate": 2.5e-4,
        "batch_size": 64,
        "policy_kwargs": {},
        "device": device,
        "clip_range": 0.1,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "n_epochs": 4,
        "gamma": 0.99,
    }

    agent = solve_env(
        train_env,
        eval_env,
        scenario,
        agent_args,
        resume=False,
    )

    save_model(agent, scenario)
