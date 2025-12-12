import json
import os
import time
import glob
import re
import warnings
import argparse
import datetime
import wandb
import torch
import numpy as np
import cv2
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler

from stable_baselines3 import PPO
from gymnasium import spaces
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
    CheckpointCallback,
)
from wandb.integration.sb3 import WandbCallback

from custom_extractor import MarioCNN
from diffusion_env import DiffusionEnv
from reward_model import RewardModel
from latent_adversary import LatentAdversary
from dataset import LatentDataset

warnings.filterwarnings("ignore")

DEFAULT_CONFIG = {
    "env": {
        "env_id": "Mario-Diffusion-Adversarial-v0",
        "frame_size": 84,
        "frame_skip": 4,
        "frame_stack": 4,
        "dataset_path": "Flaaaande/mario-png-actions",
        "n_envs": 1,
    },
    "adversary": {
        "epsilon": 0.5,
        "alpha": 0.0,
        "steps": 0,
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
        "total_timesteps": 1_000_000,
        "eval_freq": 10_000,
        "eval_episodes": 5,
        "save_freq": 50_000,
        "seed": 0,
    },
    "wandb": {
        "project": "mario-adversarial-curriculum",
        "group": "world-model-training",
        "name": "ppo-diffusion-run",
        "enable": True,
    },
}

def load_frozen_world_model(model_folder: str, device: torch.device):
    """
    Replicates the logic from model.py to load the models.
    """
    print(f"Loading World Model from {model_folder}...")

    if not os.path.isdir(model_folder):
        print(f"'{model_folder}' looks like a Repo ID. Checking cache/downloading...")
        try:
            model_path = snapshot_download(repo_id=model_folder, revision="main", allow_patterns=["*"])
        except Exception as e:
            print(f"Could not download from Hub. Assuming '{model_folder}' is a local path.")
            model_path = model_folder
    else:
        model_path = model_folder

    print(f"Loading from local path: {model_path}")

    info_path = os.path.join(model_path, "embedding_info.json")    
    with open(info_path, "r") as f:
        embedding_info = json.load(f)

    action_embedding = torch.nn.Embedding(
        num_embeddings=embedding_info["num_embeddings"], 
        embedding_dim=embedding_info["embedding_dim"]
    ).to(device)
    
    embed_path = os.path.join(model_path, "action_embedding_model.safetensors")
    action_embedding.load_state_dict(load_file(embed_path))
    
    noise_scheduler = DDIMScheduler.from_pretrained(
        model_path, 
        subfolder="noise_scheduler", 
        local_files_only=True
    )
    
    assert (
        noise_scheduler.config.prediction_type == "v_prediction"
    ), "Noise scheduler prediction type should be 'v_prediction'"
    
    # load vae FROM ROOT
    # Removing 'subfolder="vae"' forces it to look for config.json 
    # and diffusion_pytorch_model.safetensors in 'model_path' root.
    vae = AutoencoderKL.from_pretrained(
        model_path, 
        subfolder=None,
        local_files_only=True,
        low_cpu_mem_usage=False
    )
    # load unet from different model_path
    # model_path = snapshot_download(repo_id="Flaaaande/sd-model-mario", revision="main", allow_patterns=["*"])
    unet = UNet2DConditionModel.from_pretrained(
        model_path, 
        subfolder="unet", 
        local_files_only=True,
        low_cpu_mem_usage=False
    )

    components = [unet, vae, action_embedding]
    for comp in components:
        comp.eval()
        comp.to(device)
        comp.requires_grad_(False)
        
    return unet, vae, noise_scheduler, action_embedding

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

class TensionLogCallback(BaseCallback):
    """Logs the custom 'tension_score' from the DiffusionEnv"""
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        wandb.log({"env/tension_score": infos["tension_score"]})
        return True

class MultiChannelResizeObservation(TransformObservation):
    """Custom resize wrapper that handles multi-channel observations (> 4 channels)"""
    def __init__(self, env, shape):
        self.shape = shape
        self.old_observation_space = env.observation_space
        
        def resize_fn(obs):
            """Resize obs by processing each channel separately"""
            if obs.ndim != 3:
                raise ValueError(f"Expected 3D observation (H, W, C), got shape {obs.shape}")
            
            h, w, c = obs.shape
            resized_obs = np.zeros((shape[0], shape[1], c), dtype=obs.dtype)
            
            # Resize each channel separately
            for i in range(c):
                resized_obs[:, :, i] = cv2.resize(obs[:, :, i], (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            
            return resized_obs
        
        # Create the new observation space
        new_observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(shape[0], shape[1], env.observation_space.shape[2]),
            dtype=env.observation_space.dtype,
        )
        
        super().__init__(env, resize_fn, new_observation_space)

def make_diffusion_env_factory(cfg, frozen_models, dataset, adversary, device):
    def _init():
        env = DiffusionEnv(
            unet=frozen_models['unet'],
            vae=frozen_models['vae'],
            scheduler=frozen_models['scheduler'],
            action_embedding=frozen_models['action_embedding'],
            rew_model=frozen_models['reward'],
            adversary=adversary,
            dataset=dataset,
            device=device,
            frame_skip=cfg["env"]["frame_skip"],
            stack_size=cfg["env"]["frame_stack"],
            use_cfg=cfg.get("use_cfg", True)
        )
        # env = MultiChannelResizeObservation(env, shape=(cfg["env"]["frame_size"], cfg["env"]["frame_size"]))
        return env
    return _init

def build_vec_env(cfg, frozen_models, dataset, adversary, device):
    """
    Builds the Vectorized Environment (DummyVecEnv for GPU safety).
    """
    env = DummyVecEnv([
        make_diffusion_env_factory(cfg, frozen_models, dataset, adversary, device)
        for _ in range(cfg["env"]["n_envs"])
    ])
    env = VecMonitor(env)
    # VecTransposeImage is needed if the Env outputs (H, W, C) but PPO needs (C, H, W).
    # Since DiffusionEnv outputs (C, H, W) internally, we might check if this is redundant,
    # but SB3 usually expects channel-first. DiffusionEnv output is likely Channels-First already
    # based on previous code. If so, VecTransposeImage might not be needed, but VecNormalize is.
    
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    return env

def solve_env(train_env, eval_env, config, resume=False, load_path=None, wandb_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_args = {**config["ppo"], "device": device}

    run = None
    if config["wandb"]["enable"]:
        run = wandb.init(
            project=config["wandb"]["project"],
            group=config["wandb"]["group"],
            name=config["wandb"]["name"],
            id=wandb_id,
            resume="allow" if wandb_id else None,
            config=config,
            sync_tensorboard=True,
        )
        print(f"W&B run initialized: {run.url}")
        print(f"W&B run ID: {run.id}")

    if run:
        scenario = f"mario_adv_{run.id}"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = f"mario_adv_{timestamp}"

    print(f"Scenario directory: {scenario}")

    # Load or Create PPO Agent
    if resume and load_path:
        agent = PPO.load(
            load_path, env=train_env, tensorboard_log="logs/tensorboard", **ppo_args
        )
        print(f"Resumed training from {load_path}")
    else:
        policy_kwargs = dict(
            features_extractor_class=MarioCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
        agent = PPO(
            "CnnPolicy",
            train_env,
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
    
    tension_cb = TensionLogCallback()
    pbar = tqdm(total=config["train"]["total_timesteps"], desc="Training Progress")
    progress_callback = ProgressBarCallback(pbar)

    callbacks = [eval_callback, checkpoint_callback, progress_callback, tension_cb]

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
        train_env.close()
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

    scenario = f"mario_adv_{wandb_id}"
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
            if entry.startswith("mario_adv_"):
                wandb_id = entry.replace("mario_adv_", "")
                runs.append(wandb_id)

    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_folder", type=str, required=True, help="Path to trained diffusion model folder")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_CONFIG["env"]["dataset_path"])
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_CONFIG["env"]["frame_skip"])
    
    parser.add_argument("--epsilon", type=float, default=DEFAULT_CONFIG["adversary"]["epsilon"])
    parser.add_argument("--adv_steps", type=int, default=DEFAULT_CONFIG["adversary"]["steps"])

    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_CONFIG["train"]["total_timesteps"])
    parser.add_argument("--resume_id", type=str, default=None)
    parser.add_argument("--force_fresh", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_cfg", action="store_true", help="Disable classifer free guidance for faster training")

    args = parser.parse_args()

    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["env"]["dataset_path"] = args.dataset_path
    CONFIG["env"]["frame_skip"] = args.frame_skip
    CONFIG["adversary"]["epsilon"] = args.epsilon
    CONFIG["adversary"]["steps"] = args.adv_steps
    CONFIG["train"]["total_timesteps"] = args.total_timesteps
    CONFIG["use_history_cfg"] = not args.no_cfg
    
    if args.no_wandb:
        CONFIG["wandb"]["enable"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    print(f"Loading Dataset from {CONFIG['env']['dataset_path']}...")
    dataset = LatentDataset(dataset_name=CONFIG["env"]["dataset_path"])
    
    print("Loading Frozen World Models (VAE, Diffusion, Reward)...")
    unet, vae, scheduler, action_embedding = load_frozen_world_model(args.model_folder, device)
    reward_model = RewardModel(input_dim=14400).to(device).eval()
    # reward_model_path = os.path.join(args.model_folder, "reward_model.safetensors")
    # if not os.path.exists(reward_model_path):
    #     reward_model_path = hf_hub_download(repo_id=args.model_folder, filename="reward_model.safetensors", repo_type="model")
    # reward_model.load_state_dict(load_file(reward_model_path))

    # sakshamrig/smb-mlp has the reward model wweights as pickle, so load weights accordingly
    # saved as torch.save({
    #     "reg_head": reg_head.state_dict(),
    #     "cls_head": cls_head.state_dict(),
    #     "in_dim": in_dim,
    # }, "smb_mlp.pt")
    reward_model_path = os.path.join("/home/prashanth/projects/gamengen-wm/smb_mlp.pt")
    reward_state = torch.load(reward_model_path, map_location=device)
    reward_model.reg_head.load_state_dict(reward_state["reg_head"])
    reward_model.cls_head.load_state_dict(reward_state["cls_head"])

    reward_model.eval()
    reward_model.to(device)
    for param in reward_model.parameters():
        param.requires_grad = False
            
    frozen_models = {
        "unet": unet,
        "vae": vae, 
        "scheduler": scheduler,
        "action_embedding": action_embedding,
        "reward": reward_model
    }
    
    adversary = LatentAdversary(
        rew_model=reward_model,
        epsilon=CONFIG["adversary"]["epsilon"],
        alpha=CONFIG["adversary"]["alpha"],
        num_steps=CONFIG["adversary"]["steps"]
    )


    train_env = build_vec_env(CONFIG, frozen_models, dataset, adversary, device)
    obs = train_env.reset()
    # run env with random actions and store rollout to gif
    
    eval_obs = train_env.reset()
    frames = []
    for i in range(3):
        action = train_env.action_space.sample()
        eval_obs, reward, done, info = train_env.step([action])
        print(eval_obs[0].min(), eval_obs[0].max())
        frame = eval_obs[0]
        # save first frame as img
        if i == 1:
            from PIL import Image
            img = Image.fromarray(frame[:, :, :3].astype(np.uint8))
            img.save("first_frame_adv.png")
            img = Image.fromarray(frame[:, :, 3:6].astype(np.uint8))
            img.save("second_frame_adv.png")
            img = Image.fromarray(frame[:, :, 6:9].astype(np.uint8))
            img.save("third_frame_adv.png")
            img = Image.fromarray(frame[:, :, 9:12].astype(np.uint8))
            img.save("fourth_frame_adv.png")
        # break
        frames.extend([frame[:, :, :3], frame[:, :, 3:6], frame[:, :, 6:9], frame[:, :, 9:12]])
        if done:
            break
    # save frames as gif
    import imageio
    imageio.mimwrite("test_rollout.gif", frames, fps=20)    

    exit(0)
    # Eval env uses same config but could be tweaked if needed
    eval_env = build_vec_env(CONFIG, frozen_models, dataset, adversary, device)

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
        print("\nStarting fresh Adversarial Training...")

    solve_env(
        train_env,
        eval_env,
        CONFIG,
        resume=resume_flag,
        load_path=resume_path,
        wandb_id=wandb_id,
    )

    print("\nAdversarial Training Complete!")