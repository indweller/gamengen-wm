import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from collections import deque
from torch.amp import autocast
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from dataset import LatentDataset
from latent_adversary import LatentAdversary
from reward_model import RewardModel
from config_sd import BUFFER_SIZE, WIDTH, HEIGHT
from run_inference import next_latent as get_next_latent

class DiffusionEnv(gym.Env):
    def __init__(self, 
                 unet, 
                 vae, 
                 scheduler, 
                 action_embedding, 
                 rew_model: RewardModel, 
                 adversary: LatentAdversary, 
                 dataset: LatentDataset, 
                 device='cuda',
                 frame_skip=4,
                 stack_size=4,
                 num_inference_steps=20,
                 guidance_scale=1.5,
                 use_cfg=True):
        super(DiffusionEnv, self).__init__()
        
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.action_embedding = action_embedding
        self.rew_model = rew_model
        self.adversary = adversary
        self.dataset = dataset
        self.device = device
        
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.use_cfg = use_cfg
        
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        
        self.action_space = spaces.Discrete(12)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_height = HEIGHT // self.vae_scale_factor
        self.latent_width = WIDTH // self.vae_scale_factor
        self.latent_channels = self.unet.config.in_channels // (BUFFER_SIZE + 1)

        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(HEIGHT, WIDTH, 3 * self.stack_size), 
            dtype=np.uint8
        )

        self.current_latent_stack = None 
        self.steps_taken = 0
        self.max_steps = 500
        
        self.frame_buffer = deque(maxlen=self.stack_size)
        self.action_buffer = deque(maxlen=BUFFER_SIZE)

        self.time_penalty = -0.1
        self.death_penalty = -15.0

    def reset(self, seed=0, options=None):
        """
        1. Sample real latent history + action from dataset.
        2. Perturb history to maximize 'Tension' (P(Done) ~ 0.5).
        3. Set as initial state.
        4. Clear and fill frame buffer.
        """
        self.steps_taken = 0
        self.frame_buffer.clear()
        
        self.current_latent_stack, action_real = self.get_adversarial_latent_stack(done_min=0.01, done_max=0.1)
        self.action_buffer = deque([action_real[i] for i in range(BUFFER_SIZE)], maxlen=BUFFER_SIZE)
        
        with torch.no_grad():
            # Decode only the last frame of the tension stack
            current_z = self.current_latent_stack[:, -4:, :, :, :]
            obs_frame = self._process_latents(current_z.squeeze(0), is_decode=True)
            
        # Repeat the initial frame to fill the buffer
        for i in range(self.stack_size):
            self.frame_buffer.append(obs_frame[-i-1])
            
        return self._get_stacked_obs(), {}
    
    def get_adversarial_latent_stack(self, done_min, done_max):
        while True:
            with torch.no_grad():
                img_real, action_real = self.dataset.get_random_samples(num_samples=9)
                img_real, action_real = img_real.to(self.device), action_real.to(self.device)
                z_real = self.vae.encode(img_real).latent_dist.sample()
                z_real = z_real * self.vae.config.scaling_factor
                done_prob = self.rew_model(z_real.unsqueeze(0))[1].item()
            if done_prob > done_min and done_prob < done_max:
                print("Sampled real done prob:", done_prob)
                # self.save_gif(self._process_latents(img_real), filepath=f"logs/rollouts/clean_original.gif")
                # self.save_gif(self._process_latents(z_real, is_decode=True), filepath=f"logs/rollouts/clean_decode.gif")
                break

        z_real = z_real.unsqueeze(0) # Add batch dim
        z_tension = self.adversary.get_adversarial_state(z_real)
        # self.save_gif(z_tension.squeeze(0), decode=True, filepath=f"logs/rollouts/adversarial_decode.gif")
        return z_tension, action_real

    def save_gif(self, frames, filepath="output.gif"):
        imgs = []
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            imgs.append(img)
        imageio.mimwrite(filepath, imgs, fps=5)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        self.action_buffer.append(action)
        action_tensor = torch.tensor(list(self.action_buffer), device=self.device).unsqueeze(0)

        for _ in range(self.frame_skip):
            self.steps_taken += 1

            with torch.no_grad():
                # next_latent = self._predict_next_latent(self.current_latent_stack, action_tensor)
                next_latent = get_next_latent(unet=self.unet,
                                             vae=self.vae,
                                             noise_scheduler=self.scheduler,
                                             action_embedding=self.action_embedding,
                                             context_latents=self.current_latent_stack,
                                             actions=action_tensor,
                                             device=self.device,
                                             num_inference_steps=self.num_inference_steps,
                                             do_classifier_free_guidance=self.use_cfg,
                                             guidance_scale=self.guidance_scale,
                                             skip_action_conditioning=False,)
                self.current_latent_stack = torch.cat(
                    (self.current_latent_stack[:, 1:, :], next_latent.unsqueeze(1)), dim=1
                )
                pred_reg, pred_done_prob = self.rew_model(self.current_latent_stack)            

            v = pred_reg[0, 0].item()
            c = self.time_penalty * pred_reg[0, 1].item()
            
            is_dead = False # pred_done_prob.item() > 0.5
            is_timeout = self.steps_taken >= self.max_steps
            
            d = self.death_penalty if is_dead else 0.0
            
            step_reward = v + c + d
            total_reward += step_reward

            info["tension_score"] = abs(pred_done_prob.item() - 0.5)

            if is_dead or is_timeout:
                terminated = is_dead
                truncated = is_timeout
                break

        with torch.no_grad():
            final_z = self.current_latent_stack[:, -1, :]
            new_frame = self._process_latents(final_z, is_decode=True)
        
        self.frame_buffer.append(new_frame.squeeze())

        total_reward = np.clip(total_reward, -15, 15)

        return self._get_stacked_obs(), total_reward, terminated, truncated, info

    def _process_latents(self, latent, is_decode=False):
        if is_decode:
            with torch.no_grad():
                latent = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
        obs = (latent / 2 + 0.5).clamp(0, 1)
        img = (obs.permute(0, 2, 3, 1) * 255).clip(0, 255).cpu().numpy().astype(np.uint8)
        return img

    def _get_stacked_obs(self):
        frames = list(self.frame_buffer)
        stacked = np.concatenate(frames, axis=-1)
        # self.save_gif(frames, filepath=f"logs/rollouts/current_obs_{self.steps_taken}.gif")
        return stacked