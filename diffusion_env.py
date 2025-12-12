import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from collections import deque
from torch.amp import autocast
from diffusers.utils.torch_utils import randn_tensor
from dataset import LatentDataset
from latent_adversary import LatentAdversary
from reward_model import RewardModel
from config_sd import BUFFER_SIZE, WIDTH, HEIGHT

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
        
        # dataset[i] should return (z_stack, action_expert)
        # idx = np.random.randint(len(self.dataset))
        # z_real, a_real = self.dataset[idx]
        while True:
            pixel_real = self.dataset.get_random_samples(num_samples=9)
            from PIL import Image
            img = Image.fromarray(((pixel_real.permute(0, 2, 3, 1)[-1].cpu().numpy() / 2 + 0.5) * 255).clip(0, 255).astype(np.uint8))
            img.save(f"real_obs_{-1}.png")
            with torch.no_grad():
                z_real = self.vae.encode(
                    pixel_real.to(device=self.device, dtype=torch.float32)
                ).latent_dist.sample()
                z_real = z_real * self.vae.config.scaling_factor
            if F.sigmoid(self.rew_model(z_real[-4:, : , : , :].flatten())[1]).item() > 0.1 and F.sigmoid(self.rew_model(z_real[-4:, : , : , :].flatten())[1]).item() < 0.2:
                decoded_image = self.vae.decode(z_real / self.vae.config.scaling_factor, return_dict=False)[0]
                decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
                img = Image.fromarray((decoded_image.permute(0, 2, 3, 1)[-1].cpu().numpy()* 255).clip(0, 255).astype(np.uint8))
                img.save(f"real_latent_decode_{-1}.png")
                break

        z_real = z_real.unsqueeze(0).to(self.device) # Add batch dim
        z_tension = self.adversary.generate_tension_state(z_real[:,-4:, : , :].flatten(1, -1))
        # save tension decoded image
        with torch.no_grad():
            frames = []
            for i in range(4):
                decoded_image = self.vae.decode(z_tension.view(1, 4, z_real.shape[2], z_real.shape[3], z_real.shape[4])[:, i, :, :, :] / self.vae.config.scaling_factor, return_dict=False)[0]
                decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
                frames.append((decoded_image.permute(0, 2, 3, 1)[-1].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            gif_path = "tension_latent_decode.gif"
            imageio.mimwrite(gif_path, frames, fps=5)
            print(f"Saved tension latent decode gif at: {gif_path}")
        img = Image.fromarray((decoded_image.permute(0, 2, 3, 1)[-1].cpu().numpy()* 255).clip(0, 255).astype(np.uint8))
        img.save(f"tension_obs_{-1}.png")
        # exit(0)
        self.current_latent_stack = z_tension.view(
            1, 4, z_real.shape[2], z_real.shape[3], z_real.shape[4]
        )[:,-1, : , :].repeat(1, BUFFER_SIZE, 1, 1, 1)
        
        with torch.no_grad():
            # Decode only the last frame of the tension stack
            current_z = self.current_latent_stack[:, -1, :, :, :] 
            initial_frame = self._decode_latents(current_z)
            obs_frame = self._process_obs(initial_frame)
            
        # Repeat the initial frame to fill the buffer
        for _ in range(self.stack_size):
            self.frame_buffer.append(obs_frame)
            
        return self._get_stacked_obs(), {}

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        action_tensor = torch.tensor([action], device=self.device)

        for _ in range(self.frame_skip):
            self.steps_taken += 1

            with torch.no_grad():
                print("Current latent stack min/max:", self.current_latent_stack.min().item(), self.current_latent_stack.max().item())
                next_latent = self._predict_next_latent(self.current_latent_stack, action_tensor)
                print("Next latent min/max:", next_latent.min().item(), next_latent.max().item())               
                self.current_latent_stack = torch.cat(
                    (self.current_latent_stack[:, 1:, :], next_latent.unsqueeze(1)), dim=1
                )
                pred_reg, pred_done_prob = self.rew_model(self.current_latent_stack[:, -4:, :, :, :].flatten(1, -1))            

            v = pred_reg[0, 0].item()
            c = self.time_penalty * pred_reg[0, 1].item()
            
            is_dead = pred_done_prob.item() > 0.5
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
            pixel_obs = self._decode_latents(final_z)
        
        new_frame = self._process_obs(pixel_obs)
        self.frame_buffer.append(new_frame)

        total_reward = np.clip(total_reward, -15, 15)

        return self._get_stacked_obs(), total_reward, terminated, truncated, info

    def _predict_next_latent(self, context_latents, action):
        batch_size = context_latents.shape[0]
        
        shape = (
            batch_size, 
            self.latent_channels, 
            self.latent_height, 
            self.latent_width
        )
        latents = randn_tensor(shape, generator=None, device=self.device, dtype=self.unet.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        if self.use_cfg:
            encoder_hidden_states = self.action_embedding(action).repeat(2, 1, 1)
        else:
            encoder_hidden_states = self.action_embedding(action)

        # We concatenate along the sequence dimension (dim 1) -> (B, Buffer+1, C, H, W)
        latents = torch.cat([context_latents, latents.unsqueeze(1)], dim=1)

        # Fold the frames into the channel dimension -> (B, (Buffer+1)*C, H, W)
        latents = latents.view(batch_size, -1, self.latent_height, self.latent_width)

        for _, t in enumerate(timesteps):
            print(t.item())
            if self.use_cfg:
                # Clone the fused latents for the unconditional branch
                uncond_latents = latents.clone()
                
                # Zero out the history part (First BUFFER_SIZE frames)
                # We have to reshape briefly to access the sequence dimension
                # uncond_latents_reshaped = uncond_latents.view(
                #     batch_size, BUFFER_SIZE + 1, self.latent_channels, self.latent_height, self.latent_width
                # )
                # uncond_latents_reshaped[:, :BUFFER_SIZE] = torch.zeros_like(
                #     uncond_latents_reshaped[:, :BUFFER_SIZE]
                # )
                # uncond_latents = uncond_latents_reshaped.view(batch_size, -1, self.latent_height, self.latent_width)
                uncond_latents[:, :BUFFER_SIZE] = torch.zeros_like(
                    uncond_latents[:, :BUFFER_SIZE]
                )

                # Concatenate [Uncond, Cond]
                latent_model_input = torch.cat([uncond_latents, latents])
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=None,
                class_labels=torch.zeros(latent_model_input.shape[0], dtype=torch.long, device=self.device),
                return_dict=False,
            )[0]

            if self.use_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # --- Denoise ONLY the last frame ---
            reshaped_frames = latents.reshape(
                batch_size, BUFFER_SIZE + 1, self.latent_channels, self.latent_height, self.latent_width
            )
            last_frame = reshaped_frames[:, -1]
            denoised_last_frame = self.scheduler.step(
                noise_pred, t, last_frame, return_dict=False
            )[0]
            reshaped_frames[:, -1] = denoised_last_frame
            latents = reshaped_frames.reshape(batch_size, -1, self.latent_height, self.latent_width)

        # Return only the target frame
        reshaped_frames = latents.reshape(
            batch_size, BUFFER_SIZE + 1, self.latent_channels, self.latent_height, self.latent_width
        )
        print(reshaped_frames[:, -1].min().item(), reshaped_frames[:, -1].max().item())
        exit(0)
        return reshaped_frames[:, -1]

    def _decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def _process_obs(self, tensor):
        print(tensor[:10])
        obs = tensor.squeeze(0).cpu().numpy()
        img = (obs * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image
        img_pil = Image.fromarray(img.transpose(1, 2, 0))
        # save
        img_pil.save(f"obs_{self.steps_taken}.png")
        return img

    def _get_stacked_obs(self):
        frames = np.concatenate(list(self.frame_buffer), axis=0).transpose(1, 2, 0)
        return frames