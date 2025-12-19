"""
DeepRTS GameNGen-style Diffusion Model Training

This script trains a diffusion model to predict next frames in DeepRTS,
conditioned on previous frames and actions. Based on the GameNGen paper approach.

Key components:
1. Frame conditioning: Past N frames as context
2. Action conditioning: Embedded action vectors
3. Noise augmentation: For stable autoregressive generation
"""

import argparse
import os
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
        StableDiffusionPipeline
    )
    from diffusers.optimization import get_scheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not installed. Run: pip install diffusers")


@dataclass
class TrainingConfig:
    """Training configuration"""
    pretrained_model: str = "CompVis/stable-diffusion-v1-4"
    use_pretrained: bool = True
    
    dataset_path: str = "./deeprts_dataset"
    image_size: int = 64
    buffer_size: int = 8 
    
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    
    num_actions: int = 15
    action_embed_dim: int = 64
    zero_out_action_prob: float = 0.1  
    
    max_noise_level: float = 0.7
    num_noise_buckets: int = 10
    
    output_dir: str = "./deeprts_model"
    save_every: int = 1000
    validate_every: int = 500
    

class ActionEmbedding(nn.Module):
    """
    Learnable embedding for discrete actions.
    Maps action indices to dense vectors that can condition the UNet.
    """
    
    def __init__(self, num_actions: int = 15, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, action_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_indices: (batch_size,) or (batch_size, seq_len) tensor of action indices
            
        Returns:
            (batch_size, embed_dim) or (batch_size, seq_len, embed_dim) tensor
        """
        embeds = self.embedding(action_indices)
        return self.proj(embeds)


class DeepRTSDataset(Dataset):
    """
    Dataset for DeepRTS frame-action pairs.
    
    Returns sequences of (past_frames, past_actions, target_frame) for training.
    """
    
    def __init__(
        self,
        dataset_path: str,
        buffer_size: int = 8,
        image_size: int = 64,
        max_samples: Optional[int] = None
    ):
        self.dataset_path = Path(dataset_path)
        self.buffer_size = buffer_size
        self.image_size = image_size
        
        # Index all episodes
        self.samples = []
        self._index_episodes(max_samples)
        
    def _index_episodes(self, max_samples: Optional[int]):
        """Build index of all valid training samples"""
        episode_dirs = sorted(self.dataset_path.glob("episode_*"))
        
        for episode_dir in episode_dirs:
            frames_dir = episode_dir / "frames"
            actions_path = episode_dir / "actions.txt"
            
            if not frames_dir.exists() or not actions_path.exists():
                continue
            
            with open(actions_path, 'r') as f:
                actions = [int(line.strip()) for line in f.readlines()]
            
            frame_files = sorted(frames_dir.glob("*.png"))
            num_frames = len(frame_files)
            
            for i in range(self.buffer_size, num_frames):
                self.samples.append({
                    "episode_dir": episode_dir,
                    "frame_idx": i,
                    "actions": actions,
                    "num_frames": num_frames
                })
                
                if max_samples and len(self.samples) >= max_samples:
                    return
                    
        print(f"Indexed {len(self.samples)} training samples from "
              f"{len(episode_dirs)} episodes")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_frame(self, episode_dir: Path, frame_idx: int) -> torch.Tensor:
        """Load and preprocess a single frame"""
        frame_path = episode_dir / "frames" / f"{frame_idx:06d}.png"
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  
        return tensor
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        episode_dir = sample["episode_dir"]
        frame_idx = sample["frame_idx"]
        actions = sample["actions"]
        
        context_frames = []
        context_actions = []
        
        for i in range(frame_idx - self.buffer_size, frame_idx):
            frame = self._load_frame(episode_dir, i)
            context_frames.append(frame)
            if i < len(actions):
                context_actions.append(actions[i])
            else:
                context_actions.append(0) 
        
        target_frame = self._load_frame(episode_dir, frame_idx)
        
        context_frames = torch.stack(context_frames)
        context_actions = torch.tensor(context_actions, dtype=torch.long)
        
        current_action = actions[frame_idx - 1] if frame_idx - 1 < len(actions) else 0
        
        return {
            "context_frames": context_frames,
            "context_actions": context_actions,
            "target_frame": target_frame,
            "current_action": torch.tensor(current_action, dtype=torch.long)
        }


class ContextEncoder(nn.Module):
    """
    Encodes multiple past frames into a conditioning signal.
    Simple CNN-based encoder that processes each frame and aggregates.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        buffer_size: int = 8,
        hidden_dim: int = 256,
        out_dim: int = 768  
    ):
        super().__init__()
        self.buffer_size = buffer_size
        
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim)
        )
        
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (batch, buffer_size, C, H, W)
            
        Returns:
            (batch, 1, out_dim) conditioning tensor
        """
        batch_size = frames.shape[0]
        
        frames_flat = frames.view(-1, *frames.shape[2:])  
        frame_embeds = self.frame_encoder(frames_flat)  
        frame_embeds = frame_embeds.view(batch_size, self.buffer_size, -1)
        
        attn_out, _ = self.temporal_attn(frame_embeds, frame_embeds, frame_embeds)
        
        pooled = attn_out.mean(dim=1) 
        out = self.proj(pooled) 
        
        return out.unsqueeze(1) 


class SimpleDiffusionModel(nn.Module):
    """
    Simple diffusion model for when full Stable Diffusion is not available.
    Uses a U-Net architecture directly on pixel space.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        context_dim: int = 768,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
        
        self.context_proj = nn.Linear(context_dim, base_channels * 4)
        
        self.mid = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)
        
        self.dec3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1)
        self.dec1 = nn.Conv2d(base_channels * 2, out_channels, 3, padding=1)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy input (batch, C, H, W)
            timestep: Diffusion timestep (batch,)
            context: Conditioning (batch, 1, context_dim)
            
        Returns:
            Predicted noise (batch, C, H, W)
        """
        t_emb = self.time_embed(timestep.float().unsqueeze(-1) / 1000)
        
        ctx = self.context_proj(context.squeeze(1))
        
        e1 = F.silu(self.enc1(x))
        e2 = F.silu(self.enc2(e1))
        e3 = F.silu(self.enc3(e2))
        
        m = e3 + t_emb.unsqueeze(-1).unsqueeze(-1) + ctx.unsqueeze(-1).unsqueeze(-1)
        m = F.silu(self.mid(m))
        
        d3 = F.silu(self.dec3(m))
        d2 = F.silu(self.dec2(torch.cat([d3, e2], dim=1)))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1


class DeepRTSTrainer:
    """
    Trainer for the DeepRTS diffusion model.
    Handles training loop, noise augmentation, and checkpointing.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._init_models()
        
        self._init_optimizer()
        
        self.global_step = 0
        
    def _init_models(self):
        """Initialize all model components"""
        
        self.context_encoder = ContextEncoder(
            buffer_size=self.config.buffer_size,
            out_dim=768
        ).to(self.device)
        
        self.action_embed = ActionEmbedding(
            num_actions=self.config.num_actions,
            embed_dim=768
        ).to(self.device)
        
        if DIFFUSERS_AVAILABLE and self.config.use_pretrained:
            try:
                self.vae = AutoencoderKL.from_pretrained(
                    self.config.pretrained_model,
                    subfolder="vae"
                ).to(self.device)
                self.vae.eval() 
                
                self.noise_scheduler = DDPMScheduler.from_pretrained(
                    self.config.pretrained_model,
                    subfolder="scheduler"
                )
                
                self.unet = SimpleDiffusionModel(
                    in_channels=4,  
                    out_channels=4,
                    context_dim=768
                ).to(self.device)
                
                self.use_vae = True
                print("Using VAE-based model")
                
            except Exception as e:
                print(f"Could not load pretrained model: {e}")
                print("Falling back to simple pixel-space model")
                self._init_simple_model()
        else:
            self._init_simple_model()
            
    def _init_simple_model(self):
        """Initialize simple pixel-space diffusion model"""
        self.unet = SimpleDiffusionModel(
            in_channels=3,
            out_channels=3,
            context_dim=768
        ).to(self.device)
        
        self.noise_scheduler = None
        self.vae = None
        self.use_vae = False
        print("Using simple pixel-space model")
        
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        params = list(self.context_encoder.parameters()) + \
                 list(self.action_embed.parameters()) + \
                 list(self.unet.parameters())
        
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        
    def add_noise_augmentation(
        self,
        context_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise augmentation to context frames for stable autoregressive generation.
        From GameNGen paper: sample noise level per context frame from discrete buckets.
        """
        batch_size, buffer_size = context_frames.shape[:2]
        
        noise_buckets = torch.randint(
            0, self.config.num_noise_buckets,
            (batch_size, buffer_size),
            device=context_frames.device
        )
        
        noise_levels = noise_buckets.float() / self.config.num_noise_buckets
        noise_levels = noise_levels * self.config.max_noise_level
        
        noise = torch.randn_like(context_frames)
        noise_levels_expanded = noise_levels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noisy_frames = context_frames + noise * noise_levels_expanded
        
        return noisy_frames, noise_levels
    
    def training_step(self, batch: dict) -> torch.Tensor:
        """Single training step"""
        context_frames = batch["context_frames"].to(self.device)
        context_actions = batch["context_actions"].to(self.device)
        target_frame = batch["target_frame"].to(self.device)
        current_action = batch["current_action"].to(self.device)
        
        batch_size = context_frames.shape[0]
        
        noisy_context, noise_levels = self.add_noise_augmentation(context_frames)
        
        context_embed = self.context_encoder(noisy_context)
        
        if random.random() < self.config.zero_out_action_prob:
            action_embed = torch.zeros(batch_size, 1, 768, device=self.device)
        else:
            action_embed = self.action_embed(current_action).unsqueeze(1)
        
        conditioning = context_embed + action_embed
        
        if self.use_vae:
            with torch.no_grad():
                latents = self.vae.encode(target_frame).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
        else:
            latents = target_frame
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, 1000, (batch_size,),
            device=self.device,
            dtype=torch.long
        )
        
        if self.noise_scheduler:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        else:
            alpha = (1 - timesteps.float() / 1000).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            noisy_latents = alpha * latents + (1 - alpha).sqrt() * noise
        
        noise_pred = self.unet(noisy_latents, timesteps, conditioning)
        
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def train(self, dataset: DeepRTSDataset, num_epochs: Optional[int] = None):
        """Main training loop"""
        num_epochs = num_epochs or self.config.num_epochs
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.context_encoder.train()
        self.action_embed.train()
        self.unet.train()
        
        print(f"Training on {len(dataset)} samples for {num_epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                loss = self.training_step(batch)
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                self.global_step += 1
                
                pbar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
                
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        self.save_checkpoint(final=True)
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = "final" if final else f"step_{self.global_step}"
        
        checkpoint = {
            "global_step": self.global_step,
            "context_encoder": self.context_encoder.state_dict(),
            "action_embed": self.action_embed.state_dict(),
            "unet": self.unet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }
        
        torch.save(checkpoint, output_dir / f"checkpoint_{suffix}.pt")
        print(f"Saved checkpoint: checkpoint_{suffix}.pt")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint["global_step"]
        self.context_encoder.load_state_dict(checkpoint["context_encoder"])
        self.action_embed.load_state_dict(checkpoint["action_embed"])
        self.unet.load_state_dict(checkpoint["unet"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        print(f"Loaded checkpoint from step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train DeepRTS diffusion model")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="./deeprts_model",
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=8,
                       help="Number of context frames")
    parser.add_argument("--image-size", type=int, default=64,
                       help="Image size")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of training samples")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        image_size=args.image_size
    )
    
    dataset = DeepRTSDataset(
        args.dataset,
        buffer_size=config.buffer_size,
        image_size=config.image_size,
        max_samples=args.max_samples
    )
    
    trainer = DeepRTSTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(dataset)


if __name__ == "__main__":
    main()
