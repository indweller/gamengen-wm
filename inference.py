"""
Inference Script: Generate frames from trained model
Supports: single frame, rollouts, action comparisons
"""

import argparse
from pathlib import Path
from typing import List
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from train_diffusion import ContextEncoder, ActionEmbedding, SimpleDiffusionModel

ACTION_NAMES = {
    0: "IDLE", 1: "UP", 2: "UP_RIGHT", 3: "RIGHT", 4: "DOWN_RIGHT",
    5: "DOWN", 6: "DOWN_LEFT", 7: "LEFT", 8: "UP_LEFT",
    9: "ATTACK", 10: "HARVEST", 11: "BUILD_HALL", 12: "BUILD_BARRACKS",
    13: "BUILD_FARM", 14: "BUILD_UNIT"
}


class DeepRTSGenerator:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(checkpoint_path)
        
    def _load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.context_encoder = ContextEncoder(buffer_size=8, out_dim=768).to(self.device)
        self.action_embed = ActionEmbedding(num_actions=15, embed_dim=768).to(self.device)
        self.unet = SimpleDiffusionModel(in_channels=3, out_channels=3, context_dim=768).to(self.device)
        self.context_encoder.load_state_dict(checkpoint["context_encoder"])
        self.action_embed.load_state_dict(checkpoint["action_embed"])
        self.unet.load_state_dict(checkpoint["unet"])
        self.context_encoder.eval(); self.action_embed.eval(); self.unet.eval()
    
    @torch.no_grad()
    def generate_frame(self, context: torch.Tensor, action: int, steps: int = 50) -> torch.Tensor:
        ctx_emb = self.context_encoder(context)
        act_emb = self.action_embed(torch.tensor([action], device=self.device)).unsqueeze(1)
        cond = ctx_emb + act_emb
        x = torch.randn(1, 3, 64, 64, device=self.device)
        for t in reversed(range(0, 1000, 1000 // steps)):
            ts = torch.tensor([t], device=self.device)
            noise_pred = self.unet(x, ts, cond)
            alpha = 1 - t / 1000
            x = (x - (1 - alpha) ** 0.5 * noise_pred) / (alpha ** 0.5)
            if t > 0:
                alpha_prev = 1 - max(0, t - 1000 // steps) / 1000
                x = (alpha_prev ** 0.5) * x + ((1 - alpha_prev) ** 0.5) * torch.randn_like(x)
        return x[0].clamp(-1, 1)
    
    def generate_rollout(self, context: torch.Tensor, actions: List[int], steps: int = 20):
        frames = []
        ctx = context.clone()
        for action in tqdm(actions, desc="Generating"):
            frame = self.generate_frame(ctx, action, steps)
            frames.append(frame)
            ctx = torch.cat([ctx[:, 1:], frame.unsqueeze(0).unsqueeze(0)], dim=1)
        return frames
    
    def to_image(self, tensor: torch.Tensor) -> Image.Image:
        arr = ((tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        return Image.fromarray(arr)
    
    def save_gif(self, frames: List[torch.Tensor], path: str, fps: int = 10, scale: int = 4):
        imgs = [self.to_image(f).resize((64*scale, 64*scale), Image.NEAREST) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)


def load_context(dataset_path: str, idx: int = 0) -> torch.Tensor:
    ep_dirs = sorted(Path(dataset_path).glob("episode_*"))
    frames_dir = ep_dirs[idx % len(ep_dirs)] / "frames"
    frames = []
    for f in sorted(frames_dir.glob("*.png"))[:8]:
        img = Image.open(f).convert("RGB").resize((64, 64))
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(frames).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="./generated")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    
    Path(args.output).mkdir(exist_ok=True)
    gen = DeepRTSGenerator(args.model)
    ctx = load_context(args.dataset).to(gen.device)
    actions = [np.random.randint(0, 15) for _ in range(args.num_frames)]
    frames = gen.generate_rollout(ctx, actions, args.steps)
    gen.save_gif(frames, f"{args.output}/rollout.gif")
    print(f"Saved to {args.output}/rollout.gif")

if __name__ == "__main__":
    main()
