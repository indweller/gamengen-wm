"""
Mario Diffusion Model Evaluation Pipeline
Downloads model from HuggingFace and runs all evaluation metrics
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
import cv2

# Import our evaluation modules
from mario_eval_metrics import MarioEvaluator
from mario_image_quality import MarioImageQualityEvaluator
from adversarial_distribution_eval import AdversarialDistributionEvaluator


class MarioDiffusionPipeline:
    """Load and run inference on the Mario diffusion model"""
    
    def __init__(self, model_id: str = "Flaaaande/mario-sd", device: str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
        
    def load_model(self):
        """Download and load model from HuggingFace"""
        from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
        from huggingface_hub import hf_hub_download, snapshot_download
        
        print(f"Loading model from {self.model_id}...")
        
        # Download full repo
        model_path = snapshot_download(repo_id=self.model_id)
        print(f"Model downloaded to: {model_path}")
        
        # Load VAE
        try:
            self.vae = AutoencoderKL.from_pretrained(
                model_path, 
                subfolder="vae"
            ).to(self.device)
            print("✓ VAE loaded")
        except Exception as e:
            print(f"VAE loading failed: {e}")
            # Try loading from root config
            self.vae = AutoencoderKL.from_pretrained(model_path).to(self.device)
        
        # Load UNet
        try:
            self.unet = UNet2DModel.from_pretrained(
                model_path,
                subfolder="unet"
            ).to(self.device)
            print("✓ UNet loaded")
        except Exception as e:
            print(f"UNet loading issue: {e}")
        
        # Load noise scheduler
        try:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                model_path,
                subfolder="noise_scheduler"
            )
            print("✓ Noise scheduler loaded")
        except Exception as e:
            print(f"Scheduler loading issue: {e}")
            # Default scheduler
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        self.model_path = model_path
        return self
    
    def generate_frame(self, num_inference_steps: int = 50, 
                       guidance_scale: float = 7.5,
                       seed: Optional[int] = None) -> np.ndarray:
        """Generate a single Mario frame"""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Get image size from model config
        sample_size = getattr(self.unet.config, 'sample_size', 64)
        in_channels = getattr(self.unet.config, 'in_channels', 4)
        
        # Start from random noise
        latents = torch.randn(
            (1, in_channels, sample_size, sample_size),
            device=self.device
        )
        
        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # Denoise
        for t in self.noise_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.unet(latents, t).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode with VAE
        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        # Convert to numpy image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def generate_batch(self, n: int, **kwargs) -> np.ndarray:
        """Generate multiple frames"""
        frames = []
        for i in range(n):
            print(f"Generating frame {i+1}/{n}...", end='\r')
            frame = self.generate_frame(seed=i, **kwargs)
            frames.append(frame)
        print()
        return np.stack(frames)


def load_original_mario_frames(path: str = None, n: int = 20) -> np.ndarray:
    """
    Load original Mario frames for comparison
    If no path provided, creates synthetic reference frames
    """
    if path and Path(path).exists():
        frames = []
        for img_path in sorted(Path(path).glob("*.png"))[:n]:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.stack(frames)
    
    # Create synthetic reference frames (replace with real data)
    print("Warning: Using synthetic reference frames. Provide real Mario frames for accurate eval.")
    frames = np.zeros((n, 240, 256, 3), dtype=np.uint8)
    for i in range(n):
        frames[i, 200:240, :] = [172, 124, 0]   # Ground
        frames[i, 0:180, :] = [92, 148, 252]    # Sky
        frames[i, 180:200, 100:120] = [56, 204, 108]  # Pipe
    return frames


def run_full_evaluation(generated: np.ndarray, original: np.ndarray):
    """Run all three evaluation categories"""
    
    print("\n" + "=" * 60)
    print("MARIO DIFFUSION MODEL EVALUATION")
    print("=" * 60)
    
    # Resize if dimensions don't match
    if generated.shape[1:3] != original.shape[1:3]:
        print(f"Resizing generated {generated.shape} to match original {original.shape}")
        resized = []
        for frame in generated:
            r = cv2.resize(frame, (original.shape[2], original.shape[1]))
            resized.append(r)
        generated = np.stack(resized)
    
    # 1. Visual Fidelity (from mario_eval_metrics.py)
    print("\n[1/3] Visual Fidelity Metrics...")
    eval1 = MarioEvaluator()
    visual_results = eval1.evaluate_visual_fidelity(original, generated)
    print(f"  PSNR: {visual_results['psnr_mean']:.2f} dB")
    print(f"  LPIPS: {visual_results.get('lpips_mean', 'N/A')}")
    
    # 2. Image Quality (from mario_image_quality.py)
    print("\n[2/3] Image Quality Metrics...")
    eval2 = MarioImageQualityEvaluator()
    quality_results = eval2.evaluate_batch(original, generated)
    print(f"  SSIM: {quality_results['ssim_mean']:.4f}")
    print(f"  Histogram Similarity: {quality_results['histogram_similarity_mean']:.4f}")
    print(f"  Edge Similarity: {quality_results['edge_similarity_mean']:.4f}")
    
    # 3. Distribution Shift (from adversarial_distribution_eval.py)
    print("\n[3/3] Distribution Shift Metrics...")
    eval3 = AdversarialDistributionEvaluator()
    dist_results = eval3.evaluate_distribution_shift(original, generated)
    print(f"  FID: {dist_results['fid']:.2f}")
    print(f"  MMD: {dist_results['mmd']:.6f}")
    print(f"  KL Divergence: {dist_results['kl_divergence_mean']:.4f}")
    print(f"  Wasserstein: {dist_results['wasserstein_mean']:.4f}")
    
    # Combined results
    all_results = {
        "visual_fidelity": visual_results,
        "image_quality": quality_results,
        "distribution": dist_results
    }
    
    return all_results


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Mario Diffusion Model")
    parser.add_argument("--model", default="Flaaaande/mario-sd", help="HuggingFace model ID")
    parser.add_argument("--original-frames", default=None, help="Path to original Mario frames")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    args = parser.parse_args()
    
    # Load model
    pipeline = MarioDiffusionPipeline(model_id=args.model)
    pipeline.load_model()
    
    # Generate frames
    print(f"\nGenerating {args.n_samples} frames...")
    generated = pipeline.generate_batch(args.n_samples, num_inference_steps=args.steps)
    print(f"Generated shape: {generated.shape}")
    
    # Save some samples
    Path("generated_samples").mkdir(exist_ok=True)
    for i, frame in enumerate(generated[:5]):
        cv2.imwrite(f"generated_samples/gen_{i}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("Saved sample frames to generated_samples/")
    
    # Load original frames
    original = load_original_mario_frames(args.original_frames, args.n_samples)
    
    # Run evaluation
    results = run_full_evaluation(generated, original)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
