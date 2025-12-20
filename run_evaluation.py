"""
Mario Diffusion Model Evaluation Pipeline
Uses run_autoregressive.py and run_inference.py for generation
"""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "diffusers", "transformers", "accelerate",
    "datasets", "huggingface_hub", "safetensors",
    "opencv-python-headless", "scipy", "lpips", "pillow", "tqdm"
])

sys.path.insert(0, '/content/MarioGPT/GameNGen')
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
import cv2
import os
import random
from PIL import Image
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor

from mario_eval_metrics import MarioEvaluator
from mario_image_quality import MarioImageQualityEvaluator
from adversarial_distribution_eval import AdversarialDistributionEvaluator
from config_sd import BUFFER_SIZE, CFG_GUIDANCE_SCALE, TRAINING_DATASET_DICT, DEFAULT_NUM_INFERENCE_STEPS
from dataset import EpisodeDataset, collate_fn, get_single_batch
from model import load_model
from run_inference import (
    decode_and_postprocess,
    encode_conditioning_frames,
    next_latent,
    run_inference_img_conditioning_with_params,
)

class MarioGenerationPipeline:
    """Generate frames using the existing model and inference code"""
    
    def __init__(self, model_folder: str, device: str = None):
        self.model_folder = model_folder
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        self.unet = None
        self.vae = None
        self.action_embedding = None
        self.noise_scheduler = None
        self.tokenizer = None
        self.text_encoder = None
        
    def load_model(self):
        """Load model using existing model.py"""
        print(f"Loading model from {self.model_folder}...")
        self.unet, self.vae, self.action_embedding, self.noise_scheduler, self.tokenizer, self.text_encoder = load_model(
            self.model_folder, device=self.device
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        print("Model loaded!")
        return self
    
    def generate_single_frame(self, batch: dict) -> np.ndarray:
        """Generate single frame using run_inference logic"""
        img = run_inference_img_conditioning_with_params(
            self.unet,
            self.vae,
            self.noise_scheduler,
            self.action_embedding,
            self.tokenizer,
            self.text_encoder,
            batch,
            device=self.device,
            skip_action_conditioning=False,
            do_classifier_free_guidance=False,
            guidance_scale=CFG_GUIDANCE_SCALE,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        )
        return np.array(img)
    
    def generate_rollout(self, actions: list, initial_frame_context: torch.Tensor, 
                         initial_action_context: torch.Tensor) -> List[np.ndarray]:
        """Generate rollout using run_autoregressive logic"""
        all_images = []
        current_actions = initial_action_context
        context_latents = initial_frame_context

        for i in tqdm(range(len(actions)), desc="Generating rollout"):
            target_latents = next_latent(
                unet=self.unet,
                vae=self.vae,
                noise_scheduler=self.noise_scheduler,
                action_embedding=self.action_embedding,
                context_latents=context_latents.unsqueeze(0),
                device=self.device,
                skip_action_conditioning=False,
                do_classifier_free_guidance=False,
                guidance_scale=CFG_GUIDANCE_SCALE,
                num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                actions=current_actions.unsqueeze(0),
            )
            
            current_actions = torch.cat([
                current_actions[(-BUFFER_SIZE + 1):],
                torch.tensor([actions[i]]).to(self.device),
            ])
            context_latents = torch.cat([context_latents[(-BUFFER_SIZE + 1):], target_latents], dim=0)
            
            img = decode_and_postprocess(
                vae=self.vae, image_processor=self.image_processor, latents=target_latents
            )
            all_images.append(np.array(img))
        
        return all_images
    
    def generate_batch_from_dataset(self, n_samples: int, dataset_name: str = None) -> np.ndarray:
        """Generate multiple frames from dataset"""
        dataset_name = dataset_name or TRAINING_DATASET_DICT["small"]
        dataset = EpisodeDataset(dataset_name)
        
        generated_frames = []
        indices = random.sample(range(BUFFER_SIZE, len(dataset)), min(n_samples, len(dataset) - BUFFER_SIZE))
        
        for idx in tqdm(indices, desc="Generating frames"):
            batch = collate_fn([dataset[idx]])
            img = self.generate_single_frame(batch)
            generated_frames.append(img)
        
        return np.stack(generated_frames)
    
    def get_original_frames_from_dataset(self, n_samples: int, dataset_name: str = None) -> np.ndarray:
        """Get original frames from dataset for comparison"""
        dataset_name = dataset_name or TRAINING_DATASET_DICT["small"]
        dataset = EpisodeDataset(dataset_name)
        
        original_frames = []
        indices = random.sample(range(BUFFER_SIZE, len(dataset)), min(n_samples, len(dataset) - BUFFER_SIZE))
        
        for idx in indices:
            batch = collate_fn([dataset[idx]])
            img_tensor = batch["pixel_values"][0, -1] 
            img = ((img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            original_frames.append(img)
        
        return np.stack(original_frames)


def run_full_evaluation(generated: np.ndarray, original: np.ndarray):
    """Run all three evaluation categories"""
    
    print("\n" + "=" * 60)
    print("MARIO DIFFUSION MODEL EVALUATION")
    print("=" * 60)
    
    if generated.shape[1:3] != original.shape[1:3]:
        print(f"Resizing generated {generated.shape} to match original {original.shape}")
        resized = []
        for frame in generated:
            r = cv2.resize(frame, (original.shape[2], original.shape[1]))
            resized.append(r)
        generated = np.stack(resized)
    
    print("\n[1/3] Visual Fidelity Metrics...")
    eval1 = MarioEvaluator()
    visual_results = eval1.evaluate_visual_fidelity(original, generated)
    print(f"  PSNR: {visual_results['psnr_mean']:.2f} dB")
    print(f"  LPIPS: {visual_results.get('lpips_mean', 'N/A')}")
    
    print("\n[2/3] Image Quality Metrics...")
    eval2 = MarioImageQualityEvaluator()
    quality_results = eval2.evaluate_batch(original, generated)
    print(f"  SSIM: {quality_results['ssim_mean']:.4f}")
    print(f"  Histogram Similarity: {quality_results['histogram_similarity_mean']:.4f}")
    print(f"  Edge Similarity: {quality_results['edge_similarity_mean']:.4f}")
    
    print("\n[3/3] Distribution Shift Metrics...")
    eval3 = AdversarialDistributionEvaluator()
    dist_results = eval3.evaluate_distribution_shift(original, generated)
    print(f"  FID: {dist_results['fid']:.2f}")
    print(f"  MMD: {dist_results['mmd']:.6f}")
    print(f"  KL Divergence: {dist_results['kl_divergence_mean']:.4f}")
    print(f"  Wasserstein: {dist_results['wasserstein_mean']:.4f}")
    
    all_results = {
        "visual_fidelity": visual_results,
        "image_quality": quality_results,
        "distribution": dist_results
    }
    
    return all_results

MODEL_FOLDER = "Flaaaande/mario-sd" 
N_SAMPLES = 1000
DATASET = None  

pipeline = MarioGenerationPipeline(model_folder=MODEL_FOLDER)
pipeline.load_model()

print(f"\nGenerating {N_SAMPLES} frames...")
generated = pipeline.generate_batch_from_dataset(N_SAMPLES, DATASET)
print(f"Generated shape: {generated.shape}")

print(f"Loading {N_SAMPLES} original frames...")
original = pipeline.get_original_frames_from_dataset(N_SAMPLES, DATASET)
print(f"Original shape: {original.shape}")

Path("generated_samples").mkdir(exist_ok=True)
for i, frame in enumerate(generated[:5]):
    cv2.imwrite(f"generated_samples/gen_{i}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print("Saved sample frames to generated_samples/")

results = run_full_evaluation(generated, original)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
