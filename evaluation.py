import numpy as np
import torch
import cv2
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from PIL import Image

from diffusers.image_processor import VaeImageProcessor

from mario_eval_metrics import MarioEvaluator
from mario_image_quality import MarioImageQualityEvaluator
from adversarial_distribution_eval import AdversarialDistributionEvaluator
from config_sd import BUFFER_SIZE, CFG_GUIDANCE_SCALE, TRAINING_DATASET_DICT, DEFAULT_NUM_INFERENCE_STEPS
from dataset import EpisodeDataset, collate_fn
from model import load_model
from run_inference import (
    decode_and_postprocess,
    next_latent,
    run_inference_img_conditioning_with_params,
)
from skimage.metrics import structural_similarity as ssim


class MarioGenerationPipeline:
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
        self.image_processor = None

    def load_model(self):
        print(f"Loading model from {self.model_folder}...")
        self.unet, self.vae, self.action_embedding, self.noise_scheduler, self.tokenizer, self.text_encoder = load_model(
            self.model_folder, device=self.device
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def generate_single_frame(self, batch: dict) -> np.ndarray:
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

    def generate_batch_from_dataset(self, n_samples: int, dataset_name: str = None, indices: List[int] = None) -> np.ndarray:
        dataset_name = dataset_name or TRAINING_DATASET_DICT["small"]
        dataset = EpisodeDataset(dataset_name)

        if indices is None:
            print("No indices provided. Sampling random indices.")
            indices = random.sample(range(BUFFER_SIZE, len(dataset)), min(n_samples, len(dataset) - BUFFER_SIZE))

        generated_frames = []
        for idx in tqdm(indices, desc="Generating frames"):
            batch = collate_fn([dataset[idx]])
            img = self.generate_single_frame(batch)
            generated_frames.append(img)

        return np.stack(generated_frames)

    def get_original_frames_from_dataset(self, n_samples: int, dataset_name: str = None, indices: List[int] = None) -> np.ndarray:
        dataset_name = dataset_name or TRAINING_DATASET_DICT["small"]
        dataset = EpisodeDataset(dataset_name)

        if indices is None:
            print("No indices provided. Sampling random indices.")
            indices = random.sample(range(BUFFER_SIZE, len(dataset)), min(n_samples, len(dataset) - BUFFER_SIZE))

        original_frames = []
        for idx in indices:
            batch = collate_fn([dataset[idx]])
            img_tensor = batch["pixel_values"][0, -1]  # Shape: (3, H, W)
            img = ((img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            original_frames.append(img)

        return np.stack(original_frames)


def run_full_evaluation(generated: np.ndarray, original: np.ndarray):
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

def visualize_results(original, generated):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    plt.hist(original.flatten(), bins=50, alpha=0.5, label='Original', color='blue', density=True)
    plt.hist(generated.flatten(), bins=50, alpha=0.5, label='Generated', color='red', density=True)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("pixel_intensity_distribution.png")
    plt.show()

    def compute_psnr(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))

    psnr_values = [compute_psnr(original[i], generated[i]) for i in range(len(original))]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(psnr_values, kde=True, ax=axes[0], color='green')
    axes[0].set_title('Distribution of PSNR Values')
    sns.boxplot(x=psnr_values, ax=axes[1], color='green')
    axes[1].set_title('Box Plot of PSNR Values')
    plt.tight_layout()
    plt.savefig("psnr_distribution.png")
    plt.show()

    indices = random.sample(range(len(generated)), min(3, len(generated)))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    plt.suptitle("Original vs Generated vs Difference Heatmap", fontsize=16)

    for i, idx in enumerate(indices):
        axes[i, 0].imshow(original[idx])
        axes[i, 0].set_title(f"Original (Idx {idx})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(generated[idx])
        axes[i, 1].set_title(f"Generated (Idx {idx})")
        axes[i, 1].axis('off')

        diff = np.abs(original[idx].astype(np.float32) - generated[idx].astype(np.float32))
        diff_intensity = np.mean(diff, axis=2)
        im = axes[i, 2].imshow(diff_intensity, cmap='hot', vmin=0, vmax=255)
        axes[i, 2].set_title("Difference Heatmap")
        axes[i, 2].axis('off')
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("difference_heatmaps.png")
    plt.show()


if __name__ == "__main__":
    MODEL_FOLDER = "Flaaaande/mario-sd"
    N_SAMPLES = 100
    DATASET = None

    pipeline = MarioGenerationPipeline(model_folder=MODEL_FOLDER)
    pipeline.load_model()

    temp_dataset = EpisodeDataset(TRAINING_DATASET_DICT["small"])
    valid_range = range(BUFFER_SIZE, len(temp_dataset))
    
    # Generate the shared list
    shared_indices = random.sample(valid_range, min(N_SAMPLES, len(valid_range)))

    print(f"\nGenerating {N_SAMPLES} frames...")
    generated = pipeline.generate_batch_from_dataset(N_SAMPLES, DATASET, indices=shared_indices)    
    print(f"Loading {N_SAMPLES} original frames...")
    original = pipeline.get_original_frames_from_dataset(N_SAMPLES, DATASET, indices=shared_indices)

    Path("generated_samples").mkdir(exist_ok=True)
    for i, frame in enumerate(generated[:5]):
        # Save simple PNGs
        cv2.imwrite(f"generated_samples/gen_{i}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("Saved sample frames to generated_samples/")

    results = run_full_evaluation(generated, original)

    visualize_results(original, generated)