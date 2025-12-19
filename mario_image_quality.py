"""
Mario Image Generation Quality Assessment
Compares generated frames against original Super Mario Bros frames
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
import torch
import lpips


@dataclass
class ImageQualityMetrics:
    psnr: float
    ssim: float
    mse: float
    mae: float
    lpips: Optional[float]
    histogram_similarity: float
    edge_similarity: float


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Helper: Force image to 0-255 uint8 range"""
    if img.dtype == np.uint8:
        return img
    
    # If float, check range
    if img.max() <= 1.1:
        # Assume 0-1 float, scale up
        img = (img * 255.0)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def compute_mse(original: np.ndarray, generated: np.ndarray) -> float:
    """Mean Squared Error (↓ lower is better)"""
    return np.mean((original.astype(float) - generated.astype(float)) ** 2)


def compute_mae(original: np.ndarray, generated: np.ndarray) -> float:
    """Mean Absolute Error (↓ lower is better)"""
    return np.mean(np.abs(original.astype(float) - generated.astype(float)))


def compute_psnr(original: np.ndarray, generated: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (↑ higher is better)"""
    mse = compute_mse(original, generated)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(original: np.ndarray, generated: np.ndarray) -> float:
    """
    Structural Similarity Index (↑ higher is better, max=1.0)
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    orig = original.astype(float)
    gen = generated.astype(float)
    
    mu_orig = cv2.GaussianBlur(orig, (11, 11), 1.5)
    mu_gen = cv2.GaussianBlur(gen, (11, 11), 1.5)
    
    mu_orig_sq = mu_orig ** 2
    mu_gen_sq = mu_gen ** 2
    mu_orig_gen = mu_orig * mu_gen
    
    sigma_orig_sq = cv2.GaussianBlur(orig ** 2, (11, 11), 1.5) - mu_orig_sq
    sigma_gen_sq = cv2.GaussianBlur(gen ** 2, (11, 11), 1.5) - mu_gen_sq
    sigma_orig_gen = cv2.GaussianBlur(orig * gen, (11, 11), 1.5) - mu_orig_gen
    
    numerator = (2 * mu_orig_gen + C1) * (2 * sigma_orig_gen + C2)
    denominator = (mu_orig_sq + mu_gen_sq + C1) * (sigma_orig_sq + sigma_gen_sq + C2)
    
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_histogram_similarity(original: np.ndarray, generated: np.ndarray) -> float:
    """
    Histogram comparison using correlation (↑ higher is better, max=1.0)
    """
    scores = []
    for i in range(3):  # RGB channels
        hist_orig = cv2.calcHist([original], [i], None, [256], [0, 256])
        hist_gen = cv2.calcHist([generated], [i], None, [256], [0, 256])
        
        cv2.normalize(hist_orig, hist_orig)
        cv2.normalize(hist_gen, hist_gen)
        
        score = cv2.compareHist(hist_orig, hist_gen, cv2.HISTCMP_CORREL)
        scores.append(score)
    
    return float(np.mean(scores))


def compute_edge_similarity(original: np.ndarray, generated: np.ndarray) -> float:
    """
    Edge detection similarity - important for Mario's pixel art style (↑ higher is better)
    """
    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    
    edges_orig = cv2.Canny(orig_gray, 50, 150)
    edges_gen = cv2.Canny(gen_gray, 50, 150)
    
    # IoU of edges
    intersection = np.logical_and(edges_orig, edges_gen).sum()
    union = np.logical_or(edges_orig, edges_gen).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_lpips(original: np.ndarray, generated: np.ndarray) -> Optional[float]:
    """LPIPS perceptual similarity (↓ lower is better)"""
    try:
        import lpips
        import torch
        
        loss_fn = lpips.LPIPS(net='alex', verbose=False)
        
        def to_tensor(img):
            t = torch.from_numpy(img).float() / 127.5 - 1.0
            return t.permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            return float(loss_fn(to_tensor(original), to_tensor(generated)).item())
    except ImportError:
        return None


def compute_color_accuracy(original: np.ndarray, generated: np.ndarray, 
                           mario_palette: List[Tuple[int, int, int]] = None) -> Dict:
    """
    Evaluate color accuracy against Mario's limited palette
    """
    # Classic NES Mario palette (approximate)
    mario_palette = np.array([
        [0, 0, 0],       [255, 255, 255], [252, 152, 56],
        [200, 76, 12],   [92, 148, 252],  [56, 204, 108],
        [128, 208, 16],  [172, 124, 0],   [252, 188, 176]
    ])

    # Downsample for speed (every 4th pixel)
    orig_small = original[::4, ::4].reshape(-1, 3)
    gen_small = generated[::4, ::4].reshape(-1, 3)
    
    def mean_min_dist(pixels, palette):
        # Vectorized distance calculation
        # Shape: (N_pixels, 1, 3) - (1, N_palette, 3)
        dists = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        return np.mean(min_dists)

    orig_score = mean_min_dist(orig_small, mario_palette)
    gen_score = mean_min_dist(gen_small, mario_palette)
    
    return {
        "original_palette_adherence": float(orig_score),
        "generated_palette_adherence": float(gen_score),
        "palette_accuracy_ratio": float(orig_score / max(gen_score, 1e-6))
    }


class MarioImageQualityEvaluator:
    """Comprehensive image quality evaluation for Mario generation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to('cuda')
    
    def evaluate_single(self, original: np.ndarray, generated: np.ndarray) -> ImageQualityMetrics:
        """Evaluate a single frame pair"""
        
        orig_u8 = ensure_uint8(original)
        gen_u8 = ensure_uint8(generated)
        
        # Calculate Standard Metrics
        psnr = compute_psnr(orig_u8, gen_u8)
        ssim = compute_ssim(orig_u8, gen_u8)
        mse = compute_mse(orig_u8, gen_u8)
        mae = compute_mae(orig_u8, gen_u8)
        hist = compute_histogram_similarity(orig_u8, gen_u8)
        edge = compute_edge_similarity(orig_u8, gen_u8)
        
        # Calculate LPIPS (Requires float -1 to 1)
        lpips_score = None
        if self.lpips_fn is not None:
            with torch.no_grad():
                # Normalize to [-1, 1]
                t_orig = torch.from_numpy(orig_u8).float() / 127.5 - 1.0
                t_gen = torch.from_numpy(gen_u8).float() / 127.5 - 1.0
                
                # Fix dimensions (H,W,C -> 1,C,H,W)
                t_orig = t_orig.permute(2, 0, 1).unsqueeze(0).to(self.device)
                t_gen = t_gen.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                lpips_score = self.lpips_fn(t_orig, t_gen).item()

        return ImageQualityMetrics(
            psnr=psnr, ssim=ssim, mse=mse, mae=mae,
            lpips=lpips_score, histogram_similarity=hist, edge_similarity=edge
        )
    
    def evaluate_batch(self, originals: np.ndarray, generateds: np.ndarray) -> Dict:
        """Evaluate multiple frame pairs and aggregate"""
        metrics = {
            'psnr': [], 'ssim': [], 'mse': [], 'mae': [],
            'lpips': [], 'histogram_similarity': [], 'edge_similarity': []
        }
        
        for orig, gen in zip(originals, generateds):
            result = self.evaluate_single(orig, gen)
            metrics['psnr'].append(result.psnr)
            metrics['ssim'].append(result.ssim)
            metrics['mse'].append(result.mse)
            metrics['mae'].append(result.mae)
            metrics['histogram_similarity'].append(result.histogram_similarity)
            metrics['edge_similarity'].append(result.edge_similarity)
            if result.lpips is not None:
                metrics['lpips'].append(result.lpips)
        
        aggregated = {}
        for key, values in metrics.items():
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
        
        return aggregated
    
    def evaluate_with_color(self, original: np.ndarray, generated: np.ndarray) -> Dict:
        """Full evaluation including color palette analysis"""
        # Ensure format for color analysis
        orig_u8 = ensure_uint8(original)
        gen_u8 = ensure_uint8(generated)
        
        base_metrics = self.evaluate_single(orig_u8, gen_u8)
        color_metrics = compute_color_accuracy(orig_u8, gen_u8)
        
        return {
            **base_metrics.__dict__,
            **color_metrics
        }
    
    def print_comparison(self, results: Dict):
        """Print formatted comparison"""
        print("\n" + "=" * 50)
        print("MARIO IMAGE QUALITY ASSESSMENT")
        print("=" * 50)
        
        quality_indicators = {
            'psnr_mean': ('PSNR (dB)', '↑', 30),
            'ssim_mean': ('SSIM', '↑', 0.9),
            'mse_mean': ('MSE', '↓', 100),
            'lpips_mean': ('LPIPS', '↓', 0.1),
            'histogram_similarity_mean': ('Histogram Sim', '↑', 0.9),
            'edge_similarity_mean': ('Edge Sim', '↑', 0.7),
        }
        
        for key, (name, direction, threshold) in quality_indicators.items():
            if key in results:
                value = results[key]
                if direction == '↑':
                    status = "✓" if value >= threshold else "✗"
                else:
                    status = "✓" if value <= threshold else "✗"
                print(f"  {name}: {value:.4f} {direction} {status}")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    evaluator = MarioImageQualityEvaluator()
    
    # Create synthetic test data (replace with real frames)
    print("Running quality assessment demo...")
    
    # Simulate original Mario frame
    original = np.zeros((240, 256, 3), dtype=np.uint8)
    original[200:240, :] = [172, 124, 0]  # Ground
    original[0:180, :] = [92, 148, 252]   # Sky
    original[180:200, 50:70] = [56, 204, 108]  # Pipe
    
    # Simulate generated frame (with some noise/differences)
    generated = original.copy()
    noise = np.random.randint(-15, 15, original.shape, dtype=np.int16)
    generated = np.clip(generated.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Single frame evaluation
    print("\nSingle Frame Evaluation:")
    single_result = evaluator.evaluate_with_color(original, generated)
    for k, v in single_result.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Batch evaluation
    print("\nBatch Evaluation (10 frames):")
    originals = np.stack([original] * 10)
    generateds = np.stack([
        np.clip(original.astype(np.int16) + np.random.randint(-20, 20, original.shape), 0, 255).astype(np.uint8)
        for _ in range(10)
    ])
    
    batch_results = evaluator.evaluate_batch(originals, generateds)
    evaluator.print_comparison(batch_results)
