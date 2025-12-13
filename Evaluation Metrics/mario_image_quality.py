"""
Mario Image Generation Quality Assessment
Compares generated frames against original Super Mario Bros frames
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class ImageQualityMetrics:
    psnr: float
    ssim: float
    mse: float
    mae: float
    lpips: Optional[float]
    histogram_similarity: float
    edge_similarity: float


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
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


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
        return 1.0
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
    if mario_palette is None:
        mario_palette = [
            (0, 0, 0),       # Black
            (255, 255, 255), # White
            (252, 152, 56),  # Mario orange
            (200, 76, 12),   # Mario red/brown
            (92, 148, 252),  # Sky blue
            (56, 204, 108),  # Pipe green
            (128, 208, 16),  # Bright green
            (172, 124, 0),   # Ground brown
            (252, 188, 176), # Skin tone
        ]
    
    def closest_palette_distance(pixel, palette):
        min_dist = float('inf')
        for color in palette:
            dist = np.sqrt(sum((int(p) - int(c)) ** 2 for p, c in zip(pixel, color)))
            min_dist = min(min_dist, dist)
        return min_dist
    
    orig_distances = []
    gen_distances = []
    
    # Sample pixels for efficiency
    h, w = original.shape[:2]
    sample_points = [(i, j) for i in range(0, h, 4) for j in range(0, w, 4)]
    
    for i, j in sample_points:
        orig_distances.append(closest_palette_distance(original[i, j], mario_palette))
        gen_distances.append(closest_palette_distance(generated[i, j], mario_palette))
    
    return {
        "original_palette_adherence": float(np.mean(orig_distances)),
        "generated_palette_adherence": float(np.mean(gen_distances)),
        "palette_accuracy_ratio": float(np.mean(orig_distances) / max(np.mean(gen_distances), 1e-6))
    }


class MarioImageQualityEvaluator:
    """Comprehensive image quality evaluation for Mario generation"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_single(self, original: np.ndarray, generated: np.ndarray) -> ImageQualityMetrics:
        """Evaluate a single frame pair"""
        return ImageQualityMetrics(
            psnr=compute_psnr(original, generated),
            ssim=compute_ssim(original, generated),
            mse=compute_mse(original, generated),
            mae=compute_mae(original, generated),
            lpips=compute_lpips(original, generated),
            histogram_similarity=compute_histogram_similarity(original, generated),
            edge_similarity=compute_edge_similarity(original, generated)
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
            if result.lpips is not None:
                metrics['lpips'].append(result.lpips)
            metrics['histogram_similarity'].append(result.histogram_similarity)
            metrics['edge_similarity'].append(result.edge_similarity)
        
        aggregated = {}
        for key, values in metrics.items():
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
        
        return aggregated
    
    def evaluate_with_color(self, original: np.ndarray, generated: np.ndarray) -> Dict:
        """Full evaluation including color palette analysis"""
        base_metrics = self.evaluate_single(original, generated)
        color_metrics = compute_color_accuracy(original, generated)
        
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
