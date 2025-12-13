"""
Adversarial Image Distribution Shift Detection
Measures how adversarial/perturbed images differ from original distribution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
import cv2


@dataclass
class DistributionShiftMetrics:
    fid: float                    # Fréchet Inception Distance
    kl_divergence: float          # KL divergence
    js_divergence: float          # Jensen-Shannon divergence
    mmd: float                    # Maximum Mean Discrepancy
    wasserstein: float            # Wasserstein/Earth Mover distance
    cosine_shift: float           # Cosine distance in feature space
    l2_perturbation: float        # Average L2 norm of perturbation
    linf_perturbation: float      # Average L-inf norm of perturbation


# ==================== PERTURBATION METRICS ====================

def compute_l2_perturbation(original: np.ndarray, adversarial: np.ndarray) -> float:
    """L2 norm of perturbation (↓ lower = more subtle attack)"""
    diff = original.astype(float) - adversarial.astype(float)
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_linf_perturbation(original: np.ndarray, adversarial: np.ndarray) -> float:
    """L-infinity norm - max pixel change (↓ lower = more subtle)"""
    diff = np.abs(original.astype(float) - adversarial.astype(float))
    return float(np.max(diff))


def compute_l0_perturbation(original: np.ndarray, adversarial: np.ndarray, 
                            threshold: float = 1.0) -> float:
    """L0 norm - percentage of pixels changed"""
    diff = np.abs(original.astype(float) - adversarial.astype(float))
    changed = np.any(diff > threshold, axis=-1) if diff.ndim == 3 else diff > threshold
    return float(np.mean(changed) * 100)


# ==================== DISTRIBUTION DIVERGENCE ====================

def compute_kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 256) -> float:
    """KL Divergence between pixel distributions (↑ higher = more different)"""
    p_flat = p.flatten()
    q_flat = q.flatten()
    
    p_hist, _ = np.histogram(p_flat, bins=bins, range=(0, 255), density=True)
    q_hist, _ = np.histogram(q_flat, bins=bins, range=(0, 255), density=True)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    
    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    
    return float(stats.entropy(p_hist, q_hist))


def compute_js_divergence(p: np.ndarray, q: np.ndarray, bins: int = 256) -> float:
    """Jensen-Shannon Divergence - symmetric (0 to 1, ↑ higher = more different)"""
    p_flat = p.flatten()
    q_flat = q.flatten()
    
    p_hist, _ = np.histogram(p_flat, bins=bins, range=(0, 255), density=True)
    q_hist, _ = np.histogram(q_flat, bins=bins, range=(0, 255), density=True)
    
    eps = 1e-10
    p_hist = (p_hist + eps) / (p_hist + eps).sum()
    q_hist = (q_hist + eps) / (q_hist + eps).sum()
    
    m = 0.5 * (p_hist + q_hist)
    
    js = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
    return float(js)


def compute_wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein/Earth Mover Distance (↑ higher = more different)"""
    p_flat = p.flatten().astype(float)
    q_flat = q.flatten().astype(float)
    
    # Use 1D Wasserstein for efficiency
    return float(stats.wasserstein_distance(p_flat, q_flat))


# ==================== FEATURE-BASED METRICS ====================

def extract_features(images: np.ndarray) -> np.ndarray:
    """
    Extract features from images using simple statistics
    For better results, use pretrained CNN (InceptionV3, ResNet)
    """
    features = []
    for img in images:
        feat = []
        # Color statistics per channel
        for c in range(3):
            channel = img[:, :, c].astype(float)
            feat.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
                stats.skew(channel.flatten()),
                stats.kurtosis(channel.flatten())
            ])
        
        # Gradient statistics
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        feat.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.max(grad_mag)
        ])
        
        # Frequency domain
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        feat.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(np.log1p(magnitude))
        ])
        
        features.append(feat)
    
    return np.array(features)


def compute_fid(original_feats: np.ndarray, adversarial_feats: np.ndarray) -> float:
    """
    Fréchet Inception Distance (↑ higher = more different)
    Using simple features; for true FID use InceptionV3
    """
    mu1, sigma1 = np.mean(original_feats, axis=0), np.cov(original_feats, rowvar=False)
    mu2, sigma2 = np.mean(adversarial_feats, axis=0), np.cov(adversarial_feats, rowvar=False)
    
    diff = mu1 - mu2
    
    # Handle scalar covariance case
    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
        sigma2 = np.array([[sigma2]])
    
    from scipy.linalg import sqrtm
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel (↑ higher = more different)
    """
    def rbf_kernel(X, Y, gamma):
        dists = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * dists)
    
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    
    n, m = len(X), len(Y)
    
    mmd = (K_XX.sum() / (n * n) + K_YY.sum() / (m * m) - 2 * K_XY.sum() / (n * m))
    return float(max(0, mmd))


def compute_cosine_shift(original_feats: np.ndarray, adversarial_feats: np.ndarray) -> float:
    """Average cosine distance between feature centroids (0-2, ↑ higher = more different)"""
    centroid_orig = np.mean(original_feats, axis=0)
    centroid_adv = np.mean(adversarial_feats, axis=0)
    
    dot = np.dot(centroid_orig, centroid_adv)
    norm = np.linalg.norm(centroid_orig) * np.linalg.norm(centroid_adv)
    
    if norm == 0:
        return 0.0
    
    cosine_sim = dot / norm
    return float(1 - cosine_sim)  # Convert to distance


# ==================== ADVERSARIAL DETECTION ====================

def detect_adversarial_artifacts(original: np.ndarray, adversarial: np.ndarray) -> Dict:
    """Detect common adversarial perturbation patterns"""
    diff = adversarial.astype(float) - original.astype(float)
    
    # High-frequency noise detection
    gray_diff = np.mean(np.abs(diff), axis=-1)
    fft = np.fft.fft2(gray_diff)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    
    # Ratio of high-freq to low-freq energy
    low_freq = magnitude[center_h-10:center_h+10, center_w-10:center_w+10].sum()
    high_freq = magnitude.sum() - low_freq
    freq_ratio = high_freq / (low_freq + 1e-10)
    
    # Perturbation uniformity
    uniformity = np.std(np.abs(diff)) / (np.mean(np.abs(diff)) + 1e-10)
    
    # Spatial pattern detection
    diff_gray = np.mean(diff, axis=-1)
    autocorr = np.correlate(diff_gray.flatten()[:1000], diff_gray.flatten()[:1000], mode='same')
    periodicity = np.max(autocorr[len(autocorr)//2+10:]) / (autocorr[len(autocorr)//2] + 1e-10)
    
    return {
        "high_freq_ratio": float(freq_ratio),
        "perturbation_uniformity": float(uniformity),
        "spatial_periodicity": float(periodicity),
        "likely_adversarial": freq_ratio > 5 or uniformity < 0.5
    }


# ==================== MAIN EVALUATOR ====================

class AdversarialDistributionEvaluator:
    """Comprehensive evaluation of adversarial distribution shift"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_perturbation(self, original: np.ndarray, adversarial: np.ndarray) -> Dict:
        """Measure perturbation magnitude for single image pair"""
        return {
            "l2_norm": compute_l2_perturbation(original, adversarial),
            "linf_norm": compute_linf_perturbation(original, adversarial),
            "l0_percent": compute_l0_perturbation(original, adversarial),
        }
    
    def evaluate_distribution_shift(self, originals: np.ndarray, 
                                     adversarials: np.ndarray) -> Dict:
        """Measure distribution shift between original and adversarial sets"""
        
        # Extract features
        orig_feats = extract_features(originals)
        adv_feats = extract_features(adversarials)
        
        # Pixel-level divergences (aggregate)
        kl_scores = []
        js_scores = []
        wass_scores = []
        
        for orig, adv in zip(originals, adversarials):
            kl_scores.append(compute_kl_divergence(orig, adv))
            js_scores.append(compute_js_divergence(orig, adv))
            wass_scores.append(compute_wasserstein(orig, adv))
        
        # Feature-level metrics
        fid = compute_fid(orig_feats, adv_feats)
        mmd = compute_mmd(orig_feats, adv_feats)
        cosine = compute_cosine_shift(orig_feats, adv_feats)
        
        # Perturbation stats
        l2_norms = [compute_l2_perturbation(o, a) for o, a in zip(originals, adversarials)]
        linf_norms = [compute_linf_perturbation(o, a) for o, a in zip(originals, adversarials)]
        
        return {
            "fid": fid,
            "mmd": mmd,
            "cosine_shift": cosine,
            "kl_divergence_mean": float(np.mean(kl_scores)),
            "js_divergence_mean": float(np.mean(js_scores)),
            "wasserstein_mean": float(np.mean(wass_scores)),
            "l2_perturbation_mean": float(np.mean(l2_norms)),
            "l2_perturbation_std": float(np.std(l2_norms)),
            "linf_perturbation_mean": float(np.mean(linf_norms)),
            "linf_perturbation_max": float(np.max(linf_norms)),
        }
    
    def detect_adversarial_batch(self, originals: np.ndarray, 
                                  adversarials: np.ndarray) -> Dict:
        """Detect adversarial artifacts in batch"""
        detections = []
        for orig, adv in zip(originals, adversarials):
            det = detect_adversarial_artifacts(orig, adv)
            detections.append(det)
        
        return {
            "adversarial_detection_rate": np.mean([d["likely_adversarial"] for d in detections]),
            "avg_high_freq_ratio": np.mean([d["high_freq_ratio"] for d in detections]),
            "avg_uniformity": np.mean([d["perturbation_uniformity"] for d in detections]),
        }
    
    def full_evaluation(self, originals: np.ndarray, adversarials: np.ndarray) -> Dict:
        """Complete adversarial distribution analysis"""
        
        dist_metrics = self.evaluate_distribution_shift(originals, adversarials)
        detection = self.detect_adversarial_batch(originals, adversarials)
        
        self.results = {**dist_metrics, **detection}
        return self.results
    
    def print_report(self):
        """Print formatted report"""
        print("\n" + "=" * 55)
        print("ADVERSARIAL DISTRIBUTION SHIFT ANALYSIS")
        print("=" * 55)
        
        sections = {
            "Distribution Divergence": ["fid", "mmd", "cosine_shift", "kl_divergence_mean", 
                                        "js_divergence_mean", "wasserstein_mean"],
            "Perturbation Magnitude": ["l2_perturbation_mean", "l2_perturbation_std",
                                       "linf_perturbation_mean", "linf_perturbation_max"],
            "Adversarial Detection": ["adversarial_detection_rate", "avg_high_freq_ratio", 
                                      "avg_uniformity"]
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            print("-" * 40)
            for key in keys:
                if key in self.results:
                    val = self.results[key]
                    if isinstance(val, float):
                        print(f"  {key}: {val:.6f}")
                    else:
                        print(f"  {key}: {val}")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    evaluator = AdversarialDistributionEvaluator()
    
    print("Generating test data...")
    
    # Simulate original Mario frames
    n_samples = 20
    originals = np.zeros((n_samples, 240, 256, 3), dtype=np.uint8)
    for i in range(n_samples):
        originals[i, 200:240, :] = [172, 124, 0]   # Ground
        originals[i, 0:180, :] = [92, 148, 252]    # Sky
        originals[i, 100:130, 50+i*5:80+i*5] = [255, 0, 0]  # Mario (moving)
    
    # Simulate adversarial perturbations (FGSM-like)
    epsilon = 8  # Perturbation budget
    adversarials = originals.copy()
    
    # Add structured adversarial noise
    noise = np.random.randint(-epsilon, epsilon+1, originals.shape, dtype=np.int16)
    # Make noise slightly structured (adversarial patterns often are)
    for i in range(n_samples):
        noise[i] = cv2.GaussianBlur(noise[i].astype(np.float32), (3, 3), 0).astype(np.int16)
    adversarials = np.clip(originals.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Run evaluation
    print("Evaluating distribution shift...")
    results = evaluator.full_evaluation(originals, adversarials)
    evaluator.print_report()
    
    # Single pair analysis
    print("\n" + "=" * 55)
    print("SINGLE IMAGE PERTURBATION ANALYSIS")
    print("=" * 55)
    single = evaluator.evaluate_perturbation(originals[0], adversarials[0])
    for k, v in single.items():
        print(f"  {k}: {v:.4f}")
    
    artifacts = detect_adversarial_artifacts(originals[0], adversarials[0])
    print(f"\n  Likely adversarial: {artifacts['likely_adversarial']}")
