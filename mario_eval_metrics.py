"""
Super Mario Bros Game Generation Evaluation Metrics
Categories: Visual Fidelity, Gameplay, System
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ==================== VISUAL FIDELITY ====================

def compute_psnr(original: np.ndarray, generated: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (↑ higher is better)"""
    mse = np.mean((original.astype(float) - generated.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_lpips(original: np.ndarray, generated: np.ndarray) -> float:
    """
    Learned Perceptual Image Patch Similarity (↓ lower is better)
    Requires: pip install lpips torch
    """
    try:
        import lpips
        import torch
        
        loss_fn = lpips.LPIPS(net='alex', verbose=False)
        
        def to_tensor(img):
            t = torch.from_numpy(img).float() / 127.5 - 1.0
            if t.ndim == 3:
                t = t.permute(2, 0, 1).unsqueeze(0)
            return t
        
        with torch.no_grad():
            return loss_fn(to_tensor(original), to_tensor(generated)).item()
    except ImportError:
        print("Install lpips: pip install lpips torch")
        return None


def compute_fvd(real_videos: np.ndarray, generated_videos: np.ndarray, num_frames: int = 16) -> float:
    """
    Fréchet Video Distance (↓ lower is better)
    real_videos, generated_videos: shape (N, T, H, W, C)
    Requires: pip install torch torchvision
    """
    try:
        import torch
        from scipy import linalg
        
        def get_video_features(videos, model):
            """Extract I3D features from videos"""
            features = []
            for video in videos:
                indices = np.linspace(0, len(video) - 1, num_frames, dtype=int)
                clip = video[indices]
                # Normalize and reshape for model
                clip = torch.from_numpy(clip).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0
                with torch.no_grad():
                    feat = model(clip).squeeze().numpy()
                features.append(feat)
            return np.array(features)
        
        def frechet_distance(mu1, sigma1, mu2, sigma2):
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        real_flat = real_videos.reshape(len(real_videos), -1)[:, :2048]
        gen_flat = generated_videos.reshape(len(generated_videos), -1)[:, :2048]
        
        mu_real, sigma_real = np.mean(real_flat, axis=0), np.cov(real_flat, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_flat, axis=0), np.cov(gen_flat, rowvar=False)
        
        return frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    except ImportError:
        print("Install scipy: pip install scipy")
        return None


# ==================== GAMEPLAY ====================

def compute_solvability_rate(levels: List[str], solver_fn) -> float:
    """
    Solvability Rate (↑ higher is better)
    levels: list of level representations (e.g., tile strings)
    solver_fn: function that returns True if level is beatable
    """
    if not levels:
        return 0.0
    solved = sum(1 for level in levels if solver_fn(level))
    return (solved / len(levels)) * 100


def compute_tile_consistency(level: str, valid_tiles: set, tile_rules: Dict = None) -> Tuple[float, Dict]:
    """
    Tile Consistency - checks if tiles are valid and follow rules
    Returns percentage and detailed report
    """
    tiles = list(level.replace('\n', ''))
    total = len(tiles)
    
    if total == 0:
        return 0.0, {"error": "empty level"}
    
    # Check valid tiles
    valid_count = sum(1 for t in tiles if t in valid_tiles)
    validity_rate = valid_count / total
    
    # Check tile rules (e.g., pipes must be paired, ground continuity)
    rule_violations = 0
    if tile_rules:
        for rule_name, rule_fn in tile_rules.items():
            if not rule_fn(level):
                rule_violations += 1
    
    report = {
        "validity_rate": validity_rate * 100,
        "valid_tiles": valid_count,
        "total_tiles": total,
        "rule_violations": rule_violations
    }
    
    return validity_rate * 100, report


# Mario-specific tile validation
MARIO_TILES = {
    '-': 'empty/sky',
    'X': 'ground',
    'S': 'breakable brick',
    '?': 'question block',
    'Q': 'empty question block', 
    'E': 'enemy (goomba)',
    '<': 'pipe top left',
    '>': 'pipe top right',
    '[': 'pipe body left',
    ']': 'pipe body right',
    'o': 'coin',
    'B': 'bullet bill cannon',
    'b': 'bullet bill base',
}


def mario_tile_rules() -> Dict:
    """Rules for valid Mario level structure"""
    return {
        "pipes_paired": lambda lvl: lvl.count('<') == lvl.count('>'),
        "has_ground": lambda lvl: 'X' in lvl,
        "reachable_end": lambda lvl: not lvl.endswith('X' * 10),  # simplified
    }


# ==================== SYSTEM ====================

@dataclass
class SystemMetrics:
    inference_speed_fps: float
    generation_mode: str  # "real-time" or "offline"
    difficulty_control: str  # "none", "static", "dynamic"
    total_generation_time: float
    frames_generated: int


def measure_inference_speed(generate_fn, num_frames: int = 100, warmup: int = 10) -> SystemMetrics:
    """
    Measure generation speed and system characteristics
    generate_fn: function that generates one frame, returns frame array
    """
    # Warmup
    for _ in range(warmup):
        generate_fn()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_frames):
        generate_fn()
    elapsed = time.perf_counter() - start
    
    fps = num_frames / elapsed
    
    return SystemMetrics(
        inference_speed_fps=fps,
        generation_mode="real-time" if fps >= 30 else "offline",
        difficulty_control="none",  # Set based on your implementation
        total_generation_time=elapsed,
        frames_generated=num_frames
    )


# ==================== FULL EVALUATION ====================

class MarioEvaluator:
    """Complete evaluation pipeline for Mario level/frame generation"""
    
    def __init__(self, valid_tiles: set = None):
        self.valid_tiles = valid_tiles or set(MARIO_TILES.keys())
        self.results = {}
    
    def evaluate_visual_fidelity(self, original_frames: np.ndarray, 
                                  generated_frames: np.ndarray) -> Dict:
        """Evaluate visual metrics on frame pairs"""
        psnr_scores = []
        lpips_scores = []
        
        for orig, gen in zip(original_frames, generated_frames):
            psnr_scores.append(compute_psnr(orig, gen))
            lp = compute_lpips(orig, gen)
            if lp is not None:
                lpips_scores.append(lp)
        
        results = {
            "psnr_mean": np.mean(psnr_scores),
            "psnr_std": np.std(psnr_scores),
            "lpips_mean": np.mean(lpips_scores) if lpips_scores else None,
            "lpips_std": np.std(lpips_scores) if lpips_scores else None,
        }
        
        # FVD requires video sequences
        if original_frames.ndim == 5:  # (N, T, H, W, C)
            results["fvd"] = compute_fvd(original_frames, generated_frames)
        
        self.results["visual_fidelity"] = results
        return results
    
    def evaluate_gameplay(self, levels: List[str], solver_fn=None) -> Dict:
        """Evaluate gameplay metrics on generated levels"""
        
        # Default simple solver (checks basic traversability)
        if solver_fn is None:
            solver_fn = lambda lvl: 'X' in lvl and '-' in lvl
        
        solvability = compute_solvability_rate(levels, solver_fn)
        
        consistency_scores = []
        for level in levels:
            score, _ = compute_tile_consistency(level, self.valid_tiles, mario_tile_rules())
            consistency_scores.append(score)
        
        results = {
            "solvability_rate": solvability,
            "tile_consistency_mean": np.mean(consistency_scores),
            "tile_consistency_std": np.std(consistency_scores),
            "num_levels_evaluated": len(levels)
        }
        
        self.results["gameplay"] = results
        return results
    
    def evaluate_system(self, generate_fn, num_frames: int = 100) -> Dict:
        """Evaluate system performance"""
        metrics = measure_inference_speed(generate_fn, num_frames)
        
        results = {
            "inference_speed_fps": metrics.inference_speed_fps,
            "generation_mode": metrics.generation_mode,
            "difficulty_control": metrics.difficulty_control,
            "total_time_seconds": metrics.total_generation_time,
        }
        
        self.results["system"] = results
        return results
    
    def full_report(self) -> Dict:
        """Return all evaluation results"""
        return self.results
    
    def print_report(self):
        """Print formatted evaluation report"""
        print("\n" + "="*50)
        print("MARIO GENERATION EVALUATION REPORT")
        print("="*50)
        
        for category, metrics in self.results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 30)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Demo with synthetic data
    evaluator = MarioEvaluator()
    
    # Visual fidelity demo
    print("Testing Visual Fidelity...")
    orig_frames = np.random.randint(0, 256, (10, 224, 256, 3), dtype=np.uint8)
    gen_frames = orig_frames + np.random.randint(-20, 20, orig_frames.shape, dtype=np.int16)
    gen_frames = np.clip(gen_frames, 0, 255).astype(np.uint8)
    
    visual_results = evaluator.evaluate_visual_fidelity(orig_frames, gen_frames)
    
    # Gameplay demo
    print("Testing Gameplay...")
    sample_levels = [
        "--------------------\n----?--S--?--------\n--------------------\nXXXXXXXXXXXXXXXXXXXX",
        "--------------------\n--------E----------\n--------------------\nXXXXXXXX----XXXXXXXX",
        "--------------------\n----<>-------------\n----[]-------------\nXXXXXXXXXXXXXXXXXXXX",
    ]
    
    gameplay_results = evaluator.evaluate_gameplay(sample_levels)
    
    # System demo
    print("Testing System...")
    dummy_generator = lambda: np.random.randint(0, 256, (224, 256, 3), dtype=np.uint8)
    system_results = evaluator.evaluate_system(dummy_generator, num_frames=50)
    
    # Print report
    evaluator.print_report()
