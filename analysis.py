"""
Analysis Script: Does the Diffusion Model Understand Strategy?

This script provides tools to analyze whether a trained GameNGen-style diffusion
model actually captures game logic and strategic reasoning, or if it's just
learning visual patterns and "fooling" us with plausible-looking frames.

Key Analysis Methods:
1. Action Sensitivity Analysis - Does changing action change output appropriately?
2. State Consistency Tests - Do game rules remain consistent?
3. Temporal Coherence - Long-term rollout quality
4. Counterfactual Testing - What happens with impossible/illegal states?
5. Strategic Probe Tasks - Can it predict strategically correct outcomes?
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

from train_diffusion import (
    DeepRTSTrainer, TrainingConfig, DeepRTSDataset,
    ContextEncoder, ActionEmbedding, SimpleDiffusionModel
)


@dataclass
class AnalysisConfig:
    """Configuration for analysis experiments"""
    model_checkpoint: str
    dataset_path: str
    output_dir: str = "./analysis_results"
    num_samples: int = 100
    num_rollout_steps: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DiffusionAnalyzer:
    """
    Analyzer for evaluating diffusion model's understanding of game logic.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self._load_model()
        
        self.dataset = DeepRTSDataset(
            config.dataset_path,
            buffer_size=8,
            image_size=64,
            max_samples=config.num_samples * 10
        )
        
        self.results = {}
        
    def _load_model(self):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(
            self.config.model_checkpoint,
            map_location=self.device
        )
        
        training_config = checkpoint.get("config", TrainingConfig())
        
        self.context_encoder = ContextEncoder(
            buffer_size=training_config.buffer_size if hasattr(training_config, 'buffer_size') else 8,
            out_dim=768
        ).to(self.device)
        
        self.action_embed = ActionEmbedding(
            num_actions=15,
            embed_dim=768
        ).to(self.device)
        
        self.unet = SimpleDiffusionModel(
            in_channels=3,
            out_channels=3,
            context_dim=768
        ).to(self.device)
        
        self.context_encoder.load_state_dict(checkpoint["context_encoder"])
        self.action_embed.load_state_dict(checkpoint["action_embed"])
        self.unet.load_state_dict(checkpoint["unet"])
        
        self.context_encoder.eval()
        self.action_embed.eval()
        self.unet.eval()
        
        print(f"Loaded model from {self.config.model_checkpoint}")
        
    @torch.no_grad()
    def generate_frame(
        self,
        context_frames: torch.Tensor,
        action: int,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate next frame given context and action.
        Simple DDPM sampling without classifier-free guidance for analysis.
        """
        batch_size = context_frames.shape[0]
        
        context_embed = self.context_encoder(context_frames)
        
        action_tensor = torch.tensor([action], device=self.device).expand(batch_size)
        action_embed = self.action_embed(action_tensor).unsqueeze(1)
        
        conditioning = context_embed + action_embed
        
        x = torch.randn(
            batch_size, 3, 64, 64,
            device=self.device
        )
        
        for t in reversed(range(0, 1000, 1000 // num_inference_steps)):
            timestep = torch.tensor([t], device=self.device).expand(batch_size)
            
            noise_pred = self.unet(x, timestep, conditioning)
            
            alpha = (1 - t / 1000)
            alpha_prev = (1 - max(0, t - 1000 // num_inference_steps) / 1000)
            
            x = (x - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
            
            if t > 0:
                noise = torch.randn_like(x)
                x = alpha_prev.sqrt() * x + (1 - alpha_prev).sqrt() * noise
        
        return x
    
    def action_sensitivity_analysis(self) -> Dict:
        """
        Test 1: Action Sensitivity Analysis
        
        For the same context, do different actions produce meaningfully different outputs?
        A model that just generates "generic next frames" won't show action sensitivity.
        """
        print("\n" + "="*60)
        print("TEST 1: Action Sensitivity Analysis")
        print("="*60)
        
        results = {
            "action_pairs": [],
            "mean_difference": 0.0,
            "per_action_variance": []
        }
        
        num_test = min(self.config.num_samples, len(self.dataset))
        indices = np.random.choice(len(self.dataset), num_test, replace=False)
        
        all_differences = []
        
        for idx in tqdm(indices, desc="Testing action sensitivity"):
            sample = self.dataset[idx]
            context_frames = sample["context_frames"].unsqueeze(0).to(self.device)
            
            generated_frames = []
            for action in range(15):
                frame = self.generate_frame(context_frames, action, num_inference_steps=20)
                generated_frames.append(frame)
            
            frames_stack = torch.stack(generated_frames)
            
            for i in range(15):
                for j in range(i + 1, 15):
                    diff = F.mse_loss(frames_stack[i], frames_stack[j]).item()
                    all_differences.append({
                        "action_i": i,
                        "action_j": j,
                        "mse_diff": diff
                    })
        
        results["mean_difference"] = np.mean([d["mse_diff"] for d in all_differences])
        results["std_difference"] = np.std([d["mse_diff"] for d in all_differences])
        
        movement_actions = list(range(1, 9))  
        strategic_actions = list(range(9, 15)) 
        
        movement_diffs = [d["mse_diff"] for d in all_differences 
                        if d["action_i"] in movement_actions and d["action_j"] in movement_actions]
        strategic_diffs = [d["mse_diff"] for d in all_differences
                         if d["action_i"] in strategic_actions or d["action_j"] in strategic_actions]
        
        results["movement_action_mean_diff"] = np.mean(movement_diffs) if movement_diffs else 0
        results["strategic_action_mean_diff"] = np.mean(strategic_diffs) if strategic_diffs else 0
        
        print(f"\nResults:")
        print(f"  Mean MSE between different actions: {results['mean_difference']:.6f}")
        print(f"  Std: {results['std_difference']:.6f}")
        print(f"  Movement actions internal variance: {results['movement_action_mean_diff']:.6f}")
        print(f"  Strategic actions variance: {results['strategic_action_mean_diff']:.6f}")
        
        if results["mean_difference"] < 0.01:
            print("\n WARNING: Low action sensitivity - model may be ignoring actions!")
        elif results["strategic_action_mean_diff"] > results["movement_action_mean_diff"] * 1.5:
            print("\n  ✓ Strategic actions show higher variance (good sign)")
        
        self.results["action_sensitivity"] = results
        return results
    
    def temporal_coherence_analysis(self) -> Dict:
        """
        Test 2: Temporal Coherence Analysis
        
        Generate long rollouts and measure how frame quality degrades over time.
        Also check for "mode collapse" where all frames become similar.
        """
        print("\n" + "="*60)
        print("TEST 2: Temporal Coherence Analysis")
        print("="*60)
        
        results = {
            "rollout_quality": [],
            "frame_diversity": [],
            "degradation_rate": 0.0
        }
        
        num_test = min(10, len(self.dataset))  
        indices = np.random.choice(len(self.dataset), num_test, replace=False)
        
        for idx in tqdm(indices, desc="Generating rollouts"):
            sample = self.dataset[idx]
            context_frames = sample["context_frames"].unsqueeze(0).to(self.device)
            
            rollout_frames = [context_frames[0, -1]] 
            current_context = context_frames.clone()
            
            for step in range(self.config.num_rollout_steps):
                action = np.random.randint(0, 15)
                
                next_frame = self.generate_frame(current_context, action, num_inference_steps=10)
                rollout_frames.append(next_frame[0])
                
                current_context = torch.cat([
                    current_context[:, 1:],
                    next_frame.unsqueeze(1)
                ], dim=1)
            
            rollout_stack = torch.stack(rollout_frames)
            
            frame_diffs = []
            for i in range(1, len(rollout_frames)):
                diff = F.mse_loss(rollout_stack[i], rollout_stack[i-1]).item()
                frame_diffs.append(diff)
            
            results["rollout_quality"].append({
                "mean_frame_diff": np.mean(frame_diffs),
                "std_frame_diff": np.std(frame_diffs),
                "final_diff": frame_diffs[-1] if frame_diffs else 0
            })
            
            diversity_scores = []
            for _ in range(20):
                i, j = np.random.choice(len(rollout_frames), 2, replace=False)
                div = F.mse_loss(rollout_stack[i], rollout_stack[j]).item()
                diversity_scores.append(div)
            
            results["frame_diversity"].append(np.mean(diversity_scores))
        
        mean_quality = np.mean([r["mean_frame_diff"] for r in results["rollout_quality"]])
        mean_diversity = np.mean(results["frame_diversity"])
        
        print(f"\nResults:")
        print(f"  Mean frame-to-frame difference: {mean_quality:.6f}")
        print(f"  Mean frame diversity: {mean_diversity:.6f}")
        
        if mean_diversity < 0.01:
            print("\n WARNING: Low diversity - possible mode collapse!")
        else:
            print("\n  ✓ Reasonable frame diversity maintained")
        
        self.results["temporal_coherence"] = results
        return results
    
    def counterfactual_analysis(self) -> Dict:
        """
        Test 3: Counterfactual Analysis
        
        What happens when we give the model impossible or inconsistent inputs?
        A model with true understanding should produce confused/degraded outputs.
        A model just pattern-matching might not notice the inconsistency.
        """
        print("\n" + "="*60)
        print("TEST 3: Counterfactual Analysis")
        print("="*60)
        
        results = {
            "normal_outputs": [],
            "shuffled_context_outputs": [],
            "noise_context_outputs": [],
            "reversed_time_outputs": []
        }
        
        num_test = min(self.config.num_samples, len(self.dataset))
        indices = np.random.choice(len(self.dataset), num_test, replace=False)
        
        for idx in tqdm(indices, desc="Counterfactual testing"):
            sample = self.dataset[idx]
            context_frames = sample["context_frames"].unsqueeze(0).to(self.device)
            action = sample["current_action"].item()
            
            normal_out = self.generate_frame(context_frames, action, num_inference_steps=10)
            
            shuffled_context = context_frames[:, torch.randperm(context_frames.shape[1])]
            shuffled_out = self.generate_frame(shuffled_context, action, num_inference_steps=10)
            
            noise_context = torch.randn_like(context_frames)
            noise_out = self.generate_frame(noise_context, action, num_inference_steps=10)
            
            reversed_context = context_frames.flip(dims=[1])
            reversed_out = self.generate_frame(reversed_context, action, num_inference_steps=10)
            
            results["normal_outputs"].append(0) 
            results["shuffled_context_outputs"].append(
                F.mse_loss(normal_out, shuffled_out).item()
            )
            results["noise_context_outputs"].append(
                F.mse_loss(normal_out, noise_out).item()
            )
            results["reversed_time_outputs"].append(
                F.mse_loss(normal_out, reversed_out).item()
            )
        
        shuffled_diff = np.mean(results["shuffled_context_outputs"])
        noise_diff = np.mean(results["noise_context_outputs"])
        reversed_diff = np.mean(results["reversed_time_outputs"])
        
        print(f"\nResults (MSE from normal output):")
        print(f"  Shuffled context: {shuffled_diff:.6f}")
        print(f"  Noise context: {noise_diff:.6f}")
        print(f"  Reversed time: {reversed_diff:.6f}")
        
        if shuffled_diff < 0.01 and reversed_diff < 0.01:
            print("\n WARNING: Model ignores context ordering - may not understand time!")
        elif noise_diff < shuffled_diff:
            print("\n WARNING: Noise context affects less than shuffling - suspicious!")
        else:
            print("\n  ✓ Model shows appropriate sensitivity to context manipulations")
        
        self.results["counterfactual"] = {
            "shuffled_diff": shuffled_diff,
            "noise_diff": noise_diff,
            "reversed_diff": reversed_diff
        }
        return self.results["counterfactual"]
    
    def strategic_consistency_test(self) -> Dict:
        """
        Test 4: Strategic Consistency
        
        Test if the model maintains game rules and strategic consistency:
        - Resources should deplete when harvesting
        - Units should move in correct direction
        - Buildings should appear where built
        
        This is a simplified proxy test using visual pattern analysis.
        """
        print("\n" + "="*60)
        print("TEST 4: Strategic Consistency Test")
        print("="*60)
        
        results = {
            "movement_consistency": [],
            "action_effect_strength": []
        }
        
        num_test = min(50, len(self.dataset))
        indices = np.random.choice(len(self.dataset), num_test, replace=False)
        
        movement_map = {
            1: (0, -1),   # UP
            2: (1, -1),   # UP_RIGHT
            3: (1, 0),    # RIGHT
            4: (1, 1),    # DOWN_RIGHT
            5: (0, 1),    # DOWN
            6: (-1, 1),   # DOWN_LEFT
            7: (-1, 0),   # LEFT
            8: (-1, -1),  # UP_LEFT
        }
        
        for idx in tqdm(indices, desc="Testing strategic consistency"):
            sample = self.dataset[idx]
            context_frames = sample["context_frames"].unsqueeze(0).to(self.device)
            
            for action, (dx, dy) in movement_map.items():
                frame = self.generate_frame(context_frames, action, num_inference_steps=10)
                
                frame_np = frame[0].cpu().numpy().transpose(1, 2, 0)
                prev_frame_np = context_frames[0, -1].cpu().numpy().transpose(1, 2, 0)
                
                diff = np.abs(frame_np - prev_frame_np).mean(axis=2)
                
                h, w = diff.shape
                
                if dx > 0:  # Moving right
                    left_change = diff[:, :w//2].mean()
                    right_change = diff[:, w//2:].mean()
                    consistency = right_change - left_change
                elif dx < 0:  # Moving left
                    left_change = diff[:, :w//2].mean()
                    right_change = diff[:, w//2:].mean()
                    consistency = left_change - right_change
                elif dy > 0:  # Moving down
                    top_change = diff[:h//2, :].mean()
                    bottom_change = diff[h//2:, :].mean()
                    consistency = bottom_change - top_change
                elif dy < 0:  # Moving up
                    top_change = diff[:h//2, :].mean()
                    bottom_change = diff[h//2:, :].mean()
                    consistency = top_change - bottom_change
                else:
                    consistency = 0
                
                results["movement_consistency"].append({
                    "action": action,
                    "expected_direction": (dx, dy),
                    "consistency_score": consistency
                })
        
        mean_consistency = np.mean([r["consistency_score"] for r in results["movement_consistency"]])
        
        print(f"\nResults:")
        print(f"  Mean movement consistency score: {mean_consistency:.6f}")
        
        # A positive score means changes tend to happen in the expected direction
        if mean_consistency > 0:
            print("\n  ✓ Movement actions show directional consistency")
        else:
            print("\n  ⚠️  Movement actions don't show expected directional bias")
        
        self.results["strategic_consistency"] = {
            "mean_consistency": mean_consistency,
            "num_tests": len(results["movement_consistency"])
        }
        return self.results["strategic_consistency"]
    
    def run_all_analyses(self) -> Dict:
        """Run all analysis tests and generate report"""
        
        print("\n" + "="*60)
        print("DIFFUSION MODEL STRATEGY ANALYSIS")
        print("="*60)
        
        self.action_sensitivity_analysis()
        self.temporal_coherence_analysis()
        self.counterfactual_analysis()
        self.strategic_consistency_test()
        
        self._generate_report()
        
        return self.results
    
    def _generate_report(self):
        """Generate analysis report"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "analysis_results.json"
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        issues = []
        positives = []
        
        if "action_sensitivity" in self.results:
            if self.results["action_sensitivity"]["mean_difference"] < 0.01:
                issues.append("Low action sensitivity")
            else:
                positives.append("Good action sensitivity")
        
        if "counterfactual" in self.results:
            cf = self.results["counterfactual"]
            if cf["shuffled_diff"] < 0.01 and cf["reversed_diff"] < 0.01:
                issues.append("Ignores temporal ordering")
            else:
                positives.append("Sensitive to context manipulations")
        
        if "strategic_consistency" in self.results:
            if self.results["strategic_consistency"]["mean_consistency"] > 0:
                positives.append("Movement shows directional consistency")
            else:
                issues.append("No directional consistency in movement")
        
        print("\n✓ Positive indicators:")
        for p in positives:
            print(f"  - {p}")
        
        print("\n Potential issues:")
        for i in issues:
            print(f"  - {i}")
        
        # Verdict
        print("\n" + "-"*60)
        if len(issues) == 0:
            print("VERDICT: Model shows signs of understanding game logic")
        elif len(issues) <= len(positives):
            print("VERDICT: Model shows mixed results - partial understanding")
        else:
            print("VERDICT: Model may be mostly pattern-matching, not understanding strategy")
        
        print(f"\nFull results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze diffusion model for strategic understanding")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="./analysis_results",
                       help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples for analysis")
    parser.add_argument("--rollout-steps", type=int, default=50,
                       help="Number of steps for rollout analysis")
    
    args = parser.parse_args()
    
    config = AnalysisConfig(
        model_checkpoint=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_samples=args.num_samples,
        num_rollout_steps=args.rollout_steps
    )
    
    analyzer = DiffusionAnalyzer(config)
    results = analyzer.run_all_analyses()


if __name__ == "__main__":
    main()
