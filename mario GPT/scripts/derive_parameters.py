"""
Parameter Derivation Script
Derives HMM parameters from calibration data using statistical methods
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize

from src.t_score import compute_T_score_from_metrics
from src.utils import save_config, plot_t_score_distributions


def load_calibration_data(calibration_dir):
    """Load calibration data from CSV files"""
    data = {}
    for difficulty in ['low', 'transition', 'high']:
        filepath = Path(calibration_dir) / f'{difficulty}_metrics.csv'
        if filepath.exists():
            data[difficulty] = pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Calibration data not found: {filepath}")

    return data


def compute_normalization_bounds(data):
    """Compute global min/max for each metric across all difficulties"""
    all_dfs = list(data.values())
    combined = pd.concat(all_dfs, ignore_index=True)

    bounds = {
        'death_rate_max': float(combined['deaths'].max()),
        'reward_min': float(combined['reward'].min()),
        'reward_max': float(combined['reward'].max()),
        'reward_trend_min': -100.0,  # Will be computed from trends
        'reward_trend_max': 100.0,
        'time_to_complete_max': float(combined[combined['completed']]['frames'].max()) if combined['completed'].any() else 5000.0,
        'progress_variance_max': float(combined['max_x'].std()) * 2,  # 2 std devs as max
    }

    print("Normalization Bounds:")
    for key, value in bounds.items():
        print(f"  {key}: {value:.2f}")

    return bounds


def compute_metrics_from_episodes(df):
    """Compute aggregated metrics from episode data"""
    n = len(df)

    metrics = {
        'completion_rate': df['completed'].mean(),
        'death_rate': df['deaths'].mean(),
        'reward_trend': 0.0,  # Simplified for calibration
        'time_to_complete': df[df['completed']]['frames'].mean() if df['completed'].any() else 0.0,
        'progress_variance': df['max_x'].std()
    }

    return metrics


def compute_t_scores(data, weights, normalization):
    """Compute T-scores for all calibration episodes"""
    config = {
        'weights': weights,
        'normalization': normalization
    }

    t_scores = {}

    for difficulty, df in data.items():
        t_scores[difficulty] = []

        for _, row in df.iterrows():
            metrics = {
                'completion_rate': float(row['completed']),
                'death_rate': float(row['deaths']),
                'reward_trend': 0.0,
                'time_to_complete': float(row['frames']) if row['completed'] else 0.0,
                'progress_variance': 0.0  # Single episode doesn't have variance
            }

            t = compute_T_score_from_metrics(metrics, config)
            t_scores[difficulty].append(t)

    return t_scores


def fit_gaussian_distributions(t_scores):
    """Fit Gaussian distributions to T-scores for each difficulty"""
    emissions = {}

    state_map = {
        'low': 'Low',
        'transition': 'Transition',
        'high': 'High'
    }

    for difficulty, scores in t_scores.items():
        state_name = state_map[difficulty]

        mu = np.mean(scores)
        sigma = np.std(scores)

        emissions[state_name] = {
            'mu': float(mu),
            'sigma': float(sigma)
        }

        print(f"{state_name}: μ={mu:.3f}, σ={sigma:.3f}")

    return emissions


def derive_thresholds(emissions):
    """Derive T-score thresholds where distributions intersect"""
    # Simplified percentile-based thresholds
    low_mu, low_sigma = emissions['Low']['mu'], emissions['Low']['sigma']
    trans_mu, trans_sigma = emissions['Transition']['mu'], emissions['Transition']['sigma']
    high_mu, high_sigma = emissions['High']['mu'], emissions['High']['sigma']

    # Use 75th percentile of low and 25th percentile of high
    thresh_low_trans = low_mu + 0.67 * low_sigma  # ~75th percentile
    thresh_trans_high = high_mu - 0.67 * high_sigma  # ~25th percentile

    thresholds = {
        'low_transition': float(thresh_low_trans),
        'transition_high': float(thresh_trans_high)
    }

    print("Thresholds:")
    print(f"  Low → Transition: {thresh_low_trans:.3f}")
    print(f"  Transition → High: {thresh_trans_high:.3f}")

    return thresholds


def initialize_transition_matrix():
    """
    Initialize transition matrix for Option 2 design

    Design Philosophy:
    - Low (Easy): High self-loop, moderate to Transition, low to High
    - Transition (Assessment): LOW self-loop (quickly decide), high to Low/High
    - High (Hard): High self-loop, moderate to Transition, low to Low

    The Transition state has lower self-loop (0.40 vs 0.75) because it's
    an assessment state - once skill is measured, it should quickly
    transition to either Low or High difficulty.
    """
    A = np.array([
        [0.75, 0.20, 0.05],  # Low: mostly stay easy, sometimes assess, rarely go hard
        [0.30, 0.40, 0.30],  # Transition: LOWER self-loop, balanced exits to Low/High
        [0.05, 0.20, 0.75]   # High: mostly stay hard, sometimes assess, rarely go easy
    ])

    matrix_data = {
        'matrix': A.tolist(),
        'states': ['Low', 'Transition', 'High'],
        'design': 'Option 2 - Transition as Assessment State'
    }

    print("Initial Transition Matrix (Option 2 - Assessment State Design):")
    print(A)
    print("\nKey feature: Transition state has LOWER self-loop (0.40)")
    print("This ensures quick assessment → decision (Low or High)")

    return matrix_data


def optimize_metric_weights(data, normalization, initial_weights=None):
    """Optimize metric weights using Fisher's discriminant"""
    if initial_weights is None:
        initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    def fisher_score(weights):
        """Compute negative Fisher's discriminant (for minimization)"""
        weights = weights / weights.sum()  # Normalize

        config = {'weights': weights.tolist(), 'normalization': normalization}
        t_scores = compute_t_scores(data, weights.tolist(), normalization)

        means = [np.mean(t_scores[d]) for d in ['low', 'transition', 'high']]
        variances = [np.var(t_scores[d]) for d in ['low', 'transition', 'high']]

        # Between-class variance
        overall_mean = np.mean(means)
        between_var = np.var(means)

        # Within-class variance
        within_var = np.mean(variances)

        # Fisher's discriminant
        if within_var > 0:
            fisher = between_var / within_var
        else:
            fisher = 0.0

        return -fisher  # Negative for minimization

    # Optimize
    result = minimize(
        fisher_score,
        x0=initial_weights,
        method='SLSQP',
        bounds=[(0.0, 1.0)] * 5,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    )

    optimal_weights = result.x / result.x.sum()  # Ensure sum = 1

    print("Optimized Metric Weights:")
    weight_names = ['CR', 'DR', 'RT', 'TTC', 'PV']
    for name, weight in zip(weight_names, optimal_weights):
        print(f"  {name}: {weight:.3f}")

    return optimal_weights.tolist()


def create_prompts_config():
    """
    Create prompts configuration

    Design Philosophy (Option 2):
    - Low: Easy difficulty for struggling players
    - Transition: ASSESSMENT STATE with varied/unpredictable patterns to measure skill
    - High: Hard difficulty for skilled players

    The Transition state generates diverse levels to gauge player skill,
    helping the HMM decide whether to go Easy (Low) or Hard (High).
    """
    prompts = {
        'Low': "few enemies, no gaps, many pipes, low elevation, easy difficulty",
        'Transition': "varied challenges, mixed enemy density, unpredictable patterns, some gaps and pipes, skill assessment level",
        'High': "many enemies, many gaps, few pipes, high elevation, hard difficulty"
    }

    return prompts


def visualize_distributions(t_scores, save_dir):
    """Visualize T-score distributions"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert keys to match expected format
    calibration_scores = {
        'Low': t_scores['low'],
        'Transition': t_scores['transition'],
        'High': t_scores['high']
    }

    plot_t_score_distributions(
        calibration_scores,
        save_path=save_dir / 'tscore_distributions.png'
    )

    print(f"Saved visualization to {save_dir / 'tscore_distributions.png'}")


def main():
    """Main parameter derivation process"""
    print("="*60)
    print("HMM Parameter Derivation")
    print("="*60)

    BASE_DIR = Path(__file__).parent.parent
    CALIBRATION_DIR = BASE_DIR / 'calibration_data'
    CONFIG_DIR = BASE_DIR / 'config'
    FIGURES_DIR = BASE_DIR / 'figures' / 'calibration'

    CONFIG_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    print("\nLoading calibration data...")
    data = load_calibration_data(CALIBRATION_DIR)

    # Step 1: Compute normalization bounds
    print("\n" + "="*60)
    print("Step 1: Computing Normalization Bounds")
    print("="*60)
    normalization = compute_normalization_bounds(data)
    save_config(normalization, CONFIG_DIR / 'normalization_bounds.json')

    # Step 2: Optimize metric weights
    print("\n" + "="*60)
    print("Step 2: Optimizing Metric Weights")
    print("="*60)
    weights = optimize_metric_weights(data, normalization)
    save_config({'weights': weights}, CONFIG_DIR / 'metric_weights.json')

    # Step 3: Compute T-scores
    print("\n" + "="*60)
    print("Step 3: Computing T-Scores")
    print("="*60)
    t_scores = compute_t_scores(data, weights, normalization)

    # Step 4: Fit Gaussian distributions
    print("\n" + "="*60)
    print("Step 4: Fitting Gaussian Distributions")
    print("="*60)
    emissions = fit_gaussian_distributions(t_scores)
    save_config(emissions, CONFIG_DIR / 'emission_params.json')

    # Step 5: Derive thresholds
    print("\n" + "="*60)
    print("Step 5: Deriving Thresholds")
    print("="*60)
    thresholds = derive_thresholds(emissions)
    save_config(thresholds, CONFIG_DIR / 'thresholds.json')

    # Step 6: Initialize transition matrix
    print("\n" + "="*60)
    print("Step 6: Initializing Transition Matrix")
    print("="*60)
    transition_matrix = initialize_transition_matrix()
    save_config(transition_matrix, CONFIG_DIR / 'transition_matrix.json')

    # Step 7: Create prompts config
    print("\n" + "="*60)
    print("Step 7: Creating Prompts Configuration")
    print("="*60)
    prompts = create_prompts_config()
    save_config(prompts, CONFIG_DIR / 'prompts.json')

    # Step 8: Visualize
    print("\n" + "="*60)
    print("Step 8: Visualizing Distributions")
    print("="*60)
    visualize_distributions(t_scores, FIGURES_DIR)

    print("\n" + "="*60)
    print("Parameter Derivation Complete!")
    print("="*60)
    print(f"\nAll parameters saved to {CONFIG_DIR}")
    print(f"Visualizations saved to {FIGURES_DIR}")
    print("\nNext step: Run 'python scripts/train.py' to begin training")


if __name__ == "__main__":
    main()
