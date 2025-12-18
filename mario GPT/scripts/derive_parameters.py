"""
Parameter Derivation Script
Derives HMM parameters from calibration data using statistical methods.

- Fit Gaussian emission distributions to T-scores at each difficulty
- Compute optimal metric weights using Fisher's discriminant
- Derive T-score thresholds from distribution intersections
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from src.t_score import compute_T_score_from_metrics
from src.utils import save_config, set_random_seeds


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
        'death_rate_max': max(float(combined['deaths'].max()), 1.0),
        'reward_min': float(combined['reward'].min()),
        'reward_max': float(combined['reward'].max()),
        'reward_trend_min': -50.0,
        'reward_trend_max': 50.0,
        'time_to_complete_max': max(float(combined['frames'].max()), 1000.0),
        'progress_variance_max': max(float(combined['max_x'].std()) * 3, 100.0),
    }
    
    print("Normalization Bounds:")
    for key, value in bounds.items():
        print(f"  {key}: {value:.2f}")
    
    return bounds


def compute_t_scores_for_episodes(data, weights, normalization):
    """Compute T-scores for all calibration episodes"""
    t_scores = {}
    
    for difficulty, df in data.items():
        t_scores[difficulty] = []
        
        for _, row in df.iterrows():
            metrics = {
                'completion_rate': float(row['completed']),
                'death_rate': float(row['deaths']),
                'reward_trend': 0.0,  
                'time_to_complete': float(row['frames']) if row['completed'] else 0.0,
                'progress_variance': 0.0
            }
            
            config = {'weights': weights, 'normalization': normalization}
            t = compute_T_score_from_metrics(metrics, config)
            t_scores[difficulty].append(t)
    
    return t_scores


def fit_gaussian_distributions(t_scores):
    """
    Fit Gaussian distributions to T-scores for each difficulty.
    
    From Framework Section 4.3:
    - B_Low(T) ~ N(μ=0.25, σ=0.15)
    - B_Transition(T) ~ N(μ=0.50, σ=0.12)  # Tighter variance
    - B_High(T) ~ N(μ=0.75, σ=0.15)
    """
    emissions = {}
    state_map = {'low': 'Low', 'transition': 'Transition', 'high': 'High'}
    
    print("\nFitted Emission Distributions:")
    
    for difficulty, scores in t_scores.items():
        state_name = state_map[difficulty]
        
        if len(scores) > 0:
            mu = np.mean(scores)
            sigma = max(np.std(scores), 0.05)  
            
            if state_name == 'Transition':
                sigma = min(sigma, 0.12)
        else:
            defaults = {'Low': (0.25, 0.15), 'Transition': (0.50, 0.12), 'High': (0.75, 0.15)}
            mu, sigma = defaults[state_name]
        
        emissions[state_name] = {'mu': float(mu), 'sigma': float(sigma)}
        print(f"  {state_name}: μ={mu:.3f}, σ={sigma:.3f}")
    
    return emissions


def derive_thresholds(emissions):
    """
    Derive T-score thresholds where distributions intersect.
    
    Uses Gaussian percentiles from calibration data.
    """
    low_mu = emissions['Low']['mu']
    low_sigma = emissions['Low']['sigma']
    high_mu = emissions['High']['mu']
    high_sigma = emissions['High']['sigma']
    
    thresh_low_trans = low_mu + 0.674 * low_sigma  # ~75th percentile
    
    thresh_trans_high = high_mu - 0.674 * high_sigma  # ~25th percentile
    
    if thresh_low_trans >= thresh_trans_high:
        thresh_low_trans = low_mu
        thresh_trans_high = high_mu
    
    thresholds = {
        'low_transition': float(thresh_low_trans),
        'transition_high': float(thresh_trans_high)
    }
    
    print("\nDerived Thresholds:")
    print(f"  Low → Transition: {thresh_low_trans:.3f}")
    print(f"  Transition → High: {thresh_trans_high:.3f}")
    
    return thresholds


def optimize_metric_weights(data, normalization):
    """
    Optimize metric weights using Fisher's discriminant.
    
    Goal: Maximize separation between difficulty T-score distributions.
    """
    print("\nOptimizing Metric Weights...")
    
    def fisher_score(weights):
        """Compute negative Fisher's discriminant (for minimization)"""
        weights = np.abs(weights)
        weights = weights / weights.sum()
        
        t_scores = compute_t_scores_for_episodes(data, weights.tolist(), normalization)
        
        means = [np.mean(t_scores[d]) for d in ['low', 'transition', 'high']]
        variances = [np.var(t_scores[d]) + 1e-6 for d in ['low', 'transition', 'high']]
        
        between_var = np.var(means)
        
        within_var = np.mean(variances)
        
        fisher = between_var / (within_var + 1e-6)
        return -fisher
    
    initial_weights = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
    
    result = minimize(
        fisher_score,
        x0=initial_weights,
        method='Nelder-Mead',
        options={'maxiter': 100}
    )
    
    optimal = np.abs(result.x)
    optimal = optimal / optimal.sum()
    
    weight_names = ['CR', 'DR', 'RT', 'TTC', 'PV']
    print("Optimized Weights:")
    for name, w in zip(weight_names, optimal):
        print(f"  {name}: {w:.3f}")
    
    return optimal.tolist()


def create_transition_matrix():
    """
    Create transition matrix for Option 2 design.
    
    From Framework Section 3.2:
    - Low/High have high self-loop (0.70) for stability
    - Transition has LOW self-loop (0.40) for quick assessment
    """
    A = np.array([
        [0.70, 0.25, 0.05],  # Low: stable, gradual progression
        [0.20, 0.40, 0.40],  # Transition: quick decision to Low or High
        [0.05, 0.25, 0.70]   # High: stable, occasional regression
    ])
    
    print("\nTransition Matrix (Assessment State Design):")
    print(A)
    print("\nKey: Transition self-loop = 0.40 (quick assessment)")
    
    return {
        'matrix': A.tolist(),
        'states': ['Low', 'Transition', 'High'],
        'design': 'Transition as Assessment State'
    }


def visualize_distributions(t_scores, emissions, save_dir):
    """Visualize T-score distributions"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        states = ['Low', 'Transition', 'High']
        colors = ['green', 'orange', 'red']
        state_map = {'Low': 'low', 'Transition': 'transition', 'High': 'high'}
        
        for ax, state, color in zip(axes, states, colors):
            scores = t_scores[state_map[state]]
            
            ax.hist(scores, bins=20, alpha=0.5, color=color, density=True, label='Data')
            
            mu, sigma = emissions[state]['mu'], emissions[state]['sigma']
            x = np.linspace(0, 1, 100)
            y = norm.pdf(x, mu, sigma)
            ax.plot(x, y, color=color, linewidth=2, label=f'N({mu:.2f}, {sigma:.2f})')
            
            ax.set_xlabel('T-Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{state} Difficulty')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = Path(save_dir) / 'tscore_distributions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved visualization to {save_path}")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")


def main():
    """Main parameter derivation process"""
    print("=" * 60)
    print("HMM Parameter Derivation")
    print("=" * 60)
    
    set_random_seeds(42)
    
    BASE_DIR = Path(__file__).parent.parent
    CALIBRATION_DIR = BASE_DIR / 'calibration_data'
    CONFIG_DIR = BASE_DIR / 'config'
    FIGURES_DIR = BASE_DIR / 'figures' / 'calibration'
    
    CONFIG_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not (CALIBRATION_DIR / 'low_metrics.csv').exists():
        print("\nNo calibration data found. Using framework defaults.")
        
        save_config({
            'death_rate_max': 5.0,
            'reward_trend_min': -50.0,
            'reward_trend_max': 50.0,
            'time_to_complete_max': 2000.0,
            'progress_variance_max': 500.0
        }, CONFIG_DIR / 'normalization_bounds.json')
        
        save_config({'weights': [0.25, 0.20, 0.25, 0.15, 0.15]}, 
                   CONFIG_DIR / 'metric_weights.json')
        
        save_config({
            'Low': {'mu': 0.25, 'sigma': 0.15},
            'Transition': {'mu': 0.50, 'sigma': 0.12},
            'High': {'mu': 0.75, 'sigma': 0.15}
        }, CONFIG_DIR / 'emission_params.json')
        
        save_config({'low_transition': 0.35, 'transition_high': 0.65},
                   CONFIG_DIR / 'thresholds.json')
        
        save_config(create_transition_matrix(), CONFIG_DIR / 'transition_matrix.json')
        
        save_config({
            'Low': "few enemies, no gaps, many pipes, low elevation, easy",
            'Transition': "varied challenges, mixed density, skill assessment",
            'High': "many enemies, many gaps, few pipes, high elevation, hard"
        }, CONFIG_DIR / 'prompts.json')
        
        print(f"\nDefault parameters saved to {CONFIG_DIR}")
        return
    
    print("\nLoading calibration data...")
    data = load_calibration_data(CALIBRATION_DIR)
    
    print("\n" + "=" * 60)
    print("Step 1: Computing Normalization Bounds")
    print("=" * 60)
    normalization = compute_normalization_bounds(data)
    save_config(normalization, CONFIG_DIR / 'normalization_bounds.json')
    
    print("\n" + "=" * 60)
    print("Step 2: Optimizing Metric Weights")
    print("=" * 60)
    weights = optimize_metric_weights(data, normalization)
    save_config({'weights': weights}, CONFIG_DIR / 'metric_weights.json')
    
    print("\n" + "=" * 60)
    print("Step 3: Computing T-Scores")
    print("=" * 60)
    t_scores = compute_t_scores_for_episodes(data, weights, normalization)
    for diff, scores in t_scores.items():
        print(f"  {diff}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
    
    print("\n" + "=" * 60)
    print("Step 4: Fitting Emission Distributions")
    print("=" * 60)
    emissions = fit_gaussian_distributions(t_scores)
    save_config(emissions, CONFIG_DIR / 'emission_params.json')
    
    print("\n" + "=" * 60)
    print("Step 5: Deriving Thresholds")
    print("=" * 60)
    thresholds = derive_thresholds(emissions)
    save_config(thresholds, CONFIG_DIR / 'thresholds.json')
    
    print("\n" + "=" * 60)
    print("Step 6: Creating Transition Matrix")
    print("=" * 60)
    transition_matrix = create_transition_matrix()
    save_config(transition_matrix, CONFIG_DIR / 'transition_matrix.json')
    
    prompts = {
        'Low': "few enemies, no gaps, many pipes, low elevation, easy",
        'Transition': "varied challenges, mixed density, skill assessment",
        'High': "many enemies, many gaps, few pipes, high elevation, hard"
    }
    save_config(prompts, CONFIG_DIR / 'prompts.json')
    
    print("\n" + "=" * 60)
    print("Step 7: Visualizing Distributions")
    print("=" * 60)
    visualize_distributions(t_scores, emissions, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("Parameter Derivation Complete!")
    print("=" * 60)
    print(f"\nParameters saved to {CONFIG_DIR}")
    print(f"\nNext: Run 'python scripts/train.py'")


if __name__ == "__main__":
    main()
