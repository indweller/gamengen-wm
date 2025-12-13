"""
Utilities for HMM-DDA Framework
Includes logging, normalization, plotting, and configuration management
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch


class Logger:
    """Structured logger for training metrics and events"""

    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize logger

        Args:
            log_dir: Directory to save logs
            experiment_name: Optional name for this experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.csv"

        # Initialize metrics DataFrame
        self.metrics_buffer = []

    def log(self, data: Dict[str, Any], print_console: bool = True):
        """
        Log data to file and optionally print to console

        Args:
            data: Dictionary of data to log
            print_console: Whether to print to console
        """
        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()

        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

        # Add to metrics buffer if it contains episode data
        if 'episode' in data:
            self.metrics_buffer.append(data)

        if print_console:
            self._print_log(data)

    def _print_log(self, data: Dict[str, Any]):
        """Pretty print log data to console"""
        if 'episode' in data:
            ep = data.get('episode', '?')
            state = data.get('state', '?')
            reward = data.get('reward', 0)
            t_score = data.get('T_score', 0)

            print(f"[Episode {ep:5d}] State: {state:10s} | "
                  f"Reward: {reward:8.2f} | T-Score: {t_score:.3f}")

    def save_metrics(self):
        """Save buffered metrics to CSV"""
        if self.metrics_buffer:
            df = pd.DataFrame(self.metrics_buffer)
            df.to_csv(self.metrics_file, index=False)

    def load_metrics(self) -> pd.DataFrame:
        """Load metrics from CSV"""
        if self.metrics_file.exists():
            return pd.read_csv(self.metrics_file)
        return pd.DataFrame()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from logged metrics"""
        df = self.load_metrics()
        if df.empty:
            return {}

        summary = {
            'total_episodes': len(df),
            'mean_reward': df['reward'].mean(),
            'std_reward': df['reward'].std(),
            'max_reward': df['reward'].max(),
            'min_reward': df['reward'].min(),
        }

        if 'state' in df.columns:
            summary['state_distribution'] = df['state'].value_counts().to_dict()

        return summary


def normalize_value(value: float, min_val: float, max_val: float,
                   clip: bool = True) -> float:
    """
    Normalize value to [0, 1] range

    Args:
        value: Value to normalize
        min_val: Minimum value
        max_val: Maximum value
        clip: Whether to clip to [0, 1]

    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 0.5

    normalized = (value - min_val) / (max_val - min_val)

    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)

    return normalized


def denormalize_value(normalized: float, min_val: float, max_val: float) -> float:
    """
    Denormalize value from [0, 1] to original range

    Args:
        normalized: Normalized value in [0, 1]
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Original value
    """
    return normalized * (max_val - min_val) + min_val


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file

    Args:
        config_path: Path to JSON file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to JSON file

    Args:
        config: Configuration dictionary
        config_path: Path to save JSON file
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def plot_learning_curve(episodes: List[int], rewards: List[float],
                        states: Optional[List[str]] = None,
                        save_path: Optional[str] = None,
                        window: int = 50):
    """
    Plot learning curve with optional state coloring

    Args:
        episodes: Episode numbers
        rewards: Rewards per episode
        states: Optional list of states per episode
        save_path: Path to save figure
        window: Window size for moving average
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw rewards
    if states is not None:
        state_colors = {'Low': 'green', 'Transition': 'orange', 'High': 'red'}
        for state_name in ['Low', 'Transition', 'High']:
            mask = [s == state_name for s in states]
            if any(mask):
                ax.scatter([e for e, m in zip(episodes, mask) if m],
                          [r for r, m in zip(rewards, mask) if m],
                          c=state_colors[state_name], label=state_name,
                          alpha=0.3, s=20)
    else:
        ax.scatter(episodes, rewards, alpha=0.3, s=20, label='Raw Rewards')

    # Plot moving average
    if len(rewards) >= window:
        rewards_smooth = pd.Series(rewards).rolling(window=window, center=True).mean()
        ax.plot(episodes, rewards_smooth, 'k-', linewidth=2, label=f'MA({window})')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_belief_evolution(episodes: List[int], beliefs: List[List[float]],
                          save_path: Optional[str] = None):
    """
    Plot HMM belief evolution over time

    Args:
        episodes: Episode numbers
        beliefs: List of belief distributions [P(Low), P(Transition), P(High)]
        save_path: Path to save figure
    """
    beliefs_array = np.array(beliefs)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['green', 'orange', 'red']
    labels = ['P(Low)', 'P(Transition)', 'P(High)']

    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(episodes, beliefs_array[:, i], color=color,
                linewidth=2, label=label, alpha=0.8)
        ax.fill_between(episodes, 0, beliefs_array[:, i],
                        color=color, alpha=0.2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Belief Probability', fontsize=12)
    ax.set_title('HMM Belief Evolution', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_t_score_distributions(calibration_scores: Dict[str, List[float]],
                               training_scores: Optional[Dict[str, List[float]]] = None,
                               save_path: Optional[str] = None):
    """
    Plot T-score distributions for each difficulty state

    Args:
        calibration_scores: Dict of {state: [T-scores]} from calibration
        training_scores: Optional dict of {state: [T-scores]} from training
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    states = ['Low', 'Transition', 'High']
    colors = ['green', 'orange', 'red']

    for ax, state, color in zip(axes, states, colors):
        # Plot calibration distribution
        if state in calibration_scores:
            ax.hist(calibration_scores[state], bins=30, alpha=0.5,
                   color=color, label='Calibration', density=True)

        # Plot training distribution if provided
        if training_scores and state in training_scores:
            ax.hist(training_scores[state], bins=30, alpha=0.5,
                   color='blue', label='Training', density=True)

        ax.set_xlabel('T-Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{state} Difficulty', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_state_transition_diagram(transition_counts: np.ndarray,
                                  save_path: Optional[str] = None):
    """
    Plot state transition diagram

    Args:
        transition_counts: 3x3 matrix of transition counts
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    states = ['Low', 'Transition', 'High']

    # Normalize to get probabilities
    transition_probs = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-8)

    # Create heatmap
    sns.heatmap(transition_probs, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=states, yticklabels=states,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Probability'})

    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title('State Transition Probabilities', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_state_distribution(state_counts: Dict[str, int],
                            save_path: Optional[str] = None):
    """
    Plot pie chart of state distribution

    Args:
        state_counts: Dictionary of {state: count}
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    states = ['Low', 'Transition', 'High']
    colors = ['green', 'orange', 'red']

    values = [state_counts.get(s, 0) for s in states]

    ax.pie(values, labels=states, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 12})
    ax.set_title('State Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_metrics_comparison(metrics_by_state: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None):
    """
    Plot bar chart comparing metrics across states

    Args:
        metrics_by_state: Dict of {state: {metric: value}}
        save_path: Path to save figure
    """
    states = ['Low', 'Transition', 'High']
    metrics = ['completion_rate', 'avg_reward', 'avg_deaths']
    metric_labels = ['Completion Rate', 'Average Reward', 'Average Deaths']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [metrics_by_state.get(s, {}).get(metric, 0) for s in states]
        colors = ['green', 'orange', 'red']

        ax.bar(states, values, color=colors, alpha=0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def check_cuda() -> Tuple[bool, str]:
    """
    Check CUDA availability

    Returns:
        Tuple of (is_available, device_name)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, device_name
        return False, "CPU"
    except ImportError:
        return False, "CPU (PyTorch not installed)"


def create_directories(base_dir: str):
    """
    Create all necessary directories for the project

    Args:
        base_dir: Base directory path
    """
    dirs = [
        'config',
        'logs',
        'checkpoints',
        'calibration_data',
        'figures/calibration',
        'figures/training',
        'figures/evaluation',
        'report/sections',
        'report/figures'
    ]

    for d in dirs:
        Path(base_dir) / d).mkdir(parents=True, exist_ok=True)
