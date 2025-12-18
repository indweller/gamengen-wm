"""
Evaluation Script
Evaluates trained agent and generates visualizations.

- Learning curve with state coloring
- Belief evolution over time
- State distribution analysis
- Flow zone percentage (target: >60%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from collections import Counter

from src.utils import (
    plot_learning_curve,
    plot_belief_evolution,
    plot_state_distribution,
    plot_metrics_comparison,
    set_random_seeds
)


def load_training_data(checkpoint_dir, log_dir):
    """Load training metrics and HMM state"""
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    
    metrics_df = None
    
    metrics_files = sorted(checkpoint_dir.glob('metrics_*.csv'))
    if metrics_files:
        metrics_df = pd.read_csv(metrics_files[-1])
        print(f"Loaded metrics from {metrics_files[-1]}")
    
    if metrics_df is None:
        log_metrics = sorted(log_dir.glob('*_metrics.csv'))
        if log_metrics:
            metrics_df = pd.read_csv(log_metrics[-1])
            print(f"Loaded metrics from {log_metrics[-1]}")
    
    if metrics_df is None:
        print("No metrics file found")
        metrics_df = pd.DataFrame()
    
    hmm_data = {}
    hmm_files = sorted(checkpoint_dir.glob('hmm_*.json'))
    if hmm_files:
        with open(hmm_files[-1]) as f:
            hmm_data = json.load(f)
        print(f"Loaded HMM state from {hmm_files[-1]}")
    
    return metrics_df, hmm_data


def compute_metrics_by_state(metrics_df, hmm_data):
    """Compute performance metrics broken down by difficulty state"""
    state_history = hmm_data.get('state_history', [])
    
    if len(state_history) == 0 or metrics_df.empty:
        return {}
    
    if 'state' in metrics_df.columns:
        pass
    elif len(state_history) == len(metrics_df):
        metrics_df = metrics_df.copy()
        metrics_df['state'] = state_history
    else:
        print(f"Warning: state_history length ({len(state_history)}) != metrics length ({len(metrics_df)})")
        return {}
    
    metrics_by_state = {}
    
    for state in ['Low', 'Transition', 'High']:
        state_df = metrics_df[metrics_df['state'] == state]
        
        if len(state_df) > 0:
            metrics_by_state[state] = {
                'completion_rate': state_df['completed'].mean() if 'completed' in state_df else 0,
                'avg_reward': state_df['reward'].mean() if 'reward' in state_df else 0,
                'avg_deaths': state_df['deaths'].mean() if 'deaths' in state_df else 0,
                'avg_max_x': state_df['max_x'].mean() if 'max_x' in state_df else 0,
                'episode_count': len(state_df)
            }
    
    return metrics_by_state


def generate_visualizations(metrics_df, hmm_data, output_dir):
    """Generate all evaluation visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    if not metrics_df.empty and 'reward' in metrics_df.columns:
        if 'episode' not in metrics_df.columns:
            metrics_df = metrics_df.copy()
            metrics_df['episode'] = range(len(metrics_df))
        
        episodes = metrics_df['episode'].tolist()
        rewards = metrics_df['reward'].tolist()
        states = hmm_data.get('state_history', None)
        
        if states and len(states) == len(rewards):
            try:
                plot_learning_curve(
                    episodes, rewards, states=states,
                    save_path=str(output_dir / 'learning_curve.png'),
                    window=50
                )
                print("  ✓ Learning curve saved")
            except Exception as e:
                print(f"  ✗ Learning curve failed: {e}")
        else:
            try:
                plot_learning_curve(
                    episodes, rewards, states=None,
                    save_path=str(output_dir / 'learning_curve.png'),
                    window=50
                )
                print("  ✓ Learning curve saved (no state coloring)")
            except Exception as e:
                print(f"  ✗ Learning curve failed: {e}")
    
    belief_history = hmm_data.get('belief_history', [])
    if belief_history:
        try:
            belief_episodes = list(range(len(belief_history)))
            plot_belief_evolution(
                belief_episodes, belief_history,
                save_path=str(output_dir / 'belief_evolution.png')
            )
            print("  ✓ Belief evolution saved")
        except Exception as e:
            print(f"  ✗ Belief evolution failed: {e}")
    
    state_history = hmm_data.get('state_history', [])
    if state_history:
        try:
            state_counts = Counter(state_history)
            plot_state_distribution(
                state_counts,
                save_path=str(output_dir / 'state_distribution.png')
            )
            print("  ✓ State distribution saved")
        except Exception as e:
            print(f"  ✗ State distribution failed: {e}")
    
    metrics_by_state = compute_metrics_by_state(metrics_df, hmm_data)
    if metrics_by_state:
        try:
            plot_metrics_comparison(
                metrics_by_state,
                save_path=str(output_dir / 'metrics_comparison.png')
            )
            print("  ✓ Metrics comparison saved")
        except Exception as e:
            print(f"  ✗ Metrics comparison failed: {e}")


def print_summary_statistics(metrics_df, hmm_data, metrics_by_state):
    """Print comprehensive summary statistics"""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    if not metrics_df.empty:
        print("\nOverall Performance:")
        print(f"  Total episodes: {len(metrics_df)}")
        if 'completed' in metrics_df.columns:
            print(f"  Completion rate: {metrics_df['completed'].mean():.2%}")
        if 'reward' in metrics_df.columns:
            print(f"  Average reward: {metrics_df['reward'].mean():.2f}")
            print(f"  Reward std: {metrics_df['reward'].std():.2f}")
        if 'deaths' in metrics_df.columns:
            print(f"  Average deaths: {metrics_df['deaths'].mean():.2f}")
        if 'max_x' in metrics_df.columns:
            print(f"  Average max_x: {metrics_df['max_x'].mean():.1f}")
    
    state_history = hmm_data.get('state_history', [])
    if state_history:
        state_counts = Counter(state_history)
        total = len(state_history)
        
        print("\nState Distribution:")
        for state in ['Low', 'Transition', 'High']:
            count = state_counts.get(state, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"  {state}: {count} episodes ({pct:.1f}%)")
        
        transitions = sum(1 for i in range(1, len(state_history)) 
                        if state_history[i] != state_history[i-1])
        trans_freq = transitions / len(state_history) * 100 if state_history else 0
        print(f"\nTransition Frequency: {trans_freq:.1f} per 100 episodes")
    
    if metrics_by_state:
        print("\nPerformance by State:")
        for state, metrics in metrics_by_state.items():
            print(f"\n  {state}:")
            print(f"    Episodes: {metrics['episode_count']}")
            print(f"    Completion rate: {metrics['completion_rate']:.2%}")
            print(f"    Avg reward: {metrics['avg_reward']:.2f}")
            print(f"    Avg deaths: {metrics['avg_deaths']:.2f}")
    
    if not metrics_df.empty and 'reward' in metrics_df.columns:
        rewards = metrics_df['reward'].values
        median_r = np.median(rewards)
        std_r = np.std(rewards)
        flow_min = median_r - 0.5 * std_r
        flow_max = median_r + 0.5 * std_r
        
        in_flow = np.sum((rewards >= flow_min) & (rewards <= flow_max))
        flow_pct = in_flow / len(rewards) * 100
        
        print(f"\nFlow Zone Analysis (Section 7):")
        print(f"  Flow zone: [{flow_min:.1f}, {flow_max:.1f}]")
        print(f"  Episodes in flow zone: {in_flow} ({flow_pct:.1f}%)")
        print(f"  Target: >60%")
        if flow_pct >= 60:
            print("  ✓ Target achieved!")
        else:
            print(f"  ✗ Below target by {60 - flow_pct:.1f}%")


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("HMM-DDA Evaluation")
    print("=" * 60)
    
    set_random_seeds(42)
    
    BASE_DIR = Path(__file__).parent.parent
    CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
    LOG_DIR = BASE_DIR / 'logs'
    OUTPUT_DIR = BASE_DIR / 'figures' / 'evaluation'
    
    print("\nLoading training data...")
    metrics_df, hmm_data = load_training_data(CHECKPOINT_DIR, LOG_DIR)
    
    if metrics_df.empty and not hmm_data:
        print("\nNo training data found. Run training first:")
        print("  python scripts/train.py")
        return
    
    print(f"Loaded {len(metrics_df)} episodes")
    
    metrics_by_state = compute_metrics_by_state(metrics_df, hmm_data)
    
    generate_visualizations(metrics_df, hmm_data, OUTPUT_DIR)
    
    print_summary_statistics(metrics_df, hmm_data, metrics_by_state)
    
    print("\n" + "=" * 60)
    print(f"Evaluation complete!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
