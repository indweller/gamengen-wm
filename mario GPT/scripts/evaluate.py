"""
Evaluation Script
Evaluates trained agent and generates visualizations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from pathlib import Path
import pandas as pd
import json

from src.utils import (
    plot_learning_curve,
    plot_belief_evolution,
    plot_state_distribution,
    plot_metrics_comparison
)


def load_training_data(log_dir, checkpoint_dir):
    """Load training metrics and HMM state"""
    # Load metrics from logger
    metrics_file = list(Path(log_dir).glob('*_metrics.csv'))
    if metrics_file:
        metrics_df = pd.read_csv(metrics_file[0])
    else:
        # Load from checkpoint
        metrics_df = pd.read_csv(checkpoint_dir / 'metrics_final.csv')

    # Load HMM state
    hmm_file = checkpoint_dir / 'hmm_final.json'
    with open(hmm_file, 'r') as f:
        hmm_data = json.load(f)

    return metrics_df, hmm_data


def compute_metrics_by_state(metrics_df, hmm_data):
    """Compute performance metrics broken down by difficulty state"""
    state_history = hmm_data.get('state_history', [])

    if len(state_history) == 0:
        return {}

    # Add state to metrics_df
    if 'state' not in metrics_df.columns and len(state_history) == len(metrics_df):
        metrics_df['state'] = state_history

    metrics_by_state = {}

    for state in ['Low', 'Transition', 'High']:
        state_df = metrics_df[metrics_df.get('state', '') == state]

        if len(state_df) > 0:
            metrics_by_state[state] = {
                'completion_rate': state_df['completed'].mean(),
                'avg_reward': state_df['reward'].mean(),
                'avg_deaths': state_df['deaths'].mean(),
                'avg_max_x': state_df['max_x'].mean(),
                'episode_count': len(state_df)
            }

    return metrics_by_state


def generate_visualizations(metrics_df, hmm_data, output_dir):
    """Generate all evaluation visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # 1. Learning curve with state coloring
    if 'episode' not in metrics_df.columns:
        metrics_df['episode'] = range(len(metrics_df))

    episodes = metrics_df['episode'].tolist()
    rewards = metrics_df['reward'].tolist()
    states = hmm_data.get('state_history', None)

    if len(states) == len(rewards):
        plot_learning_curve(
            episodes, rewards, states=states,
            save_path=output_dir / 'learning_curve.png',
            window=50
        )
        print(f"  ✓ Learning curve saved")
    else:
        print(f"  ⚠ Skipping learning curve (state history length mismatch)")

    # 2. Belief evolution
    belief_history = hmm_data.get('belief_history', [])
    if belief_history:
        belief_episodes = range(len(belief_history))
        plot_belief_evolution(
            list(belief_episodes), belief_history,
            save_path=output_dir / 'belief_evolution.png'
        )
        print(f"  ✓ Belief evolution saved")

    # 3. State distribution
    if states:
        from collections import Counter
        state_counts = Counter(states)
        plot_state_distribution(
            state_counts,
            save_path=output_dir / 'state_distribution.png'
        )
        print(f"  ✓ State distribution saved")

    # 4. Metrics by state
    metrics_by_state = compute_metrics_by_state(metrics_df, hmm_data)
    if metrics_by_state:
        plot_metrics_comparison(
            metrics_by_state,
            save_path=output_dir / 'metrics_comparison.png'
        )
        print(f"  ✓ Metrics comparison saved")


def print_summary_statistics(metrics_df, hmm_data, metrics_by_state):
    """Print comprehensive summary statistics"""
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)

    # Overall stats
    print("\nOverall Performance:")
    print(f"  Total episodes: {len(metrics_df)}")
    print(f"  Completion rate: {metrics_df['completed'].mean():.2%}")
    print(f"  Average reward: {metrics_df['reward'].mean():.2f}")
    print(f"  Average deaths: {metrics_df['deaths'].mean():.2f}")
    print(f"  Average max_x: {metrics_df['max_x'].mean():.0f}")

    # State distribution
    state_history = hmm_data.get('state_history', [])
    if state_history:
        from collections import Counter
        state_counts = Counter(state_history)
        total = len(state_history)

        print("\nState Distribution:")
        for state in ['Low', 'Transition', 'High']:
            count = state_counts.get(state, 0)
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {state}: {count} episodes ({percentage:.1f}%)")

    # Transition frequency
    transitions = 0
    for i in range(1, len(state_history)):
        if state_history[i] != state_history[i-1]:
            transitions += 1

    transition_freq = transitions / len(state_history) * 100 if state_history else 0
    print(f"\nTransition Frequency: {transition_freq:.1f} transitions per 100 episodes")

    # Performance by state
    if metrics_by_state:
        print("\nPerformance by State:")
        for state, metrics in metrics_by_state.items():
            print(f"\n  {state}:")
            print(f"    Episodes: {metrics['episode_count']}")
            print(f"    Completion rate: {metrics['completion_rate']:.2%}")
            print(f"    Avg reward: {metrics['avg_reward']:.2f}")
            print(f"    Avg deaths: {metrics['avg_deaths']:.2f}")

    # Flow zone analysis (simple heuristic)
    median_reward = metrics_df['reward'].median()
    std_reward = metrics_df['reward'].std()
    flow_zone_min = median_reward - 0.5 * std_reward
    flow_zone_max = median_reward + 0.5 * std_reward

    in_flow_zone = ((metrics_df['reward'] >= flow_zone_min) &
                   (metrics_df['reward'] <= flow_zone_max)).sum()
    flow_zone_percentage = in_flow_zone / len(metrics_df) * 100

    print(f"\nFlow Zone Analysis:")
    print(f"  Flow zone: [{flow_zone_min:.1f}, {flow_zone_max:.1f}]")
    print(f"  Episodes in flow zone: {in_flow_zone} ({flow_zone_percentage:.1f}%)")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained HMM-DDA agent')
    parser.add_argument('--checkpoint', type=str, default='checkpoints',
                       help='Path to checkpoint directory')
    args = parser.parse_args()

    print("="*60)
    print("HMM-DDA Evaluation")
    print("="*60)

    BASE_DIR = Path(__file__).parent.parent
    CHECKPOINT_DIR = BASE_DIR / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    LOG_DIR = BASE_DIR / 'logs'
    OUTPUT_DIR = BASE_DIR / 'figures' / 'evaluation'

    # Load training data
    print("\nLoading training data...")
    metrics_df, hmm_data = load_training_data(LOG_DIR, CHECKPOINT_DIR)
    print(f"Loaded {len(metrics_df)} episodes")

    # Compute metrics by state
    metrics_by_state = compute_metrics_by_state(metrics_df, hmm_data)

    # Generate visualizations
    generate_visualizations(metrics_df, hmm_data, OUTPUT_DIR)

    # Print summary
    print_summary_statistics(metrics_df, hmm_data, metrics_by_state)

    print("\n" + "="*60)
    print(f"Evaluation complete! Figures saved to {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
