#!/usr/bin/env python3
"""
HMM-DDA Complete Run Pipeline
Unified script that imports and orchestrates all src/ and scripts/ modules.

Usage:
    python run_pipeline.py [--skip-calibration] [--episodes N] [--device cpu|cuda]

Requires: src/ and scripts/ folders in the same directory
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from collections import deque, Counter
from tqdm import tqdm


try:
    from src.mario_env import MarioGridEnv
    from src.level_generator import LevelGenerator
    from src.hmm_controller import HMM_DDA
    from src.metrics_collector import MetricsCollector
    from src.t_score import (
        compute_T_score,
        compute_T_score_from_metrics,
        get_metric_contributions,
        interpret_T_score
    )
    from src.utils import (
        Logger,
        set_random_seeds,
        check_cuda,
        load_config,
        save_config,
        normalize_value,
        denormalize_value,
        plot_learning_curve,
        plot_belief_evolution,
        plot_state_distribution,
        plot_metrics_comparison,
        plot_t_score_distributions,
        plot_state_transition_diagram,
    )

    from scripts.calibrate import (
        HeuristicAgent,
        GymEnvWrapper,
        create_ppo_agent,
        train_baseline_agent,
        collect_calibration_data,
    )

    from scripts.derive_parameters import (
        load_calibration_data,
        compute_normalization_bounds,
        compute_t_scores_for_episodes,
        fit_gaussian_distributions,
        derive_thresholds,
        optimize_metric_weights,
        create_transition_matrix,
        visualize_distributions,
    )

    from scripts.train import (
        CONFIG as TRAIN_CONFIG,
        HeuristicAgent as TrainHeuristicAgent,
        AdaptiveEnv,
        create_ppo_agent as create_train_ppo_agent,
        run_episode,
    )

    from scripts.evaluate import (
        load_training_data,
        compute_metrics_by_state,
        generate_visualizations,
        print_summary_statistics,
    )
except ImportError as e:
    print(f"\nERROR: Could not import modules. Ensure {PROJECT_ROOT} contains 'src' and 'scripts' folders.")
    print(f"Details: {e}")
    raise e

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


PIPELINE_CONFIG = {
    'total_episodes': 1000,
    'calibration_episodes_per_difficulty': 50,
    'baseline_training_episodes': 200,
    'hmm_update_frequency': 10,
    'metrics_window': 10,
    'checkpoint_frequency': 500,
    'log_frequency': 50,
    'max_steps_per_episode': 2000,
    'adaptation_frequency': 500,
    'train_freq': 10,
    'train_timesteps': 2048,
    'seed': 42,
}


def phase1_calibration(generator: LevelGenerator, device: str, config: dict, output_dir: Path):
    """
    Phase 1: Calibration
    Imports and uses functions from scripts/calibrate.py
    """
    print("\n" + "=" * 60)
    print("PHASE 1: CALIBRATION")
    print("Using: scripts/calibrate.py")
    print("=" * 60)

    calibration_dir = output_dir / 'calibration_data'
    calibration_dir.mkdir(parents=True, exist_ok=True)

    print("\nStep 1.1: Calling train_baseline_agent() from scripts/calibrate.py...")
    agent = train_baseline_agent(
        generator=generator,
        n_episodes=config['baseline_training_episodes'],
        device=device
    )

    print("\nStep 1.2: Calling collect_calibration_data() from scripts/calibrate.py...")
    calibration_results = {}

    for difficulty in ['Low', 'Transition', 'High']:
        df = collect_calibration_data(
            agent=agent,
            generator=generator,
            difficulty_name=difficulty,
            n_episodes=config['calibration_episodes_per_difficulty']
        )

        csv_path = calibration_dir / f'{difficulty.lower()}_metrics.csv'
        df.to_csv(csv_path, index=False)
        calibration_results[difficulty] = df

        print(f"  {difficulty}: CR={df['completed'].mean():.2%}, Deaths={df['deaths'].mean():.2f}")

    print(f"\nCalibration data saved to {calibration_dir}")
    return calibration_results, agent



def phase2_derive_parameters(calibration_dir: Path, output_dir: Path):
    """
    Phase 2: Parameter Derivation
    Imports and uses functions from scripts/derive_parameters.py
    """
    print("\n" + "=" * 60)
    print("PHASE 2: PARAMETER DERIVATION")
    print("Using: scripts/derive_parameters.py")
    print("=" * 60)

    config_dir = output_dir / 'config'
    figures_dir = output_dir / 'figures' / 'calibration'
    config_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nStep 2.1: Calling load_calibration_data()...")
    data = load_calibration_data(calibration_dir)

    print("\nStep 2.2: Calling compute_normalization_bounds()...")
    normalization = compute_normalization_bounds(data)
    save_config(normalization, str(config_dir / 'normalization_bounds.json'))

    print("\nStep 2.3: Calling optimize_metric_weights()...")
    weights = optimize_metric_weights(data, normalization)
    save_config({'weights': weights}, str(config_dir / 'metric_weights.json'))

    print("\nStep 2.4: Calling compute_t_scores_for_episodes()...")
    t_scores = compute_t_scores_for_episodes(data, weights, normalization)
    for diff, scores in t_scores.items():
        print(f"  {diff}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

    print("\nStep 2.5: Calling fit_gaussian_distributions()...")
    emissions = fit_gaussian_distributions(t_scores)
    save_config(emissions, str(config_dir / 'emission_params.json'))

    print("\nStep 2.6: Calling derive_thresholds()...")
    thresholds = derive_thresholds(emissions)
    save_config(thresholds, str(config_dir / 'thresholds.json'))

    print("\nStep 2.7: Calling create_transition_matrix()...")
    transition_matrix = create_transition_matrix()
    save_config(transition_matrix, str(config_dir / 'transition_matrix.json'))

    prompts = {
        'Low': "few enemies, no gaps, many pipes, low elevation, easy",
        'Transition': "varied challenges, mixed density, skill assessment",
        'High': "many enemies, many gaps, few pipes, high elevation, hard"
    }
    save_config(prompts, str(config_dir / 'prompts.json'))

    print("\nStep 2.8: Calling visualize_distributions()...")
    visualize_distributions(t_scores, emissions, figures_dir)

    print(f"\nParameters saved to {config_dir}")

    return {
        'weights': weights,
        'normalization': normalization,
        'emissions': emissions,
        'thresholds': thresholds,
        'transition_matrix': transition_matrix,
    }


def phase3_training(generator: LevelGenerator, device: str, config: dict,
                    output_dir: Path, derived_params: dict):
    """
    Phase 3: Training with HMM-DDA
    """
    print("\n" + "=" * 60)
    print("PHASE 3: TRAINING WITH HMM-DDA")
    print("Using: scripts/train.py, src/hmm_controller.py, src/metrics_collector.py")
    print("=" * 60)

    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    config_dir = output_dir / 'config'

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    t_score_config = {
        'weights': derived_params['weights'],
        'normalization': derived_params['normalization']
    }

    print("\nInitializing HMM_DDA...")
    config_path = str(config_dir) if (config_dir / 'transition_matrix.json').exists() else None
    hmm = HMM_DDA(config_path)
    print(f"  {hmm}")

    print("\nInitializing AdaptiveEnv...")
    env = AdaptiveEnv(generator, hmm)

    print(f"\nCalling create_ppo_agent() (SB3: {SB3_AVAILABLE})...")
    agent = create_train_ppo_agent(env, device, TRAIN_CONFIG.get('ppo_config'))

    print("\nInitializing MetricsCollector...")
    collector = MetricsCollector(max_size=2000, window_size=config['metrics_window'])

    print("Initializing Logger...")
    logger = Logger(str(log_dir), experiment_name='hmm_dda_training')

    episode_rewards = deque(maxlen=100)
    state_counts = {'Low': 0, 'Transition': 0, 'High': 0}

    print(f"\nStarting training loop...")
    print(f"  Total episodes: {config['total_episodes']}")

    for episode in tqdm(range(config['total_episodes']), desc="Training"):
        current_state = hmm.get_current_state()
        state_counts[current_state] += 1

        metrics = run_episode(env, agent, max_steps=config['max_steps_per_episode'])

        collector.add_episode(metrics)
        episode_rewards.append(metrics['reward'])

        if (episode + 1) % config['hmm_update_frequency'] == 0:
            if len(collector) >= config['metrics_window']:
                T = compute_T_score(collector, t_score_config, window=config['metrics_window'])
                new_state = hmm.update(T)
                belief = hmm.get_belief()

                if new_state != current_state or (episode + 1) % config['log_frequency'] == 0:
                    log_data = {
                        'episode': episode + 1,
                        'state': new_state,
                        'prev_state': current_state,
                        'belief_low': float(belief[0]),
                        'belief_transition': float(belief[1]),
                        'belief_high': float(belief[2]),
                        'T_score': float(T),
                        'reward': float(metrics['reward']),
                        'avg_reward_100': float(np.mean(episode_rewards)),
                        'completed': bool(metrics['completed']),
                        'deaths': int(metrics['deaths']),
                        'max_x': float(metrics['max_x']),
                        'completion_rate': collector.get_completion_rate(),
                    }
                    logger.log(log_data, print_console=(episode + 1) % config['log_frequency'] == 0)

        if SB3_AVAILABLE and (episode + 1) % config['train_freq'] == 0:
            agent.learn(total_timesteps=config['train_timesteps'], reset_num_timesteps=False)

        if (episode + 1) % config['checkpoint_frequency'] == 0:
            hmm.save_state(str(checkpoint_dir / f'hmm_{episode+1}.json'))
            if SB3_AVAILABLE:
                agent.save(str(checkpoint_dir / f'ppo_{episode+1}'))
            collector.save_to_csv(str(checkpoint_dir / f'metrics_{episode+1}.csv'))

            print(f"\n[Checkpoint {episode+1}]")
            print(f"  State: {hmm.get_current_state()}, Belief: {hmm.get_belief().round(3)}")
            print(f"  Avg Reward: {np.mean(episode_rewards):.2f}, CR: {collector.get_completion_rate():.2%}")

        if (episode + 1) % config['adaptation_frequency'] == 0:
            hmm.adapt_transition_matrix()

    hmm.save_state(str(checkpoint_dir / 'hmm_final.json'))
    if SB3_AVAILABLE:
        agent.save(str(checkpoint_dir / 'ppo_final'))
    logger.save_metrics()
    collector.save_to_csv(str(checkpoint_dir / 'metrics_final.csv'))
    env.close()

    print(f"\nTraining complete. Checkpoints saved to {checkpoint_dir}")

    return {
        'hmm': hmm,
        'collector': collector,
        'state_counts': state_counts,
        'final_avg_reward': float(np.mean(episode_rewards)),
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
    }

def phase4_evaluation(training_results: dict, output_dir: Path):
    """
    Phase 4: Evaluation
    """
    print("\n" + "=" * 60)
    print("PHASE 4: EVALUATION")
    print("Using: scripts/evaluate.py, src/utils.py")
    print("=" * 60)

    figures_dir = output_dir / 'figures' / 'evaluation'
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = training_results['checkpoint_dir']
    log_dir = training_results['log_dir']
    hmm = training_results['hmm']

    print("\nCalling load_training_data()...")
    metrics_df, hmm_data = load_training_data(checkpoint_dir, log_dir)

    if not hmm_data:
        hmm_data = {
            'state_history': hmm.state_history,
            'belief_history': [b.tolist() for b in hmm.belief_history],
            't_score_history': hmm.t_score_history,
        }

    print("\nCalling compute_metrics_by_state()...")
    metrics_by_state = compute_metrics_by_state(metrics_df, hmm_data)

    print("\nCalling generate_visualizations()...")
    generate_visualizations(metrics_df, hmm_data, figures_dir)

    print("\nCalling print_summary_statistics()...")
    print_summary_statistics(metrics_df, hmm_data, metrics_by_state)

    if not metrics_df.empty and 'reward' in metrics_df.columns:
        rewards = metrics_df['reward'].values
        median_r = np.median(rewards)
        std_r = np.std(rewards)
        flow_min = median_r - 0.5 * std_r
        flow_max = median_r + 0.5 * std_r
        in_flow = np.sum((rewards >= flow_min) & (rewards <= flow_max))
        flow_pct = in_flow / len(rewards) * 100
        print(f"\nAdditional Flow Zone Analysis:")
        print(f"  Flow zone: [{flow_min:.1f}, {flow_max:.1f}]")
        print(f"  Episodes in flow: {in_flow}/{len(rewards)} ({flow_pct:.1f}%)")

    print(f"\nFigures saved to {figures_dir}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='HMM-DDA Complete Pipeline')
        parser.add_argument('--skip-calibration', action='store_true',
                            help='Skip calibration and use default parameters')
        parser.add_argument('--episodes', type=int, default=1000,
                            help='Number of training episodes')
        parser.add_argument('--calibration-episodes', type=int, default=50,
                            help='Calibration episodes per difficulty')
        parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                            help='Device for training')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
        parser.add_argument('--output-dir', type=str, default='./hmm_dda_output',
                            help='Output directory')
        args = parser.parse_args()

    print("=" * 60)
    print("HMM-DDA COMPLETE PIPELINE")
    print("=" * 60)

    set_random_seeds(args.seed)
    cuda_available, device_name = check_cuda()
    device = args.device if args.device == 'cpu' or cuda_available else 'cpu'
    print(f"\nDevice: {device} ({device_name})")
    print(f"SB3 Available: {SB3_AVAILABLE}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PIPELINE_CONFIG.copy()
    config['total_episodes'] = args.episodes
    config['calibration_episodes_per_difficulty'] = args.calibration_episodes
    save_config(config, str(output_dir / 'run_config.json'))

    print("\nInitializing LevelGenerator...")
    generator = LevelGenerator(device=device)

    if not args.skip_calibration:
        phase1_calibration(generator, device, config, output_dir)
        calibration_dir = output_dir / 'calibration_data'
        derived_params = phase2_derive_parameters(calibration_dir, output_dir)
    else:
        print("\nSkipping calibration, using default parameters...")
        config_dir = output_dir / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        derived_params = {
            'weights': [0.25, 0.20, 0.25, 0.15, 0.15],
            'normalization': {
                'death_rate_max': 5.0,
                'reward_trend_min': -50.0,
                'reward_trend_max': 50.0,
                'time_to_complete_max': 2000.0,
                'progress_variance_max': 500.0,
            },
            'emissions': {
                'Low': {'mu': 0.25, 'sigma': 0.15},
                'Transition': {'mu': 0.50, 'sigma': 0.12},
                'High': {'mu': 0.75, 'sigma': 0.15}
            },
            'thresholds': {'low_transition': 0.35, 'transition_high': 0.65},
            'transition_matrix': {
                'matrix': [[0.70, 0.25, 0.05], [0.20, 0.40, 0.40], [0.05, 0.25, 0.70]],
                'states': ['Low', 'Transition', 'High'],
            }
        }
        save_config(derived_params['normalization'], str(config_dir / 'normalization_bounds.json'))
        save_config({'weights': derived_params['weights']}, str(config_dir / 'metric_weights.json'))
        save_config(derived_params['emissions'], str(config_dir / 'emission_params.json'))
        save_config(derived_params['thresholds'], str(config_dir / 'thresholds.json'))
        save_config(derived_params['transition_matrix'], str(config_dir / 'transition_matrix.json'))
        save_config({
            'Low': "few enemies, no gaps, many pipes, low elevation, easy",
            'Transition': "varied challenges, mixed density, skill assessment",
            'High': "many enemies, many gaps, few pipes, high elevation, hard"
        }, str(config_dir / 'prompts.json'))

    training_results = phase3_training(generator, device, config, output_dir, derived_params)
    phase4_evaluation(training_results, output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        get_ipython()
        print("Detected Notebook Environment.")
        class NotebookArgs:
            skip_calibration = False
            episodes = 500
            calibration_episodes = 50
            device = 'cpu'
            seed = 42
            output_dir = './hmm_dda_output'

        main(NotebookArgs())
    except NameError:
        main()
