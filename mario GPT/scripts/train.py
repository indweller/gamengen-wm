"""
Training Script
Main training loop with HMM-driven difficulty adaptation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from tqdm import tqdm
import torch

from src.hmm_controller import HMM_DDA
from src.metrics_collector import MetricsCollector
from src.t_score import compute_T_score
from src.level_generator import LevelGenerator
from src.mario_env_wrapper import MarioEnvWrapper
from src.utils import Logger, set_random_seeds, check_cuda, load_config

from stable_baselines3 import PPO


# Configuration
CONFIG = {
    'total_episodes': 1000,           # Minimal demo
    'hmm_update_frequency': 5,        # Update HMM every 5 episodes
    'metrics_window': 5,              # Window for T-score
    'checkpoint_frequency': 100,      # Checkpoint every 100 episodes
    'log_frequency': 10,              # Log every 10 episodes
    'device': 'cuda',                 # CUDA GPU
    'ppo_config': {
        'learning_rate': 3e-4,
        'n_steps': 512,
        'batch_size': 64,
        'n_epochs': 4,
        'gamma': 0.99,
        'gae_lambda': 0.95
    }
}


def train():
    """Main training function"""
    print("="*60)
    print("HMM-DDA Training with Adaptive Difficulty")
    print("="*60)

    # Setup paths
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / 'config'
    CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
    LOG_DIR = BASE_DIR / 'logs'

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # Check CUDA
    cuda_available, device_name = check_cuda()
    device = CONFIG['device'] if cuda_available else 'cpu'
    print(f"Using device: {device} ({device_name})")

    # Set random seeds
    set_random_seeds(42)

    # Initialize components
    print("\nInitializing components...")

    # 1. HMM Controller
    hmm = HMM_DDA(str(CONFIG_DIR))
    print(f"HMM initialized: {hmm}")

    # 2. Metrics Collector
    collector = MetricsCollector(max_size=1000, window_size=CONFIG['metrics_window'])

    # 3. Level Generator
    generator = LevelGenerator(device=device)

    # 4. Logger
    logger = Logger(str(LOG_DIR), experiment_name='training')

    # 5. Load T-score config
    metric_weights_config = load_config(CONFIG_DIR / 'metric_weights.json')
    normalization_config = load_config(CONFIG_DIR / 'normalization_bounds.json')

    t_score_config = {
        'weights': metric_weights_config['weights'],
        'normalization': normalization_config,
        'window_size': CONFIG['metrics_window']
    }

    # 6. PPO Agent
    print("\nInitializing PPO agent...")
    dummy_env = MarioEnvWrapper()
    agent = PPO(
        'CnnPolicy',
        dummy_env,
        verbose=0,
        device=device,
        **CONFIG['ppo_config']
    )

    # Training loop
    print("\n" + "="*60)
    print("Starting Training Loop")
    print("="*60)

    for episode in tqdm(range(CONFIG['total_episodes']), desc="Training"):
        # Get current difficulty state and prompt
        current_state = hmm.get_current_state()
        prompt = hmm.get_prompt()

        # Generate level
        level = generator.generate(prompt)

        # Create environment
        env = MarioEnvWrapper()
        obs = env.reset()

        # Run episode
        done = False
        episode_steps = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            episode_steps += 1

            if episode_steps > 5000:  # Max steps per episode
                break

        # Collect metrics
        metrics = env.get_episode_metrics()
        collector.add_episode(metrics)

        # Update HMM periodically
        if (episode + 1) % CONFIG['hmm_update_frequency'] == 0 and len(collector) >= CONFIG['metrics_window']:
            # Compute T-score
            T = compute_T_score(collector, t_score_config, window=CONFIG['metrics_window'])

            # Update HMM
            new_state = hmm.update(T)
            belief = hmm.get_belief()

            # Log HMM update
            if (episode + 1) % CONFIG['log_frequency'] == 0:
                log_data = {
                    'episode': episode + 1,
                    'state': new_state,
                    'prev_state': current_state,
                    'belief_low': float(belief[0]),
                    'belief_transition': float(belief[1]),
                    'belief_high': float(belief[2]),
                    'T_score': float(T),
                    'reward': float(metrics['reward']),
                    'completed': bool(metrics['completed']),
                    'deaths': int(metrics['deaths']),
                    'max_x': float(metrics['max_x'])
                }
                logger.log(log_data)

        # Train agent
        if (episode + 1) % 10 == 0:
            agent.learn(total_timesteps=2048, reset_num_timesteps=False)

        # Checkpoint
        if (episode + 1) % CONFIG['checkpoint_frequency'] == 0:
            # Save agent
            agent_path = CHECKPOINT_DIR / f'agent_{episode+1}'
            agent.save(agent_path)

            # Save HMM state
            hmm_path = CHECKPOINT_DIR / f'hmm_{episode+1}.json'
            hmm.save_state(str(hmm_path))

            # Save metrics
            collector.save_to_csv(CHECKPOINT_DIR / f'metrics_{episode+1}.csv')

            print(f"\nCheckpoint saved at episode {episode+1}")

        # Adaptation every 200 episodes
        if (episode + 1) % 200 == 0:
            hmm.adapt_transition_matrix(hmm.state_history, hmm.t_score_history)

        env.close()

    # Final save
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    agent.save(CHECKPOINT_DIR / 'agent_final')
    hmm.save_state(str(CHECKPOINT_DIR / 'hmm_final.json'))
    logger.save_metrics()
    collector.save_to_csv(CHECKPOINT_DIR / 'metrics_final.csv')

    # Print summary
    state_dist = hmm.get_state_distribution()
    transition_freq = hmm.get_transition_frequency()

    print("\nTraining Summary:")
    print(f"  Total episodes: {CONFIG['total_episodes']}")
    print(f"  State distribution:")
    for state, percentage in state_dist.items():
        print(f"    {state}: {percentage:.1%}")
    print(f"  Transition frequency: {transition_freq:.1f} per 100 episodes")

    print(f"\nFinal state: {hmm.get_current_state()}")
    print(f"Final belief: {hmm.get_belief()}")

    print("\nNext step: Run 'python scripts/evaluate.py' to evaluate the trained agent")


if __name__ == "__main__":
    train()
