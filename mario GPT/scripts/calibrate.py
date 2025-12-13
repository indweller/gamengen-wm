"""
Calibration Script
Collects empirical data at fixed difficulties to derive HMM parameters
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

from src.level_generator import LevelGenerator
from src.mario_env_wrapper import MarioEnvWrapper
from src.metrics_collector import MetricsCollector
from src.utils import Logger, set_random_seeds, check_cuda

from stable_baselines3 import PPO


def train_baseline_agent(n_episodes=500, device='cuda'):
    """
    Train a baseline PPO agent on mixed difficulties

    Args:
        n_episodes: Number of training episodes
        device: Device to use

    Returns:
        Trained PPO agent
    """
    print("="*60)
    print("Training Baseline Agent on Mixed Difficulties")
    print("="*60)

    # Create generator
    generator = LevelGenerator(device=device)

    # Mixed difficulty prompts
    prompts = [
        "few enemies, no gaps, many pipes, low elevation, easy",
        "some enemies, few gaps, some pipes, medium elevation, moderate",
        "many enemies, many gaps, few pipes, high elevation, hard"
    ]

    # Create a dummy environment for PPO initialization
    env = MarioEnvWrapper()

    # Create PPO agent
    model = PPO('CnnPolicy', env, verbose=1, device=device,
                n_steps=512, batch_size=64, n_epochs=4,
                learning_rate=3e-4)

    # Training loop with random difficulty selection
    for episode in tqdm(range(n_episodes), desc="Training baseline"):
        # Random difficulty
        prompt = prompts[episode % len(prompts)]
        level = generator.generate(prompt)

        # Create new environment
        env = MarioEnvWrapper()
        obs = env.reset()

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)

        # Learn from experience
        if episode % 10 == 0:
            model.learn(total_timesteps=2048, reset_num_timesteps=False)

    print(f"Baseline agent trained for {n_episodes} episodes")
    return model


def collect_calibration_data(agent, difficulty_name, prompt, n_episodes=100, device='cuda'):
    """
    Collect calibration data at fixed difficulty

    Args:
        agent: Trained PPO agent
        difficulty_name: Name of difficulty level
        prompt: Text prompt for level generation
        n_episodes: Number of episodes to collect
        device: Device to use

    Returns:
        DataFrame of episode metrics
    """
    print(f"\nCollecting {n_episodes} episodes for {difficulty_name} difficulty...")

    generator = LevelGenerator(device=device)
    collector = MetricsCollector()

    for episode in tqdm(range(n_episodes), desc=f"{difficulty_name} calibration"):
        # Generate level
        level = generator.generate(prompt, seed=episode)

        # Create environment
        env = MarioEnvWrapper()
        obs = env.reset()

        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)

        # Collect metrics
        metrics = env.get_episode_metrics()
        collector.add_episode(metrics)

        env.close()

    # Convert to DataFrame
    df = pd.DataFrame(list(collector.buffer))
    return df


def main():
    """Main calibration process"""
    print("="*60)
    print("HMM-DDA Calibration Process")
    print("="*60)

    # Configuration
    BASE_DIR = Path(__file__).parent.parent
    CALIBRATION_DIR = BASE_DIR / 'calibration_data'
    CALIBRATION_DIR.mkdir(exist_ok=True)

    # Check CUDA
    cuda_available, device_name = check_cuda()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"Using device: {device} ({device_name})")

    # Set seeds for reproducibility
    set_random_seeds(42)

    # Define fixed difficulty prompts (Option 2 Design)
    # Low=Easy, Transition=Assessment, High=Hard
    prompts = {
        'Low': "few enemies, no gaps, many pipes, low elevation, easy difficulty",
        'Transition': "varied challenges, mixed enemy density, unpredictable patterns, some gaps and pipes, skill assessment level",
        'High': "many enemies, many gaps, few pipes, high elevation, hard difficulty"
    }

    # Step 1: Train baseline agent
    print("\n" + "="*60)
    print("Step 1: Training Baseline Agent")
    print("="*60)
    agent = train_baseline_agent(n_episodes=500, device=device)

    # Save baseline agent
    agent_path = BASE_DIR / 'checkpoints' / 'baseline_agent'
    agent.save(agent_path)
    print(f"Saved baseline agent to {agent_path}")

    # Step 2: Collect calibration data for each difficulty
    print("\n" + "="*60)
    print("Step 2: Collecting Calibration Data")
    print("="*60)

    calibration_results = {}

    for difficulty_name, prompt in prompts.items():
        df = collect_calibration_data(
            agent=agent,
            difficulty_name=difficulty_name,
            prompt=prompt,
            n_episodes=100,  # Reduced for minimal demo
            device=device
        )

        # Save to CSV
        csv_path = CALIBRATION_DIR / f'{difficulty_name.lower()}_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved {difficulty_name} metrics to {csv_path}")

        calibration_results[difficulty_name] = df

    # Step 3: Summary statistics
    print("\n" + "="*60)
    print("Calibration Summary")
    print("="*60)

    for difficulty_name, df in calibration_results.items():
        print(f"\n{difficulty_name} Difficulty:")
        print(f"  Episodes collected: {len(df)}")
        print(f"  Completion rate: {df['completed'].mean():.2%}")
        print(f"  Average deaths: {df['deaths'].mean():.2f}")
        print(f"  Average reward: {df['reward'].mean():.2f}")
        print(f"  Average frames: {df['frames'].mean():.0f}")
        print(f"  Average max_x: {df['max_x'].mean():.0f}")

    print("\n" + "="*60)
    print("Calibration Complete!")
    print("="*60)
    print(f"\nNext step: Run 'python scripts/derive_parameters.py' to derive HMM parameters")


if __name__ == "__main__":
    main()
