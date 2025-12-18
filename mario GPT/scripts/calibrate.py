"""
Calibration Script
Collects empirical data at fixed difficulties to derive HMM parameters.

From Framework Section 6: Before training with DDA, we need baseline
performance data at each difficulty level to calibrate emission parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mario_env import MarioEnvWrapper
from src.level_generator import LevelGenerator
from src.metrics_collector import MetricsCollector
from src.utils import set_random_seeds, check_cuda

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("stable-baselines3 not installed. Using heuristic agent.")


class HeuristicAgent:
    """Simple heuristic agent for when SB3 is not available"""
    
    def __init__(self):
        self.jump_cooldown = 0
    
    def predict(self, obs, deterministic=False):
        self.jump_cooldown = max(0, self.jump_cooldown - 1)
        
        # Simple strategy: move right, jump occasionally
        if np.random.random() < 0.2 and self.jump_cooldown == 0:
            self.jump_cooldown = 10
            return 2, None  # Right + jump
        elif np.random.random() < 0.05:
            return 5, None  # Jump
        else:
            return 1, None  # Right


class GymEnvWrapper:
    """Wrapper to make MarioEnvWrapper compatible with SB3"""
    
    def __init__(self, level_generator, state='Low'):
        self.generator = level_generator
        self.state = state
        self.env = None
        self._create_env()
    
    def _create_env(self):
        level = self.generator.generate_for_state(self.state)
        self.env = MarioEnvWrapper(level)
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def reset(self):
        level = self.generator.generate_for_state(self.state)
        self.env.load_new_level(level)
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        if self.env:
            self.env.close()


def create_ppo_agent(env, device='cpu'):
    """Create PPO agent for training"""
    if not SB3_AVAILABLE:
        return HeuristicAgent()
    
    def make_env():
        return env
    
    vec_env = DummyVecEnv([make_env])
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        verbose=0,
        device=device,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
    )
    
    return model


def train_baseline_agent(generator, n_episodes=200, device='cpu'):
    """
    Train a baseline agent on mixed difficulties.
    
    This agent will be used to collect calibration data at each
    fixed difficulty level.
    """
    print("=" * 60)
    print("Training Baseline Agent on Mixed Difficulties")
    print("=" * 60)
    
    states = ['Low', 'Transition', 'High']
    
    if SB3_AVAILABLE:
        env = GymEnvWrapper(generator, 'Transition')
        agent = create_ppo_agent(env, device)
        
        for episode in tqdm(range(n_episodes), desc="Training baseline"):
            env.state = states[episode % 3]
            
            obs = env.reset()
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
            
            if (episode + 1) % 20 == 0:
                agent.learn(total_timesteps=1024, reset_num_timesteps=False)
        
        env.close()
    else:
        agent = HeuristicAgent()
        print("Using heuristic agent (install stable-baselines3 for PPO)")
    
    return agent


def collect_calibration_data(agent, generator, difficulty_name, n_episodes=50):
    """
    Collect calibration data at fixed difficulty.
    
    Args:
        agent: Trained agent
        generator: Level generator
        difficulty_name: 'Low', 'Transition', or 'High'
        n_episodes: Number of episodes to collect
        
    Returns:
        DataFrame of episode metrics
    """
    print(f"\nCollecting {n_episodes} episodes for {difficulty_name} difficulty...")
    
    collector = MetricsCollector()
    
    for episode in tqdm(range(n_episodes), desc=f"{difficulty_name}"):
        level = generator.generate_for_state(difficulty_name, seed=episode)
        env = MarioEnvWrapper(level)
        
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 2000:
            action, _ = agent.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        metrics = env.get_episode_metrics()
        collector.add_episode(metrics)
        env.close()
    
    df = pd.DataFrame(list(collector.buffer))
    return df


def main():
    """Main calibration process"""
    print("=" * 60)
    print("HMM-DDA Calibration Process")
    print("=" * 60)
    
    BASE_DIR = Path(__file__).parent.parent
    CALIBRATION_DIR = BASE_DIR / 'calibration_data'
    CALIBRATION_DIR.mkdir(exist_ok=True)
    
    cuda_available, device_name = check_cuda()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"Using device: {device} ({device_name})")
    
    set_random_seeds(42)
    
    generator = LevelGenerator(device=device)
    
    print("\n" + "=" * 60)
    print("Step 1: Training Baseline Agent")
    print("=" * 60)
    agent = train_baseline_agent(generator, n_episodes=200, device=device)
    
    print("\n" + "=" * 60)
    print("Step 2: Collecting Calibration Data")
    print("=" * 60)
    
    calibration_results = {}
    
    for difficulty in ['Low', 'Transition', 'High']:
        df = collect_calibration_data(
            agent=agent,
            generator=generator,
            difficulty_name=difficulty,
            n_episodes=50
        )
        
        csv_path = CALIBRATION_DIR / f'{difficulty.lower()}_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved {difficulty} metrics to {csv_path}")
        
        calibration_results[difficulty] = df
    
    print("\n" + "=" * 60)
    print("Calibration Summary")
    print("=" * 60)
    
    for difficulty, df in calibration_results.items():
        print(f"\n{difficulty} Difficulty:")
        print(f"  Episodes: {len(df)}")
        print(f"  Completion rate: {df['completed'].mean():.2%}")
        print(f"  Average deaths: {df['deaths'].mean():.2f}")
        print(f"  Average reward: {df['reward'].mean():.2f}")
        print(f"  Average frames: {df['frames'].mean():.0f}")
        print(f"  Average max_x: {df['max_x'].mean():.1f}")
    
    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print(f"\nNext: Run 'python scripts/derive_parameters.py'")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
