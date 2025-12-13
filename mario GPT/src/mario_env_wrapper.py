"""
Mario Environment Wrapper
Wraps gym-super-mario-bros environment and tracks episode metrics
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import gym


class MarioEnvWrapper:
    """
    Wrapper for Mario environment that tracks metrics during gameplay
    """

    def __init__(self, level_data: Optional[str] = None, env_name: str = 'SuperMarioBros-v0'):
        """
        Initialize Mario environment wrapper

        Args:
            level_data: Optional level string from MarioGPT
            env_name: Gym environment name
        """
        self.level_data = level_data
        self.env_name = env_name

        # Try to create gym environment
        try:
            import gym_super_mario_bros
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
            from nes_py.wrappers import JoypadSpace

            # Create base environment
            self.env = gym_super_mario_bros.make(env_name)
            # Simplify action space
            self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

            print(f"Created Mario environment: {env_name}")

        except ImportError:
            print("Warning: gym-super-mario-bros not installed. Using dummy environment.")
            self.env = self._create_dummy_env()

        # Episode metrics tracking
        self.reset_metrics()

    def reset_metrics(self):
        """Reset episode metrics"""
        self.episode_reward = 0.0
        self.episode_frames = 0
        self.max_x_pos = 0
        self.deaths = 0
        self.completed = False
        self.prev_lives = 2  # Mario starts with 2 lives
        self.prev_x_pos = 0

    def reset(self) -> np.ndarray:
        """
        Reset environment and metrics

        Returns:
            Initial observation
        """
        self.reset_metrics()
        obs = self.env.reset()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment and update metrics

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)

        # Update metrics
        self.episode_reward += reward
        self.episode_frames += 1

        # Track x position
        x_pos = info.get('x_pos', 0)
        self.max_x_pos = max(self.max_x_pos, x_pos)

        # Detect deaths (life decrease)
        current_lives = info.get('life', 2)
        if current_lives < self.prev_lives:
            self.deaths += 1
        self.prev_lives = current_lives

        # Check if level completed
        if info.get('flag_get', False):
            self.completed = True

        self.prev_x_pos = x_pos

        return obs, reward, done, info

    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the completed episode

        Returns:
            Dictionary of metrics
        """
        return {
            'completed': self.completed,
            'deaths': self.deaths,
            'reward': self.episode_reward,
            'frames': self.episode_frames,
            'max_x': self.max_x_pos
        }

    def render(self, mode: str = 'human'):
        """Render the environment"""
        return self.env.render(mode=mode)

    def close(self):
        """Close the environment"""
        if hasattr(self.env, 'close'):
            self.env.close()

    def _create_dummy_env(self):
        """Create a dummy environment for testing without gym-super-mario-bros"""

        class DummyMarioEnv:
            """Minimal dummy Mario environment for testing"""

            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(84, 84, 4), dtype=np.uint8
                )
                self.action_space = gym.spaces.Discrete(7)  # SIMPLE_MOVEMENT has 7 actions
                self.step_count = 0
                self.max_steps = 500

            def reset(self):
                self.step_count = 0
                return np.zeros((84, 84, 4), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1

                obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)

                # Random reward
                reward = np.random.randn() * 0.1

                # Random termination
                done = self.step_count >= self.max_steps or np.random.rand() < 0.001

                # Info with metrics
                info = {
                    'x_pos': self.step_count * 2,  # Simulated progress
                    'life': 2,
                    'flag_get': done and np.random.rand() < 0.3  # 30% chance of completion
                }

                return obs, reward, done, info

            def render(self, mode='human'):
                pass

            def close(self):
                pass

        return DummyMarioEnv()

    @property
    def observation_space(self):
        """Get observation space"""
        return self.env.observation_space

    @property
    def action_space(self):
        """Get action space"""
        return self.env.action_space

    def __repr__(self) -> str:
        """String representation"""
        return f"MarioEnvWrapper(env={self.env_name})"


def test_mario_env():
    """Test the Mario environment wrapper"""
    print("Testing Mario Environment Wrapper...")

    env = MarioEnvWrapper()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run a short episode
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode finished after {i+1} steps")
            break

    metrics = env.get_episode_metrics()
    print(f"\nEpisode Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"Total reward: {total_reward}")

    env.close()
    print("\nMario Environment test complete")


if __name__ == "__main__":
    test_mario_env()
