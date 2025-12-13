"""
Metrics Collector for tracking episode performance
Maintains a buffer of recent episodes and computes aggregated metrics
"""

from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np
from scipy import stats


class MetricsCollector:
    """
    Collects and aggregates episode metrics for T-score computation
    """

    def __init__(self, max_size: int = 1000, window_size: int = 10):
        """
        Initialize metrics collector

        Args:
            max_size: Maximum number of episodes to store in buffer
            window_size: Default window size for metric computation
        """
        self.max_size = max_size
        self.window_size = window_size

        # Buffer to store episode metrics
        self.buffer = deque(maxlen=max_size)

    def add_episode(self, metrics: Dict[str, Any]):
        """
        Add episode metrics to buffer

        Args:
            metrics: Dictionary containing:
                - completed: bool - whether episode was completed
                - deaths: int - number of deaths
                - reward: float - total reward
                - frames: int - number of frames (timesteps)
                - max_x: float - maximum x position reached
        """
        required_keys = ['completed', 'deaths', 'reward', 'frames', 'max_x']
        for key in required_keys:
            if key not in metrics:
                raise ValueError(f"Missing required metric: {key}")

        self.buffer.append(metrics)

    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent n episodes from buffer

        Args:
            n: Number of recent episodes to get (defaults to window_size)

        Returns:
            List of episode metrics
        """
        if n is None:
            n = self.window_size

        n = min(n, len(self.buffer))
        return list(self.buffer)[-n:]

    def get_completion_rate(self, window: Optional[int] = None) -> float:
        """
        Compute completion rate over recent window

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Completion rate in [0, 1]
        """
        recent = self.get_recent(window)
        if not recent:
            return 0.0

        completed_count = sum(1 for ep in recent if ep['completed'])
        return completed_count / len(recent)

    def get_death_rate(self, window: Optional[int] = None) -> float:
        """
        Compute average death rate over recent window

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Average deaths per episode
        """
        recent = self.get_recent(window)
        if not recent:
            return 0.0

        total_deaths = sum(ep['deaths'] for ep in recent)
        return total_deaths / len(recent)

    def get_reward_trend(self, window: Optional[int] = None) -> float:
        """
        Compute reward trend (linear regression slope) over recent window

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Slope of linear regression on rewards
        """
        recent = self.get_recent(window)
        if len(recent) < 2:
            return 0.0

        rewards = [ep['reward'] for ep in recent]
        x = np.arange(len(rewards))

        # Linear regression
        slope, _, _, _, _ = stats.linregress(x, rewards)

        return slope

    def get_time_to_complete(self, window: Optional[int] = None) -> float:
        """
        Compute average time to complete (frames) for completed episodes

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Average frames for completed episodes (0 if none completed)
        """
        recent = self.get_recent(window)
        completed = [ep for ep in recent if ep['completed']]

        if not completed:
            return 0.0

        avg_frames = np.mean([ep['frames'] for ep in completed])
        return avg_frames

    def get_progress_variance(self, window: Optional[int] = None) -> float:
        """
        Compute variance of progress (max_x) across recent window
        Lower variance = more consistent performance

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Standard deviation of max_x values
        """
        recent = self.get_recent(window)
        if len(recent) < 2:
            return 0.0

        max_x_values = [ep['max_x'] for ep in recent]
        return np.std(max_x_values)

    def get_average_reward(self, window: Optional[int] = None) -> float:
        """
        Compute average reward over recent window

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Average reward
        """
        recent = self.get_recent(window)
        if not recent:
            return 0.0

        return np.mean([ep['reward'] for ep in recent])

    def get_average_max_x(self, window: Optional[int] = None) -> float:
        """
        Compute average max_x over recent window

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Average max_x
        """
        recent = self.get_recent(window)
        if not recent:
            return 0.0

        return np.mean([ep['max_x'] for ep in recent])

    def get_all_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        Get all computed metrics at once

        Args:
            window: Window size (defaults to self.window_size)

        Returns:
            Dictionary of all metrics
        """
        return {
            'completion_rate': self.get_completion_rate(window),
            'death_rate': self.get_death_rate(window),
            'reward_trend': self.get_reward_trend(window),
            'time_to_complete': self.get_time_to_complete(window),
            'progress_variance': self.get_progress_variance(window),
            'average_reward': self.get_average_reward(window),
            'average_max_x': self.get_average_max_x(window),
            'episode_count': len(self.get_recent(window))
        }

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

    def __len__(self) -> int:
        """Return number of episodes in buffer"""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation"""
        return f"MetricsCollector(size={len(self)}/{self.max_size}, window={self.window_size})"

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all episodes in buffer

        Returns:
            Dictionary of summary statistics
        """
        if not self.buffer:
            return {}

        all_episodes = list(self.buffer)

        rewards = [ep['reward'] for ep in all_episodes]
        max_xs = [ep['max_x'] for ep in all_episodes]
        frames = [ep['frames'] for ep in all_episodes]
        deaths = [ep['deaths'] for ep in all_episodes]
        completed = [ep['completed'] for ep in all_episodes]

        return {
            'total_episodes': len(all_episodes),
            'completion_rate': sum(completed) / len(completed),
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'max_x_mean': np.mean(max_xs),
            'max_x_std': np.std(max_xs),
            'frames_mean': np.mean(frames),
            'deaths_mean': np.mean(deaths),
            'deaths_total': sum(deaths),
        }

    def save_to_csv(self, filepath: str):
        """
        Save buffer contents to CSV

        Args:
            filepath: Path to save CSV file
        """
        import pandas as pd

        if not self.buffer:
            print("Warning: Buffer is empty, nothing to save")
            return

        df = pd.DataFrame(list(self.buffer))
        df.to_csv(filepath, index=False)
        print(f"Saved {len(self.buffer)} episodes to {filepath}")

    def load_from_csv(self, filepath: str):
        """
        Load episodes from CSV into buffer

        Args:
            filepath: Path to CSV file
        """
        import pandas as pd

        df = pd.DataFrame(filepath)

        self.clear()

        for _, row in df.iterrows():
            metrics = row.to_dict()
            self.add_episode(metrics)

        print(f"Loaded {len(self.buffer)} episodes from {filepath}")
