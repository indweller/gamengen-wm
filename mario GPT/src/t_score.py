"""
T-Score Computation Module
Computes weighted performance score from multiple metrics
"""

from typing import Dict, Any
import numpy as np
from .metrics_collector import MetricsCollector
from .utils import normalize_value


def compute_T_score(collector: MetricsCollector,
                    config: Dict[str, Any],
                    window: int = None) -> float:
    """
    Compute T-score from metrics collector

    T-score is a weighted combination of normalized metrics:
    T = w1*CR + w2*DR_score + w3*RT_score + w4*TTC_score + w5*PV_score

    Where:
    - CR: Completion Rate
    - DR_score: Death Rate score (inverted, lower deaths = higher score)
    - RT_score: Reward Trend score (normalized slope)
    - TTC_score: Time-To-Complete score (inverted, faster = higher score)
    - PV_score: Progress Variance score (inverted, consistent = higher score)

    Args:
        collector: MetricsCollector instance
        config: Configuration dictionary containing:
            - weights: [w1, w2, w3, w4, w5]
            - normalization: bounds for each metric
        window: Window size for metrics computation

    Returns:
        T-score in [0, 1]
    """
    if len(collector) == 0:
        return 0.5  # Neutral score if no data

    # Get configuration
    weights = config.get('weights', [0.2, 0.2, 0.2, 0.2, 0.2])
    normalization = config.get('normalization', {})

    if window is None:
        window = config.get('window_size', collector.window_size)

    # Ensure we have enough episodes
    if len(collector.get_recent(window)) < 2:
        return 0.5  # Neutral score if insufficient data

    # 1. Completion Rate (already in [0, 1])
    cr = collector.get_completion_rate(window)

    # 2. Death Rate Score (inverted)
    dr = collector.get_death_rate(window)
    dr_max = normalization.get('death_rate_max', 10.0)
    # Transform: lower deaths → higher score
    dr_score = 1.0 / (1.0 + dr / dr_max)

    # 3. Reward Trend Score
    rt = collector.get_reward_trend(window)
    rt_min = normalization.get('reward_trend_min', -100.0)
    rt_max = normalization.get('reward_trend_max', 100.0)
    rt_score = normalize_value(rt, rt_min, rt_max, clip=True)

    # 4. Time-to-Complete Score (inverted)
    ttc = collector.get_time_to_complete(window)
    if ttc == 0:  # No completions
        ttc_score = 0.0
    else:
        ttc_max = normalization.get('time_to_complete_max', 5000.0)
        # Transform: faster completion → higher score
        ttc_score = 1.0 / (1.0 + ttc / ttc_max)

    # 5. Progress Variance Score (inverted)
    pv = collector.get_progress_variance(window)
    pv_max = normalization.get('progress_variance_max', 1000.0)
    # Transform: lower variance → higher score
    pv_score = 1.0 / (1.0 + pv / pv_max)

    # Compute weighted T-score
    T = (weights[0] * cr +
         weights[1] * dr_score +
         weights[2] * rt_score +
         weights[3] * ttc_score +
         weights[4] * pv_score)

    # Ensure T is in [0, 1]
    T = np.clip(T, 0.0, 1.0)

    return float(T)


def compute_T_score_from_metrics(metrics: Dict[str, float],
                                  config: Dict[str, Any]) -> float:
    """
    Compute T-score directly from pre-computed metrics

    Args:
        metrics: Dictionary containing:
            - completion_rate
            - death_rate
            - reward_trend
            - time_to_complete
            - progress_variance
        config: Configuration dictionary

    Returns:
        T-score in [0, 1]
    """
    weights = config.get('weights', [0.2, 0.2, 0.2, 0.2, 0.2])
    normalization = config.get('normalization', {})

    # 1. Completion Rate
    cr = metrics.get('completion_rate', 0.0)

    # 2. Death Rate Score
    dr = metrics.get('death_rate', 0.0)
    dr_max = normalization.get('death_rate_max', 10.0)
    dr_score = 1.0 / (1.0 + dr / dr_max)

    # 3. Reward Trend Score
    rt = metrics.get('reward_trend', 0.0)
    rt_min = normalization.get('reward_trend_min', -100.0)
    rt_max = normalization.get('reward_trend_max', 100.0)
    rt_score = normalize_value(rt, rt_min, rt_max, clip=True)

    # 4. Time-to-Complete Score
    ttc = metrics.get('time_to_complete', 0.0)
    if ttc == 0:
        ttc_score = 0.0
    else:
        ttc_max = normalization.get('time_to_complete_max', 5000.0)
        ttc_score = 1.0 / (1.0 + ttc / ttc_max)

    # 5. Progress Variance Score
    pv = metrics.get('progress_variance', 0.0)
    pv_max = normalization.get('progress_variance_max', 1000.0)
    pv_score = 1.0 / (1.0 + pv / pv_max)

    # Compute weighted T-score
    T = (weights[0] * cr +
         weights[1] * dr_score +
         weights[2] * rt_score +
         weights[3] * ttc_score +
         weights[4] * pv_score)

    T = np.clip(T, 0.0, 1.0)

    return float(T)


def get_metric_contributions(collector: MetricsCollector,
                             config: Dict[str, Any],
                             window: int = None) -> Dict[str, float]:
    """
    Get individual contributions of each metric to the T-score

    Useful for debugging and understanding which metrics drive the score

    Args:
        collector: MetricsCollector instance
        config: Configuration dictionary
        window: Window size for metrics

    Returns:
        Dictionary of metric contributions
    """
    if len(collector) == 0:
        return {}

    weights = config.get('weights', [0.2, 0.2, 0.2, 0.2, 0.2])
    normalization = config.get('normalization', {})

    if window is None:
        window = config.get('window_size', collector.window_size)

    if len(collector.get_recent(window)) < 2:
        return {}

    # Compute individual scores
    cr = collector.get_completion_rate(window)

    dr = collector.get_death_rate(window)
    dr_max = normalization.get('death_rate_max', 10.0)
    dr_score = 1.0 / (1.0 + dr / dr_max)

    rt = collector.get_reward_trend(window)
    rt_min = normalization.get('reward_trend_min', -100.0)
    rt_max = normalization.get('reward_trend_max', 100.0)
    rt_score = normalize_value(rt, rt_min, rt_max, clip=True)

    ttc = collector.get_time_to_complete(window)
    ttc_score = 0.0 if ttc == 0 else 1.0 / (1.0 + ttc / normalization.get('time_to_complete_max', 5000.0))

    pv = collector.get_progress_variance(window)
    pv_max = normalization.get('progress_variance_max', 1000.0)
    pv_score = 1.0 / (1.0 + pv / pv_max)

    return {
        'completion_rate': {
            'raw': cr,
            'score': cr,
            'contribution': weights[0] * cr
        },
        'death_rate': {
            'raw': dr,
            'score': dr_score,
            'contribution': weights[1] * dr_score
        },
        'reward_trend': {
            'raw': rt,
            'score': rt_score,
            'contribution': weights[2] * rt_score
        },
        'time_to_complete': {
            'raw': ttc,
            'score': ttc_score,
            'contribution': weights[3] * ttc_score
        },
        'progress_variance': {
            'raw': pv,
            'score': pv_score,
            'contribution': weights[4] * pv_score
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate T-score configuration

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check weights
    if 'weights' not in config:
        raise ValueError("Config must contain 'weights'")

    weights = config['weights']
    if len(weights) != 5:
        raise ValueError(f"Expected 5 weights, got {len(weights)}")

    if not all(0 <= w <= 1 for w in weights):
        raise ValueError("All weights must be in [0, 1]")

    weight_sum = sum(weights)
    if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
        raise ValueError(f"Weights must sum to 1, got {weight_sum}")

    # Check normalization
    if 'normalization' not in config:
        raise ValueError("Config must contain 'normalization'")

    required_norm_keys = [
        'death_rate_max',
        'reward_trend_min',
        'reward_trend_max',
        'time_to_complete_max',
        'progress_variance_max'
    ]

    normalization = config['normalization']
    for key in required_norm_keys:
        if key not in normalization:
            raise ValueError(f"Normalization missing required key: {key}")

    return True
