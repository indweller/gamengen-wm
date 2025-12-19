"""
T-Score Computation Module

Computes weighted performance score from multiple metrics.
This score is the observation that feeds into the HMM.

T = w1*CR + w2*DR_score + w3*RT_score + w4*TTC_score + w5*PV_score

Where:
- CR: Completion Rate (already in [0,1])
- DR_score: Death Rate (inverted: fewer deaths → higher score)
- RT_score: Reward Trend (positive slope → higher score)
- TTC_score: Time to Complete (faster → higher score)
- PV_score: Progress Variance (consistent → higher score)

Interpretation: 
- T > 0.65: Strong signal for High state
- 0.35 < T < 0.65: Ambiguous; stay in Transition
- T < 0.35: Strong signal for Low state
"""

import numpy as np
from typing import Dict, Any, Optional
from .metrics_collector import MetricsCollector


def normalize_value(value: float, min_val: float, max_val: float, clip: bool = True) -> float:
    """Normalize value to [0, 1]"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def compute_T_score(collector: MetricsCollector,
                    config: Optional[Dict[str, Any]] = None,
                    window: int = 10) -> float:
    """
    Compute T-score from metrics collector.
    
    Args:
        collector: MetricsCollector with episode data
        config: Configuration with weights and normalization bounds
        window: Number of recent episodes to consider
        
    Returns:
        T-score in [0, 1]
    """
    if len(collector) == 0:
        return 0.5 
    
    if config is None:
        config = {
            'weights': [0.25, 0.20, 0.25, 0.15, 0.15],  # CR, DR, RT, TTC, PV
            'normalization': {
                'death_rate_max': 5.0,
                'reward_trend_min': -50.0,
                'reward_trend_max': 50.0,
                'time_to_complete_max': 2000.0,
                'progress_variance_max': 500.0,
            }
        }
    
    weights = config.get('weights', [0.25, 0.20, 0.25, 0.15, 0.15])
    norm = config.get('normalization', {})
    
    if len(collector.get_recent(window)) < 2:
        return 0.5  
    
    cr = collector.get_completion_rate(window)
    
    dr = collector.get_death_rate(window)
    dr_max = norm.get('death_rate_max', 5.0)
    dr_score = 1.0 / (1.0 + dr / dr_max)
    
    rt = collector.get_reward_trend(window)
    rt_min = norm.get('reward_trend_min', -50.0)
    rt_max = norm.get('reward_trend_max', 50.0)
    rt_score = normalize_value(rt, rt_min, rt_max)
    
    ttc = collector.get_time_to_complete(window)
    if ttc == 0:
        ttc_score = 0.2 
    else:
        ttc_max = norm.get('time_to_complete_max', 2000.0)
        ttc_score = 1.0 / (1.0 + ttc / ttc_max)
    
    pv = collector.get_progress_variance(window)
    pv_max = norm.get('progress_variance_max', 500.0)
    pv_score = 1.0 / (1.0 + pv / pv_max)
    
    T = (weights[0] * cr +
         weights[1] * dr_score +
         weights[2] * rt_score +
         weights[3] * ttc_score +
         weights[4] * pv_score)
    
    return float(np.clip(T, 0.0, 1.0))


def compute_T_score_from_metrics(metrics: Dict[str, float],
                                  config: Optional[Dict[str, Any]] = None) -> float:
    """
    Compute T-score from pre-computed metrics dict.
    
    Args:
        metrics: Dict with completion_rate, death_rate, reward_trend, etc.
        config: Configuration
        
    Returns:
        T-score in [0, 1]
    """
    if config is None:
        config = {
            'weights': [0.25, 0.20, 0.25, 0.15, 0.15],
            'normalization': {
                'death_rate_max': 5.0,
                'reward_trend_min': -50.0,
                'reward_trend_max': 50.0,
                'time_to_complete_max': 2000.0,
                'progress_variance_max': 500.0,
            }
        }
    
    weights = config.get('weights', [0.25, 0.20, 0.25, 0.15, 0.15])
    norm = config.get('normalization', {})
    
    cr = metrics.get('completion_rate', 0.0)
    
    dr = metrics.get('death_rate', 0.0)
    dr_score = 1.0 / (1.0 + dr / norm.get('death_rate_max', 5.0))
    
    rt = metrics.get('reward_trend', 0.0)
    rt_score = normalize_value(rt, 
                               norm.get('reward_trend_min', -50.0),
                               norm.get('reward_trend_max', 50.0))
    
    ttc = metrics.get('time_to_complete', 0.0)
    if ttc == 0:
        ttc_score = 0.2
    else:
        ttc_score = 1.0 / (1.0 + ttc / norm.get('time_to_complete_max', 2000.0))
    
    pv = metrics.get('progress_variance', 0.0)
    pv_score = 1.0 / (1.0 + pv / norm.get('progress_variance_max', 500.0))
    
    T = (weights[0] * cr +
         weights[1] * dr_score +
         weights[2] * rt_score +
         weights[3] * ttc_score +
         weights[4] * pv_score)
    
    return float(np.clip(T, 0.0, 1.0))


def get_metric_contributions(collector: MetricsCollector,
                             config: Optional[Dict[str, Any]] = None,
                             window: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Get breakdown of each metric's contribution to T-score.
    Useful for debugging and understanding what drives difficulty changes.
    """
    if config is None:
        config = {
            'weights': [0.25, 0.20, 0.25, 0.15, 0.15],
            'normalization': {
                'death_rate_max': 5.0,
                'reward_trend_min': -50.0,
                'reward_trend_max': 50.0,
                'time_to_complete_max': 2000.0,
                'progress_variance_max': 500.0,
            }
        }
    
    weights = config['weights']
    norm = config['normalization']
    
    cr = collector.get_completion_rate(window)
    dr = collector.get_death_rate(window)
    dr_score = 1.0 / (1.0 + dr / norm.get('death_rate_max', 5.0))
    rt = collector.get_reward_trend(window)
    rt_score = normalize_value(rt, norm.get('reward_trend_min', -50), norm.get('reward_trend_max', 50))
    ttc = collector.get_time_to_complete(window)
    ttc_score = 0.2 if ttc == 0 else 1.0 / (1.0 + ttc / norm.get('time_to_complete_max', 2000))
    pv = collector.get_progress_variance(window)
    pv_score = 1.0 / (1.0 + pv / norm.get('progress_variance_max', 500))
    
    return {
        'completion_rate': {'raw': cr, 'score': cr, 'contribution': weights[0] * cr},
        'death_rate': {'raw': dr, 'score': dr_score, 'contribution': weights[1] * dr_score},
        'reward_trend': {'raw': rt, 'score': rt_score, 'contribution': weights[2] * rt_score},
        'time_to_complete': {'raw': ttc, 'score': ttc_score, 'contribution': weights[3] * ttc_score},
        'progress_variance': {'raw': pv, 'score': pv_score, 'contribution': weights[4] * pv_score},
    }


def interpret_T_score(T: float) -> str:
    """
    Interpret T-score according to framework thresholds.
    
    Returns:
        Signal interpretation string
    """
    if T > 0.65:
        return "High (player performing well, increase difficulty)"
    elif T < 0.35:
        return "Low (player struggling, decrease difficulty)"
    else:
        return "Ambiguous (stay in assessment/transition)"
