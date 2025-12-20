# HMM-DDA: Hidden Markov Model-based Dynamic Difficulty Adjustment

A framework for adaptive difficulty in procedurally generated Super Mario levels using Hidden Markov Models and reinforcement learning.

## Overview

This framework implements a three-state HMM that tracks player skill through performance metrics and adjusts level difficulty accordingly:

- **Low**: Easy difficulty (few enemies, no gaps, confidence building)
- **Transition**: Assessment state (varied challenges to gauge skill)
- **High**: Hard difficulty (many enemies, gaps, mastery challenge)

The Transition state is crucial—it's not a medium difficulty, but rather a decision point where the HMM quickly determines whether the player should move to Low or High.

## Project Structure

```
├── src/                          # Core framework modules
│   ├── __init__.py
│   ├── hmm_controller.py         # HMM state management and Bayesian updates
│   ├── level_generator.py        # Level generation (MarioGPT or procedural)
│   ├── mario_env.py              # Mario physics simulation
│   ├── metrics_collector.py      # Episode metric collection
│   ├── t_score.py                # Performance scoring (observation for HMM)
│   └── utils.py                  # Logging, plotting, config management
│
├── scripts/                      # Pipeline scripts
│   ├── calibrate.py              # Phase 1: Collect baseline data at fixed difficulties
│   ├── derive_parameters.py      # Phase 2: Fit HMM parameters from calibration
│   ├── train.py                  # Phase 3: Train RL agent with HMM adaptation
│   ├── evaluate.py               # Phase 4: Analyze results and generate visualizations
│   └── run_pipeline.py           # Orchestrate all phases
│
├── config/                       # HMM parameters (generated during calibration)
│   ├── normalization_bounds.json
│   ├── metric_weights.json
│   ├── emission_params.json
│   ├── thresholds.json
│   ├── transition_matrix.json
│   └── prompts.json
│
├── checkpoints/                  # Training checkpoints
│   ├── hmm_*.json
│   ├── ppo_*.zip
│   └── metrics_*.csv
│
└── figures/                      # Output visualizations
    ├── calibration/
    └── evaluation/
```

## Key Concepts

### T-Score (Section 4.2)

Performance metric combining five normalized sub-scores:
- **Completion Rate** (CR): % of levels completed
- **Death Rate** (DR): Penalizes deaths
- **Reward Trend** (RT): Positive slope indicates improvement
- **Time to Complete** (TTC): Rewards fast completion
- **Progress Variance** (PV): Penalizes inconsistent performance

**Interpretation:**
- T > 0.65 → High skill signal
- 0.35 < T < 0.65 → Ambiguous (stay in Transition)
- T < 0.35 → Low skill signal

### HMM States (Section 3.2)

Three-state Markov model with design emphasis on assessment:

```
Transition Matrix:
            Low   Trans  High
Low        [0.70  0.25  0.05]  (stable, gradual progression)
Trans      [0.20  0.40  0.40]  (LOW self-loop for quick decisions)
High       [0.05  0.25  0.70]  (stable, occasional regression)
```

Key: Transition has 0.40 self-loop (vs 0.70 for Low/High) to force quick assessment.

## Quick Start

### 1. Basic Setup

```bash
# Install dependencies
pip install gymnasium stable-baselines3 pandas numpy scipy matplotlib seaborn

# Optional: MarioGPT for better level generation
pip install mario-gpt torch
```

### 2. Run Complete Pipeline

```bash
# Full pipeline (calibration → parameters → training → evaluation)
python scripts/run_pipeline.py --episodes 1000 --device cpu

# Skip calibration (use defaults)
python scripts/run_pipeline.py --skip-calibration --episodes 1000

# Custom settings
python scripts/run_pipeline.py \
  --episodes 5000 \
  --calibration-episodes 50 \
  --device cuda \
  --output-dir ./results
```

### 3. Individual Phases

```bash
# Phase 1: Calibration (collect baseline data)
python scripts/calibrate.py

# Phase 2: Derive HMM parameters
python scripts/derive_parameters.py

# Phase 3: Train with HMM
python scripts/train.py

# Phase 4: Evaluate results
python scripts/evaluate.py
```

## Pipeline Details

### Phase 1: Calibration
**Input:** PPO agent, level generator  
**Output:** `calibration_data/{low,transition,high}_metrics.csv`

Collects 50 episodes at each fixed difficulty with a pre-trained baseline agent to understand baseline performance.

### Phase 2: Parameter Derivation
**Input:** Calibration data  
**Output:** `config/` (5 JSON files)

- Computes normalization bounds from calibration data
- Optimizes metric weights via Fisher's discriminant
- Fits Gaussian emission distributions: N(μ, σ) for each state
- Derives T-score thresholds from distribution intersections

### Phase 3: Training
**Input:** HMM parameters, PPO agent  
**Output:** `checkpoints/` (HMM states, RL model, metrics)

Main training loop:
1. Generate level based on current HMM state
2. Run episode, collect metrics
3. Every N episodes: compute T-score, update HMM belief
4. HMM emits new state → level generator adapts

### Phase 4: Evaluation
**Input:** Training checkpoints  
**Output:** Visualizations in `figures/evaluation/`

Generates:
- Learning curve (with state coloring)
- Belief evolution over time
- State distribution pie chart
- Performance metrics by state
- Flow zone analysis

## Configuration

All parameters saved as JSON in `config/`:

```json
{
  "metric_weights.json": {
    "weights": [0.25, 0.20, 0.25, 0.15, 0.15]
  },
  "emission_params.json": {
    "Low": {"mu": 0.25, "sigma": 0.15},
    "Transition": {"mu": 0.50, "sigma": 0.12},
    "High": {"mu": 0.75, "sigma": 0.15}
  },
  "thresholds.json": {
    "low_transition": 0.35,
    "transition_high": 0.65
  }
}
```

## Module Reference

### `hmm_controller.py`
- `HMM_DDA`: Main controller class
  - `update(T_score)`: Bayesian update step
  - `get_current_state()`: Current belief argmax
  - `get_prompt()`: MarioGPT prompt for current state
  - `adapt_transition_matrix()`: Online tuning to prevent oscillation

### `t_score.py`
- `compute_T_score()`: From MetricsCollector
- `compute_T_score_from_metrics()`: From pre-computed metrics dict
- `get_metric_contributions()`: Breakdown of each component
- `interpret_T_score()`: Human-readable interpretation

### `metrics_collector.py`
- `MetricsCollector`: Circular buffer of episode metrics
  - `add_episode()`: Add metrics
  - `get_completion_rate()`: Windowed stat
  - `get_reward_trend()`: Linear regression slope
  - `get_all_metrics()`: All five metrics at once

### `level_generator.py`
- `LevelGenerator`: Generates ASCII levels
  - Primary: MarioGPT (if installed)
  - Fallback: Procedural with difficulty parameters
- `generate_for_state()`: Generate for Low/Transition/High

### `mario_env.py`
- `MarioGridEnv`: Physics-based platformer
  - No NES emulation; pure Python
  - Tracks metrics for T-score computation
- `MarioEnvWrapper`: Compatibility layer

## Training Configuration

Edit `scripts/train.py`:

```python
CONFIG = {
    'total_episodes': 5000,
    'hmm_update_frequency': 10,      # Update HMM every N episodes
    'metrics_window': 10,             # Window for T-score
    'checkpoint_frequency': 500,
    'log_frequency': 50,
    'max_steps_per_episode': 2000,
    'adaptation_frequency': 500,      # Adapt transition matrix
    'ppo_config': {
        'learning_rate': 3e-4,
        'n_steps': 512,
        'batch_size': 64,
        # ... see scripts/train.py
    }
}
```

## Expected Results

After training ~5000 episodes:
- **Flow Zone**: >60% of episodes in balanced reward range
- **State Distribution**: Roughly 30-40% Low, 20-30% Transition, 30-40% High
- **Transition Frequency**: 1-3 transitions per 100 episodes (stable)
- **Learning Curve**: Consistent upward trend with occasional plateaus

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MarioGPT not loading | Install with `pip install mario-gpt torch`, or use procedural fallback |
| CUDA out of memory | Use `--device cpu` or reduce `batch_size` in `train.py` |
| Calibration fails | Ensure `stable-baselines3` is installed; uses heuristic agent as fallback |
| No state transitions | Check HMM transition matrix; may need to adjust self-loop probabilities |

## References

- Framework: Sections 3-7 in accompanying PDF
- Paper: Hidden Markov Models for Dynamic Difficulty Adjustment in Games
- T-Score metrics: Section 4.2
- HMM design: Section 3.2 (Transition as assessment state)

## License

[Specify your license here]

## Citation

```bibtex
@software{hmm_dda,
  title={HMM-DDA: Hidden Markov Model-based Dynamic Difficulty Adjustment},
  author={[Your Name]},
  year={2024}
}
```
