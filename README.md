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

## Citation

```bibtex
@software{hmm_dda,
  title={HMM-DDA: Hidden Markov Model-based Dynamic Difficulty Adjustment},
  author={[Your Name]},
  year={2024}
}
```
