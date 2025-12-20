# DeepRTS GameNGen: Testing Diffusion Models for Strategic Reasoning

This project adapts the GameNGen approach (originally for DOOM) to the DeepRTS real-time strategy game. The goal is to test whether diffusion models can capture **strategic reasoning** or if they're just creating visually plausible frames.

## Why DeepRTS?

DeepRTS is ideal for this experiment because:
- **Low-res, fast**: 64x64 frames, 6M+ steps/second
- **Strategic complexity**: Resource management, unit building, combat
- **15 discrete actions**: Clear action-outcome relationships
- **Open source**: Easy to modify and analyze

## Project Structure

```
deeprts_gamengen/
├── collect_dataset.py      # Step 1: Collect frame-action pairs
├── train_diffusion.py      # Step 2: Train diffusion model
├── analyze_model.py        # Step 3: Analyze strategic understanding
├── requirements.txt
└── README.md
```

## Complete Workflow

### Step 1: Collect Frame-Action Dataset

```bash
# Install DeepRTS (requires C++ build tools)
git clone https://github.com/cair/deep-rts.git
cd deep-rts && pip install .

# Collect dataset
python collect_dataset.py \
    --output ./deeprts_dataset \
    --episodes 500 \
    --map 15x15-2-FFA \
    --frame-size 64 \
    --frame-skip 4 \
    --agent heuristic
```

**Parameters:**
- `--episodes`: More episodes = more diverse training data
- `--frame-skip`: Record every Nth frame (4 is good balance)
- `--agent`: `random` or `heuristic` (heuristic is more realistic)

### Step 2: Train Diffusion Model

```bash
python train_diffusion.py \
    --dataset ./deeprts_dataset \
    --output ./deeprts_model \
    --epochs 100 \
    --batch-size 8 \
    --buffer-size 8 \
    --image-size 64
```

**Key training features:**
- **Context conditioning**: Uses past 8 frames
- **Action embedding**: Learnable 768-dim embeddings for 15 actions
- **Noise augmentation**: Gaussian noise on context frames for stable rollouts

### Step 3: Analyze the Model

```bash
python analyze_model.py \
    --model ./deeprts_model/checkpoint_final.pt \
    --dataset ./deeprts_dataset \
    --output ./analysis_results \
    --num-samples 100
```

## Analysis Methods

The `analyze_model.py` script runs 4 tests to determine if the model understands strategy:

### Test 1: Action Sensitivity Analysis
- Same context, different actions → different outputs?
- A model ignoring actions just generates "generic next frames"
- **Good sign**: High variance between different action outputs

### Test 2: Temporal Coherence
- Generate 50-step rollouts, measure quality degradation
- Check for "mode collapse" (all frames becoming similar)
- **Good sign**: Maintained diversity over long rollouts

### Test 3: Counterfactual Analysis
- What if we shuffle context frame order?
- What if we use random noise as context?
- What if we reverse time?
- **Good sign**: Model outputs change significantly with manipulated inputs

### Test 4: Strategic Consistency
- Do movement actions cause changes in the expected direction?
- MOVE_RIGHT should affect right side of frame more than left
- **Good sign**: Directional consistency scores > 0

## Interpreting Results

The analysis outputs a verdict:
- **"Understanding game logic"**: Model shows action sensitivity, temporal coherence, counterfactual sensitivity, and strategic consistency
- **"Mixed results"**: Some positive signals but also issues
- **"Pattern-matching"**: Model mostly ignores actions/context structure

## Comparison to Original GameNGen

| Aspect | GameNGen (DOOM) | This Project (DeepRTS) |
|--------|-----------------|------------------------|
| Frame size | 240×320 | 64×64 |
| Action space | Variable (buttons) | 15 discrete |
| Game type | FPS | RTS |
| Strategic depth | Aim/shoot/move | Build/harvest/combat |
| Analysis | Human rating | Automated tests |

## Why This Matters

The core question: **Can diffusion models actually learn game logic, or are they just sophisticated video predictors?**

RTS games are a harder test than FPS because:
1. **Delayed consequences**: Building a unit now affects combat later
2. **Multi-entity tracking**: Must maintain state of many units
3. **Resource constraints**: Actions have costs and dependencies
4. **Strategic planning**: Optimal play requires lookahead

If a diffusion model can maintain RTS game rules, it suggests genuine world-model capabilities. If it fails, it tells us the limits of frame prediction approaches.

## Next Steps After Analysis

1. **If model passes tests**: 
   - Try harder maps (31x31, 4+ players)
   - Test on unseen scenarios
   - Compare to explicit world models

2. **If model fails tests**:
   - Try explicit state supervision
   - Add auxiliary losses for game state prediction
   - Use larger models / more data

## Citation

Based on:
```
@article{valevski2024diffusion,
  title={Diffusion Models Are Real-Time Game Engines},
  author={Valevski, Dani and Leviathan, Yaniv and Arar, Moab and Fruchter, Shlomi},
  journal={arXiv preprint arXiv:2408.14837},
  year={2024}
}

@inproceedings{andersen2018deep,
  title={Deep RTS: A Game Environment for Deep Reinforcement Learning},
  author={Andersen, Per-Arne and Goodwin, Morten and Granmo, Ole-Christoffer},
  booktitle={IEEE Conference on Computational Intelligence and Games},
  year={2018}
}
```
