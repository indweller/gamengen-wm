# DeepRTS GameNGen: Testing Diffusion Models for Strategic Reasoning

This adapts the GameNGen approach (originally for DOOM) to the DeepRTS real-time strategy game. The goal is to test whether diffusion models can capture **strategic reasoning** or if they're just creating visually plausible frames.

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
