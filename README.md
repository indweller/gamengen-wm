# Adversarial GameNGen - Dynamic Difficulty adjustment with Diffusion Models for Super Mario Bros

This project trains a world model to simulate Super Mario Bros gameplay. The implementation includes training and evaluation code for RL agents, adversarial initialization, reward models, and latent diffusion models.

## Installation

### Requirements

Install dependencies for training the RL agent:

```bash
pip install -r requirements.txt
```

For training the diffusion model use:

```bash
pip install -r requirements2.txt
```

The two requirements are not compatible and require two separate virtual environments.

### Training

- **PPO Agent**: `python train_ppo_mario.py` (default config solves level 1-1)
- **Dataset Generation**: `python generate_dataset.py --episodes <no_episodes> --output <gif/parquet> --upload --hf_repo <repo_name>`
- **Finetuning VAE**: `python finetune_autoencoder.py --hf_model_folder <repo_name>`
    
    Before running the Diffusion train code, make sure to change the <REPO_NAME> inside config_sd.py
- **Diffusion Model**: 
```
    python train_text_to_image.py \
    --dataset_name Flaaaande/mario-png-actions \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --train_batch_size 12 \
    --dataloader_num_workers 18 \
    --num_train_epochs 3 \
    --validation_steps 1000 \
    --use_cfg \
    --output_dir sd-model-finetuned \
    --push_to_hub \
    --lr_scheduler cosine \
    --report_to wandb
```
- **Diffusion RL environement with Adversarial Initialization**: `python train_adversarial.py --model_folder Flaaaande/mario-sd`
    - Loads trained diffusion model, VAE and reward models from HuggingFace
    - use `--adv_steps 0` for unperturbed environment

### Evaluation & Inference

- **Evaluation metrics (diffusion)**: `python run_evaluation.py`
- **Run Inference (1 step game simulation)**: `python run_inference.py --model_folder ./sd-model-finetuned`
- **Autoregressive game simulation**: `python adversarial_dist_eval.py`

### Notes

- `Mario-GPT` and `deeprts` branches contain respective implementations.
- All codes accept command line arguments for configuration. Please refer to the code files for details. 
- Jupyter notebooks contain exploratory codes for the corresponding portions.
- View rollouts and visualization [here](https://drive.google.com/drive/folders/1dZOtOZOf5_zoiFqR9MaaVuUpSEDJ8TnA?usp=drive_link)

## References

- [GameNGen Paper Implementation](https://github.com/arnaudstiegler/gameNgen-repro)
- [Super Mario Bros Dataset](https://github.com/rafaelcp/smbdataset)