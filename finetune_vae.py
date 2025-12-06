from datetime import datetime
import argparse
import math

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config_sd import PRETRAINED_MODEL_NAME_OR_PATH
from png_dataset import PNGDataset  # <-- our custom dataset

import wandb

# Fine-tuning parameters
NUM_EPOCHS = 2
NUM_WARMUP_STEPS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0
EVAL_STEP = 1000
IMAGE_SIZE = 512  # adjust if you want another size


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune VAE model on PNG dataset")
    parser.add_argument(
        "--hf_model_folder",
        type=str,
        required=True,
        help="HuggingFace model folder to save the model to",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder containing PNG images (recursively searched)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Target image size (height=width)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for test/validation",
    )
    return parser.parse_args()


def make_decoder_trainable(model: AutoencoderKL):
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(True)


def eval_model(model: AutoencoderKL, test_loader: DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    test_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for batch in progress_bar:
            data = batch["pixel_values"].to(device)
            reconstruction = model(data).sample
            loss = F.mse_loss(reconstruction, data, reduction="mean")
            test_loss += loss.item()

            # Log a few reconstructions
            recon = model.decode(model.encode(data).latent_dist.sample()).sample
            wandb.log(
                {
                    "original": [wandb.Image(img) for img in data],
                    "reconstructed": [wandb.Image(img) for img in recon],
                }
            )

    return test_loss / len(test_loader)


def main():
    args = parse_args()
    wandb.init(
        project="gamengen-vae-training",
        config={
            # Model parameters
            "model": PRETRAINED_MODEL_NAME_OR_PATH,
            # Training parameters
            "num_epochs": NUM_EPOCHS,
            "eval_step": EVAL_STEP,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": NUM_WARMUP_STEPS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "hf_model_folder": args.hf_model_folder,
            "data_root": args.data_root,
            "image_size": args.image_size,
        },
        name=f"vae-finetuning-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    )

    # =====================
    # Dataset Setup (PNG)
    # =====================
    full_dataset = PNGDataset(root_dir=args.data_root)
    n_total = len(full_dataset)
    n_test = max(1, int(math.floor(n_total * args.test_size)))
    n_train = n_total - n_test

    train_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
    )

    # =====================
    # Model Setup
    # =====================
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vae.to(device)
    make_decoder_trainable(model)

    # =====================
    # Optimizer & Scheduler
    # =====================
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    num_training_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            data = batch["pixel_values"].to(device)
            optimizer.zero_grad()

            reconstruction = model(data).sample
            loss = F.mse_loss(reconstruction, data, reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({"loss": loss.item(), "lr": current_lr})

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                }
            )

            step += 1
            if step % EVAL_STEP == 0:
                test_loss = eval_model(model, test_loader)

                # save model to hub
                model.save_pretrained(
                    "test",
                    repo_id=args.hf_model_folder,
                    push_to_hub=True,
                )
                wandb.log({"test_loss": test_loss})


if __name__ == "__main__":
    main()
