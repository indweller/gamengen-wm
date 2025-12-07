from datetime import datetime
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import AutoencoderKL
from transformers import get_cosine_schedule_with_warmup

from datasets import load_dataset
from torchvision import transforms

from config_sd import PRETRAINED_MODEL_NAME_OR_PATH

import wandb

# -------------------------
# Hyperparameters
# -------------------------
NUM_EPOCHS = 2
NUM_WARMUP_STEPS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0
EVAL_STEP = 1000
TEST_SIZE = 0.1  # fraction of data for validation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion VAE on Flaaaande/mario-png"
    )

    parser.add_argument(
        "--hf_model_repo",
        type=str,
        required=True,
        help=(
            "Hugging Face model repo ID to push the finetuned VAE to "
            "(e.g. 'yourname/gamengen-vae')"
        ),
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data to use for validation (default: 0.1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (default: 16)",
    )

    return parser.parse_args()


def make_decoder_trainable(model: AutoencoderKL):
    """Freeze encoder, train decoder only."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)


def build_dataloaders(test_size: float, batch_size: int):
    """
    Load Flaaaande/mario-png and return train/test dataloaders.
    Dataset has:
      - split: train
      - column: 'image' (H x W x C, uint8)
    """

    # Load full dataset
    dataset = load_dataset("Flaaaande/mario-png")  # default config, has 'train' split

    # Train/val split
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    # Transform: HWC uint8 -> CHW float in [-1, 1]
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # HWC [0,255] -> CHW [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
        ]
    )

    def hf_transform(batch):
        # batch["image"] is a list of 3D arrays (H,W,C) with uint8
        imgs = [np.array(x, dtype=np.uint8) for x in batch["image"]]
        pixel_values = [image_transform(img) for img in imgs]
        batch["pixel_values"] = pixel_values
        return batch

    # Apply transform lazily
    train_dataset.set_transform(hf_transform)
    test_dataset.set_transform(hf_transform)

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        return {"pixel_values": pixel_values}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader


def eval_model(model: AutoencoderKL, test_loader: DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    test_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for i, batch in enumerate(progress_bar):
            data = batch["pixel_values"].to(device)
            reconstruction = model(data).sample
            loss = F.mse_loss(reconstruction, data, reduction="mean")
            test_loss += loss.item()

            # Log first batch reconstructions
            if i == 0:
                with torch.no_grad():
                    latents = model.encode(data).latent_dist.sample()
                    recon = model.decode(latents).sample
                wandb.log(
                    {
                        "original": [wandb.Image(img) for img in data[:4]],
                        "reconstructed": [wandb.Image(img) for img in recon[:4]],
                    }
                )

    return test_loss / len(test_loader)


def main():
    args = parse_args()

    wandb.init(
        project="gamengen-vae-training",
        config={
            "model": PRETRAINED_MODEL_NAME_OR_PATH,
            "num_epochs": NUM_EPOCHS,
            "eval_step": EVAL_STEP,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": NUM_WARMUP_STEPS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "hf_model_repo": args.hf_model_repo,
            "dataset": "Flaaaande/mario-png",
            "test_size": args.test_size,
        },
        name=f"vae-finetuning-mario-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    )

    # -------------------------
    # Data
    # -------------------------
    train_loader, test_loader = build_dataloaders(
        test_size=args.test_size, batch_size=args.batch_size
    )

    # -------------------------
    # Model
    # -------------------------
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vae.to(device)
    make_decoder_trainable(model)

    # -------------------------
    # Optimizer & Scheduler
    # -------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    num_training_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = min(NUM_WARMUP_STEPS, max(num_training_steps // 2, 0))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # -------------------------
    # Training Loop
    # -------------------------
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
                    "step": step,
                    "epoch": epoch,
                }
            )

            step += 1

            if step % EVAL_STEP == 0:
                test_loss = eval_model(model, test_loader)
                wandb.log({"test_loss": test_loss, "step": step})

                # Push checkpoint to Hub
                model.save_pretrained(
                    args.hf_model_repo,
                    push_to_hub=True,
                )

    # -------------------------
    # Final eval + save
    # -------------------------
    final_test_loss = eval_model(model, test_loader)
    wandb.log({"final_test_loss": final_test_loss})

    model.save_pretrained(
        args.hf_model_repo,
        push_to_hub=True,
    )


if __name__ == "__main__":
    main()
