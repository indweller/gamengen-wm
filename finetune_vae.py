from datetime import datetime
import argparse
import math

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config_sd import PRETRAINED_MODEL_NAME_OR_PATH

from datasets import load_dataset
from torchvision import transforms

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
    parser = argparse.ArgumentParser(description="Fine-tune VAE model on HF image dataset")

    parser.add_argument(
        "--hf_model_folder",
        type=str,
        required=True,
        help="HuggingFace model repo id to push the model to (e.g. user/vae-finetune)",
    )

    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        required=True,
        help="HuggingFace dataset repo id to load images from (e.g. user/mario-frames)",
    )

    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of the training split in the dataset (default: train)",
    )

    parser.add_argument(
        "--val_split",
        type=str,
        default="validation",
        help="Name of the validation split in the dataset (if missing, will be created via train_test_split)",
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
        help="Fraction of data to use for test/validation if val_split does not exist",
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

            # Optionally log a few reconstructions (here: first batch only)
            recon = model.decode(model.encode(data).latent_dist.sample()).sample
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
            "hf_dataset_repo": args.hf_dataset_repo,
            "image_size": args.image_size,
        },
        name=f"vae-finetuning-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    )

    # =====================
    # Dataset Setup (HF)
    # =====================

    # Load dataset dict; e.g. {"train": Dataset, "validation": Dataset}
    raw_dataset = load_dataset(args.hf_dataset_repo)

    if args.train_split not in raw_dataset:
        raise ValueError(f"Train split '{args.train_split}' not found in dataset")

    if args.val_split in raw_dataset:
        train_dataset = raw_dataset[args.train_split]
        test_dataset = raw_dataset[args.val_split]
    else:
        # Create a validation split from the training split
        split = raw_dataset[args.train_split].train_test_split(
            test_size=args.test_size, seed=42
        )
        train_dataset = split["train"]
        test_dataset = split["test"]

    # Transform: resize -> tensor -> normalize to [-1, 1]
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
        ]
    )

    def hf_transform(examples):
        # expects column "image" with PIL Images
        images = [img.convert("RGB") for img in examples["image"]]
        pixel_values = [image_transform(img) for img in images]
        return {"pixel_values": pixel_values}

    train_dataset.set_transform(hf_transform)
    test_dataset.set_transform(hf_transform)

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        return {"pixel_values": pixel_values}

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
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
    warmup_steps = min(NUM_WARMUP_STEPS, num_training_steps // 2) if num_training_steps > 0 else 0

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
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

                # Save model to Hugging Face Hub
                model.save_pretrained(
                    args.hf_model_folder,
                    push_to_hub=True,
                )
                wandb.log({"test_loss": test_loss})

    # Final eval + save (optional)
    final_test_loss = eval_model(model, test_loader)
    wandb.log({"final_test_loss": final_test_loss})
    model.save_pretrained(args.hf_model_folder, push_to_hub=True)


if __name__ == "__main__":
    main()
