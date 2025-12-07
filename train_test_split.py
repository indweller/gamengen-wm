from datasets import load_dataset, DatasetDict

# 1. Load original dataset
ds = load_dataset("Flaaaande/mario-png", cache_dir="E:\Datasets\mario-png")

# 2. Split ONCE
split = ds["train"].train_test_split(test_size=0.1, seed=42)

# 3. Rename key if you want
dataset_dict = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
})

# 4. Push to HF
dataset_dict.push_to_hub("Flaaaande/mario-png-split")
