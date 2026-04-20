import os
import random
from pathlib import Path

# Paths - Updated for current structure
processed_dir = Path("processed")
multimodal_dir = processed_dir / "multimodal"

# Ensure directory exists
processed_dir.mkdir(parents=True, exist_ok=True)

# Get all .npz files
files = sorted([f.name for f in multimodal_dir.glob("*.npz")])
random.seed(42)  # For reproducibility
random.shuffle(files)

# Calculate split sizes (80% train, 10% val, 10% test)
total = len(files)
train_split = int(total * 0.8)
val_split = int(total * 0.1)

train_files = files[:train_split]
val_files = files[train_split : train_split + val_split]
test_files = files[train_split + val_split :]

# Write to text files
def write_list(filename, file_list):
    with open(processed_dir / filename, "w", encoding="utf-8") as f:
        for item in file_list:
            f.write(f"{item}\n")
    print(f"Created {filename} with {len(file_list)} samples")

print(f"\n{'='*60}")
print(f"📊 Creating dataset splits from {multimodal_dir}")
print(f"{'='*60}")
print(f"Found {total} processed samples.")
write_list("train_samples.txt", train_files)
write_list("val_samples.txt", val_files)
write_list("test_samples.txt", test_files)
print(f"{'='*60}\n")
