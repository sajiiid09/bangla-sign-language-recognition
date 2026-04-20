"""Create synthetic training data for SPOTER model."""

import torch
import numpy as np
from pathlib import Path
import os

# Create data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Parameters
seq_len = 48
pose_dim = 33 * 3  # x, y, z for 33 pose points
num_classes = 60
num_train_samples = 1000
num_val_samples = 200

# Generate synthetic training data
train_data = torch.randn(num_train_samples, seq_len, pose_dim)
train_labels = torch.randint(0, num_classes, (num_train_samples,))

# Generate synthetic validation data
val_data = torch.randn(num_val_samples, seq_len, pose_dim)
val_labels = torch.randint(0, num_classes, (num_val_samples,))

# Save as .npz files
train_path = data_dir / "train_data.npz"
val_path = data_dir / "val_data.npz"

np.savez(train_path, x=train_data.numpy(), y=train_labels.numpy())
np.savez(val_path, x=val_data.numpy(), y=val_labels.numpy())

print("✅ Created training data:")
print(f"  Train: {train_data.shape}")
print(f"  Val: {val_data.shape}")
print(f"  Classes: {num_classes}")
print(f"  Pose dimensions: {pose_dim}")
print(f"  Sequence length: {seq_len}")
print(f"  Data saved to: {data_dir}")
