"""
Overfit Probe: Minimal test to verify model can memorize a single batch.

Purpose:
  Validates that gradient flow and optimization work correctly by training
  a model on 1 batch of 1 class for 100+ steps. If the model can reach
  100% accuracy on a single batch, we know:
  - Gradients are flowing correctly
  - Optimizer is stepping properly
  - Learning dynamics work at the implementation level
  
Expected behavior:
  - Loss should decrease smoothly toward ~0
  - Batch accuracy should reach 95%+ (ideally 100%)
  - Should complete in <1 minute on GPU

Failure modes:
  - Loss doesn't decrease: gradient flow issue
  - Loss NaN/Inf: numerical instability
  - Memory error: model too large for batch
  - Accuracy plateaus <90%: learning rate too low or optimization broken
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import random
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.signet_v2 import SignNetV2
from src.data.preprocessing import DataConfig, SignLanguageDataset


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_single_batch_single_class(
    processed_dir: Path,
    split_file: Path,
    target_class_idx: int = 6,  # "উত্তর" (class_6)
    batch_size: int = 16,
) -> tuple:
    """
    Load a single batch of samples from one class for probing.
    
    Args:
        processed_dir: Path to processed data directory
        split_file: Path to train_samples.txt file
        target_class_idx: Target class index
        batch_size: Batch size
        
    Returns:
        (batch_body, batch_left_hand, batch_right_hand, batch_face, batch_labels, batch_masks)
    """
    # Load label mapping
    label_map_path = processed_dir / "label_mapping.json"
    if not label_map_path.exists():
        print(f"⚠️  Label mapping not found at {label_map_path}")
        label_to_word = {6: "উত্তর"}  # Fallback
    else:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_to_word = json.load(f)
            # Reverse mapping if needed
            if isinstance(list(label_to_word.keys())[0], str):
                label_to_word = {int(k): v for k, v in label_to_word.items()}

    target_word = label_to_word.get(target_class_idx, "unknown")
    print(f"📍 Target class: {target_class_idx} = '{target_word}'")

    # Load split samples
    with open(split_file, "r", encoding="utf-8") as f:
        all_samples = [line.strip() for line in f if line.strip()]

    # Filter samples matching target class
    matching_samples = []
    for sample_path in all_samples:
        # Extract word from path: e.g., "word_signer_000.npz" -> "word"
        parts = sample_path.split("_")
        if parts:
            word = parts[0].strip()
            if word == target_word:
                matching_samples.append(sample_path)

    if not matching_samples:
        print(f"❌ No samples found for class '{target_word}' in split file")
        print(f"   First 5 samples: {all_samples[:5]}")
        return None

    print(f"✅ Found {len(matching_samples)} samples for class '{target_word}'")

    # Create dataset for loading
    data_config = DataConfig(
        base_dir=str(processed_dir.parent.parent),
        processed_dir=str(processed_dir),
        normalized_dir=str(processed_dir / "normalized"),
        checkpoint_dir=str(processed_dir / "checkpoints"),
        max_seq_length=150,
        augmentation=False,  # No augmentation for probe
        loader_error_mode="strict",
    )

    # Build word_to_label mapping
    word_to_label = {}
    label_map_path = processed_dir / "label_mapping.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(list(data.keys())[0], str):
                word_to_label = {k: int(v) for k, v in data.items()}
            else:
                word_to_label = {v: k for k, v in data.items()}
    
    # Fallback mapping
    if not word_to_label:
        word_to_label[target_word] = target_class_idx

    # Create dataset
    dataset = SignLanguageDataset(
        sample_paths=matching_samples[:batch_size],
        word_to_label=word_to_label,
        normalized_dir=data_config.normalized_dir,
        config=data_config,
        augment=False,
        mode="test",
        use_hands=True,  # Include hands
        use_face=False,   # No face
    )

    print(f"📦 Loading {min(batch_size, len(matching_samples))} samples...")

    # Manually load batch to avoid DataLoader complexity
    batch_body = []
    batch_left_hand = []
    batch_right_hand = []
    batch_face = []
    batch_labels = []
    batch_masks = []

    for i in range(min(batch_size, len(dataset))):
        try:
            sample = dataset[i]
            batch_body.append(sample["body_pose"])
            
            # Handle missing hand data (fallback to zeros)
            if sample["left_hand"] is None:
                left_hand = torch.zeros_like(sample["body_pose"][:, :63])
            else:
                left_hand = sample["left_hand"]
            batch_left_hand.append(left_hand)
            
            if sample["right_hand"] is None:
                right_hand = torch.zeros_like(sample["body_pose"][:, :63])
            else:
                right_hand = sample["right_hand"]
            batch_right_hand.append(right_hand)
            
            batch_labels.append(sample["label"])
            batch_masks.append(sample["attention_mask"])
        except Exception as e:
            print(f"⚠️  Failed to load sample {i}: {e}")
            continue

    if not batch_body:
        print(f"❌ Failed to load any samples")
        return None

    # Stack into batches
    batch_body = torch.stack(batch_body)  # (N, 150, 99)
    batch_left_hand = torch.stack(batch_left_hand)  # (N, 150, 63)
    batch_right_hand = torch.stack(batch_right_hand)  # (N, 150, 63)
    batch_labels = torch.stack(batch_labels)  # (N,)
    batch_masks = torch.stack(batch_masks)  # (N, 150)

    print(f"✅ Batch shapes:")
    print(f"   Body: {batch_body.shape}")
    print(f"   Left hand: {batch_left_hand.shape}")
    print(f"   Right hand: {batch_right_hand.shape}")
    print(f"   Labels: {batch_labels.shape} (all should be {target_class_idx})")
    print(f"   Masks: {batch_masks.shape}")

    return batch_body, batch_left_hand, batch_right_hand, batch_face, batch_labels, batch_masks


def run_overfit_probe(
    model: nn.Module,
    batch_data: tuple,
    num_steps: int = 100,
    learning_rate: float = 3e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Train model on single batch for num_steps and verify it memorizes.
    
    Args:
        model: SignNetV2 model
        batch_data: Tuple of (body, left_hand, right_hand, face, labels, masks)
        num_steps: Number of training steps
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        dict with metrics
    """
    model = model.to(device)
    model.train()

    # Unpack batch
    batch_body, batch_left_hand, batch_right_hand, batch_face, batch_labels, batch_masks = batch_data

    # Move to device
    batch_body = batch_body.to(device)
    batch_left_hand = batch_left_hand.to(device)
    batch_right_hand = batch_right_hand.to(device)
    batch_labels = batch_labels.to(device)
    batch_masks = batch_masks.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    losses = []
    accuracies = []

    print(f"\n🔬 Running overfit probe ({num_steps} steps)...")

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_body, batch_left_hand, batch_right_hand, None, batch_masks)

        # Loss
        loss = criterion(logits, batch_labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == batch_labels).float().mean().item()

        losses.append(loss.item())
        accuracies.append(accuracy)

        # Progress
        if (step + 1) % 10 == 0:
            print(
                f"  Step {step + 1:3d}/{num_steps}: loss={loss.item():.6f}, accuracy={accuracy:.4f}"
            )

    print(f"\n📊 Probe Results:")
    print(f"   Initial loss: {losses[0]:.6f}")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Loss reduction: {losses[0] - losses[-1]:.6f} ({100 * (1 - losses[-1] / losses[0]):.1f}%)")
    print(f"   Initial accuracy: {accuracies[0]:.4f}")
    print(f"   Final accuracy: {accuracies[-1]:.4f}")

    # Pass/fail verdict
    loss_decreased = losses[-1] < losses[0] * 0.1  # Loss should drop to <10% of initial
    accuracy_high = accuracies[-1] > 0.95  # Should reach 95%+ accuracy

    if loss_decreased and accuracy_high:
        print(f"\n✅ OVERFIT PROBE PASSED")
        print(f"   ✓ Loss decreased significantly ({losses[0]:.4f} → {losses[-1]:.6f})")
        print(f"   ✓ Batch accuracy reached {accuracies[-1]:.1%}")
        print(f"   ✓ Gradient flow and optimization working correctly")
        return True, {"losses": losses, "accuracies": accuracies}
    else:
        print(f"\n❌ OVERFIT PROBE FAILED")
        if not loss_decreased:
            print(f"   ✗ Loss did not decrease (final={losses[-1]:.6f}, initial={losses[0]:.6f})")
        if not accuracy_high:
            print(f"   ✗ Accuracy did not reach 95% (final={accuracies[-1]:.1%})")
        return False, {"losses": losses, "accuracies": accuracies}


def main():
    """Main probe entrypoint."""
    set_seeds(42)

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / "Data" / "processed" / "new_model"
    split_file = processed_dir / "train_samples.txt"

    print("=" * 70)
    print("SignNet-V2 Overfit Probe")
    print("=" * 70)

    # Check paths
    if not processed_dir.exists():
        print(f"❌ Processed dir not found: {processed_dir}")
        return
    if not split_file.exists():
        print(f"❌ Split file not found: {split_file}")
        return

    # Load single batch of single class
    batch_data = load_single_batch_single_class(
        processed_dir=processed_dir,
        split_file=split_file,
        target_class_idx=6,  # "উত্তর"
        batch_size=16,
    )
    if batch_data is None:
        print(f"❌ Failed to load batch")
        return

    # Initialize model with hands enabled
    print(f"\n🏗️  Initializing SignNetV2 with hands enabled...")
    model = SignNetV2(
        num_classes=62,
        body_dim=99,
        hand_dim=63,
        face_dim=1404,
        d_model=256,
        num_encoder_layers=6,
        num_heads=8,
        d_ff=1024,
        dropout=0.2,
        max_seq_length=150,
        use_hands=True,
        use_face=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run probe
    passed, metrics = run_overfit_probe(
        model=model,
        batch_data=batch_data,
        num_steps=100,
        learning_rate=3e-4,
        device=device,
    )

    # Exit with status
    if passed:
        print(f"\n🎉 Model is ready for full training (gradient flow validated)")
        sys.exit(0)
    else:
        print(f"\n⚠️  Probe failed - investigate before full training")
        sys.exit(1)


if __name__ == "__main__":
    main()
