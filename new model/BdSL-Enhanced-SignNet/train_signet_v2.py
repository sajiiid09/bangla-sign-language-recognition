"""
Main Training Script for SignNet-V2
====================================

Complete training pipeline for Bengali Sign Language recognition.
This script orchestrates:
1. Data loading and preprocessing
2. Model initialization
3. Training with advanced techniques
4. Evaluation and comparison with baseline

Usage:
    python train_signet_v2.py --config config.yaml

Author: BDSL Recognition Team
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import numpy as np
import random
import wandb
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.signet_v2 import SignNetV2, count_parameters
from src.data.preprocessing import DataConfig, create_data_loaders
from src.training.trainer import TrainingConfig, SignNetTrainer
from src.evaluation.evaluator import (
    EvaluationConfig,
    SignNetEvaluator,
    ComparativeAnalyzer,
)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sample_list(file_path: str) -> List[str]:
    """Load sample paths from text file."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_word_mapping(samples: List[Dict]) -> tuple:
    """Create word to label and label to word mappings."""
    all_words = sorted(set([s["word"] for s in samples]))
    word_to_label = {word: idx for idx, word in enumerate(all_words)}
    label_to_word = {idx: word for idx, word in enumerate(all_words)}
    return word_to_label, label_to_word


def parse_metadata(video_path: str) -> Dict:
    """Parse metadata from video filename."""
    path = Path(video_path)
    filename = path.stem
    parts = filename.split("__")

    if len(parts) != 5:
        return None

    return {
        "word": parts[0],
        "signer": parts[1],
        "session": parts[2],
        "repetition": parts[3],
        "grammar": parts[4],
        "full_path": video_path,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train SignNet-V2 for BdSL recognition"
    )

    # Configuration
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/raco/Repos/bangla-sign-language-recognition",
    )
    parser.add_argument("--processed_dir", type=str, default="Data/processed/new_model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument(
        "--wandb_project", type=str, default="bangla-sign-language-recognition"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Set random seeds
    set_seeds(args.seed)

    # Setup paths
    base_path = Path(args.base_dir)
    processed_dir = base_path / args.processed_dir
    checkpoint_dir = processed_dir / "checkpoints" / "signet_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("🚀 SignNet-V2 Training Pipeline")
    print(f"{'=' * 70}")
    print(f"   Base directory: {base_path}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Mixed precision: {args.use_amp}")

    # Load sample lists
    train_samples = load_sample_list(processed_dir / "train_samples.txt")
    val_samples = load_sample_list(processed_dir / "val_samples.txt")
    test_samples = load_sample_list(processed_dir / "test_samples.txt")

    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(train_samples)} samples")
    print(f"   Val: {len(val_samples)} samples")
    print(f"   Test: {len(test_samples)} samples")

    # Parse metadata and create word mapping
    train_metadata = [parse_metadata(s) for s in train_samples]
    val_metadata = [parse_metadata(s) for s in val_samples]
    test_metadata = [parse_metadata(s) for s in test_samples]

    train_metadata = [m for m in train_metadata if m is not None]
    val_metadata = [m for m in val_metadata if m is not None]
    test_metadata = [m for m in test_metadata if m is not None]

    all_metadata = train_metadata + val_metadata + test_metadata
    word_to_label, label_to_word = create_word_mapping(all_metadata)
    num_classes = len(word_to_label)

    print(f"\n📚 Classes: {num_classes} unique Bengali words")

    # Save label mapping
    label_mapping = {
        "word_to_label": word_to_label,
        "label_to_word": {str(k): v for k, v in label_to_word.items()},
    }
    with open(checkpoint_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=None,
        name=f"SignNet-V2_{len(train_samples)}samples_{args.epochs}epochs",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_classes": num_classes,
            "seed": args.seed,
            "use_amp": args.use_amp,
        },
    )

    # Create data config
    data_config = DataConfig(
        base_dir=str(base_path),
        processed_dir=args.processed_dir,
        normalized_dir=str(processed_dir / "normalized"),
        checkpoint_dir=str(checkpoint_dir),
        max_seq_length=150,
        augmentation=True,
    )

    # Create data loaders (Note: dataset only has body_pose)
    train_loader, val_loader, test_loader = create_data_loaders(
        config=data_config,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        word_to_label=word_to_label,
        batch_size=args.batch_size,
        num_workers=2,
        use_hands=False,
        use_face=False,
    )

    print(f"\n📦 Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Create training config
    training_config = TrainingConfig(
        num_classes=num_classes,
        body_dim=data_config.body_dim,
        hand_dim=data_config.hand_dim,
        face_dim=data_config.face_dim,
        d_model=128,
        num_encoder_layers=4,
        num_heads=8,
        d_ff=512,
        dropout=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.05,
        label_smoothing=0.1,
        early_stopping_patience=25,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=1,
        use_amp=args.use_amp,
        mixup_alpha=0.2,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Initialize model (Note: dataset only has body_pose)
    model = SignNetV2(
        num_classes=num_classes,
        body_dim=data_config.body_dim,
        hand_dim=data_config.hand_dim,
        face_dim=data_config.face_dim,
        d_model=training_config.d_model,
        num_encoder_layers=training_config.num_encoder_layers,
        num_heads=training_config.num_heads,
        d_ff=training_config.d_ff,
        dropout=training_config.dropout,
        max_seq_length=data_config.max_seq_length,
        use_face=False,  # Dataset only has body_pose
        use_hands=False,  # Dataset only has body_pose
    )

    # Count parameters
    params = count_parameters(model)
    print(f"\n🧠 Model: SignNet-V2")
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    print(f"   Model size: {params['total'] * 4 / 1024**2:.2f} MB")

    # Watch model with WandB
    wandb.watch(model, log_freq=100)

    # Setup trainer
    trainer = SignNetTrainer(
        config=training_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        start_epoch = trainer.load_checkpoint(Path(args.resume))

    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    test_input = torch.randn(2, data_config.max_seq_length, data_config.body_dim).to(
        device
    )
    test_mask = torch.ones(2, data_config.max_seq_length).to(device)

    with torch.no_grad():
        if device.type == "cuda" and args.use_amp:
            with torch.cuda.amp.autocast():
                test_output = model(
                    test_input,  # body_pose
                    None,  # left_hand
                    None,  # right_hand
                    None,  # face
                    test_mask,  # attention_mask
                )
        else:
            test_output = model(
                test_input,  # body_pose
                None,  # left_hand
                None,  # right_hand
                None,  # face
                test_mask,  # attention_mask
            )

    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   ✅ Forward pass successful!")

    # Train model
    history = trainer.train(start_epoch=start_epoch)

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate best model
    print(f"\n📊 Evaluating best model...")

    eval_config = EvaluationConfig(
        checkpoint_dir=str(checkpoint_dir), num_classes=num_classes
    )

    evaluator = SignNetEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        label_to_word=label_to_word,
        config=eval_config,
    )

    results = evaluator.evaluate()
    evaluator.print_results(results)
    evaluator.save_results(results)
    evaluator.generate_visualizations(results)

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")

    # Log final metrics to WandB
    wandb.log(
        {
            "test/accuracy": results["test_accuracy"],
            "test/precision": results["test_precision"],
            "test/recall": results["test_recall"],
            "test/f1_score": results["test_f1"],
            "test/top5_accuracy": results["top_5_accuracy"],
        }
    )

    # Finish WandB
    wandb.finish()

    # Print final summary
    print(f"\n{'=' * 70}")
    print("🎉 TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n📁 Output Directory: {checkpoint_dir}")
    print(f"\n📊 Test Results:")
    print(
        f"   Top-1 Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy'] * 100:.2f}%)"
    )
    print(
        f"   Top-5 Accuracy: {results['top_5_accuracy']:.4f} ({results['top_5_accuracy'] * 100:.2f}%)"
    )
    print(f"   Precision: {results['test_precision']:.4f}")
    print(f"   Recall: {results['test_recall']:.4f}")
    print(f"   F1-Score: {results['test_f1']:.4f}")
    print(f"\n🧠 Model Information:")
    print(f"   Architecture: SignNet-V2")
    print(f"   Total Parameters: {params['total']:,}")
    print(f"   Model Size: {params['total'] * 4 / 1024**2:.2f} MB")
    print(
        f"   Best Val Accuracy: {trainer.best_val_acc:.4f} ({trainer.best_val_acc * 100:.2f}%)"
    )
    print(f"\n🌐 WandB:")
    print(f"   Project: {args.wandb_project}")
    print(f"   All metrics and artifacts logged")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
