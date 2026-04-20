"""
Training Pipeline for SignNet-V2
=================================

Advanced training pipeline with:
- Mixed precision training
- Learning rate scheduling (OneCycleLR with warmup)
- Gradient clipping and accumulation
- Early stopping
- Model checkpointing
- WandB integration
- Mixup augmentation

Author: BDSL Recognition Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import wandb
import time


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    num_classes: int = 72
    body_dim: int = 99
    hand_dim: int = 63
    face_dim: int = 1404
    d_model: int = 128
    num_encoder_layers: int = 4
    num_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.2

    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    early_stopping_patience: int = 25
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True

    # Augmentation
    mixup_alpha: float = 0.2

    # Paths
    base_dir: str = "/home/raco/Repos/bangla-sign-language-recognition"
    checkpoint_dir: str = "Data/processed/new_model/checkpoints"

    # Logging
    log_interval: int = 10
    save_interval: int = 5


class Lookahead(Optimizer):
    """
    Lookahead Optimizer wrapper.

    Based on: "Lookahead Optimizer: k steps forward, 1 step back"
    (Zhang et al., 2019)
    """

    def __init__(self, optimizer: Optimizer, la_steps: int = 5, alpha: float = 0.5):
        """
        Initialize Lookahead optimizer.

        Args:
            optimizer: Inner optimizer
            la_steps: Number of lookahead steps
            alpha: Lookahead alpha parameter
        """
        self.optimizer = optimizer
        self.la_steps = la_steps
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self._la_step_count = 0

        # Initialize slow weights
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_state = self.optimizer.state[p]
                    param_state["slow_param"] = p.data.clone()

    def __getattr__(self, name):
        """Delegate attribute access to underlying optimizer."""
        return getattr(self.optimizer, name)

    def step(self, closure=None):
        """Perform optimization step."""
        loss = self.optimizer.step(closure)

        self._la_step_count += 1

        if self._la_step_count % self.la_steps == 0:
            self._lookahead()

        return loss

    def _lookahead(self):
        """Perform lookahead update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.optimizer.state[p]
                if "slow_param" not in param_state:
                    continue

                slow_p = param_state["slow_param"]
                # Update slow weights: slow_p = slow_p + alpha * (fast_p - slow_p)
                slow_p.add_(p.data - slow_p, alpha=self.alpha)
                # Update fast weights to match slow weights
                p.data.copy_(slow_p)

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


class Mixup:
    """Mixup augmentation for sign language data."""

    def __init__(self, alpha: float = 0.2):
        """
        Initialize mixup.

        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def mixup_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, float]:
        """
        Apply mixup to a batch.

        Args:
            x: Input features (body_pose, optional hands, optional face)
            y: Labels

        Returns:
            Tuple of mixed inputs, labels, and lambda value
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x[0].size(0)
        index = torch.randperm(batch_size, device=x[0].device, dtype=torch.long)

        mixed_x = []
        for xi in x:
            if xi is not None:
                mixed_x.append(lam * xi + (1 - lam) * xi[index])
            else:
                mixed_x.append(None)

        return mixed_x, index, lam

    def mixup_criterion(
        self,
        criterion: nn.Module,
        logits: torch.Tensor,
        y: torch.Tensor,
        index: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute mixup loss.

        Args:
            criterion: Loss function
            logits: Model predictions
            y: Original labels
            index: Permuted indices
            lam: Mixup lambda

        Returns:
            Mixup loss
        """
        return lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[index])


class SignNetTrainer:
    """Trainer for SignNet-V2 model."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        checkpoint_dir: Path,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model: SignNet-V2 model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

        # Mixed precision scaler (use new API)
        self.scaler = (
            torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
            if config.use_amp
            else None
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Lookahead optimizer
        self.optimizer = Lookahead(self.optimizer, la_steps=5, alpha=0.5)

        # Learning rate scheduler
        total_steps = (
            len(train_loader) * config.epochs // config.gradient_accumulation_steps
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate * 3,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader) // config.gradient_accumulation_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=3.0,
            final_div_factor=10.0,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # Mixup
        self.mixup = Mixup(alpha=config.mixup_alpha)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.no_improve_count = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        accumulation_steps = self.config.gradient_accumulation_steps

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch + 1}", leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            body_pose = batch["body_pose"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].squeeze().to(self.device)

            left_hand = batch.get("left_hand")
            right_hand = batch.get("right_hand")
            face = batch.get("face")

            if left_hand is not None:
                left_hand = left_hand.to(self.device)
            if right_hand is not None:
                right_hand = right_hand.to(self.device)
            if face is not None:
                face = face.to(self.device)

            # Apply mixup with probability
            apply_mixup = self.config.mixup_alpha > 0 and np.random.random() < 0.5

            if apply_mixup:
                x_tuple = (body_pose, left_hand, right_hand, face)
                x_mixed, index, lam = self.mixup.mixup_data(x_tuple, labels)
                body_pose, left_hand, right_hand, face = x_mixed

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        body_pose, left_hand, right_hand, face, attention_mask
                    )

                    if apply_mixup:
                        loss = self.mixup.mixup_criterion(
                            self.criterion, logits, labels, index, lam
                        )
                    else:
                        loss = self.criterion(logits, labels)

                    loss = loss / accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

            else:
                logits = self.model(
                    body_pose, left_hand, right_hand, face, attention_mask
                )

                if apply_mixup:
                    loss = self.mixup.mixup_criterion(
                        self.criterion, logits, labels, index, lam
                    )
                else:
                    loss = self.criterion(logits, labels)

                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()

            # Metrics
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)

                if apply_mixup:
                    # Mixup metrics are approximate
                    correct = (predictions == labels).sum().item() * lam + (
                        predictions == labels[index]
                    ).sum().item() * (1 - lam)
                else:
                    correct = (predictions == labels).sum().item()

                total_loss += loss.item() * accumulation_steps * len(labels)
                total_correct += correct
                total_samples += len(labels)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * accumulation_steps:.4f}",
                    "acc": f"{correct / len(labels):.4f}",
                }
            )
            
            # Log batch-level metrics to WandB every N batches
            if (batch_idx + 1) % 5 == 0:
                wandb.log({
                    "train/batch_loss": loss.item() * accumulation_steps,
                    "train/batch_accuracy": correct / len(labels),
                    "train/batch": batch_idx + 1,
                })

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        all_predictions = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            # Unpack batch
            body_pose = batch["body_pose"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].squeeze().to(self.device)

            left_hand = batch.get("left_hand")
            right_hand = batch.get("right_hand")
            face = batch.get("face")

            if left_hand is not None:
                left_hand = left_hand.to(self.device)
            if right_hand is not None:
                right_hand = right_hand.to(self.device)
            if face is not None:
                face = face.to(self.device)

            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        body_pose, left_hand, right_hand, face, attention_mask
                    )
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(
                    body_pose, left_hand, right_hand, face, attention_mask
                )
                loss = self.criterion(logits, labels)

            # Metrics
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item() * len(labels)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def save_checkpoint(self, is_best: bool = False, is_intermediate: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_accuracy": self.best_val_acc,
            "val_loss": self.history["val_loss"][-1]
            if self.history["val_loss"]
            else float("inf"),
            "config": self.config.__dict__,
            "history": self.history,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            print(f"   ✨ Best model saved! Val Acc: {self.best_val_acc:.4f}")

        # Save intermediate checkpoint
        if is_intermediate:
            torch.save(
                checkpoint,
                self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth",
            )

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pth")

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load checkpoint and return starting epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint.get("val_accuracy", 0.0)
        self.history = checkpoint.get("history", self.history)

        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"   Best val accuracy: {self.best_val_acc:.4f}")

        return self.current_epoch

    def train(self, start_epoch: int = 0, max_epochs: Optional[int] = None) -> Dict:
        """
        Complete training loop.

        Args:
            start_epoch: Starting epoch
            max_epochs: Maximum epochs (uses config if None)

        Returns:
            Training history dictionary
        """
        if max_epochs is None:
            max_epochs = self.config.epochs

        print(f"\n🚀 Starting training for {max_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Mixed precision: {self.config.use_amp}")
        print(f"   Checkpoint directory: {self.checkpoint_dir}")

        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Get learning rate
            current_lr = self.scheduler.get_last_lr()[0]

            # Log to history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            # Log to WandB in real-time
            try:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "learning_rate": current_lr,
                    }
                )
            except Exception:
                pass  # W&B logging not available

            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1}/{max_epochs} Summary:")
            print(
                f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)"
            )
            print(
                f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc * 100:.2f}%)"
            )
            print(f"   LR: {current_lr:.6f}")

            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            # Save checkpoint
            self.save_checkpoint(
                is_best=is_best,
                is_intermediate=(epoch + 1) % self.config.save_interval == 0,
            )

            # Early stopping
            if self.no_improve_count >= self.config.early_stopping_patience:
                print(
                    f"\n⏹️  Early stopping triggered after {self.config.early_stopping_patience} epochs without improvement"
                )
                print(
                    f"   Best validation accuracy: {self.best_val_acc:.4f} ({self.best_val_acc * 100:.2f}%)"
                )
                break

        print(f"\n✅ Training complete!")
        print(f"   Total epochs: {self.current_epoch + 1}")
        print(
            f"   Best validation accuracy: {self.best_val_acc:.4f} ({self.best_val_acc * 100:.2f}%)"
        )

        return self.history


def setup_training(
    config: TrainingConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> SignNetTrainer:
    """
    Setup training components.

    Args:
        config: Training configuration
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on

    Returns:
        Configured SignNetTrainer
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = SignNetTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    return trainer


if __name__ == "__main__":
    # Test training setup
    from src.models.signet_v2 import SignNetV2
    from src.data.preprocessing import DataConfig, create_data_loaders

    print("✅ Training components imported")
    print("\nTesting training configuration...")

    config = TrainingConfig()
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Mixed precision: {config.use_amp}")
    print(f"   Mixup alpha: {config.mixup_alpha}")

    print("\n✅ Training pipeline setup complete")
