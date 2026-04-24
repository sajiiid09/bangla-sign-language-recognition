# Training Operations

## Quick Start

1. Configure environment and dependencies.
2. Set W&B credentials.
3. Run train/evaluate scripts with explicit run names and output paths.
4. Log, compare, and archive artifacts.

## W&B Configuration

### Preferred Credential Flow

- Store API key in `.env` or environment variable.
- Validate with a dedicated check script before long runs.

Example environment variable names used in legacy docs:

- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY`

### Authentication Reliability Notes

A prior incident showed corrupted `wandb` installs can remove expected methods.
Recovery pattern:

1. Uninstall broken `wandb` package.
2. Reinstall latest compatible version.
3. Re-run authentication checks before training.

## Logging Schema (Canonical)

### Batch Level

- `train/batch_loss`
- `train/batch_accuracy`
- `train/batch`

### Epoch Level

- `train/loss`
- `train/accuracy`
- `val/loss`
- `val/accuracy`
- `learning_rate`

### Final Test

- `test/accuracy`
- `test/precision`
- `test/recall`
- `test/f1_score`
- `test/top5_accuracy`

### Visual Artifacts

- Confusion matrix (raw and normalized)
- Per-class and per-signer plots
- Top-k accuracy plots
- Model comparison visuals

## Training Command Patterns

### Primary Track

Run SignNet-V2 scripts from `new model/BdSL-Enhanced-SignNet` with explicit:

- input split/sample files
- checkpoint output directory
- run name and project metadata
- device and mixed precision flags

### Comparison Track

Run SPOTER scripts from `comparison model/BDSLW_SPOTER` with matching split assumptions whenever feasible.

## Reproducibility Checklist

Before starting a long run:

1. Confirm dataset snapshot and split files.
2. Confirm model config and optimizer schedule.
3. Confirm logging target (W&B project/entity/run name).
4. Confirm checkpoint directory and resume policy.
5. Confirm random seed strategy.

After run completion:

1. Save model and optimizer checkpoints.
2. Export run summary and key charts.
3. Record evaluation metrics by class and signer.
4. Link run artifacts in comparison reports.

## Troubleshooting Matrix

### Missing W&B Auth

- Check `.env` presence and key naming.
- Validate API key from W&B account settings.
- Retry login in a clean environment.

### Logging Stops Mid-Run

- Verify network and W&B mode.
- Use offline mode if needed and sync later.
- Check package integrity and script-side error handling.

### Low Accuracy with Healthy Training Loop

- Verify real landmark extraction.
- Inspect split leakage/imbalance.
- Inspect class and expression balance.
- Compare against prior checkpoints before hyperparameter changes.

## Operational Policy

- Prefer one canonical run recipe per model track.
- Treat ad-hoc experimental recipes as temporary unless promoted into this file.
- Do not claim cross-model improvements without artifact-backed, split-aligned evaluation.
