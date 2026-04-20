# W&B Real-Time Logging Setup Guide

## Overview
This guide explains how to set up W&B (Weights & Biases) for real-time monitoring of your training pipeline.

## Prerequisites
- W&B account at https://wandb.ai
- W&B API key (found in your account settings)
- `wandb` Python package installed

## Installation

```bash
pip install wandb python-dotenv
```

## Setup Steps

### Option 1: Using Environment Variables (Recommended)

#### Windows PowerShell:
```powershell
# Set environment variable for current session
$env:WANDB_API_KEY = "your_api_key_here"

# To make it permanent, add to your system environment variables
[System.Environment]::SetEnvironmentVariable("WANDB_API_KEY", "your_api_key_here", [System.EnvironmentVariableTarget]::User)
```

#### Windows Command Prompt:
```cmd
set WANDB_API_KEY=your_api_key_here
```

#### Linux/Mac:
```bash
export WANDB_API_KEY="your_api_key_here"
```

### Option 2: Using .env File

1. Create a `.env` file in the project root:
```
WANDB_API_KEY=your_api_key_here
```

2. The training script will automatically load this file

### Option 3: W&B CLI

```bash
wandb login
# Enter your API key when prompted
```

## Finding Your API Key

1. Go to https://wandb.ai/settings/profile
2. Scroll down to "API keys"
3. Copy your API key

## Real-Time Logging Features

The training pipeline now logs the following metrics in real-time:

### Batch-Level Metrics (Every 5 batches)
- `train/batch_loss`: Training loss at batch level
- `train/batch_accuracy`: Training accuracy at batch level
- `train/batch`: Batch index

### Epoch-Level Metrics
- `train/loss`: Average training loss per epoch
- `train/accuracy`: Training accuracy per epoch
- `val/loss`: Validation loss per epoch
- `val/accuracy`: Validation accuracy per epoch
- `learning_rate`: Current learning rate

### Evaluation Metrics
- `test/accuracy`: Test accuracy
- `test/precision`: Test precision
- `test/recall`: Test recall
- `test/f1_score`: Test F1 score
- `test/top5_accuracy`: Top-5 accuracy

### Visualizations
- `eval/confusion_matrix`: Confusion matrix
- `eval/confusion_matrix_normalized`: Normalized confusion matrix
- `eval/per_signer_accuracy`: Per-signer accuracy plot
- `eval/per_class_accuracy`: Per-class accuracy plot
- `eval/top_k_accuracy`: Top-K accuracy plot
- `eval/model_comparison`: Model comparison visualization

## Running Training with W&B

### Basic Training
```bash
python train_signet_v2_optimized.py --base_dir "." --processed_dir "../../Data/processed/new_model" --epochs 3 --batch_size 16 --learning_rate 3e-4
```

### With Custom Project Name
```bash
python train_signet_v2_optimized.py --base_dir "." --processed_dir "../../Data/processed/new_model" --epochs 3 --batch_size 16 --learning_rate 3e-4 --wandb_project "my-custom-project"
```

## Viewing Results

1. Go to https://wandb.ai
2. Navigate to your project
3. Click on the latest run to see:
   - Real-time training curves
   - Batch-level metrics
   - Epoch summaries
   - Evaluation visualizations
   - System metrics (GPU, memory, CPU)

## Troubleshooting

### "ImportError: No module named 'wandb'"
```bash
pip install wandb
```

### "API key not found"
Make sure to set `WANDB_API_KEY` environment variable or create `.env` file with your API key

### "Permission denied" errors
Ensure your API key is valid at https://wandb.ai/settings/profile

### Offline Mode
If you don't have internet, use:
```bash
export WANDB_MODE=offline
```
Data will be saved locally and synced when connection is restored.

## Advanced Configuration

### Disable W&B Temporarily
```bash
export WANDB_DISABLED=true
```

### Change Entity (Team)
Edit `train_signet_v2_optimized.py`:
```python
wandb.init(
    project=args.wandb_project,
    entity="your-team-name",  # Change this
    name=f"SignNet-V2_{len(train_samples)}samples_{args.epochs}epochs",
    ...
)
```

### Change Logging Frequency
In `src/training/trainer.py`, modify the batch logging condition:
```python
# Log every 10 batches instead of 5
if (batch_idx + 1) % 10 == 0:
    wandb.log({...})
```

## Example W&B Dashboard Features

- **Line Charts**: Training/validation loss and accuracy over time
- **Confusion Matrix**: See which classes are misclassified
- **System Metrics**: Monitor GPU, memory, and CPU usage
- **Hyperparameter Importance**: Automatic correlation analysis
- **Model Comparison**: Compare different runs side-by-side

## Next Steps

1. Set your API key using one of the methods above
2. Run training: `python train_signet_v2_optimized.py --base_dir "." --processed_dir "../../Data/processed/new_model" --epochs 3`
3. Watch real-time updates at https://wandb.ai
4. Compare runs and optimize hyperparameters

## Support

- W&B Documentation: https://docs.wandb.ai
- W&B Community: https://community.wandb.ai
- GitHub Issues: https://github.com/wandb/wandb/issues
