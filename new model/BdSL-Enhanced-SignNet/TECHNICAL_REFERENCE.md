# W&B Integration - Technical Reference

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Training Pipeline with Real-Time W&B Logging    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │   train_signet_v2_optimized.py   │
        │  - Load API key from .env        │
        │  - Initialize W&B with login()   │
        │  - Create project & run          │
        └──────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │    src/training/trainer.py       │
        │  - Train loop with batches       │
        │  - Log batch metrics every 5     │
        │  - Log epoch metrics             │
        │  - Save checkpoints              │
        └──────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │  src/evaluation/evaluator.py     │
        │  - Evaluate on test set          │
        │  - Generate visualizations       │
        │  - Log confusion matrices        │
        │  - Log accuracy plots            │
        └──────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │   W&B Cloud (wandb.ai)           │
        │  - Real-time dashboards          │
        │  - Metric tracking               │
        │  - Visualization storage         │
        │  - Run comparison tools          │
        └──────────────────────────────────┘
```

## Code Integration Points

### 1. API Key Authentication (`train_signet_v2_optimized.py`)

```python
# Line 27: Import dotenv for .env support
from dotenv import load_dotenv

# Lines 187-196: Load API key and authenticate
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("✅ W&B authenticated with API key")

# Lines 198-207: Initialize W&B run
wandb.init(
    project=args.wandb_project,
    entity=None,
    name=f"SignNet-V2_{len(train_samples)}samples_{args.epochs}epochs",
    config={...}
)
```

### 2. Model Monitoring (`train_signet_v2_optimized.py`)

```python
# Lines 300-302: Watch model parameters
wandb.watch(model, log_freq=100)
wandb.log({"model/total_parameters": params['total']})
```

### 3. Batch-Level Logging (`src/training/trainer.py`)

```python
# Lines 405-415: Log metrics every 5 batches
if (batch_idx + 1) % 5 == 0:
    wandb.log({
        "train/batch_loss": loss.item() * accumulation_steps,
        "train/batch_accuracy": correct / len(labels),
        "train/batch": batch_idx + 1,
    })
```

### 4. Epoch-Level Logging (`src/training/trainer.py`)

```python
# Lines 564-575: Log after each epoch
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
```

### 5. Visualization Logging (`src/evaluation/evaluator.py`)

```python
# Lines 450-461: Log confusion matrix
wandb.log(
    {
        "eval/confusion_matrix": wandb.Image(
            str(self.output_dir / "confusion_matrix.png")
        ),
        "eval/confusion_matrix_normalized": wandb.Image(
            str(self.output_dir / "confusion_matrix_normalized.png")
        ),
    }
)
```

### 6. Final Metrics (`train_signet_v2_optimized.py`)

```python
# Lines 381-391: Log test metrics
wandb.log(
    {
        "test/accuracy": results["test_accuracy"],
        "test/precision": results["test_precision"],
        "test/recall": results["test_recall"],
        "test/f1_score": results["test_f1"],
        "test/top5_accuracy": results["top_5_accuracy"],
    }
)
wandb.finish()
```

## Logging Frequency

| Metric Type | Frequency | Purpose |
|-------------|-----------|---------|
| Batch Loss/Accuracy | Every 5 batches | Fine-grained training progress |
| Epoch Summary | After each epoch | Overall epoch performance |
| Learning Rate | After each epoch | LR schedule tracking |
| Test Metrics | After training ends | Final model evaluation |
| Visualizations | After training ends | Analysis and debugging |

## Data Flow

```
Training Data (.npz files)
        │
        ▼
   Data Loader
        │
        ▼
   ┌─────────────────┐
   │  Training Loop  │────► Log batch metrics (every 5 batches)
   └─────────────────┘           │
        │                         ▼
        ├─────────────────────► W&B Cloud (batch level)
        │
        ▼
   ┌──────────────┐
   │  Validation  │────► Log epoch metrics
   └──────────────┘           │
        │                     ▼
        ├──────────────────► W&B Cloud (epoch level)
        │
        ▼
   ┌──────────────┐
   │ Evaluation   │────► Generate visualizations
   └──────────────┘           │
        │                     ▼
        ├──────────────────► Log plots & metrics
        │                     │
        │                     ▼
        └──────────────────► W&B Cloud (final)
```

## W&B Logging Schema

### Metrics Structure
```python
{
    "epoch": int,
    "train/loss": float,
    "train/accuracy": float,
    "train/batch_loss": float,
    "train/batch_accuracy": float,
    "train/batch": int,
    "val/loss": float,
    "val/accuracy": float,
    "test/accuracy": float,
    "test/precision": float,
    "test/recall": float,
    "test/f1_score": float,
    "test/top5_accuracy": float,
    "learning_rate": float,
}
```

### Images Structure
```python
{
    "eval/confusion_matrix": wandb.Image,
    "eval/confusion_matrix_normalized": wandb.Image,
    "eval/per_signer_accuracy": wandb.Image,
    "eval/per_class_accuracy": wandb.Image,
    "eval/top_k_accuracy": wandb.Image,
    "eval/model_comparison": wandb.Image,
}
```

## Configuration Flow

```
.env file (or environment variable)
    │
    ▼
load_dotenv() reads WANDB_API_KEY
    │
    ▼
wandb.login(key=api_key)
    │
    ▼
wandb.init(project=..., config={...})
    │
    ▼
W&B Session Created
    │
    ▼
wandb.log({...}) during training
    │
    ▼
Real-time dashboard at wandb.ai
```

## API Calls

### Initialization
```python
wandb.login(key=api_key)              # Authenticate with W&B
wandb.init(project=..., config=...)   # Create new run
```

### Logging
```python
wandb.log({...})                      # Log metrics/images
wandb.watch(model, log_freq=100)      # Watch model gradients
```

### Finalization
```python
wandb.finish()                        # End run gracefully
```

## Error Handling

**Before Integration:**
```python
try:
    wandb.init(...)
except Exception as e:
    print("WandB failed, continuing without it")
```

**After Integration:**
```python
wandb.init(...)  # Will fail if key is invalid
wandb.log(...)   # Will fail if project doesn't exist
```

**Recommendation:** Set up API key correctly to avoid errors.

## Performance Impact

- **Batch Logging:** ~1-2ms overhead per log call
- **Epoch Logging:** ~10-50ms overhead
- **Network:** Async uploads (doesn't block training)
- **Storage:** ~1-2MB per run (metrics + images)

## Scalability

- ✅ Handles 100+ epochs
- ✅ Supports batch-level metrics
- ✅ Efficient image compression
- ✅ Cloud-based storage (unlimited)
- ✅ Real-time syncing

## Debugging Tips

1. **Check API Key:**
   ```bash
   echo $WANDB_API_KEY  # Linux/Mac
   echo %WANDB_API_KEY% # Windows
   ```

2. **Enable Debug Mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Offline Mode:**
   ```bash
   export WANDB_MODE=offline
   ```

4. **Network Issues:**
   ```python
   wandb.init(..., settings=wandb.Settings(init_timeout=60))
   ```

## Best Practices

1. ✅ Store API key in `.env`, not in code
2. ✅ Use meaningful project names
3. ✅ Add descriptive run names
4. ✅ Include hyperparameters in config
5. ✅ Log visualizations for debugging
6. ✅ Compare multiple runs
7. ✅ Archive old runs for reference

## Example W&B URL Structure

```
https://wandb.ai/
    ├── [username]
    │   ├── bengla-sign-language-recognition    (project)
    │   │   ├── run-1: SignNet-V2_433samples_3epochs
    │   │   ├── run-2: SignNet-V2_433samples_5epochs
    │   │   └── run-3: SignNet-V2_433samples_10epochs (current)
```

## Integration Checklist

- ✅ W&B module installed
- ✅ python-dotenv installed
- ✅ API key in `.env` or environment
- ✅ Authentication configured
- ✅ Batch logging implemented
- ✅ Epoch logging implemented
- ✅ Visualization logging implemented
- ✅ Test metrics logging implemented
- ✅ Error handling removed (uses proper auth)
- ✅ Documentation created

---

**Ready for production W&B logging! 🚀📊**
