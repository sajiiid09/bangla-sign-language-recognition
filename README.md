# Bangla Sign Language Recognition

This project implements Bangla Sign Language (BdSL) recognition systems using computer vision and machine learning approaches.

## Project Structure

- `new model/Emotion-Integrated-Sign-Interpretation-model/` - Multimodal fusion model with emotion recognition

## Getting Started

### Prerequisites

- Python 3.12+
- UV package manager
- CUDA-capable GPU (optional but recommended)
- WandB account (for experiment tracking)
- WandB API key (from https://wandb.ai/settings)

### Installation

```bash
uv sync
source .venv/bin/activate
```

### Configuration

1. Get your WandB entity name from your profile
2. Add WandB configuration to `.env`:

```bash
# Copy example file
cp .env.example .env

# Edit .env with your credentials
vim .env
```

Configure with:
- `WANDB_PROJECT`: Project name (default: `BB3lAowfaCGkIlsby`)
- `WANDB_ENTITY`: Your WandB entity name (default: `wandb_v1`)
- `WANDB_API_KEY`: Your API key from WandB settings

## Usage

### WandB Tracking

This project uses Weights & Biases (WandB) to track training runs and compare model performance.

## Project Status

### ✅ Completed

**Data Processing:**
- Combined manifest from 3 video directories: 833 total samples
- Signers: S01 (281), S02 (337), S05 (215)
- Unique words: 77
- Generated 833 landmark files (organized by word)
- Created train/val/test splits (70/15/15):
  - Train: S01 (281 samples, 57 words)
  - Val: S02 (337 samples, 60 words)
  - Test: S05 (215 samples, 60 words)

**Model Training:**
- Trained fusion model on GPU for 10 epochs
- Checkpoint: `fusion_model.pt` (21MB)
- Final val_acc: 15% (expected with random test data)

**Code Updates:**
- Fixed dataset to load npz from correct path structure
- Added `build_vocab_from_samples()` for dynamic vocabulary
- Removed RAG/LLM/AI tutor functionality
- Simplified demo to pure sign language recognition
- Updated project with benchmark documentation

**RAG/LLM Removal:**
- ✅ Removed entire `brain/` directory (RAG, LLM, AI tutor)
- ✅ Updated demo to remove brain/ imports
- ✅ Cleaned up `.env.example` (no LLM configs)
- ✅ Updated `requirements.txt` (removed google-genai)

### 📂 Dataset Structure

```
Data/
├── raw_inkiad/          # 281 source videos
├── raw_santonu/         # 337 source videos  
├── raw_sumaiya/        # 215 source videos
└── processed/
    ├── manifest.csv         # 833 samples combined
    ├── landmarks/          # 833 .npz files
    │   └── word/filename.npz
    ├── splits/            # train/val/test JSON files
    └── benchmarks/         # Metrics documentation
```

### ⚠️  Known Issues

**MediaPipe API Version:**
- **Issue:** Code uses old API (`mp.solutions.holistic`)
- **Current:** mediapipe 0.10.32 uses new API (`mp.tasks.vision`)
- **Impact:** Cannot extract real landmarks from videos
- **Workaround:** Currently using random test landmark data
- **Fix Required:** Extract real landmarks using MediaPipe for production

**Training Data Quality:**
- **Issue:** Landmark files contain random data instead of actual features
- **Impact:** Low validation accuracy (~15%)
- **Fix:** Run landmark extraction from videos before production training

## Usage

### Tracking Experiments with WandB

This project uses Weights & Biases (WandB) for comprehensive experiment tracking.

**Setup:**
1. Create WandB account at https://wandb.ai
2. Get your entity name and API key
3. Configure `.env` file with credentials

**WandB Configuration:**
```bash
# Copy example file
cp .env.example .env

# Edit with your credentials
vim .env

# Set these values:
WANDB_PROJECT=BB3lAowfaCGkIlsby
WANDB_ENTITY=your_wandb_username
WANDB_API_KEY=your_wandb_api_key_here
```

### Data Processing

**Generate manifest and landmarks:**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

# Note: Currently using random test data
# For production, extract real landmarks from videos
```

**Create splits:**
Splits are auto-generated at `Data/processed/splits/`

### Training

**Train model with current landmarks (test data):**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

PYTHONPATH=. python3 train/train_fusion.py \
  ../../Data/processed/manifest.csv \
  ../../Data/processed/landmarks \
  --epochs 40 \
  --batch-size 64 \
  --lr 3e-4 \
  --device cuda \
  --train-signers S01 S02 S05 \
  --val-signers S02 \
  --test-signers S05
```

**Note:** For production training, first extract real landmarks from videos.

### Training with WandB

**Train SPOTER v1 (Baseline):**
```bash
cd "comparison model/BDSLW_SPOTER"
source ../../.venv/bin/activate

python train.py \
  train_data.npz val_data.npz \
  --epochs 40 \
  --batch-size 64 \
  --lr 3e-4 \
  --device cuda \
  --run-name spoter_v1 \
  --wandb-project BB3lAowfaCGkIlsby \
  --wandb-entity wandb_v1
```

**Train Fusion v2 (Multimodal):**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

PYTHONPATH=. python3 train/train_fusion.py \
  ../../Data/processed/manifest.csv \
  ../../Data/processed/landmarks \
  --epochs 40 \
  --batch-size 64 \
  --lr 3e-4 \
  --device cuda \
  --train-signers S01 S02 S05 \
  --val-signers S02 \
  --test-signers S05 \
  --run-name fusion_v2 \
  --wandb-project BB3lAowfaCGkIlsby \
  --wandb-entity wandb_v1
```

**Training without WandB (for testing):**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

PYTHONPATH=. python3 train/train_fusion.py \
  ../../Data/processed/manifest.csv \
  ../../Data/processed/landmarks \
  --epochs 40 \
  --batch-size 64 \
  --lr 3e-4 \
  --device cuda \
  --train-signers S01 S02 S05 \
  --val-signers S02 \
  --test-signers S05 \
  --no-wandb
```

**WandB Features:**
- Experiments named: `spoter_v1` and `fusion_v2`
- Tracks: loss, accuracy, learning rate
- Confusion matrices logged (as images and CSV artifacts)
- Model checkpoints saved (all epochs + best + final)
- Both sign and grammar tasks tracked (for fusion model)
- View results at: https://wandb.ai/wandb_v1/BB3lAowfaCGkIlsby

### Run Demo

**Real-time recognition with webcam:**

**Real-time recognition with webcam:**
```bash
cd "new model/Emotion-Integrated-Sign-Interpretation-model"
source ../../.venv/bin/activate

PYTHONPATH=. python3 demo/realtime_demo.py \
  fusion_model.pt \
  --manifest ../../Data/processed/manifest.csv \
  --device cuda \
  --buffer 48 \
  --stable-frames 10 \
  --min-conf 0.60 \
  --font-path demo/kalpurush.ttf
```

**Demo Controls:**
- Press `c` - Clear sentence buffer
- Press `q` or ESC - Quit demo

**Demo Displays:**
- Predicted word from sign language
- Grammar tag (neutral/question/negation/happy/sad)
- Confidence score
- Current sentence buffer
- FPS counter

### Benchmark Evaluation

See `Data/benchmarks/README.md` for benchmark folder structure and metrics to track.

## Files Modified

1. `pyproject.toml` - Updated mediapipe version constraint
2. `train/dataset.py` - Fixed npz loading path and vocabulary building
3. `train/vocab.py` - Added `build_vocab_from_samples()` function
4. `Data/processed/manifest.csv` - Created with all 833 samples
5. `Data/processed/landmarks/` - Created 833 landmark files (test data)
6. `Data/processed/splits/` - Created train/val/test split files
7. `Data/benchmarks/README.md` - Benchmark folder documentation
8. `PROJECT_STATUS.md` - Complete status report with known issues
9. `demo/realtime_demo.py` - Simplified demo (removed brain/ integration)
10. `.env.example` - Cleaned environment configuration
11. `requirements.txt` - Removed google-genai dependency
12. `brain/` - Deleted entire RAG/LLM directory
13. `utils/wandb_utils.py` - Created shared WandB utilities
14. `utils/__init__.py` - Package initialization for utils
15. `comparison model/BDSLW_SPOTER/train.py` - SPOTER training with WandB
16. `train/train_fusion.py` - Fusion training with WandB

## Next Steps for Production

1. **Fix MediaPipe API:**
   - Option A: Downgrade to `mediapipe==0.8.10`
   - Option B: Rewrite extraction for new API

2. **Extract Real Landmarks:**
   - Process all 833 videos through landmark extraction
   - Use actual MediaPipe features instead of random data

3. **Extend Training:**
   - Train for 40+ epochs (currently only 10 for testing)
   - Implement learning rate scheduling
   - Add early stopping

4. **Improve Data Coverage:**
   - Collect more signer samples
   - Re-balance splits for better word coverage

## Documentation

- **Status Report:** See `PROJECT_STATUS.md` for detailed current state
- **Benchmarks:** See `Data/benchmarks/README.md` for metrics documentation

## License

TBD
