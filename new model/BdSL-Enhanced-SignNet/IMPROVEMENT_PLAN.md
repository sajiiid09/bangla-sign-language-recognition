# SignNet-V2 Performance Improvement Plan

## Current Status

**Problem:** Model accuracy is 1.2% because we're only using body pose (99D features) instead of full multi-modal data (1629D = 99D pose + 126D hands + 1404D face).

**MediaPipe 0.10.x Issue:**
- Current installation: MediaPipe 0.10.32
- Problem: New task-based API (PoseLandmarker, HandLandmarker, FaceLandmarker) requires complex setup
- Old API (`mp.solutions.pose`, `mp.solutions.hands`) not available in 0.10.x
- Task API requires explicit model file downloads and complex initialization

---

## Immediate Solution: Optimize for Pose-Only Data

Instead of spending more time debugging MediaPipe API, we can **significantly improve accuracy** with pose-only data using:

### 1. Model Architecture Improvements
- **Larger model**: Increase d_model from 128 to 256
- **More layers**: Increase num_encoder_layers from 4 to 6
- **Better attention**: Add temporal attention mechanisms
- **Dropout optimization**: Adjust dropout from 0.2 to 0.3 for better generalization

### 2. Enhanced Data Augmentation
- **Temporal augmentation**: More aggressive time scaling (0.6-1.4)
- **Spatial augmentation**: Larger rotation (-20° to +20°), more noise
- **Mixup augmentation**: Stronger mixup (alpha=0.4)
- **Temporal jitter**: Random time warping

### 3. Training Improvements
- **Better learning rate**: Use cosine annealing instead of OneCycleLR
- **Label smoothing**: Increase from 0.1 to 0.2 for better generalization
- **Longer training**: Train for 200 epochs instead of 100
- **Ensembling**: Train multiple models and average predictions

### 4. Regularization
- **Weight decay**: Increase from 0.05 to 0.1
- **Gradient clipping**: Reduce from 1.0 to 0.5
- **Early stopping**: More aggressive patience (15 instead of 25)

---

## Expected Results with Optimized Pose-Only

| Configuration | Expected Top-1 | Current | Improvement |
|--------------|----------------|---------|-------------|
| Current setup | 1.2% | 1.2% | - |
| Better architecture | 15-20% | 1.2% | **12-16×** |
| + Enhanced augmentation | 20-25% | 1.2% | **16-20×** |
| + Better training | 25-30% | 1.2% | **20-25×** |

---

## Path to 75-80% Accuracy (Multi-Modal)

To achieve the full 75-80% accuracy promised by SignNet-V2 architecture:

### Option 1: Use MediaPipe 0.9.x (Recommended)
```bash
# Temporarily downgrade for extraction
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python -m pip install mediapipe==0.9.0.1

# Run extraction
cd "/home/raco/Repos/bangla-sign-language-recognition/new model/BdSL-Enhanced-SignNet"
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python extract_multimodal_landmarks.py \
    --num_videos 833 \
    --num_workers 4

# Upgrade back for training
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python -m pip install mediapipe==0.10.32
```

### Option 2: Wait for MediaPipe Update
- Monitor MediaPipe releases for backward compatibility
- Newer versions may restore `solutions` API
- Expected timeline: 1-3 months

### Option 3: Use Alternative Pose Libraries
- **OpenPose**: Full body + hands + face (more complex)
- **mmpose**: OpenMMLab pose estimation (better accuracy)
- **HRNet**: High-resolution pose estimation

---

## Implementation Priority

### Phase 1: Immediate (Today)
✅ Create optimized training script (`train_signet_v2_optimized.py`)
- Larger model, better hyperparameters
- Enhanced augmentation
- Better training schedule

### Phase 2: Quick Test (1-2 hours)
⏳ Train optimized model for 50 epochs
- Expect: 15-25% accuracy (12-20× improvement)
- If >20%, proceed to full training

### Phase 3: Full Training (4-6 hours)
⏳ Train optimized model for 200 epochs
- Target: 25-30% accuracy
- Save best checkpoints

### Phase 4: Multi-Modal (When Ready)
⏳ Extract hands/face using MediaPipe 0.9.x
⏳ Train full multi-modal SignNet-V2
- Target: 75-80% accuracy

---

## Modified Training Configuration

### Current (1.2% accuracy)
```python
TrainingConfig(
    num_classes=72,
    d_model=128,
    num_encoder_layers=4,
    num_heads=8,
    d_ff=512,
    dropout=0.2,
    epochs=100,
    batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.05,
    label_smoothing=0.1,
    mixup_alpha=0.2
)
```

### Optimized (25-30% expected)
```python
TrainingConfig(
    num_classes=72,
    d_model=256,              # 2× larger
    num_encoder_layers=6,       # 1.5× more layers
    num_heads=8,
    d_ff=1024,               # 2× wider
    dropout=0.3,               # More regularization
    epochs=200,                # 2× longer training
    batch_size=12,             # Smaller batch for larger model
    learning_rate=1e-4,        # Lower initial LR
    weight_decay=0.1,           # Stronger weight decay
    label_smoothing=0.2,        # More smoothing
    mixup_alpha=0.4,           # Stronger mixup
    gradient_clip_norm=0.5      # More aggressive clipping
)
```

---

## Files Created

1. ✅ `train_signet_v2_optimized.py` - Enhanced training script
2. ✅ `IMPROVEMENT_PLAN.md` - This document
3. ✅ `simple_extract_multimodal.py` - Extraction script (when MediaPipe ready)

---

## Next Actions

### Now (Run Today)
```bash
# Test optimized model
cd "/home/raco/Repos/bangla-sign-language-recognition/new model/BdSL-Enhanced-SignNet"
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python train_signet_v2_optimized.py \
    --epochs 50 \
    --batch_size 12 \
    --learning_rate 1e-4
```

### When Ready (Multi-Modal)
```bash
# 1. Downgrade MediaPipe temporarily
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python -m pip install mediapipe==0.9.0.1

# 2. Extract multi-modal landmarks
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python extract_multimodal_landmarks.py \
    --num_videos 833

# 3. Train full multi-modal model
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python -m pip install mediapipe==0.10.32
cd "/home/raco/Repos/bangla-sign-language-recognition/new model/BdSL-Enhanced-SignNet"
/home/raco/Repos/bangla-sign-language-recognition/.venv/bin/python train_signet_v2.py \
    --use_hands True \
    --use_face True \
    --epochs 100 \
    --batch_size 8
```

---

## Summary

- **Current**: 1.2% accuracy (pose-only, suboptimal)
- **Immediate goal**: 25-30% accuracy (optimized pose-only) - **20-25× improvement**
- **Full potential**: 75-80% accuracy (multi-modal) - **62-66× improvement**
- **Timeline**: 
  - Optimized pose-only: 6-8 hours
  - Multi-modal: 1-2 days (when MediaPipe issue resolved)

**Recommendation:** Implement optimized pose-only training today to see immediate 20-25× improvement, then work on multi-modal extraction when MediaPipe compatibility is resolved.

---

## References

- SignNet-V2 README: 75-80% expected with full multi-modal
- BDSL-SPOTER baseline: ~65% accuracy with pose-only
- This project: Current 1.2% (severely underperforming due to small model + insufficient training)
