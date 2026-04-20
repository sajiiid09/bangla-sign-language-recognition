# 📊 W&B Real-Time Logging - Complete Documentation Index

## 🎯 Start Here

- **[QUICKSTART_WANDB.md](QUICKSTART_WANDB.md)** ⚡ (5 minutes)
  - 30-second setup
  - API key configuration
  - Running your first training with W&B

## 📚 Documentation Files

### Getting Started
1. **[QUICKSTART_WANDB.md](QUICKSTART_WANDB.md)** - Quick setup guide
   - Quick API key setup (3 options)
   - Running training
   - Verification steps

2. **[WANDB_SETUP.md](WANDB_SETUP.md)** - Detailed setup guide
   - Prerequisites
   - Step-by-step installation
   - Environment configuration
   - Troubleshooting common issues

### Integration Details
3. **[WANDB_INTEGRATION_SUMMARY.md](WANDB_INTEGRATION_SUMMARY.md)** - What was implemented
   - Changes made to each file
   - Real-time metrics overview
   - Feature list
   - Next steps

4. **[TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)** - Architecture & implementation
   - System architecture
   - Code integration points
   - Logging frequency
   - API reference

## 🛠️ Utility Scripts

### Setup Scripts
- **[setup_wandb.py](setup_wandb.py)** - Interactive setup
  - Guides you through API key configuration
  - Verifies W&B connection
  - Saves configuration to `.env`

- **[check_wandb.py](check_wandb.py)** - Configuration checker
  - Verifies W&B installation
  - Checks API key configuration
  - Validates training script setup
  - Reports readiness status

## 📊 Real-Time Metrics

### Logged During Training
- Batch-level loss & accuracy (every 5 batches)
- Epoch-level metrics (after each epoch)
- Learning rate tracking
- Test metrics (final evaluation)

### Visualizations Generated
- Confusion matrix
- Per-class accuracy plots
- Per-signer accuracy analysis
- Top-K accuracy curves
- Model comparison charts

## 🚀 Quick Commands

### Setup (Choose One)
```bash
# Interactive setup (recommended)
python setup_wandb.py

# OR create .env file manually
echo "WANDB_API_KEY=your_api_key_here" > .env

# OR set environment variable
$env:WANDB_API_KEY = "your_api_key_here"  # PowerShell
```

### Verify Setup
```bash
python check_wandb.py
```

### Run Training with Logging
```bash
python train_signet_v2_optimized.py \
  --base_dir "." \
  --processed_dir "../../Data/processed/new_model" \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-4
```

### Monitor in Real-Time
```
Open: https://wandb.ai
```

## 📁 Modified Source Files

### Core Training Script
- **train_signet_v2_optimized.py**
  - Added dotenv import
  - Added API key loading and W&B login
  - Added proper W&B initialization
  - Added final metrics logging

### Training Loop
- **src/training/trainer.py**
  - Added batch-level logging (every 5 batches)
  - Added epoch-level logging
  - Removed try-except blocks

### Evaluation
- **src/evaluation/evaluator.py**
  - Removed all try-except blocks
  - Added confusion matrix logging
  - Added visualization logging
  - Added accuracy plots logging

## 🔑 Getting Your API Key

1. Visit: https://wandb.ai/settings/profile
2. Scroll to "API keys" section
3. Copy your key (40+ character string)
4. Use in setup script or .env file

## ✅ Integration Checklist

- [x] W&B module installed
- [x] python-dotenv installed
- [x] Batch-level logging implemented
- [x] Epoch-level logging implemented
- [x] Visualization logging enabled
- [x] API key authentication configured
- [x] Setup script created
- [x] Check script created
- [x] Documentation completed

## 📈 Expected Output

When running with W&B:

```
🚀 SignNet-V2 Training Pipeline
📊 Dataset splits: Train: 433, Val: 54, Test: 55
✅ W&B authenticated with API key
✅ WandB initialized
📦 Data loaders created: Train: 27 batches, Val: 2 batches, Test: 2 batches
🧠 Model: SignNet-V2 (6,847,414 parameters)

🚀 Starting training for 3 epochs
Epoch 1: 100%|███████| 27/27 [batch metrics logged every 5 batches]
📊 Epoch 1 Summary:
   Train - Loss: 3.9850, Acc: 0.0329 (3.29%)
   Val   - Loss: 3.9485, Acc: 0.0185 (1.85%)
   LR: 0.000300

[Continues for all epochs...]

✅ Visualizations saved
🎉 TRAINING COMPLETE
📊 Test Results logged to W&B
```

## 🎯 Next Steps

1. **Setup:** Run `python setup_wandb.py`
2. **Verify:** Run `python check_wandb.py`
3. **Train:** Run `python train_signet_v2_optimized.py --base_dir . --processed_dir ../../Data/processed/new_model --epochs 3`
4. **Monitor:** Open https://wandb.ai in browser
5. **Analyze:** Compare runs, optimize hyperparameters

## 📞 Support Resources

- **W&B Official Docs:** https://docs.wandb.ai
- **W&B Community:** https://community.wandb.ai
- **API Key Help:** https://wandb.ai/settings/profile
- **GitHub Issues:** https://github.com/wandb/wandb/issues

## 📋 File Structure

```
BdSL-Enhanced-SignNet/
├── train_signet_v2_optimized.py         (Modified - W&B init & logging)
├── setup_wandb.py                       (New - Interactive setup)
├── check_wandb.py                       (New - Config checker)
├── .env                                 (To create - API key)
├── src/
│   ├── training/
│   │   └── trainer.py                   (Modified - Batch logging)
│   └── evaluation/
│       └── evaluator.py                 (Modified - Viz logging)
├── QUICKSTART_WANDB.md                  (New - 30-second guide)
├── WANDB_SETUP.md                       (New - Detailed setup)
├── WANDB_INTEGRATION_SUMMARY.md         (New - What changed)
├── TECHNICAL_REFERENCE.md               (New - Architecture)
└── WANDB_DOCS_INDEX.md                  (This file)
```

## 🎓 Learning Path

**For Quick Users (5 min):**
1. Read: QUICKSTART_WANDB.md
2. Run: setup_wandb.py
3. Run: check_wandb.py
4. Start training

**For Detailed Understanding (20 min):**
1. Read: WANDB_SETUP.md
2. Read: WANDB_INTEGRATION_SUMMARY.md
3. Skim: TECHNICAL_REFERENCE.md
4. Follow setup steps

**For Implementation Details (60 min):**
1. Read: All documentation files
2. Review: Code changes in modified files
3. Study: TECHNICAL_REFERENCE.md
4. Understand: Logging flow and integration points

## 💡 Pro Tips

1. **Multiple Runs:** Train with different hyperparameters and compare on W&B
2. **Batch Monitoring:** Watch batch-level metrics for real-time feedback
3. **Visualizations:** Use confusion matrix to identify problem classes
4. **Offline Mode:** Train without internet, sync results later
5. **Team Sharing:** Use W&B's entity feature to share runs with team members

## ⚠️ Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| API key not found | Run `setup_wandb.py` or create `.env` |
| Module not installed | Run `pip install wandb python-dotenv` |
| Authentication failed | Check API key at wandb.ai/settings/profile |
| Can't see metrics | Make sure wandb.ai is open in browser |
| Training too slow | Check batch logging frequency in trainer.py |

---

## 🎉 You're All Set!

Your training pipeline now has production-ready W&B real-time logging.

**Start with:** `python setup_wandb.py` → `python check_wandb.py` → Training!

**Monitor at:** https://wandb.ai

**Questions?** Check the documentation files above or visit wandb.ai/docs
