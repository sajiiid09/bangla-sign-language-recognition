# 🚀 W&B Real-Time Logging - Quick Start

## ⚡ 30-Second Setup

### Step 1: Get Your API Key
1. Go to https://wandb.ai/settings/profile
2. Find "API keys" section
3. Copy your key

### Step 2: Set API Key (Choose One)

**Option A: Interactive Setup (Easiest)**
```bash
python setup_wandb.py
# Follow the prompts
```

**Option B: Create .env File**
Create file named `.env` in project root:
```
WANDB_API_KEY=paste_your_api_key_here
```

**Option C: Environment Variable (PowerShell)**
```powershell
$env:WANDB_API_KEY = "paste_your_api_key_here"
```

### Step 3: Verify Setup
```bash
python check_wandb.py
```

### Step 4: Run Training
```bash
python train_signet_v2_optimized.py \
  --base_dir "." \
  --processed_dir "../../Data/processed/new_model" \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-4
```

### Step 5: Watch Real-Time Results
Open https://wandb.ai in your browser and see metrics update live!

---

## 📊 What Gets Logged

### Every 5 Batches (Real-Time)
- ⚡ Training loss
- ⚡ Training accuracy
- ⚡ Batch number

### Every Epoch (Real-Time)
- 📈 Validation loss
- 📈 Validation accuracy  
- 📉 Learning rate

### After Training (Final)
- 🎯 Test accuracy
- 🎯 Test precision
- 🎯 Test recall
- 🎯 F1-score
- 🎯 Top-5 accuracy
- 📊 Confusion matrix
- 📊 Per-class accuracy plots
- 📊 Per-signer accuracy plots

---

## 🎯 Example: Running with W&B

```bash
# Terminal 1: Start training
python train_signet_v2_optimized.py --epochs 5 --batch_size 16

# Terminal 2: Watch real-time dashboard
# Open: https://wandb.ai
```

**You'll see:**
- ✅ Metrics updating every batch
- ✅ Live charts updating
- ✅ System resource usage
- ✅ All visualizations and plots

---

## 📁 Files Added/Modified

### New Files
- ✅ `setup_wandb.py` - Interactive setup
- ✅ `check_wandb.py` - Configuration checker
- ✅ `WANDB_SETUP.md` - Detailed guide
- ✅ `WANDB_INTEGRATION_SUMMARY.md` - Integration details
- ✅ `.env` - API key storage (create this)

### Modified Files
- ✅ `train_signet_v2_optimized.py` - Added W&B login and logging
- ✅ `src/training/trainer.py` - Added batch-level logging
- ✅ `src/evaluation/evaluator.py` - Enabled visualization logging

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key not found" | Run `python setup_wandb.py` or create `.env` file |
| "Module not found" | Install: `pip install wandb python-dotenv` |
| "Permission denied" | Check API key is valid at wandb.ai/settings/profile |
| Can't see metrics | Make sure browser is open to https://wandb.ai |

---

## 💡 Pro Tips

1. **Multiple Runs**: Run training multiple times to compare results on W&B
2. **Batch Logging**: Check metrics every 5 batches in real-time
3. **Visualizations**: All plots automatically generated and uploaded
4. **Offline Mode**: Set `WANDB_DISABLED=true` to train without internet
5. **Compare Runs**: Use W&B's comparison feature to optimize hyperparameters

---

## 📚 Documentation

- **Quick Setup**: This file (you are here!)
- **Detailed Setup**: `WANDB_SETUP.md`
- **Integration Info**: `WANDB_INTEGRATION_SUMMARY.md`
- **W&B Docs**: https://docs.wandb.ai

---

## ✨ Next Steps

1. ✅ Set API key (30 seconds)
2. ✅ Run `python setup_wandb.py` or create `.env`
3. ✅ Verify with `python check_wandb.py`
4. ✅ Start training!
5. ✅ Open https://wandb.ai to watch live

---

**Ready? Start training with real-time monitoring! 🚀📊**

```bash
python train_signet_v2_optimized.py --base_dir "." --processed_dir "../../Data/processed/new_model" --epochs 3
```
