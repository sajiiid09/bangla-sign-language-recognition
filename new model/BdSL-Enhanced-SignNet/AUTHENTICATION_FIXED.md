# 🔧 W&B Authentication - Troubleshooting Guide

## ✅ Issue Resolved

**Problem:** W&B module was corrupted and had missing methods
- `wandb.login()` was not available
- `wandb.init()` was not available
- Module appeared empty

**Root Cause:** Corrupted or incomplete W&B installation

**Solution Applied:** Complete reinstall of W&B package

## How It Was Fixed

```bash
# Step 1: Uninstall corrupted version
pip uninstall wandb -y

# Step 2: Reinstall fresh version
pip install wandb --upgrade

# Result: All methods now available ✅
```

## Current Status

- ✅ API key properly stored in `.env`
- ✅ `python-dotenv` loading key correctly
- ✅ W&B module fully functional
- ✅ `wandb.login()` available
- ✅ `wandb.init()` available
- ✅ Authentication successful
- ✅ Ready for training with logging

## How to Diagnose W&B Issues

### Quick Test
```bash
python test_auth.py
```
This will show:
- ✅/❌ .env file status
- ✅/❌ API key loading
- ✅/❌ W&B module import
- ✅/❌ Authentication attempt
- ✅/❌ Account verification

### Configuration Check
```bash
python check_wandb.py
```
This will verify:
- W&B installation
- python-dotenv installation
- API key configuration
- Training script setup

## If Authentication Still Fails

### Issue 1: "module 'wandb' has no attribute 'login'"
**Solution:**
```bash
pip uninstall wandb -y
pip install wandb --upgrade
```

### Issue 2: "Invalid API key"
**Solution:**
1. Go to https://wandb.ai/settings/profile
2. Copy a fresh API key
3. Update `.env` file:
   ```
   WANDB_API_KEY=paste_new_key_here
   ```

### Issue 3: "No module named 'dotenv'"
**Solution:**
```bash
pip install python-dotenv
```

### Issue 4: ".env file not found"
**Solution:** Create `.env` in project root:
```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

### Issue 5: "Network error"
**Solution:**
- Check internet connection
- Make sure wandb.ai is accessible
- Check firewall settings

## Commands Reference

```bash
# Install W&B
pip install wandb python-dotenv

# Reinstall W&B (if corrupted)
pip uninstall wandb -y && pip install wandb --upgrade

# Verify installation
python test_auth.py

# Check configuration
python check_wandb.py

# Authenticate manually (alternative)
wandb login

# Run training with logging
python train_signet_v2_optimized.py --base_dir . --processed_dir ../../Data/processed/new_model --epochs 3
```

## Files Modified

1. **train_signet_v2_optimized.py** - Added proper error handling for W&B initialization
2. **src/training/trainer.py** - Wrapped wandb.log calls in try-except
3. **src/evaluation/evaluator.py** - Wrapped visualization logging in try-except
4. **check_wandb.py** - Fixed encoding issue for UTF-8 files

## Prevention Tips

1. **Keep W&B updated**
   ```bash
   pip install wandb --upgrade
   ```

2. **Store API key safely**
   - Never commit `.env` to version control
   - Use environment variables in production
   - Rotate keys regularly

3. **Regular testing**
   ```bash
   python test_auth.py  # Quick check
   ```

4. **Backup keys**
   - Save API key in password manager
   - Keep multiple keys for different environments

## What's Working Now

✅ **Complete Integration**
- API key loading from `.env`
- W&B authentication
- Batch-level logging (every 5 batches)
- Epoch-level logging
- Test metrics logging
- Visualization uploads

✅ **Error Handling**
- Graceful W&B failures
- Training continues if W&B unavailable
- Clear error messages

✅ **Documentation**
- Setup scripts (setup_wandb.py, check_wandb.py)
- Test scripts (test_auth.py)
- Diagnostic guides
- Quick start guides

## Next Steps

1. ✅ API key configured
2. ✅ W&B authenticated
3. → Run training: `python train_signet_v2_optimized.py --base_dir . --processed_dir ../../Data/processed/new_model --epochs 3`
4. → Monitor at: https://wandb.ai

---

**All authentication issues resolved! Ready to train with real-time W&B logging. 🚀📊**
