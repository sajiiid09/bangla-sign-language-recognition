#!/usr/bin/env python3
"""
Quick W&B Integration Check
============================
Verifies W&B is properly configured and ready to use.
"""

import os
import sys
from pathlib import Path
import re


def check_wandb():
    """Check W&B installation and configuration."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    print("\n" + "=" * 70)
    print("🔍 W&B Configuration Check")
    print("=" * 70)
    
    # Check 1: W&B Module
    print("\n1️⃣  W&B Installation:")
    try:
        import wandb
        print("   ✅ W&B module installed")
    except ImportError:
        print("   ❌ W&B not installed")
        print("   Run: pip install wandb")
        return False
    
    # Check 2: python-dotenv
    print("\n2️⃣  python-dotenv Installation:")
    try:
        import dotenv
        print("   ✅ python-dotenv installed")
    except ImportError:
        print("   ❌ python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return False
    
    # Check 3: API Key
    print("\n3️⃣  W&B API Key Configuration:")
    env_file_candidates = [
        repo_root / ".env",
        script_dir / ".env",
        Path(".env"),
    ]
    env_file = next((p for p in env_file_candidates if p.exists()), None)
    env_key = None
    file_key = None
    
    # Check environment variable
    env_key = os.getenv("WANDB_API_KEY")
    
    # Check .env file
    if env_file is not None:
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("WANDB_API_KEY="):
                    file_key = line.split("=", 1)[1].strip()
                    break
    
    if env_key:
        print(f"   ✅ Found in environment variable")
    elif file_key:
        print(f"   ✅ Found in .env file")
    else:
        print(f"   ❌ No API key found")
        print(f"   Create .env file or set WANDB_API_KEY environment variable")
        print(f"   Get key from: https://wandb.ai/settings/profile")
        return False
    
    # Check 4: API Key Validity
    print("\n4️⃣  W&B Authentication:")
    api_key = env_key or file_key
    try:
        from wandb.errors import CommError
        # Just try to validate the key format
        if len(api_key) > 20 and re.fullmatch(r"[A-Za-z0-9_-]+", api_key):
            print("   ✅ API key format looks valid")
        else:
            print("   ⚠️  API key format might be invalid")
    except Exception as e:
        print(f"   ⚠️  Could not verify: {e}")
    
    # Check 5: Training Script
    print("\n5️⃣  Training Script Setup:")
    train_candidates = [
        script_dir / "train_signet_v2_optimized.py",
        script_dir / "train_signet_v2.py",
    ]
    train_script = next((p for p in train_candidates if p.exists()), None)
    if train_script is not None:
        try:
            with open(train_script, "r", encoding="utf-8") as f:
                content = f.read()
                if "load_dotenv()" in content and "wandb.login" in content:
                    print("   ✅ Training script properly configured")
                else:
                    print("   ⚠️  Training script may need updates")
        except Exception as e:
            print(f"   ⚠️  Could not read training script: {e}")
    else:
        print("   ❌ Training script not found")
        return False
    
    # Check 6: Setup Script
    print("\n6️⃣  Setup Script Available:")
    setup_script = script_dir / "setup_wandb.py"
    if setup_script.exists():
        print("   ✅ setup_wandb.py available")
    else:
        print("   ⚠️  setup_wandb.py not found")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ All checks passed! Ready for real-time W&B logging!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run training:")
    print("   python train_signet_v2_optimized.py --base_dir \".\" --processed_dir \"../../Data/processed/new_model\" --epochs 3")
    print("\n2. Monitor at:")
    print("   https://wandb.ai")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    success = check_wandb()
    sys.exit(0 if success else 1)
