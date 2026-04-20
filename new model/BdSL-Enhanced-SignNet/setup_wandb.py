#!/usr/bin/env python3
"""
W&B Setup Script
================
Helps you configure W&B API key for real-time logging.

Usage:
    python setup_wandb.py
"""

import os
from pathlib import Path


def main():
    print("\n" + "=" * 70)
    print("🎯 W&B Setup for Bengali Sign Language Recognition")
    print("=" * 70)
    
    # Check if .env exists
    env_file = Path(".env")
    existing_key = None
    
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("WANDB_API_KEY="):
                    existing_key = line.split("=", 1)[1].strip()
                    break
    
    # Check environment variable
    env_var_key = os.getenv("WANDB_API_KEY")
    
    print("\n📋 Current Status:")
    print(f"   .env file exists: {'✅ Yes' if env_file.exists() else '❌ No'}")
    if existing_key:
        masked_key = existing_key[:10] + "..." + existing_key[-5:] if len(existing_key) > 20 else "***"
        print(f"   API key in .env: ✅ {masked_key}")
    else:
        print(f"   API key in .env: ❌ Not found")
    
    if env_var_key:
        masked_key = env_var_key[:10] + "..." + env_var_key[-5:] if len(env_var_key) > 20 else "***"
        print(f"   API key in environment: ✅ {masked_key}")
    else:
        print(f"   API key in environment: ❌ Not found")
    
    print("\n📝 Enter your W&B API key:")
    print("   (Get it from https://wandb.ai/settings/profile)")
    print("   (Press Enter to skip if already configured)")
    
    api_key = input("\n🔑 API Key: ").strip()
    
    if not api_key:
        if existing_key or env_var_key:
            print("\n✅ Using existing API key configuration")
        else:
            print("\n⚠️  No API key provided. Set it later with:")
            print("   export WANDB_API_KEY='your_api_key_here'")
        return
    
    # Save to .env file
    with open(env_file, "w") as f:
        f.write(f"WANDB_API_KEY={api_key}\n")
    
    print(f"\n✅ API key saved to .env file")
    
    # Verify
    try:
        import wandb
        wandb.login(key=api_key, relogin=True)
        print("✅ W&B authentication successful!")
        
        # Get user info
        api = wandb.Api()
        user = api.user()
        print(f"   Logged in as: {user.username}")
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Please check your API key")
        return
    
    print("\n" + "=" * 70)
    print("🚀 Ready to train with real-time W&B logging!")
    print("=" * 70)
    print("\nRun training with:")
    print("   python train_signet_v2_optimized.py --base_dir \".\" --processed_dir \"../../Data/processed/new_model\" --epochs 3 --batch_size 16")
    print("\nView results at: https://wandb.ai")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
