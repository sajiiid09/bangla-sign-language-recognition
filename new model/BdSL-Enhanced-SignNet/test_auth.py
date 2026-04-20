#!/usr/bin/env python3
"""
W&B Authentication Diagnostic
==============================
Debugs why W&B authentication is failing.
"""

import os
import sys
from pathlib import Path

print("\n" + "=" * 70)
print("🔍 W&B Authentication Diagnostic")
print("=" * 70)

# Step 1: Check .env file
print(f"\n1️⃣  .env File Check:")
env_file = Path(".env")
if env_file.exists():
    print(f"   ✅ .env file exists")
    with open(env_file, "r") as f:
        content = f.read()
    
    if "WANDB_API_KEY=" in content:
        print(f"   ✅ WANDB_API_KEY found in .env")
    else:
        print(f"   ❌ WANDB_API_KEY not in .env file")
        sys.exit(1)
else:
    print(f"   ❌ .env file not found")
    sys.exit(1)

# Step 2: Load API key
print(f"\n2️⃣  Loading API Key:")
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    
    if api_key:
        print(f"   ✅ API key loaded from .env")
        print(f"   Format: {api_key[:30]}...")
        print(f"   Length: {len(api_key)} characters")
    else:
        print(f"   ❌ API key is None after loading")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Failed to load from .env: {e}")
    sys.exit(1)

# Step 3: Check W&B module
print(f"\n3️⃣  W&B Module Check:")
try:
    import wandb
    print(f"   ✅ wandb module imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import wandb: {e}")
    sys.exit(1)

# Step 4: Authenticate with W&B
print(f"\n4️⃣  W&B Authentication:")
try:
    print(f"   Attempting login with provided API key...")
    wandb.login(key=api_key, relogin=True)
    print(f"   ✅ Successfully authenticated!")
    
except Exception as e:
    print(f"   ❌ Authentication FAILED: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    # Analyze error
    error_str = str(e).lower()
    if "invalid" in error_str:
        print(f"\n   💡 Possible causes:")
        print(f"      • Invalid or expired API key")
        print(f"      • Key format incorrect")
    elif "network" in error_str or "connection" in error_str:
        print(f"\n   💡 Network issue:")
        print(f"      • Check internet connection")
        print(f"      • Check if wandb.ai is accessible")
    
    sys.exit(1)

# Step 5: Verify login
print(f"\n5️⃣  Verify Login:")
try:
    api = wandb.Api()
    user = api.user()
    print(f"   ✅ Logged in as: {user.username}")
    print(f"   ✅ Account verified!")
    
except Exception as e:
    print(f"   ⚠️  Could not verify user account: {e}")
    print(f"   But login may still be successful")

# Summary
print("\n" + "=" * 70)
print("✅ AUTHENTICATION SUCCESSFUL!")
print("=" * 70)
print("\nYou can now run training with W&B logging enabled:")
print("  python train_signet_v2_optimized.py --base_dir . --processed_dir ../../Data/processed/new_model --epochs 3")
print("\n" + "=" * 70 + "\n")
