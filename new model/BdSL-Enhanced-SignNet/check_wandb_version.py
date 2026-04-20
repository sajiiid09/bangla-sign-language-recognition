#!/usr/bin/env python3
"""Check W&B version and available methods"""

import wandb
import inspect

print("W&B Available Methods:")
print("=" * 70)

# List all public methods
methods = [m for m in dir(wandb) if not m.startswith('_')]
print(f"Total methods: {len(methods)}")

# Check for auth methods
print("\nAuthentication-related methods:")
auth_methods = [m for m in methods if 'login' in m.lower() or 'auth' in m.lower() or 'api' in m.lower()]
if auth_methods:
    for method in auth_methods:
        print(f"  ✓ {method}")
else:
    print("  ✗ No login/auth methods found")

# Check init method
print("\nInit method signature:")
if hasattr(wandb, 'init'):
    sig = inspect.signature(wandb.init)
    print(f"  wandb.init({sig})")

# Check what we can use
print("\nAvailable core methods:")
core = ['init', 'log', 'finish', 'save', 'artifact', 'summary', 'config']
for c in core:
    if hasattr(wandb, c):
        print(f"  ✓ wandb.{c}")
    else:
        print(f"  ✗ wandb.{c}")
