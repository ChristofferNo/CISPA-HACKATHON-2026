#!/usr/bin/env python3
"""
Diagnostic script to check VAR and RAR setup
"""
import sys
from pathlib import Path

print("="*60)
print("VAR/RAR SETUP DIAGNOSTIC")
print("="*60)

# Check directories
var_root = Path("VAR")
rar_root = Path("RAR/1d-tokenizer")

print(f"\n1. Directory structure:")
print(f"   VAR root: {var_root.absolute()}")
print(f"   Exists: {var_root.exists()}")

if var_root.exists():
    print(f"   Contents:")
    for item in sorted(var_root.iterdir()):
        print(f"     - {item.name} ({'dir' if item.is_dir() else 'file'})")

print(f"\n   RAR root: {rar_root.absolute()}")
print(f"   Exists: {rar_root.exists()}")

if rar_root.exists():
    print(f"   Contents:")
    for item in sorted(rar_root.iterdir()):
        print(f"     - {item.name} ({'dir' if item.is_dir() else 'file'})")

# Check VAR checkpoints
print(f"\n2. VAR checkpoints:")
var_ckpt = var_root / "checkpoints"
print(f"   Directory: {var_ckpt}")
print(f"   Exists: {var_ckpt.exists()}")

if var_ckpt.exists():
    files = list(var_ckpt.glob("*.pth"))
    if files:
        print(f"   Found {len(files)} .pth files:")
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024**2)
            print(f"     - {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"   ⚠️  No .pth files found!")
        print(f"   All files:")
        for f in sorted(var_ckpt.iterdir()):
            print(f"     - {f.name}")

# Check RAR checkpoints
print(f"\n3. RAR checkpoints:")
rar_ckpt = rar_root / "checkpoints"
print(f"   Directory: {rar_ckpt}")
print(f"   Exists: {rar_ckpt.exists()}")

if rar_ckpt.exists():
    files = list(rar_ckpt.glob("*.bin"))
    if files:
        print(f"   Found {len(files)} .bin files:")
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024**2)
            print(f"     - {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"   ⚠️  No .bin files found!")
        print(f"   All files:")
        for f in sorted(rar_ckpt.iterdir()):
            print(f"     - {f.name}")

# Check VAR models directory
print(f"\n4. VAR models directory:")
var_models = var_root / "models"
print(f"   Directory: {var_models}")
print(f"   Exists: {var_models.exists()}")

if var_models.exists():
    print(f"   Python files:")
    for f in sorted(var_models.glob("*.py")):
        print(f"     - {f.name}")
    
    # Check for __init__.py
    init_file = var_models / "__init__.py"
    print(f"   Has __init__.py: {init_file.exists()}")

# Try importing
print(f"\n5. Import test:")

# Test VAR import
sys.path.insert(0, str(var_root.absolute()))
try:
    import models
    print(f"   ✓ Successfully imported 'models'")
    print(f"     Location: {models.__file__}")
    print(f"     Has build_vae_var: {hasattr(models, 'build_vae_var')}")
    if hasattr(models, 'build_vae_var'):
        print(f"     build_vae_var: {models.build_vae_var}")
except Exception as e:
    print(f"   ✗ Failed to import 'models': {e}")
finally:
    sys.path.pop(0)

# Test RAR import
sys.path.insert(0, str(rar_root.absolute()))
try:
    import demo_util
    print(f"   ✓ Successfully imported 'demo_util'")
    print(f"     Location: {demo_util.__file__}")
except Exception as e:
    print(f"   ✗ Failed to import 'demo_util': {e}")
finally:
    sys.path.pop(0)

print(f"\n{'='*60}")
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nRECOMMENDATIONS:")

if not (var_ckpt / "vae_ch160v4096z32.pth").exists():
    print("  ⚠️  VAR checkpoint missing!")
    print("     Expected: task2/VAR/checkpoints/vae_ch160v4096z32.pth")
    print("     You may need to download it")

if not (rar_ckpt / "maskgit-vqgan-imagenet-f16-256.bin").exists():
    print("  ⚠️  RAR tokenizer checkpoint missing!")
    print("     Expected: task2/RAR/1d-tokenizer/checkpoints/maskgit-vqgan-imagenet-f16-256.bin")
    print("     Run: cd task2/RAR/1d-tokenizer && python download_rar_checkpoints.py")

if not var_models.exists():
    print("  ⚠️  VAR models directory missing!")
    print("     Expected: task2/VAR/models/")
    print("     This is required for loading VAR models")

print()