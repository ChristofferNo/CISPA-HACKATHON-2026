#!/usr/bin/env python3
"""
VAR/RAR Setup Diagnostic Tool

Checks that all required files and dependencies are in place
for running the data provenance detection solution.
"""

import sys
from pathlib import Path


def check_directory(path: Path, name: str) -> bool:
    """Check if directory exists and show contents."""
    print(f"\n{name}:")
    print(f"  Path: {path.absolute()}")

    if not path.exists():
        print(f"  ✗ NOT FOUND")
        return False

    print(f"  ✓ EXISTS")
    print(f"  Contents:")

    for item in sorted(path.iterdir()):
        item_type = "dir" if item.is_dir() else "file"
        print(f"    - {item.name} ({item_type})")

    return True


def check_checkpoints(ckpt_dir: Path, pattern: str, name: str) -> bool:
    """Check for checkpoint files."""
    print(f"\n{name} Checkpoints:")
    print(f"  Directory: {ckpt_dir}")

    if not ckpt_dir.exists():
        print(f"  ✗ Directory not found")
        return False

    files = list(ckpt_dir.glob(pattern))

    if not files:
        print(f"  ✗ No {pattern} files found")
        print(f"  Available files:")
        for f in sorted(ckpt_dir.iterdir()):
            print(f"    - {f.name}")
        return False

    print(f"  ✓ Found {len(files)} checkpoint(s):")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024**2)
        print(f"    - {f.name} ({size_mb:.1f} MB)")

    return True


def test_import(root: Path, module: str, name: str) -> bool:
    """Test if module can be imported."""
    sys.path.insert(0, str(root.absolute()))

    try:
        imported = __import__(module)
        print(f"  ✓ Successfully imported '{module}'")
        print(f"    Location: {imported.__file__}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to import '{module}'")
        print(f"    Error: {e}")
        return False

    finally:
        sys.path.pop(0)


def main():
    print("="*60)
    print("VAR/RAR SETUP DIAGNOSTIC")
    print("="*60)

    var_root = Path("VAR")
    rar_root = Path("RAR/1d-tokenizer")

    # Track results
    all_ok = True

    # Check directories
    print("\n" + "="*60)
    print("1. DIRECTORY STRUCTURE")
    print("="*60)

    var_ok = check_directory(var_root, "VAR Root")
    rar_ok = check_directory(rar_root, "RAR Root")

    all_ok = all_ok and var_ok and rar_ok

    # Check VAR components
    if var_ok:
        var_models = var_root / "models"
        var_models_ok = check_directory(var_models, "VAR Models")

        if var_models_ok:
            init_file = var_models / "__init__.py"
            if init_file.exists():
                print(f"    ✓ Has __init__.py")
            else:
                print(f"    ✗ Missing __init__.py")
                all_ok = False
        else:
            all_ok = False

    # Check checkpoints
    print("\n" + "="*60)
    print("2. CHECKPOINTS")
    print("="*60)

    if var_ok:
        var_ckpt_ok = check_checkpoints(
            var_root / "checkpoints",
            "*.pth",
            "VAR"
        )

        # Check specific file
        vae_ckpt = var_root / "checkpoints" / "vae_ch160v4096z32.pth"
        if vae_ckpt.exists():
            print(f"\n  ✓ Found required VAE checkpoint: {vae_ckpt.name}")
        else:
            print(f"\n  ✗ Missing required VAE checkpoint: vae_ch160v4096z32.pth")
            all_ok = False
    else:
        var_ckpt_ok = False

    if rar_ok:
        rar_ckpt_ok = check_checkpoints(
            rar_root / "checkpoints",
            "*.bin",
            "RAR"
        )

        # Check specific files
        required_rar = [
            "maskgit-vqgan-imagenet-f16-256.bin",
            "rar_b.bin",
            "rar_l.bin",
            "rar_xl.bin",
            "rar_xxl.bin"
        ]

        print(f"\n  Checking required RAR checkpoints:")
        for filename in required_rar:
            ckpt = rar_root / "checkpoints" / filename
            if ckpt.exists():
                print(f"    ✓ {filename}")
            else:
                print(f"    ✗ {filename}")
                all_ok = False
    else:
        rar_ckpt_ok = False

    all_ok = all_ok and var_ckpt_ok and rar_ckpt_ok

    # Test imports
    print("\n" + "="*60)
    print("3. IMPORT TEST")
    print("="*60)

    if var_ok:
        print("\nVAR Models:")
        var_import_ok = test_import(var_root, "models", "VAR")

        if var_import_ok:
            # Check for specific function
            sys.path.insert(0, str(var_root.absolute()))
            try:
                import models
                if hasattr(models, 'build_vae_var'):
                    print(f"    ✓ Has build_vae_var function")
                else:
                    print(f"    ✗ Missing build_vae_var function")
                    all_ok = False
            finally:
                sys.path.pop(0)
        else:
            all_ok = False

    if rar_ok:
        print("\nRAR Demo Utils:")
        rar_import_ok = test_import(rar_root, "demo_util", "RAR")
        all_ok = all_ok and rar_import_ok

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if all_ok:
        print("\n✓ All checks passed! You're ready to run solution.py")
    else:
        print("\n✗ Some checks failed. See recommendations below.")

        print("\nRECOMMENDATIONS:")

        if not var_ok:
            print("  • Clone or copy VAR repository to ./VAR")

        if not rar_ok:
            print("  • Clone or copy RAR repository to ./RAR/1d-tokenizer")

        if var_ok and not var_ckpt_ok:
            print("  • Download VAR checkpoints:")
            print("    - vae_ch160v4096z32.pth → ./VAR/checkpoints/")

        if rar_ok and not rar_ckpt_ok:
            print("  • Download RAR checkpoints:")
            print("    - Run: cd RAR/1d-tokenizer && python download_rar_checkpoints.py")
            print("    - Or manually download to ./RAR/1d-tokenizer/checkpoints/:")
            print("      - maskgit-vqgan-imagenet-f16-256.bin (tokenizer)")
            print("      - rar_b.bin, rar_l.bin, rar_xl.bin, rar_xxl.bin (generators)")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
