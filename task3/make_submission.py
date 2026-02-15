import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ----------------------------
# IO helpers
# ----------------------------
def load_png_to_tensor(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3x32x32
    return x

def tensor_to_u8(x01: torch.Tensor) -> np.ndarray:
    # x01: 1x3xHxW in [0,1]
    x = x01.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()  # HWC
    u8 = (x * 255.0 + 0.5).astype(np.uint8)
    return u8

def u8_to_tensor(u8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x

def quantize_roundtrip(x01: torch.Tensor) -> torch.Tensor:
    """Simulate PNG quantization robustly: float -> uint8 -> float."""
    u8 = tensor_to_u8(x01)
    return u8_to_tensor(u8)

def save_png_u8(u8: np.ndarray, p: Path) -> None:
    Image.fromarray(u8, mode="RGB").save(p)

def ensure_rgb32(u8: np.ndarray) -> None:
    assert u8.shape == (32, 32, 3), f"Expected 32x32x3, got {u8.shape}"
    assert u8.dtype == np.uint8, f"Expected uint8, got {u8.dtype}"


# ----------------------------
# Model helpers
# ----------------------------
@torch.no_grad()
def logits_from_u8(model, u8: np.ndarray) -> torch.Tensor:
    x = u8_to_tensor(u8)
    return model(x)[0]  # (C,)

def top1_top2_margin(logits: torch.Tensor) -> tuple[int, int, float]:
    vals, idx = torch.topk(logits, 2)
    c = int(idx[0].item())
    t = int(idx[1].item())
    margin = float((vals[0] - vals[1]).item())
    return c, t, margin


# ----------------------------
# Boundary search
# ----------------------------
def push_to_boundary(
    model,
    start_u8: np.ndarray,
    steps: int,
    lr: float,
    margin_stop: float,
    check_every: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """
    Minimize (logit_top1 - logit_top2) using PGD-like updates in float space,
    with frequent PNG-quantization roundtrips.
    """
    rng = np.random.default_rng(seed)

    # Start from a quantized image
    u8 = start_u8.copy()
    ensure_rgb32(u8)

    # Convert to float tensor for optimization
    x = u8_to_tensor(u8).clone().requires_grad_(True)

    best_u8 = u8.copy()
    best_margin = float("inf")
    best_info = {}

    for k in range(1, steps + 1):
        # Important: quantize roundtrip to stay realistic
        with torch.no_grad():
            x_q = quantize_roundtrip(x)
        x = x_q.clone().requires_grad_(True)

        logits = model(x)[0]
        # compute dynamic top1/top2 on current x
        vals, idx = torch.topk(logits, 2)
        c = idx[0]
        t = idx[1]
        margin = (vals[0] - vals[1])

        loss = margin
        loss.backward()

        with torch.no_grad():
            g = x.grad
            # sign step (L_inf style)
            x -= lr * g.sign()
            x.clamp_(0.0, 1.0)

        if k % check_every == 0 or k == 1:
            # evaluate after quantization
            u8_cur = tensor_to_u8(quantize_roundtrip(x))
            logits_cur = logits_from_u8(model, u8_cur)
            c2, t2, m2 = top1_top2_margin(logits_cur)

            if m2 < best_margin:
                best_margin = m2
                best_u8 = u8_cur.copy()
                best_info = {"step": k, "c": c2, "t": t2, "margin": m2}

            print(
                f"[boundary] step {k:04d}  top1={c2} top2={t2}  margin={m2:.3e}"
            )

            if m2 <= margin_stop:
                break

        # tiny random dithering every now and then can help escape flat regions
        if k % (check_every * 5) == 0:
            with torch.no_grad():
                jitter = torch.from_numpy(rng.integers(-1, 2, size=(1, 3, 32, 32))).float() / 255.0
                x.add_(0.25 * jitter).clamp_(0, 1)

    return best_u8, best_info


# ----------------------------
# Farming: jitter near-boundary images
# ----------------------------
def jitter_u8(u8: np.ndarray, n_edits: int, max_delta: int, rng: random.Random) -> np.ndarray:
    """Apply n_edits random pixel-channel edits of +/- up to max_delta."""
    out = u8.copy()
    H, W, C = out.shape
    for _ in range(n_edits):
        y = rng.randrange(H)
        x = rng.randrange(W)
        c = rng.randrange(C)
        delta = rng.randint(-max_delta, max_delta)
        out[y, x, c] = np.uint8(np.clip(int(out[y, x, c]) + delta, 0, 255))
    return out

def farm_near_boundary(
    model,
    seed_u8: np.ndarray,
    want: int,
    margin_keep: float,
    tries: int,
    n_edits: int,
    max_delta: int,
    seed: int,
) -> list[np.ndarray]:
    """
    Generate many variants around a near-boundary seed.
    Keep those with small top1-top2 margin after quantization.
    """
    rng = random.Random(seed)
    kept = []
    best_seen = float("inf")

    for i in range(1, tries + 1):
        cand = jitter_u8(seed_u8, n_edits=n_edits, max_delta=max_delta, rng=rng)
        logits = logits_from_u8(model, cand)
        _, _, m = top1_top2_margin(logits)

        best_seen = min(best_seen, m)
        if m <= margin_keep:
            kept.append(cand)
            if len(kept) % 10 == 0 or len(kept) == 1:
                print(f"[farm] kept {len(kept)}/{want}  margin={m:.3e}  best_seen={best_seen:.3e}")
            if len(kept) >= want:
                break

        if i % 200 == 0:
            print(f"[farm] tried {i}/{tries}  kept {len(kept)}/{want}  best_seen={best_seen:.3e}")

    return kept


# ----------------------------
# Zip creation (no subfolders)
# ----------------------------
def make_zip(zip_path: Path, png_paths: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for p in png_paths:
            z.write(p, arcname=p.name)


def main():
    ap = argparse.ArgumentParser(description="Generate near-boundary PNGs and zip for Chimera submission.")
    ap.add_argument("--model", type=Path, default=Path("model.pt"))
    ap.add_argument("--base", type=Path, default=Path("images/000.png"),
                    help="Base image path (32x32 RGB).")
    ap.add_argument("--outdir", type=Path, default=Path("submission_images"))
    ap.add_argument("--zip", type=Path, default=Path("my_submission.zip"))
    ap.add_argument("--n", type=int, default=100, help="How many PNGs to output (<=1000).")

    # boundary search params
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--margin_stop", type=float, default=1e-6)
    ap.add_argument("--check_every", type=int, default=10)

    # farming params
    ap.add_argument("--margin_keep", type=float, default=5e-6)
    ap.add_argument("--farm_tries", type=int, default=20000)
    ap.add_argument("--n_edits", type=int, default=6)
    ap.add_argument("--max_delta", type=int, default=1)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    assert 1 <= args.n <= 1000

    # Load model
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.eval()

    # Load base image
    base_u8 = tensor_to_u8(load_png_to_tensor(args.base))
    ensure_rgb32(base_u8)

    print(f"Base: {args.base}")
    base_logits = logits_from_u8(model, base_u8)
    c, t, m = top1_top2_margin(base_logits)
    print(f"[start] top1={c} top2={t} margin={m:.3e}")

    # Step A: push a single image to boundary
    near_u8, info = push_to_boundary(
        model=model,
        start_u8=base_u8,
        steps=args.steps,
        lr=args.lr,
        margin_stop=args.margin_stop,
        check_every=args.check_every,
        seed=args.seed,
    )
    print(f"[best] {info}")

    # Save the near-boundary seed
    seed_path = args.outdir / "seed_near_boundary.png"
    save_png_u8(near_u8, seed_path)

    # Step B: farm many near-boundary variants for submission
    want = args.n
    print(f"\nFarming {want} images with margin_keep={args.margin_keep:.1e} ...")
    kept = farm_near_boundary(
        model=model,
        seed_u8=near_u8,
        want=want,
        margin_keep=args.margin_keep,
        tries=args.farm_tries,
        n_edits=args.n_edits,
        max_delta=args.max_delta,
        seed=args.seed + 1337,
    )

    if len(kept) == 0:
        print("\n[WARN] Farm kept 0 images. Try loosening --margin_keep (e.g. 1e-4) or increasing --steps.")
        print("Still writing just the near-boundary seed as 000.png so you can submit/test.")
        kept = [near_u8]

    # Write numbered PNGs
    png_paths = []
    for i, u8 in enumerate(kept[:args.n]):
        name = f"{i:03d}.png"
        p = args.outdir / name
        save_png_u8(u8, p)
        png_paths.append(p)

    # Build zip (must contain only PNG files, no directories)
    # We create the zip in the *challenge* folder by default (args.zip)
    make_zip(args.zip, png_paths)
    size_mb = args.zip.stat().st_size / (1024 * 1024)
    print(f"\nCreated {args.zip} with {len(png_paths)} PNGs ({size_mb:.2f} MB).")
    print("Next: run your submission script (sample_submission.py) from this folder.")

if __name__ == "__main__":
    main()
