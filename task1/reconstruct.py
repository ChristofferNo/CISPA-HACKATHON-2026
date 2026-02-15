"""
reconstruct.py — Phase 2: Optimization-based image reconstruction.

If Phase 1 (membership inference in solve.py) didn't find a clear gap,
this script iteratively optimizes candidate images using the logits API
to reconstruct the 100 training images.

STRATEGY:
  1. Seed 100 candidates from the best auxiliary images (10 per class)
  2. Each iteration: perturb candidates, query API, keep improvements
  3. Auto-handles rate limiting (10 min between calls)
  4. Saves progress after each iteration so you can interrupt and resume

USAGE:
  python task1/reconstruct.py                    # Start fresh (seeds from auxiliary)
  python task1/reconstruct.py --resume           # Resume from saved progress
  python task1/reconstruct.py --max-iters 50     # Limit iterations
"""

import torch
import numpy as np
import requests
import os
import sys
import time
import json
import tempfile
import argparse
import copy

# ================================
# CONFIGURATION
# ================================
BASE_URL = "http://35.192.205.84:80"
API_KEY = "c8286483e3f08d5579bea4e972a7d21b"
TASK_ID = "12-data-reconstruction"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

PROGRESS_FILE = os.path.join(WORK_DIR, "reconstruct_progress.json")
CANDIDATES_FILE = os.path.join(WORK_DIR, "reconstruct_candidates.npz")

# Perturbation parameters
INITIAL_NOISE_STD = 0.05     # Starting noise magnitude
MIN_NOISE_STD = 0.005        # Minimum noise after decay
NOISE_DECAY = 0.995          # Decay per iteration
PATCH_SIZE = 4               # Size of patches for patch-swap perturbations
N_PERTURB_PIXELS = 50        # Number of pixels to perturb per image per iteration
INTERPOLATION_WEIGHT = 0.1   # How much to interpolate with another same-class image


def query_logits(images_np):
    """
    Send images to the logits API. Returns the JSON response.
    Automatically retries if rate-limited (429).
    """
    import re

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
        np.savez_compressed(tmp_path, images=images_np.astype(np.float32))

    try:
        while True:
            with open(tmp_path, "rb") as f:
                files = {"npz": (tmp_path, f, "application/octet-stream")}
                response = requests.post(
                    f"{BASE_URL}/{TASK_ID}/logits",
                    files=files,
                    headers={"X-API-Key": API_KEY},
                    timeout=(10, 300),
                )

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                try:
                    detail = response.json().get("detail", "")
                    match = re.search(r"(\d+)\s*seconds", detail)
                    wait = int(match.group(1)) + 5 if match else 605
                except Exception:
                    wait = 605
                print(f"  Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
                print(f"  Retrying...")
                continue

            else:
                print(f"  API error {response.status_code}: {response.text}")
                return None
    finally:
        os.unlink(tmp_path)


def compute_scores(api_result, true_labels):
    """
    Extract per-image confidence scores from API response.
    Returns list of (confidence, loss) tuples.
    """
    results_list = api_result.get("results", [])
    scores = []

    for i, entry in enumerate(results_list):
        true_label = int(true_labels[i])

        if "logits" in entry:
            logit_vec = np.array(entry["logits"], dtype=np.float64)
            logit_vec = logit_vec - logit_vec.max()
            probs = np.exp(logit_vec) / np.exp(logit_vec).sum()
            confidence = probs[true_label]
            loss = -np.log(confidence + 1e-10)
        elif "probabilities" in entry:
            probs = np.array(entry["probabilities"], dtype=np.float64)
            confidence = probs[true_label]
            loss = -np.log(confidence + 1e-10)
        elif "loss" in entry:
            loss = float(entry["loss"])
            confidence = np.exp(-loss)
        elif "confidence" in entry:
            confidence = float(entry["confidence"])
            loss = -np.log(confidence + 1e-10)
        else:
            confidence = 0.0
            loss = 999.0

        scores.append((confidence, loss))

    return scores


def seed_candidates(aux_images, aux_labels, scores_file=None):
    """
    Create initial 100 candidates: 10 per class, choosing the most confident
    auxiliary images for each class.

    If confidence_scores.json exists (from Phase 1), use those scores.
    Otherwise, pick random samples per class.
    """
    n_classes = 10
    n_per_class = 10

    # Try to load Phase 1 scores
    class_scores = {c: [] for c in range(n_classes)}

    if scores_file and os.path.exists(scores_file):
        print(f"  Loading Phase 1 scores from {scores_file}...")
        with open(scores_file, "r") as f:
            data = json.load(f)
        for entry in data["scores"]:
            idx = entry["index"]
            label = int(aux_labels[idx])
            class_scores[label].append((idx, entry["confidence"], entry["loss"]))

        # Sort each class by loss (ascending = most confident first)
        for c in class_scores:
            class_scores[c].sort(key=lambda x: x[2])
    else:
        print("  No Phase 1 scores found. Using random seeds per class.")
        for idx in range(len(aux_labels)):
            label = int(aux_labels[idx])
            class_scores[label].append((idx, 0.0, 0.0))
        # Shuffle for random selection
        for c in class_scores:
            np.random.shuffle(class_scores[c])

    # Pick top n_per_class from each class
    selected_indices = []
    candidate_labels = []
    for c in range(n_classes):
        available = class_scores[c][:n_per_class]
        for idx, _, _ in available:
            selected_indices.append(idx)
            candidate_labels.append(c)
        # Pad if not enough images in this class
        while len(selected_indices) % n_per_class != 0 or len(selected_indices) < (c + 1) * n_per_class:
            # Duplicate the best one
            if available:
                selected_indices.append(available[0][0])
                candidate_labels.append(c)
            else:
                break

    # Limit to 100
    selected_indices = selected_indices[:100]
    candidate_labels = candidate_labels[:100]

    candidates = aux_images[selected_indices].copy()
    candidate_labels = np.array(candidate_labels, dtype=np.int64)

    print(f"  Seeded {len(candidates)} candidates from {n_classes} classes")
    for c in range(n_classes):
        count = (candidate_labels == c).sum()
        print(f"    Class {c}: {count} candidates")

    return candidates, candidate_labels, selected_indices


def perturb_candidates(candidates, candidate_labels, aux_images, aux_labels,
                       noise_std, iteration):
    """
    Generate perturbed versions of candidates using multiple strategies.

    Strategies (randomly chosen per image):
    1. Gaussian noise addition
    2. Pixel-level perturbation (random pixels)
    3. Patch swap from same-class auxiliary image
    4. Interpolation with another same-class auxiliary image
    """
    n = len(candidates)
    perturbed = candidates.copy()

    for i in range(n):
        strategy = np.random.choice(["noise", "pixel", "patch", "interpolate"],
                                     p=[0.3, 0.3, 0.2, 0.2])
        label = int(candidate_labels[i])

        if strategy == "noise":
            # Add Gaussian noise to entire image
            noise = np.random.randn(*perturbed[i].shape).astype(np.float32) * noise_std
            perturbed[i] = perturbed[i] + noise

        elif strategy == "pixel":
            # Perturb random pixels
            c, h, w = perturbed[i].shape
            n_pixels = min(N_PERTURB_PIXELS, h * w)
            for _ in range(n_pixels):
                ch = np.random.randint(0, c)
                py = np.random.randint(0, h)
                px = np.random.randint(0, w)
                perturbed[i][ch, py, px] += np.random.randn() * noise_std * 2

        elif strategy == "patch":
            # Swap a random patch with one from a same-class auxiliary image
            same_class = np.where(aux_labels == label)[0]
            if len(same_class) > 0:
                donor_idx = np.random.choice(same_class)
                _, h, w = perturbed[i].shape
                py = np.random.randint(0, max(1, h - PATCH_SIZE))
                px = np.random.randint(0, max(1, w - PATCH_SIZE))
                # Blend patch (don't fully replace — blend with weight)
                blend = 0.3 + np.random.rand() * 0.4  # 0.3 to 0.7
                perturbed[i][:, py:py+PATCH_SIZE, px:px+PATCH_SIZE] = (
                    (1 - blend) * perturbed[i][:, py:py+PATCH_SIZE, px:px+PATCH_SIZE] +
                    blend * aux_images[donor_idx][:, py:py+PATCH_SIZE, px:px+PATCH_SIZE]
                )

        elif strategy == "interpolate":
            # Interpolate with a random same-class auxiliary image
            same_class = np.where(aux_labels == label)[0]
            if len(same_class) > 0:
                donor_idx = np.random.choice(same_class)
                weight = INTERPOLATION_WEIGHT * (0.5 + np.random.rand())
                perturbed[i] = (1 - weight) * perturbed[i] + weight * aux_images[donor_idx]

    # Clip to valid range [0, 1]
    perturbed = np.clip(perturbed, 0.0, 1.0)
    return perturbed


def save_progress(candidates, candidate_labels, best_losses, iteration, noise_std):
    """Save current state so we can resume later."""
    np.savez_compressed(CANDIDATES_FILE,
                        images=candidates.astype(np.float32),
                        labels=candidate_labels)

    progress = {
        "iteration": iteration,
        "noise_std": float(noise_std),
        "best_losses": [float(l) for l in best_losses],
        "avg_loss": float(np.mean(best_losses)),
        "min_loss": float(np.min(best_losses)),
        "max_loss": float(np.max(best_losses)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

    # Also save a submission-ready file
    submission_path = os.path.join(WORK_DIR, "submission.npz")
    np.savez_compressed(submission_path, images=candidates.astype(np.float32))

    print(f"  Progress saved (iter {iteration}, avg_loss={np.mean(best_losses):.4f})")


def load_progress():
    """Load saved progress if it exists."""
    if not os.path.exists(PROGRESS_FILE) or not os.path.exists(CANDIDATES_FILE):
        return None

    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)

    data = np.load(CANDIDATES_FILE)
    candidates = data["images"]
    candidate_labels = data["labels"]

    print(f"  Resuming from iteration {progress['iteration']}")
    print(f"  Previous avg loss: {progress['avg_loss']:.4f}")
    print(f"  Noise std: {progress['noise_std']:.6f}")

    return {
        "candidates": candidates,
        "candidate_labels": candidate_labels,
        "iteration": progress["iteration"],
        "noise_std": progress["noise_std"],
        "best_losses": np.array(progress["best_losses"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Optimization-based reconstruction")
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    parser.add_argument("--max-iters", type=int, default=200, help="Maximum iterations")
    parser.add_argument("--noise-std", type=float, default=None, help="Override initial noise std")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2: Optimization-based Image Reconstruction")
    print("=" * 60)
    print()

    # Load auxiliary dataset
    print("Loading auxiliary dataset...")
    dataset = torch.load(os.path.join(WORK_DIR, "auxiliary_dataset.pt"), weights_only=False)
    aux_images = dataset["images"].numpy().astype(np.float32)  # (1000, 3, 32, 32)
    aux_labels = dataset["labels"].numpy()                      # (1000,)
    print(f"  {len(aux_images)} auxiliary images loaded")
    print()

    # Initialize or resume
    if args.resume:
        print("Attempting to resume from saved progress...")
        saved = load_progress()
        if saved is None:
            print("  No saved progress found. Starting fresh.")
            args.resume = False

    if args.resume and saved is not None:
        candidates = saved["candidates"]
        candidate_labels = saved["candidate_labels"]
        start_iter = saved["iteration"] + 1
        noise_std = saved["noise_std"]
        best_losses = saved["best_losses"]
    else:
        print("Seeding initial candidates...")
        scores_file = os.path.join(WORK_DIR, "confidence_scores.json")
        candidates, candidate_labels, seed_indices = seed_candidates(
            aux_images, aux_labels, scores_file
        )
        start_iter = 0
        noise_std = args.noise_std if args.noise_std else INITIAL_NOISE_STD
        best_losses = np.full(100, 999.0)  # Will be updated after first query

    print()
    print(f"Starting optimization loop (iters {start_iter} to {args.max_iters - 1})")
    print(f"  Noise std: {noise_std:.6f}")
    print(f"  Rate limit: ~10 min between API calls")
    print()

    for iteration in range(start_iter, args.max_iters):
        iter_start = time.time()
        print(f"--- Iteration {iteration} ---")

        if iteration == start_iter and not args.resume:
            # First iteration: just query the seeds to get baseline scores
            print("  Querying baseline scores for seeded candidates...")
            perturbed = candidates.copy()
        else:
            # Generate perturbations
            print(f"  Generating perturbations (noise_std={noise_std:.6f})...")
            perturbed = perturb_candidates(
                candidates, candidate_labels,
                aux_images, aux_labels,
                noise_std, iteration
            )

        # Query the API
        print(f"  Querying API with {len(perturbed)} candidates...")
        result = query_logits(perturbed)

        if result is None:
            print("  API call failed! Saving progress and stopping.")
            save_progress(candidates, candidate_labels, best_losses, iteration - 1, noise_std)
            break

        # Compute scores
        scores = compute_scores(result, candidate_labels)
        new_losses = np.array([loss for _, loss in scores])

        # Compare with best: keep improvements, revert others
        improved = 0
        for i in range(len(candidates)):
            if new_losses[i] < best_losses[i]:
                candidates[i] = perturbed[i]
                best_losses[i] = new_losses[i]
                improved += 1

        # Decay noise
        noise_std = max(MIN_NOISE_STD, noise_std * NOISE_DECAY)

        # Report
        elapsed = time.time() - iter_start
        print(f"  Improved: {improved}/{len(candidates)} images")
        print(f"  Avg loss: {np.mean(best_losses):.4f} | "
              f"Min: {np.min(best_losses):.4f} | "
              f"Max: {np.max(best_losses):.4f}")
        print(f"  Elapsed: {elapsed:.1f}s | Next noise_std: {noise_std:.6f}")

        # Save progress
        save_progress(candidates, candidate_labels, best_losses, iteration, noise_std)

        # Check convergence
        if improved == 0 and iteration > start_iter + 5:
            print("\n  No improvements in this iteration. Consider:")
            print("  - Increasing noise std: --noise-std 0.1")
            print("  - The current candidates may be near-optimal")

        print()

    print("=" * 60)
    print("Optimization complete!")
    print(f"  Final avg loss: {np.mean(best_losses):.4f}")
    print(f"  Submission saved to: {os.path.join(WORK_DIR, 'submission.npz')}")
    print()
    print("To submit, run:")
    print("  python task1/task_template.py  (set SUBMIT=True and FILE_PATH)")
    print("=" * 60)


if __name__ == "__main__":
    main()
