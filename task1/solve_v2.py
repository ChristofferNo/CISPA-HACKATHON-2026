"""
solve_v2.py — Cluster-first data recondstruction pipeline.

KEY INSIGHT: The evaluation metric is MSE to the nearest training image.
K-means centroids minimize average squared distance to cluster members,
which directly aligns with MSE. Logit-based optimization drifts images
away from pixel-space centers, hurting MSE even when confidence rises.

PIPELINE:
  Phase A: Per-class k-means clustering (no API needed)
  Phase B: Merge with optimization progress (no API needed)
  Phase C: Light drift-bounded optimization (few API calls)
  Phase D: Assemble and submit

USAGE:
  python task1/solve_v2.py                        # Full pipeline (clustering + merge + light opt)
  python task1/solve_v2.py --skip-optimize         # Clustering + merge only (no API calls)
  python task1/solve_v2.py --submit                # Submit after pipeline
  python task1/solve_v2.py --k 15                  # Use 15 centroids per class
  python task1/solve_v2.py --use-medoids            # Use actual images instead of centroids
"""

import torch
import numpy as np
import requests
import os
import sys
import time
import json
import re
import tempfile
import argparse
from collections import Counter

# ================================
# CONFIGURATION
# ================================
BASE_URL = "http://35.192.205.84:80"
API_KEY = "c8286483e3f08d5579bea4e972a7d21b"
TASK_ID = "12-data-reconstruction"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================
# API HELPERS (copied from solve.py to be standalone)
# ================================
def query_logits(images_np):
    """Send up to 100 images to the logits API. Returns JSON or None."""
    assert len(images_np) <= 100, f"API limit is 100 images, got {len(images_np)}"
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
        np.savez_compressed(tmp_path, images=images_np.astype(np.float32))
    try:
        while True:
            with open(tmp_path, "rb") as f:
                resp = requests.post(
                    f"{BASE_URL}/{TASK_ID}/logits",
                    files={"npz": (tmp_path, f, "application/octet-stream")},
                    headers={"X-API-Key": API_KEY},
                    timeout=(10, 300),
                )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                try:
                    detail = resp.json().get("detail", "")
                    match = re.search(r"(\d+)\s*seconds", detail)
                    wait = int(match.group(1)) + 5 if match else 605
                except Exception:
                    wait = 605
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  API error {resp.status_code}: {resp.text}")
                return None
    finally:
        os.unlink(tmp_path)


def extract_scores(api_result):
    """Extract per-image scores from API response."""
    scores = []
    for entry in api_result.get("results", []):
        logits = np.array(entry["logits"], dtype=np.float64)
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
        max_class = int(np.argmax(probs))
        max_conf = float(probs[max_class])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_logit = float(logits.max())
        scores.append({
            "max_conf": max_conf, "max_class": max_class,
            "entropy": entropy, "max_logit": max_logit,
        })
    return scores


def save_submission(images_np, filename="submission_v2.npz"):
    """Save images as a valid submission file."""
    path = os.path.join(WORK_DIR, filename)
    images_np = np.clip(images_np, 0.0, 1.0).astype(np.float32)
    assert images_np.shape == (100, 3, 32, 32), f"Bad shape: {images_np.shape}"
    np.savez_compressed(path, images=images_np)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def submit_solution(npz_path):
    """Submit a .npz file to the evaluation server."""
    print(f"\nSubmitting {npz_path}...")
    with open(npz_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (os.path.basename(npz_path), f, "application/octet-stream")},
            timeout=(10, 120),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    if resp.status_code == 200:
        print(f"  Success! Response: {body}")
    else:
        print(f"  Failed ({resp.status_code}): {body}")
    return body


def load_auxiliary():
    dataset = torch.load(os.path.join(WORK_DIR, "auxiliary_dataset.pt"), weights_only=False)
    images = dataset["images"].numpy().astype(np.float32)
    labels = dataset["labels"].numpy()
    return images, labels


# ================================
# PHASE A: Per-Class Clustering
# ================================
def kmeans_per_class(images, labels, k=10, max_iter=30):
    """
    Run k-means per class. Returns:
      centroids: (10*k, 3, 32, 32) — cluster centers
      centroid_labels: (10*k,) — class label for each centroid
      assignments: dict[class] -> array of cluster assignments
      class_images: dict[class] -> images for that class
    """
    all_centroids = []
    all_labels = []
    all_assignments = {}
    all_class_images = {}

    for cls in range(10):
        cls_mask = labels == cls
        cls_imgs = images[cls_mask]
        all_class_images[cls] = cls_imgs
        if len(cls_imgs) == 0:
            continue

        n_k = min(k, len(cls_imgs))
        flat = cls_imgs.reshape(len(cls_imgs), -1)

        rng = np.random.RandomState(seed=cls * 42 + 7)
        init_idx = rng.choice(len(flat), n_k, replace=False)
        centers = flat[init_idx].copy()

        for _ in range(max_iter):
            dists = np.sum((flat[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for ki in range(n_k):
                mask = assignments == ki
                if np.any(mask):
                    new_centers[ki] = flat[mask].mean(axis=0)
                else:
                    new_centers[ki] = centers[ki]

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        centroids = np.clip(centers.reshape(n_k, 3, 32, 32), 0.0, 1.0).astype(np.float32)
        all_centroids.append(centroids)
        all_labels.extend([cls] * n_k)
        all_assignments[cls] = assignments

    centroids = np.concatenate(all_centroids, axis=0)
    centroid_labels = np.array(all_labels)
    return centroids, centroid_labels, all_assignments, all_class_images


def find_nearest_aux(centroid, class_images):
    """Find the actual auxiliary image closest (MSE) to a centroid."""
    flat_centroid = centroid.reshape(1, -1)
    flat_imgs = class_images.reshape(len(class_images), -1)
    mses = np.mean((flat_imgs - flat_centroid) ** 2, axis=1)
    best_idx = int(np.argmin(mses))
    return class_images[best_idx].copy(), float(mses[best_idx])


def find_medoids(class_images, assignments, k):
    """Find medoids (actual images closest to cluster center) for each cluster."""
    medoids = []
    flat = class_images.reshape(len(class_images), -1)
    for ki in range(k):
        mask = assignments == ki
        if not np.any(mask):
            continue
        cluster_flat = flat[mask]
        cluster_imgs = class_images[mask]
        # Medoid = point with minimum total distance to all other points in cluster
        if len(cluster_flat) == 1:
            medoids.append(cluster_imgs[0].copy())
        else:
            center = cluster_flat.mean(axis=0)
            dists = np.sum((cluster_flat - center) ** 2, axis=1)
            medoids.append(cluster_imgs[int(np.argmin(dists))].copy())
    return medoids


# ================================
# PHASE B: Merge with Optimization Progress
# ================================
def load_optimized_candidates():
    """Load optimized candidates if available. Returns (images, labels) or None."""
    candidates_file = os.path.join(WORK_DIR, "optimize_candidates.npz")
    progress_file = os.path.join(WORK_DIR, "optimize_progress.json")

    if not os.path.exists(candidates_file) or not os.path.exists(progress_file):
        return None

    data = np.load(candidates_file)
    candidates = data["images"].copy()

    with open(progress_file) as f:
        progress = json.load(f)
    candidate_labels = np.array(progress.get("candidate_labels", [0] * len(candidates)))
    best_scores = np.array(progress.get("best_scores", [0.0] * len(candidates)))

    print(f"  Loaded {len(candidates)} optimized candidates (avg score: {best_scores.mean():.4f})")
    return candidates, candidate_labels, best_scores


def compute_mse(img1, img2):
    """MSE between two images."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def merge_candidates(centroids, centroid_labels, class_images,
                     optimized=None, opt_labels=None, opt_scores=None,
                     use_medoids=False, assignments=None, k=10,
                     n_per_class=10):
    """
    Merge clustering results with optimization progress.

    For each class, build a candidate pool from:
      1. Centroids (or medoids)
      2. Nearest-neighbor auxiliary images to each centroid
      3. Optimized candidates that are close to cluster centers

    Select best n_per_class per class by diversity + proximity scoring.
    """
    final_images = []
    final_labels = []

    for cls in range(10):
        cls_centroids = centroids[centroid_labels == cls]
        cls_aux = class_images.get(cls, np.array([]))
        pool = []  # list of (image, source, score)

        # Source 1: Centroids
        for i, c in enumerate(cls_centroids):
            pool.append((c, f"centroid_{i}", 0.0))

        # Source 2: Medoids (actual images closest to cluster center)
        if use_medoids and cls in assignments and len(cls_aux) > 0:
            medoids = find_medoids(cls_aux, assignments[cls], min(k, len(cls_aux)))
            for i, m in enumerate(medoids):
                pool.append((m, f"medoid_{i}", 0.0))

        # Source 3: Nearest auxiliary image to each centroid
        if len(cls_aux) > 0:
            for i, c in enumerate(cls_centroids):
                nn_img, nn_mse = find_nearest_aux(c, cls_aux)
                pool.append((nn_img, f"nn_centroid_{i}", nn_mse))

        # Source 4: Optimized candidates close to centroids
        if optimized is not None and opt_labels is not None:
            opt_mask = opt_labels == cls
            opt_cls = optimized[opt_mask]
            opt_cls_scores = opt_scores[opt_mask] if opt_scores is not None else np.zeros(opt_mask.sum())

            for i, opt_img in enumerate(opt_cls):
                # Compute MSE to nearest centroid
                if len(cls_centroids) > 0:
                    mses_to_centroids = [compute_mse(opt_img, c) for c in cls_centroids]
                    min_mse = min(mses_to_centroids)
                else:
                    min_mse = 999.0

                # Only include if reasonably close to a centroid
                # Threshold: median MSE between centroids and their nearest aux images
                pool.append((opt_img, f"optimized_{i}", min_mse))

        # Deduplicate: remove images that are too similar (MSE < 0.0001)
        unique_pool = []
        for img, src, score in pool:
            is_dup = False
            for existing_img, _, _ in unique_pool:
                if compute_mse(img, existing_img) < 0.0001:
                    is_dup = True
                    break
            if not is_dup:
                unique_pool.append((img, src, score))
        pool = unique_pool

        # Select best n_per_class using diversity-aware selection
        selected = select_diverse(pool, n_per_class, cls_centroids)

        for img, src, _ in selected:
            final_images.append(img)
            final_labels.append(cls)

        n_selected = len(selected)
        sources = Counter(src.split("_")[0] for _, src, _ in selected)
        print(f"  Class {cls}: {n_selected} selected — {dict(sources)}")

    return np.array(final_images, dtype=np.float32), np.array(final_labels)


def select_diverse(pool, n, centroids):
    """
    Select n images from pool maximizing diversity.

    Strategy: greedy farthest-point sampling seeded from centroids.
    This ensures selected images are spread across the image space,
    covering different modes/clusters.
    """
    if len(pool) <= n:
        return pool

    # Score each candidate: prefer images that are close to centroids
    # (good MSE representatives) but far from already-selected images (diversity)
    scored = []
    for img, src, mse_score in pool:
        # Proximity to nearest centroid (lower = better representative)
        if len(centroids) > 0:
            centroid_dists = [compute_mse(img, c) for c in centroids]
            proximity = min(centroid_dists)
        else:
            proximity = 0.0

        # Prefer centroids and medoids (source bonus)
        source_type = src.split("_")[0]
        if source_type == "centroid":
            source_bonus = 0.0  # best
        elif source_type == "medoid":
            source_bonus = 0.001
        elif source_type == "nn":
            source_bonus = 0.002
        else:  # optimized
            source_bonus = proximity  # penalize based on drift

        scored.append((img, src, mse_score, source_bonus))

    # Sort by source_bonus (ascending = better)
    scored.sort(key=lambda x: x[3])

    # Greedy diverse selection
    selected = [scored[0]]  # start with best-scored
    remaining = scored[1:]

    while len(selected) < n and remaining:
        # Find candidate most distant from all already-selected
        best_idx = -1
        best_min_dist = -1

        for i, (img, src, mse, sb) in enumerate(remaining):
            min_dist = min(compute_mse(img, sel_img) for sel_img, _, _, _ in selected)
            # Combined score: diversity (higher=better) - proximity penalty
            combined = min_dist - sb * 0.1
            if combined > best_min_dist:
                best_min_dist = combined
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    # Convert back to (img, src, score) format
    return [(img, src, mse) for img, src, mse, _ in selected[:n]]


# ================================
# PHASE C: Light Drift-Bounded Optimization
# ================================
def light_optimize(candidates, candidate_labels, images, labels,
                   reference_images, max_iters=5, noise_std=0.01,
                   drift_threshold=0.01):
    """
    Light optimization with drift prevention.

    Only perturbs candidates with small noise, and rejects any perturbation
    that moves the image too far from its reference (original centroid/seed).

    Args:
        candidates: (100, 3, 32, 32) current candidates
        candidate_labels: (100,) class labels
        images: (1000, 3, 32, 32) all auxiliary images
        labels: (1000,) all aux labels
        reference_images: (100, 3, 32, 32) original seeds (centroids/medoids)
        max_iters: max optimization iterations
        noise_std: perturbation magnitude (small!)
        drift_threshold: max MSE from reference before rejecting
    """
    print(f"\n  Light optimization: {max_iters} iters, noise={noise_std}, drift_limit={drift_threshold}")

    best_scores = np.full(100, -999.0)

    for iteration in range(max_iters):
        # Generate 1 perturbation per image (light touch)
        perturbed = candidates.copy()
        for i in range(100):
            noise = np.random.randn(*perturbed[i].shape).astype(np.float32) * noise_std
            perturbed[i] = np.clip(perturbed[i] + noise, 0.0, 1.0)

            # Drift check: reject if too far from reference
            if reference_images is not None:
                drift = compute_mse(perturbed[i], reference_images[i])
                if drift > drift_threshold:
                    perturbed[i] = candidates[i]  # revert

        # Query API
        result = query_logits(perturbed)
        if result is None:
            print(f"  API failed at iter {iteration}. Stopping.")
            break

        scores = extract_scores(result)

        # Accept improvements only
        improved = 0
        for i in range(100):
            new_score = scores[i]["max_conf"]
            if best_scores[i] == -999.0:
                best_scores[i] = new_score
                candidates[i] = perturbed[i]
            elif new_score > best_scores[i]:
                candidates[i] = perturbed[i]
                best_scores[i] = new_score
                improved += 1

        avg = float(np.mean(best_scores))
        print(f"  Iter {iteration}: improved {improved}/100, avg_conf={avg:.4f}")

    return candidates


# ================================
# MAIN PIPELINE
# ================================
def run_pipeline(k=10, use_medoids=False, skip_optimize=False,
                 max_opt_iters=5, noise_std=0.01, drift_threshold=0.01,
                 do_submit=False):
    """Run the full cluster-first pipeline."""

    print("=" * 60)
    print("solve_v2: Cluster-First Reconstruction Pipeline")
    print("=" * 60)

    # Load data
    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images ({images.shape})")

    # --- PHASE A: Clustering ---
    print(f"\n--- Phase A: Per-class k-means clustering (k={k}) ---")
    centroids, centroid_labels, assignments, class_images = kmeans_per_class(images, labels, k=k)
    print(f"  Computed {len(centroids)} centroids across {len(set(centroid_labels.tolist()))} classes")

    # --- PHASE B: Merge with optimization progress ---
    print("\n--- Phase B: Merge with optimization progress ---")
    opt_data = load_optimized_candidates()
    if opt_data is not None:
        opt_imgs, opt_labels, opt_scores = opt_data
    else:
        opt_imgs = opt_labels = opt_scores = None
        print("  No optimized candidates found. Using clustering only.")

    candidates, candidate_labels = merge_candidates(
        centroids, centroid_labels, class_images,
        optimized=opt_imgs, opt_labels=opt_labels, opt_scores=opt_scores,
        use_medoids=use_medoids, assignments=assignments, k=k,
        n_per_class=10,
    )

    assert len(candidates) == 100, f"Expected 100 candidates, got {len(candidates)}"
    print(f"\n  Merged candidates: {len(candidates)} images")
    print(f"  Class distribution: {dict(sorted(Counter(candidate_labels.tolist()).items()))}")

    # Save reference images for drift prevention
    reference_images = candidates.copy()

    # --- PHASE C: Light optimization ---
    if not skip_optimize:
        print("\n--- Phase C: Light drift-bounded optimization ---")
        candidates = light_optimize(
            candidates, candidate_labels, images, labels,
            reference_images=reference_images,
            max_iters=max_opt_iters, noise_std=noise_std,
            drift_threshold=drift_threshold,
        )
    else:
        print("\n--- Phase C: Skipped (--skip-optimize) ---")

    # --- PHASE D: Assemble and submit ---
    print("\n--- Phase D: Assemble submission ---")
    path = save_submission(candidates, filename="submission_v2.npz")

    # Also save with different k values for comparison
    print(f"\n  Final stats:")
    print(f"    Shape: {candidates.shape}, dtype: {candidates.dtype}")
    print(f"    Value range: [{candidates.min():.3f}, {candidates.max():.3f}]")
    print(f"    Mean: {candidates.mean():.3f}")

    if do_submit:
        submit_solution(path)

    return path


# ================================
# VARIANT: Try multiple k values
# ================================
def sweep_k_values(k_values=None, use_medoids=False, do_submit=False):
    """Run pipeline with different k values, save each, optionally submit best."""
    if k_values is None:
        k_values = [5, 8, 10, 12, 15, 20]

    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images")

    opt_data = load_optimized_candidates()
    if opt_data is not None:
        opt_imgs, opt_labels, opt_scores = opt_data
    else:
        opt_imgs = opt_labels = opt_scores = None

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"  k={k}")
        print(f"{'='*60}")

        centroids, centroid_labels, assignments, class_images = kmeans_per_class(images, labels, k=k)

        candidates, candidate_labels = merge_candidates(
            centroids, centroid_labels, class_images,
            optimized=opt_imgs, opt_labels=opt_labels, opt_scores=opt_scores,
            use_medoids=use_medoids, assignments=assignments, k=k,
            n_per_class=10,
        )

        filename = f"submission_v2_k{k}.npz"
        path = save_submission(candidates, filename=filename)

    if do_submit:
        # Submit the default k=10 version
        default_path = os.path.join(WORK_DIR, "submission_v2.npz")
        if os.path.exists(default_path):
            submit_solution(default_path)


# ================================
# CLI
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster-first reconstruction pipeline")
    parser.add_argument("--k", type=int, default=10, help="Number of k-means centroids per class")
    parser.add_argument("--use-medoids", action="store_true", help="Use medoids (actual images) instead of centroids")
    parser.add_argument("--skip-optimize", action="store_true", help="Skip light optimization (no API calls)")
    parser.add_argument("--max-opt-iters", type=int, default=5, help="Max light optimization iterations")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Perturbation noise std for light optimization")
    parser.add_argument("--drift-threshold", type=float, default=0.01, help="Max MSE drift from reference")
    parser.add_argument("--submit", action="store_true", help="Submit result to server")
    parser.add_argument("--sweep", action="store_true", help="Try multiple k values")
    args = parser.parse_args()

    if args.sweep:
        sweep_k_values(use_medoids=args.use_medoids, do_submit=args.submit)
    else:
        run_pipeline(
            k=args.k,
            use_medoids=args.use_medoids,
            skip_optimize=args.skip_optimize,
            max_opt_iters=args.max_opt_iters,
            noise_std=args.noise_std,
            drift_threshold=args.drift_threshold,
            do_submit=args.submit,
        )
