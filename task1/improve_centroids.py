"""
improve_centroids.py — Multi-seed centroid improvement.

Keeps Variant C composition (~57c + 43m) but improves centroid quality by:
1. Running k-means with many different seeds
2. Collecting all centroids + medoids across seeds
3. Picking the best per class by proxy MSE (nearest aux distance)
4. Replacing only the weakest images from Variant C

NO API calls. Output-only.

USAGE:
  python task1/improve_centroids.py                # Generate improved submission
  python task1/improve_centroids.py --submit       # Generate and submit
"""

import torch
import numpy as np
import requests
import os
import json
import argparse
from collections import Counter

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "http://35.192.205.84:80"
API_KEY = "c8286483e3f08d5579bea4e972a7d21b"
TASK_ID = "12-data-reconstruction"


def load_auxiliary():
    dataset = torch.load(os.path.join(WORK_DIR, "auxiliary_dataset.pt"), weights_only=False)
    images = dataset["images"].numpy().astype(np.float32)
    labels = dataset["labels"].numpy()
    return images, labels


def compute_mse(img1, img2):
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def kmeans_one_class(cls_images, k, seed, max_iter=30):
    """Run k-means on one class with given seed. Returns centroids, assignments."""
    n_k = min(k, len(cls_images))
    flat = cls_images.reshape(len(cls_images), -1)

    rng = np.random.RandomState(seed=seed)
    init_idx = rng.choice(len(flat), n_k, replace=False)
    centers = flat[init_idx].copy()

    for _ in range(max_iter):
        dists = np.zeros((len(flat), n_k), dtype=np.float32)
        for ki in range(n_k):
            diff = flat - centers[ki]
            dists[:, ki] = np.sum(diff ** 2, axis=1)
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
    return centroids, assignments


def get_medoid(cls_images, assignments, ki):
    """Get medoid for cluster ki."""
    flat = cls_images.reshape(len(cls_images), -1)
    mask = assignments == ki
    if not np.any(mask):
        return None
    cluster_flat = flat[mask]
    cluster_imgs = cls_images[mask]
    center = cluster_flat.mean(axis=0)
    dists = np.sum((cluster_flat - center) ** 2, axis=1)
    return cluster_imgs[int(np.argmin(dists))].copy()


def score_image(img, cls_aux_flat):
    """Score an image by MSE to nearest same-class aux image (lower = better)."""
    flat = img.reshape(1, -1)
    mses = np.mean((cls_aux_flat - flat) ** 2, axis=1)
    return float(np.min(mses))


def is_near_dup(img, selected, threshold=0.0005):
    for s in selected:
        if compute_mse(img, s) < threshold:
            return True
    return False


def save_submission(images_np, filename):
    path = os.path.join(WORK_DIR, filename)
    images_np = np.clip(images_np, 0.0, 1.0).astype(np.float32)
    assert images_np.shape == (100, 3, 32, 32), f"Bad shape: {images_np.shape}"
    np.savez_compressed(path, images=images_np)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def submit_solution(npz_path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=50,
                        help="Number of random seeds to try per class")
    parser.add_argument("--k", type=int, default=10,
                        help="k-means k per seed")
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-SEED CENTROID IMPROVEMENT")
    print(f"  Seeds: {args.n_seeds}, k={args.k}")
    print("=" * 70)

    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images")

    # Target composition matching Variant C: ~6 centroids + ~4 medoids per class
    # (C had 57c+43m total, roughly 5-6c + 4-5m per class)
    # We'll use the same class-level ratios C had
    # C's per-class: 0:6c+4m, 1:5c+5m, 2:6c+4m, 3:6c+4m, 4:5c+5m,
    #                5:5c+5m, 6:6c+4m, 7:6c+4m, 8:5c+5m, 9:7c+3m
    # Simplified: target 6 centroids + 4 medoids per class (60c+40m)
    # But let's be flexible and just pick the best 10 per class from the full pool

    # Step 1: Generate candidate pool from many seeds
    print(f"\nGenerating candidates from {args.n_seeds} seeds...")

    class_pools = {cls: [] for cls in range(10)}  # list of (image, source_type, proxy_score)

    for cls in range(10):
        cls_mask = labels == cls
        cls_images = images[cls_mask]
        cls_aux_flat = cls_images.reshape(len(cls_images), -1)

        for seed in range(args.n_seeds):
            centroids, assignments = kmeans_one_class(cls_images, args.k, seed=seed)

            for ki in range(len(centroids)):
                score = score_image(centroids[ki], cls_aux_flat)
                class_pools[cls].append((centroids[ki], "centroid", score, seed))

                medoid = get_medoid(cls_images, assignments, ki)
                if medoid is not None:
                    mscore = score_image(medoid, cls_aux_flat)
                    class_pools[cls].append((medoid, "medoid", mscore, seed))

        print(f"  Class {cls}: {len(class_pools[cls])} candidates")

    # Step 2: For each class, pick best 10 with composition target ~6c+4m
    # Strategy: pick best candidates overall, but ensure at least 4 centroids
    # and at least 3 medoids per class for balance
    print(f"\nSelecting best 10 per class (min 4 centroids, min 3 medoids)...")

    MIN_CENTROIDS = 4
    MIN_MEDOIDS = 3
    N_PER_CLASS = 10

    final_images = []
    final_labels = []
    final_sources = []

    for cls in range(10):
        pool = class_pools[cls]
        # Sort by proxy score (ascending = better)
        pool.sort(key=lambda x: x[2])

        # Separate centroids and medoids
        centroid_pool = [(img, src, sc, sd) for img, src, sc, sd in pool if src == "centroid"]
        medoid_pool = [(img, src, sc, sd) for img, src, sc, sd in pool if src == "medoid"]

        selected = []
        selected_sources = []

        # First: pick MIN_CENTROIDS best centroids (with dedup)
        for img, src, sc, sd in centroid_pool:
            if len([s for s in selected_sources if s == "centroid"]) >= MIN_CENTROIDS:
                break
            if not is_near_dup(img, selected):
                selected.append(img)
                selected_sources.append("centroid")

        # Second: pick MIN_MEDOIDS best medoids (with dedup)
        for img, src, sc, sd in medoid_pool:
            if len([s for s in selected_sources if s == "medoid"]) >= MIN_MEDOIDS:
                break
            if not is_near_dup(img, selected):
                selected.append(img)
                selected_sources.append("medoid")

        # Fill remaining from combined pool (best proxy score, any type)
        remaining_needed = N_PER_CLASS - len(selected)
        if remaining_needed > 0:
            combined = centroid_pool + medoid_pool
            combined.sort(key=lambda x: x[2])
            for img, src, sc, sd in combined:
                if len(selected) >= N_PER_CLASS:
                    break
                if not is_near_dup(img, selected):
                    selected.append(img)
                    selected_sources.append(src)

        sources_count = Counter(selected_sources)
        scores = [score_image(img, images[labels == cls].reshape(-1, 3072))
                  for img in selected]
        avg_score = np.mean(scores)
        print(f"  Class {cls}: {dict(sources_count)}, avg proxy MSE: {avg_score:.6f}")

        for img, src in zip(selected, selected_sources):
            final_images.append(img)
            final_labels.append(cls)
            final_sources.append(src)

    final_images = np.clip(np.array(final_images[:100], dtype=np.float32), 0.0, 1.0)
    final_labels = np.array(final_labels[:100])

    total_sources = Counter(final_sources)
    print(f"\n  Total composition: {dict(total_sources)}")

    # Step 3: Compare with Variant C
    print(f"\n{'='*70}")
    print("COMPARISON WITH VARIANT C")
    print("=" * 70)

    # Compute proxy metrics for new submission
    all_nearest = []
    for i, img in enumerate(final_images):
        cls = final_labels[i]
        cls_aux_flat = images[labels == cls].reshape(-1, 3072)
        all_nearest.append(score_image(img, cls_aux_flat))
    new_avg = np.mean(all_nearest)

    # Load Variant C
    vc_path = os.path.join(WORK_DIR, "submission_ablation_C.npz")
    if os.path.exists(vc_path):
        vc_data = np.load(vc_path)["images"]
        vc_nearest = []
        for i, img in enumerate(vc_data):
            cls = i // 10  # C had 10 per class in order
            cls_aux_flat = images[labels == cls].reshape(-1, 3072)
            vc_nearest.append(score_image(img, cls_aux_flat))
        vc_avg = np.mean(vc_nearest)

        print(f"  Variant C      MSE→NearAux: {vc_avg:.6f}")
        print(f"  Multi-seed     MSE→NearAux: {new_avg:.6f}")
        print(f"  Delta:                       {new_avg - vc_avg:+.6f}")

        # Per-class comparison
        print(f"\n  Per-class MSE→NearAux:")
        print(f"  {'Cls':<5} {'Var C':>10} {'New':>10} {'Delta':>10}")
        print(f"  {'-'*37}")
        for cls in range(10):
            c_cls = np.mean(vc_nearest[cls*10:(cls+1)*10])
            n_cls = np.mean(all_nearest[cls*10:(cls+1)*10])
            marker = " <-- worse" if n_cls > c_cls else ""
            print(f"  {cls:<5} {c_cls:>10.6f} {n_cls:>10.6f} {n_cls-c_cls:>+10.6f}{marker}")
    else:
        print(f"  New MSE→NearAux: {new_avg:.6f}")
        print(f"  (Variant C not found for comparison)")

    # Diversity check
    all_div = []
    for cls in range(10):
        cls_imgs = final_images[final_labels == cls]
        div = 0.0
        cnt = 0
        for i in range(len(cls_imgs)):
            for j in range(i+1, len(cls_imgs)):
                div += compute_mse(cls_imgs[i], cls_imgs[j])
                cnt += 1
        all_div.append(div / max(cnt, 1))
    print(f"\n  Avg intra-class diversity: {np.mean(all_div):.6f}")

    # Save
    path = save_submission(final_images, "submission_multiseed.npz")

    if args.submit:
        submit_solution(path)


if __name__ == "__main__":
    main()
