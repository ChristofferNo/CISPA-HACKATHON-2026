"""
expand_coverage.py — Replace redundant Variant C candidates with new cluster modes.

INSIGHT: Prior improvements refined existing modes but didn't discover new ones.
The leaderboard metric rewards covering MORE training images, not being closer
to the same ones. We need to reach into unexplored parts of the distribution.

STRATEGY:
  1. Load Variant C as base
  2. For each class, identify the most redundant pair (closest pair)
  3. Run k-means with many NEW seeds to discover alternative cluster modes
  4. Find candidates that are MAXIMALLY FAR from all existing images (new modes)
  5. Replace the redundant member with the most distant new candidate
  6. Repeat for 15-25 total replacements (1-3 per class)

NO API calls. NO modification of optimization state.

USAGE:
  python task1/expand_coverage.py                # Generate + compare
  python task1/expand_coverage.py --submit       # Generate and submit
"""

import torch
import numpy as np
import requests
import os
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
    flat = cls_images.reshape(len(cls_images), -1)
    mask = assignments == ki
    if not np.any(mask):
        return None
    cluster_flat = flat[mask]
    cluster_imgs = cls_images[mask]
    center = cluster_flat.mean(axis=0)
    dists = np.sum((cluster_flat - center) ** 2, axis=1)
    return cluster_imgs[int(np.argmin(dists))].copy()


def min_dist_to_set(img, img_set):
    """Min MSE from img to any image in the set."""
    if len(img_set) == 0:
        return float('inf')
    flat = img.reshape(1, -1).astype(np.float64)
    set_flat = np.array(img_set).reshape(len(img_set), -1).astype(np.float64)
    mses = np.mean((set_flat - flat) ** 2, axis=1)
    return float(np.min(mses))


def compute_class_diversity(imgs):
    n = len(imgs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += compute_mse(imgs[i], imgs[j])
            count += 1
    return total / count


def find_redundant_slots(cls_imgs, n_slots=3):
    """
    Find the n most redundant slots. A slot is "redundant" if removing it
    causes the least diversity loss (= it's close to another image).
    Returns list of (local_idx, min_dist_to_others).
    """
    n = len(cls_imgs)
    slot_scores = []
    for i in range(n):
        others = [cls_imgs[j] for j in range(n) if j != i]
        min_dist = min_dist_to_set(cls_imgs[i], others)
        slot_scores.append((i, min_dist))

    # Sort by min_dist ascending (most redundant = closest to another image)
    slot_scores.sort(key=lambda x: x[1])
    return slot_scores[:n_slots]


def proxy_mse(img, cls_aux_flat):
    flat = img.reshape(1, -1)
    mses = np.mean((cls_aux_flat - flat) ** 2, axis=1)
    return float(np.min(mses))


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
    parser.add_argument("--replacements-per-class", type=int, default=2,
                        help="Number of slots to replace per class (default 2 = 20 total)")
    parser.add_argument("--n-seeds", type=int, default=100,
                        help="Number of new seeds to explore")
    args = parser.parse_args()

    n_replace = args.replacements_per_class
    total_target = n_replace * 10

    print("=" * 70)
    print("EXPAND COVERAGE: Discover new cluster modes")
    print(f"  Replacements per class: {n_replace} ({total_target} total)")
    print(f"  Exploration seeds: {args.n_seeds}")
    print("=" * 70)

    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images")

    # Load Variant C
    vc_path = os.path.join(WORK_DIR, "submission_ablation_C.npz")
    if not os.path.exists(vc_path):
        print("ERROR: submission_ablation_C.npz not found!")
        return
    vc_images = np.load(vc_path)["images"].copy()
    print(f"Loaded Variant C: {vc_images.shape}")

    new_images = vc_images.copy()
    total_replaced = 0

    for cls in range(10):
        cls_aux = images[labels == cls]
        cls_aux_flat = cls_aux.reshape(len(cls_aux), -1)
        cls_start = cls * 10
        cls_imgs = new_images[cls_start:cls_start + 10]

        # Step 1: Find most redundant slots
        redundant = find_redundant_slots(cls_imgs, n_slots=n_replace)
        print(f"\n  Class {cls}: redundant slots = "
              f"{[(idx, f'{dist:.6f}') for idx, dist in redundant]}")

        # Step 2: Discover new modes from many seeds
        # Collect centroids and medoids from diverse seeds
        new_candidates = []  # (image, source_str)
        for seed in range(100, 100 + args.n_seeds):
            # Try different k values too for mode diversity
            for k in [5, 8, 10, 15, 20]:
                centroids, assignments = kmeans_one_class(cls_aux, k, seed=seed)
                for ki in range(len(centroids)):
                    new_candidates.append((centroids[ki], f"centroid_s{seed}_k{k}"))
                    medoid = get_medoid(cls_aux, assignments, ki)
                    if medoid is not None:
                        new_candidates.append((medoid, f"medoid_s{seed}_k{k}"))

        print(f"    Generated {len(new_candidates)} new candidates")

        # Step 3: Score candidates by DISTANCE from ALL current class images
        # We want candidates that explore NEW parts of the space
        scored = []
        for cand_img, cand_src in new_candidates:
            dist = min_dist_to_set(cand_img, cls_imgs)
            scored.append((cand_img, cand_src, dist))

        # Sort by distance descending (most novel first)
        scored.sort(key=lambda x: -x[2])

        # Step 4: Replace redundant slots with most novel candidates
        for slot_idx, slot_dist in redundant:
            global_idx = cls_start + slot_idx
            old_img = new_images[global_idx].copy()
            old_proxy = proxy_mse(old_img, cls_aux_flat)

            # Find best replacement: most distant from remaining images after removal
            remaining = [new_images[cls_start + i] for i in range(10) if i != slot_idx]

            best_cand = None
            best_src = None
            best_dist = -1.0

            for cand_img, cand_src, _ in scored[:200]:  # check top 200 most novel
                # Must not be near-duplicate of any remaining image
                d = min_dist_to_set(cand_img, remaining)
                if d < 0.002:
                    continue
                if d > best_dist:
                    best_dist = d
                    best_cand = cand_img
                    best_src = cand_src

            if best_cand is not None:
                new_proxy = proxy_mse(best_cand, cls_aux_flat)
                new_images[global_idx] = best_cand

                # Recompute diversity
                old_div = compute_class_diversity(cls_imgs)
                new_cls_imgs = new_images[cls_start:cls_start + 10]
                new_div = compute_class_diversity(new_cls_imgs)

                print(f"    Slot {slot_idx}: replaced (proxy={old_proxy:.6f}, "
                      f"redundancy={slot_dist:.6f}) -> {best_src.split('_')[0]} "
                      f"(proxy={new_proxy:.6f}, novelty={best_dist:.6f}), "
                      f"div {old_div:.6f}->{new_div:.6f}")

                # Update cls_imgs reference for next replacement in same class
                cls_imgs = new_images[cls_start:cls_start + 10]
                total_replaced += 1
            else:
                print(f"    Slot {slot_idx}: no suitable novel candidate found")

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON ({total_replaced} images replaced)")
    print("=" * 70)

    vc_nearest = []
    new_nearest = []
    vc_divs = []
    new_divs = []

    for cls in range(10):
        cls_aux_flat = images[labels == cls].reshape(-1, 3072)
        vc_cls = vc_images[cls * 10:(cls + 1) * 10]
        new_cls = new_images[cls * 10:(cls + 1) * 10]

        for img in vc_cls:
            vc_nearest.append(proxy_mse(img, cls_aux_flat))
        for img in new_cls:
            new_nearest.append(proxy_mse(img, cls_aux_flat))

        vc_divs.append(compute_class_diversity(vc_cls))
        new_divs.append(compute_class_diversity(new_cls))

    vc_avg_mse = np.mean(vc_nearest)
    new_avg_mse = np.mean(new_nearest)
    vc_avg_div = np.mean(vc_divs)
    new_avg_div = np.mean(new_divs)

    print(f"\n{'Metric':<25} {'Variant C':>12} {'Expanded':>12} {'Delta':>12}")
    print("-" * 63)
    print(f"{'MSE→NearAux':<25} {vc_avg_mse:>12.6f} {new_avg_mse:>12.6f} {new_avg_mse-vc_avg_mse:>+12.6f}")
    print(f"{'Avg Diversity':<25} {vc_avg_div:>12.6f} {new_avg_div:>12.6f} {new_avg_div-vc_avg_div:>+12.6f}")

    print(f"\n  Per-class detail:")
    print(f"  {'Cls':<5} {'C div':>10} {'New div':>10} {'C mse':>10} {'New mse':>10}")
    print(f"  {'-'*47}")
    for cls in range(10):
        c_div = vc_divs[cls]
        n_div = new_divs[cls]
        c_mse = np.mean(vc_nearest[cls*10:(cls+1)*10])
        n_mse = np.mean(new_nearest[cls*10:(cls+1)*10])
        print(f"  {cls:<5} {c_div:>10.6f} {n_div:>10.6f} {c_mse:>10.6f} {n_mse:>10.6f}")

    # Per-image diff
    per_img_diff = np.mean((new_images.astype(np.float64) - vc_images.astype(np.float64)) ** 2,
                            axis=(1, 2, 3))
    n_changed = int(np.sum(per_img_diff > 1e-10))
    print(f"\n  Images changed: {n_changed}/100")
    print(f"  Diversity change: {new_avg_div - vc_avg_div:+.6f} ({(new_avg_div/vc_avg_div - 1)*100:+.1f}%)")

    path = save_submission(new_images, "submission_expanded.npz")

    if args.submit:
        submit_solution(path)


if __name__ == "__main__":
    main()
