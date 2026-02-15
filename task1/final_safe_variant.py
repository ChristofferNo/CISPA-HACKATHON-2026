"""
final_safe_variant.py — Minimal safe improvement over Variant C.

Strategy:
  1. Reproduce Variant C exactly
  2. Within each class, find the most similar pair of images
  3. Replace the weaker member of each such pair with the best available
     alternative that maximizes coverage (farthest from all remaining)
  4. Only replace bottom 5-10 weakest slots across all classes
  5. Verify diversity improved; abort if not

NO API calls. NO modification of optimization state.

USAGE:
  python task1/final_safe_variant.py                # Generate + compare
  python task1/final_safe_variant.py --submit       # Generate and submit
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


def proxy_mse(img, cls_aux_flat):
    flat = img.reshape(1, -1)
    mses = np.mean((cls_aux_flat - flat) ** 2, axis=1)
    return float(np.min(mses))


def compute_class_diversity(imgs):
    """Avg pairwise MSE within a set of images."""
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


def find_most_similar_pair(imgs):
    """Find the pair with lowest pairwise MSE (most redundant)."""
    n = len(imgs)
    best_mse = float('inf')
    best_i, best_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            mse = compute_mse(imgs[i], imgs[j])
            if mse < best_mse:
                best_mse = mse
                best_i, best_j = i, j
    return best_i, best_j, best_mse


def min_dist_to_set(img, imgs):
    """Min MSE from img to any image in imgs."""
    if len(imgs) == 0:
        return float('inf')
    return min(compute_mse(img, other) for other in imgs)


def reproduce_variant_c(images, labels):
    """Exact reproduction of Variant C from ablation_variants.py."""
    all_images = []
    all_labels = []
    all_sources = []
    class_data = {}  # store per-class intermediates for replacement pool

    for cls in range(10):
        cls_mask = labels == cls
        cls_images = images[cls_mask]
        cls_aux_flat = cls_images.reshape(len(cls_images), -1)

        # K-means with seed=cls (matching cluster_submission_test.py)
        centroids, assignments = kmeans_one_class(cls_images, 10, seed=cls)

        # Build pool: centroids + medoids
        pool = []
        for i, c in enumerate(centroids):
            pool.append((c, f"centroid_{i}"))

        for ki in range(len(centroids)):
            medoid = get_medoid(cls_images, assignments, ki)
            if medoid is not None:
                pool.append((medoid, f"medoid_{ki}"))

        # Score by MSE to class mean (matching ablation_variants.py build_variant_c)
        class_mean = cls_aux_flat.mean(axis=0).reshape(3, 32, 32)
        scored_pool = []
        for img, src in pool:
            mse_to_mean = compute_mse(img, class_mean)
            scored_pool.append((img, src, mse_to_mean))
        scored_pool.sort(key=lambda x: x[2])

        selected = scored_pool[:10]
        for img, src, _ in selected:
            all_images.append(img)
            all_labels.append(cls)
            all_sources.append(src)

        # Store full pool + class data for later replacement
        class_data[cls] = {
            "centroids": centroids,
            "assignments": assignments,
            "cls_images": cls_images,
            "full_pool": scored_pool,  # all candidates sorted by MSE-to-mean
        }

    all_images = np.clip(np.array(all_images, dtype=np.float32), 0.0, 1.0)
    all_labels = np.array(all_labels)
    return all_images, all_labels, all_sources, class_data


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
    parser.add_argument("--max-replacements", type=int, default=10,
                        help="Max images to replace across all classes")
    args = parser.parse_args()

    print("=" * 70)
    print("FINAL SAFE VARIANT: Minimal diversity improvement over Variant C")
    print("=" * 70)

    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images")

    # Step 1: Reproduce Variant C
    print(f"\n--- Reproducing Variant C ---")
    vc_images, vc_labels, vc_sources, class_data = reproduce_variant_c(images, labels)

    # Verify against saved file
    vc_path = os.path.join(WORK_DIR, "submission_ablation_C.npz")
    if os.path.exists(vc_path):
        saved = np.load(vc_path)["images"]
        mse = float(np.mean((vc_images.astype(np.float64) - saved.astype(np.float64)) ** 2))
        print(f"  Verification vs saved C: MSE={mse:.2e} {'[MATCH]' if mse < 1e-10 else '[MISMATCH]'}")

    # Step 2: Analyze per-class redundancy
    print(f"\n--- Analyzing per-class redundancy ---")
    replacement_candidates = []  # (cls, slot_idx, pair_mse, slot_proxy_mse)

    for cls in range(10):
        cls_imgs = vc_images[vc_labels == cls]
        cls_sources_list = [vc_sources[i] for i in range(len(vc_sources))
                           if vc_labels[i] == cls]
        cls_aux_flat = images[labels == cls].reshape(-1, 3072)

        # Find most similar pair
        idx_i, idx_j, pair_mse = find_most_similar_pair(cls_imgs)

        # Compute proxy MSE for both members
        proxy_i = proxy_mse(cls_imgs[idx_i], cls_aux_flat)
        proxy_j = proxy_mse(cls_imgs[idx_j], cls_aux_flat)

        # The weaker member (higher proxy MSE = farther from aux) is the replacement target
        if proxy_i >= proxy_j:
            target_idx = idx_i
            target_proxy = proxy_i
        else:
            target_idx = idx_j
            target_proxy = proxy_j

        global_idx = cls * 10 + target_idx
        replacement_candidates.append({
            "cls": cls,
            "local_idx": target_idx,
            "global_idx": global_idx,
            "pair_mse": pair_mse,
            "target_proxy": target_proxy,
            "source": cls_sources_list[target_idx] if target_idx < len(cls_sources_list) else "?",
        })

        diversity = compute_class_diversity(cls_imgs)
        print(f"  Class {cls}: diversity={diversity:.6f}, "
              f"most similar pair=({idx_i},{idx_j}) MSE={pair_mse:.6f}, "
              f"replace slot {target_idx} ({cls_sources_list[target_idx]}, proxy={target_proxy:.6f})")

    # Sort by pair similarity (most redundant first = most beneficial to replace)
    replacement_candidates.sort(key=lambda x: x["pair_mse"])

    # Step 3: Replace weakest slots
    n_replace = min(args.max_replacements, len(replacement_candidates))
    print(f"\n--- Replacing {n_replace} most redundant slots ---")

    new_images = vc_images.copy()
    new_sources = list(vc_sources)
    replacements_made = 0

    # Also gather additional replacement candidates from multiple seeds
    extra_seeds = list(range(20, 70))  # seeds 20-69, avoiding seed=cls used by C

    for rc in replacement_candidates[:n_replace]:
        cls = rc["cls"]
        local_idx = rc["local_idx"]
        global_idx = rc["global_idx"]
        cls_imgs_current = new_images[cls * 10:(cls + 1) * 10]
        cls_aux = images[labels == cls]
        cls_aux_flat = cls_aux.reshape(len(cls_aux), -1)

        # Build replacement pool from extra seeds
        candidates = []
        for seed in extra_seeds:
            centroids, assignments = kmeans_one_class(cls_aux, 10, seed=seed)
            for ki in range(len(centroids)):
                candidates.append((centroids[ki], "centroid"))
                medoid = get_medoid(cls_aux, assignments, ki)
                if medoid is not None:
                    candidates.append((medoid, "medoid"))

        # Find the best replacement: maximizes min-distance to remaining images
        remaining = [cls_imgs_current[i] for i in range(10) if i != local_idx]

        best_replacement = None
        best_replacement_src = None
        best_min_dist = -1.0
        best_proxy = float('inf')

        for cand_img, cand_src in candidates:
            # Skip if near-duplicate of any remaining image
            md = min_dist_to_set(cand_img, remaining)
            if md < 0.001:
                continue

            # Prefer candidates that:
            # 1. Are far from existing images (coverage)
            # 2. Have reasonable proxy MSE (not terrible)
            cand_proxy = proxy_mse(cand_img, cls_aux_flat)

            # Score: coverage-weighted, with mild proxy penalty
            score = md - cand_proxy * 0.1

            if score > best_min_dist:
                best_min_dist = score
                best_replacement = cand_img
                best_replacement_src = cand_src
                best_proxy = cand_proxy

        if best_replacement is not None:
            old_proxy = rc["target_proxy"]
            old_div = compute_class_diversity(cls_imgs_current)

            # Apply replacement
            new_images[global_idx] = best_replacement
            new_sources[global_idx] = f"{best_replacement_src}_replaced"

            new_cls_imgs = new_images[cls * 10:(cls + 1) * 10]
            new_div = compute_class_diversity(new_cls_imgs)

            div_change = new_div - old_div
            print(f"  Class {cls} slot {local_idx}: "
                  f"replaced {rc['source']} (proxy={old_proxy:.6f}) "
                  f"with {best_replacement_src} (proxy={best_proxy:.6f}), "
                  f"diversity {old_div:.6f} -> {new_div:.6f} ({div_change:+.6f})")

            if div_change < -0.001:
                # Revert if diversity got worse
                new_images[global_idx] = vc_images[global_idx]
                new_sources[global_idx] = vc_sources[global_idx]
                print(f"    -> REVERTED (diversity decreased)")
            else:
                replacements_made += 1
        else:
            print(f"  Class {cls} slot {local_idx}: no suitable replacement found")

    print(f"\n  Total replacements made: {replacements_made}")

    # Step 4: Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print("=" * 70)

    # Compute metrics for both
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

    print(f"\n{'Metric':<25} {'Variant C':>12} {'Final':>12} {'Delta':>12}")
    print("-" * 63)
    print(f"{'MSE→NearAux':<25} {vc_avg_mse:>12.6f} {new_avg_mse:>12.6f} {new_avg_mse-vc_avg_mse:>+12.6f}")
    print(f"{'Avg Diversity':<25} {vc_avg_div:>12.6f} {new_avg_div:>12.6f} {new_avg_div-vc_avg_div:>+12.6f}")

    print(f"\n  Per-class:")
    print(f"  {'Cls':<5} {'C div':>10} {'New div':>10} {'C mse':>10} {'New mse':>10}")
    print(f"  {'-'*47}")
    for cls in range(10):
        print(f"  {cls:<5} {vc_divs[cls]:>10.6f} {new_divs[cls]:>10.6f} "
              f"{np.mean(vc_nearest[cls*10:(cls+1)*10]):>10.6f} "
              f"{np.mean(new_nearest[cls*10:(cls+1)*10]):>10.6f}")

    # Source composition
    total_sources = Counter(s.split("_")[0] for s in new_sources)
    print(f"\n  Source composition: {dict(total_sources)}")

    # Decision
    div_improved = new_avg_div > vc_avg_div
    mse_not_worse = new_avg_mse <= vc_avg_mse * 1.05  # allow 5% proxy MSE increase
    safe_to_submit = div_improved and mse_not_worse

    print(f"\n  Diversity improved: {div_improved} ({new_avg_div-vc_avg_div:+.6f})")
    print(f"  Proxy MSE acceptable: {mse_not_worse} ({new_avg_mse-vc_avg_mse:+.6f})")
    print(f"  SAFE TO SUBMIT: {safe_to_submit}")

    # Save
    path = save_submission(new_images, "submission_final_safe.npz")

    # Count changed images
    per_img_diff = np.mean((new_images.astype(np.float64) - vc_images.astype(np.float64)) ** 2,
                            axis=(1, 2, 3))
    n_changed = int(np.sum(per_img_diff > 1e-10))
    print(f"  Images changed vs Variant C: {n_changed}/100")

    if args.submit:
        if safe_to_submit:
            submit_solution(path)
        else:
            print("\n  NOT SUBMITTING: safety check failed.")
            print("  Variant C remains the best confirmed submission.")
            print("  Use --submit with Variant C's file if you want to resubmit that.")


if __name__ == "__main__":
    main()
