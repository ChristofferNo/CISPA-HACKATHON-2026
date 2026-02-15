"""
ablation_medoid_ratio.py — Explore increasing medoid ratios while preserving coverage.

Variant C (57 centroids + 43 medoids) improved the leaderboard. Now we test whether
pushing the ratio further toward medoids helps more.

Strategy: Use farthest-point sampling on centroids to pick the N_centroid "anchor"
centroids that maximally cover the class distribution, then fill remaining slots
with medoids that add the most coverage (farthest from already-selected images).

Variants:
  D: 5 centroids + 5 medoids per class (50/50)
  E: 4 centroids + 6 medoids per class (40/60)
  F: 2 centroids + 8 medoids per class (20/80, medoid-heavy)

NO API/logits calls. NO modification of optimization state.

USAGE:
  python task1/ablation_medoid_ratio.py                # Generate all + compare
  python task1/ablation_medoid_ratio.py --submit D     # Submit variant D
  python task1/ablation_medoid_ratio.py --submit E     # Submit variant E
  python task1/ablation_medoid_ratio.py --submit F     # Submit variant F
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


# ================================================================
# DATA LOADING
# ================================================================
def load_auxiliary():
    dataset = torch.load(os.path.join(WORK_DIR, "auxiliary_dataset.pt"), weights_only=False)
    images = dataset["images"].numpy().astype(np.float32)
    labels = dataset["labels"].numpy()
    return images, labels


# ================================================================
# K-MEANS (seed=cls, matching cluster_submission_test.py)
# ================================================================
def compute_kmeans(images, labels, k=10):
    all_centroids = []
    all_labels = []
    all_assignments = {}
    all_class_images = {}

    for cls in range(10):
        cls_mask = labels == cls
        cls_images = images[cls_mask]
        all_class_images[cls] = cls_images
        if len(cls_images) == 0:
            continue
        n_k = min(k, len(cls_images))
        flat = cls_images.reshape(len(cls_images), -1)

        rng = np.random.RandomState(seed=cls)
        init_idx = rng.choice(len(flat), n_k, replace=False)
        centers = flat[init_idx].copy()

        for _ in range(30):
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

        centroids = centers.reshape(n_k, 3, 32, 32)
        all_centroids.append(centroids)
        all_labels.extend([cls] * n_k)
        all_assignments[cls] = assignments

    centroids = np.clip(np.concatenate(all_centroids, axis=0).astype(np.float32), 0.0, 1.0)
    centroid_labels = np.array(all_labels)
    return centroids, centroid_labels, all_assignments, all_class_images


# ================================================================
# MEDOID COMPUTATION
# ================================================================
def compute_all_medoids(class_images, assignments, k=10):
    """
    For each class, find the medoid (actual image nearest to cluster center)
    for every cluster. Returns dict[cls] -> list of (medoid_image, cluster_idx).
    """
    medoids_by_class = {}
    for cls in range(10):
        cls_imgs = class_images.get(cls, np.array([]))
        cls_assign = assignments.get(cls, np.array([]))
        if len(cls_imgs) == 0:
            medoids_by_class[cls] = []
            continue

        flat = cls_imgs.reshape(len(cls_imgs), -1)
        medoids = []
        for ki in range(k):
            mask = cls_assign == ki
            if not np.any(mask):
                continue
            cluster_flat = flat[mask]
            cluster_imgs = cls_imgs[mask]
            center = cluster_flat.mean(axis=0)
            dists = np.sum((cluster_flat - center) ** 2, axis=1)
            best_idx = int(np.argmin(dists))
            medoids.append((cluster_imgs[best_idx].copy(), ki))
        medoids_by_class[cls] = medoids
    return medoids_by_class


# ================================================================
# UTILITIES
# ================================================================
def compute_mse(img1, img2):
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def farthest_point_sampling(images, n_select):
    """
    Greedy farthest-point sampling: select n images that are maximally spread out.
    Returns indices into the input array.
    """
    if len(images) <= n_select:
        return list(range(len(images)))

    flat = images.reshape(len(images), -1).astype(np.float64)
    # Start with the image closest to the global mean (most central)
    mean = flat.mean(axis=0)
    dists_to_mean = np.sum((flat - mean) ** 2, axis=1)
    first = int(np.argmin(dists_to_mean))

    selected = [first]
    # min_dists[i] = min distance from image i to any selected image
    min_dists = np.sum((flat - flat[first]) ** 2, axis=1)

    for _ in range(n_select - 1):
        # Pick the point farthest from all selected points
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        new_dists = np.sum((flat - flat[next_idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return selected


def is_near_duplicate(img, existing_images, threshold=0.0005):
    """Check if img is a near-duplicate of any image in existing_images."""
    for ex in existing_images:
        if compute_mse(img, ex) < threshold:
            return True
    return False


# ================================================================
# VARIANT BUILDER
# ================================================================
def build_variant(centroids, centroid_labels, medoids_by_class,
                  n_centroids_per_class, n_medoids_per_class, label):
    """
    Build a variant with a specific centroid/medoid ratio.

    Strategy:
      1. Select n_centroids anchor centroids via farthest-point sampling
         (maximizes distribution coverage).
      2. Fill remaining slots with medoids, also via farthest-point from
         already-selected images (maximizes diversity).
      3. Dedup: skip medoids that are near-duplicates of selected images.
    """
    images = []
    labels = []
    source_log = []
    n_total = n_centroids_per_class + n_medoids_per_class

    for cls in range(10):
        cls_centroids = centroids[centroid_labels == cls]
        cls_medoids = medoids_by_class.get(cls, [])
        selected = []
        selected_sources = []

        # Step 1: Select anchor centroids via farthest-point sampling
        n_c = min(n_centroids_per_class, len(cls_centroids))
        if n_c > 0:
            fps_indices = farthest_point_sampling(cls_centroids, n_c)
            for idx in fps_indices:
                selected.append(cls_centroids[idx])
                selected_sources.append("centroid")

        # Step 2: Add medoids, prioritized by distance from selected images
        # Sort medoids by their distance to nearest already-selected image (desc)
        remaining_medoids = []
        for med_img, ki in cls_medoids:
            if not is_near_duplicate(med_img, selected):
                remaining_medoids.append(med_img)

        n_m = min(n_medoids_per_class, len(remaining_medoids))
        if n_m > 0 and len(selected) > 0:
            # Farthest-point from already-selected
            selected_flat = np.array(selected).reshape(len(selected), -1).astype(np.float64)
            remaining_flat = np.array(remaining_medoids).reshape(len(remaining_medoids), -1).astype(np.float64)

            added = 0
            available = list(range(len(remaining_medoids)))
            while added < n_m and available:
                best_idx = -1
                best_min_dist = -1.0
                for i in available:
                    dists = np.sum((selected_flat - remaining_flat[i]) ** 2, axis=1)
                    min_dist = float(np.min(dists))
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = i

                if best_idx >= 0:
                    selected.append(remaining_medoids[best_idx])
                    selected_sources.append("medoid")
                    selected_flat = np.vstack([selected_flat, remaining_flat[best_idx:best_idx+1]])
                    available.remove(best_idx)
                    added += 1
                else:
                    break
        elif n_m > 0:
            # No centroids selected, just take medoids
            for med_img in remaining_medoids[:n_m]:
                selected.append(med_img)
                selected_sources.append("medoid")

        # Step 3: If we still need more images (not enough medoids), fill with centroids
        while len(selected) < n_total and len(cls_centroids) > len([s for s in selected_sources if s == "centroid"]):
            for c in cls_centroids:
                if len(selected) >= n_total:
                    break
                if not is_near_duplicate(c, selected):
                    selected.append(c)
                    selected_sources.append("centroid_fill")

        # Trim to target
        selected = selected[:n_total]
        selected_sources = selected_sources[:n_total]

        for img, src in zip(selected, selected_sources):
            images.append(img)
            labels.append(cls)
            source_log.append(src)

        sources = Counter(selected_sources)
        print(f"  Class {cls}: {dict(sources)}")

    images = np.clip(np.array(images[:100], dtype=np.float32), 0.0, 1.0)
    labels_arr = np.array(labels[:100])
    return images, labels_arr, source_log


# ================================================================
# PROXY METRICS
# ================================================================
def compute_proxy_metrics(submission, submission_labels, aux_images, aux_labels):
    all_nearest_mses = []
    all_mean_mses = []
    all_diversities = []
    per_class = {}

    for cls in range(10):
        sub_mask = submission_labels == cls
        sub_imgs = submission[sub_mask]
        aux_mask = aux_labels == cls
        cls_aux = aux_images[aux_mask]
        cls_aux_flat = cls_aux.reshape(len(cls_aux), -1)
        class_mean = cls_aux_flat.mean(axis=0).reshape(1, 3, 32, 32)

        nearest_mses = []
        mean_mses = []
        for img in sub_imgs:
            flat_img = img.reshape(1, -1)
            mses = np.mean((cls_aux_flat - flat_img) ** 2, axis=1)
            nearest_mses.append(float(np.min(mses)))
            mean_mses.append(compute_mse(img, class_mean[0]))

        diversity = 0.0
        count = 0
        for i in range(len(sub_imgs)):
            for j in range(i + 1, len(sub_imgs)):
                diversity += compute_mse(sub_imgs[i], sub_imgs[j])
                count += 1
        diversity = diversity / max(count, 1)

        per_class[cls] = {
            "avg_mse_nearest_aux": float(np.mean(nearest_mses)) if nearest_mses else 0.0,
            "avg_mse_to_mean": float(np.mean(mean_mses)) if mean_mses else 0.0,
            "intra_diversity": diversity,
        }
        all_nearest_mses.extend(nearest_mses)
        all_mean_mses.extend(mean_mses)
        all_diversities.append(diversity)

    return {
        "avg_mse_to_nearest_aux": float(np.mean(all_nearest_mses)),
        "avg_mse_to_class_mean": float(np.mean(all_mean_mses)),
        "avg_intra_diversity": float(np.mean(all_diversities)),
        "per_class": per_class,
    }


# ================================================================
# SUBMISSION
# ================================================================
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


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Medoid ratio ablation")
    parser.add_argument("--submit", type=str, choices=["D", "E", "F"],
                        help="Submit variant D, E, or F")
    args = parser.parse_args()

    print("=" * 70)
    print("MEDOID RATIO ABLATION: Finding optimal centroid/medoid balance")
    print("=" * 70)

    # Load data
    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images")

    # Compute centroids
    print(f"\nComputing k-means centroids (seed=cls, k=10)...")
    centroids, centroid_labels, assignments, class_images = compute_kmeans(images, labels, k=10)
    print(f"  Total centroids: {len(centroids)}")

    # Compute all medoids
    print("Computing medoids for all clusters...")
    medoids_by_class = compute_all_medoids(class_images, assignments, k=10)
    total_medoids = sum(len(v) for v in medoids_by_class.values())
    print(f"  Total medoids: {total_medoids}")

    # Load Variant C for comparison baseline
    vc_path = os.path.join(WORK_DIR, "submission_ablation_C.npz")
    va_path = os.path.join(WORK_DIR, "submission_ablation_A.npz")

    # ============================================================
    # BUILD VARIANTS
    # ============================================================
    variant_configs = {
        "D": (5, 5, "50% centroids + 50% medoids (FPS-selected)"),
        "E": (4, 6, "40% centroids + 60% medoids (FPS-selected)"),
        "F": (2, 8, "20% centroids + 80% medoids (FPS-selected)"),
    }

    variant_results = {}

    for name, (n_c, n_m, desc) in variant_configs.items():
        print(f"\n{'='*70}")
        print(f"VARIANT {name}: {desc}")
        print(f"  Target: {n_c} centroids + {n_m} medoids per class")
        print("=" * 70)

        v_images, v_labels, v_sources = build_variant(
            centroids, centroid_labels, medoids_by_class,
            n_centroids_per_class=n_c, n_medoids_per_class=n_m,
            label=name,
        )
        source_counts = Counter(v_sources)
        print(f"  Total composition: {dict(source_counts)}")
        v_path = save_submission(v_images, f"submission_ablation_{name}.npz")
        variant_results[name] = (v_images, v_labels, v_path, source_counts)

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("PROXY METRIC COMPARISON")
    print("=" * 70)

    all_variants = {}

    # Load previous variants for reference
    if os.path.exists(va_path):
        va_data = np.load(va_path)["images"]
        va_labels_arr = np.repeat(np.arange(10), 10)
        all_variants["A (100% centroid)"] = (va_data, va_labels_arr)

    if os.path.exists(vc_path):
        vc_data = np.load(vc_path)["images"]
        vc_labels_arr = np.repeat(np.arange(10), 10)
        all_variants["C (57c+43m, prev best)"] = (vc_data, vc_labels_arr)

    for name in ["D", "E", "F"]:
        v_imgs, v_labels, _, sc = variant_results[name]
        n_c, n_m, desc = variant_configs[name]
        all_variants[f"{name} ({n_c}c+{n_m}m/class)"] = (v_imgs, v_labels)

    all_metrics = {}
    for vname, (v_imgs, v_labels) in all_variants.items():
        all_metrics[vname] = compute_proxy_metrics(v_imgs, v_labels, images, labels)

    # Summary table
    print(f"\n{'Variant':<30} {'MSE→NearAux':>12} {'MSE→Mean':>12} {'Diversity':>12}")
    print("-" * 68)
    for vname, m in all_metrics.items():
        print(f"{vname:<30} {m['avg_mse_to_nearest_aux']:>12.6f} "
              f"{m['avg_mse_to_class_mean']:>12.6f} {m['avg_intra_diversity']:>12.6f}")

    # Per-class detail
    print(f"\n--- Per-class MSE to nearest aux ---")
    short_names = {v: v.split("(")[0].strip() for v in all_metrics}
    print(f"{'Cls':<5}", end="")
    for vname in all_metrics:
        print(f"{short_names[vname]:>10}", end="")
    print()
    print("-" * (5 + 10 * len(all_metrics)))
    for cls in range(10):
        print(f"{cls:<5}", end="")
        for vname, m in all_metrics.items():
            val = m["per_class"].get(cls, {}).get("avg_mse_nearest_aux", 0.0)
            print(f"{val:>10.6f}", end="")
        print()

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("=" * 70)

    # Reference: Variant A (pure centroids)
    if "A (100% centroid)" in all_metrics:
        ref_mse = all_metrics["A (100% centroid)"]["avg_mse_to_nearest_aux"]
        print(f"\nBaseline A (pure centroids) MSE→NearAux: {ref_mse:.6f}")
    else:
        ref_mse = None

    best_name = None
    best_mse = float("inf")
    for vname, m in all_metrics.items():
        mse = m["avg_mse_to_nearest_aux"]
        if mse < best_mse:
            best_mse = mse
            best_name = vname
        if ref_mse is not None:
            delta = mse - ref_mse
            print(f"  {vname}: {mse:.6f} (delta vs A: {delta:+.6f})")

    print(f"\n  BEST by proxy MSE: {best_name} ({best_mse:.6f})")

    # Check diversity isn't collapsing
    print(f"\n  Coverage check (intra-class diversity — higher = more spread):")
    for vname, m in all_metrics.items():
        div = m["avg_intra_diversity"]
        print(f"    {vname}: {div:.6f}")

    if ref_mse is not None:
        ref_div = all_metrics["A (100% centroid)"]["avg_intra_diversity"]
        for vname, m in all_metrics.items():
            if "A " in vname:
                continue
            div_ratio = m["avg_intra_diversity"] / ref_div
            if div_ratio < 0.3:
                print(f"  WARNING: {vname} diversity dropped to {div_ratio:.0%} of baseline — coverage collapse risk!")

    # ============================================================
    # SUBMIT
    # ============================================================
    if args.submit and args.submit in variant_results:
        _, _, path, _ = variant_results[args.submit]
        submit_solution(path)


if __name__ == "__main__":
    main()
