"""
ablation_variants.py — Controlled ablation to identify which solve_v2 additions hurt MSE.

Produces three submission variants using IDENTICAL centroid computation (matching
the successful cluster_submission_test.py seed=cls), then measures which additions
degrade local MSE proxy metrics.

Variants:
  A: Pure centroids only (exact reproduction of cluster_submission_test)
  B: Centroids + optimized candidates (no medoids)
  C: Centroids + medoids (no optimized candidates)

NO API/logits calls. NO modification to optimization state. Output-only.

USAGE:
  python task1/ablation_variants.py                # Generate all variants + comparison
  python task1/ablation_variants.py --submit A     # Submit variant A
  python task1/ablation_variants.py --submit B     # Submit variant B
  python task1/ablation_variants.py --submit C     # Submit variant C
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


def load_optimized_candidates():
    """Load optimized candidates from the optimization pipeline."""
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
    return candidates, candidate_labels, best_scores


# ================================================================
# K-MEANS (exact match to cluster_submission_test.py)
# ================================================================
def compute_kmeans_centroids(images, labels, n_centroids_per_class=10):
    """
    Pure numpy k-means: n centroids per class.
    Uses seed=cls to match cluster_submission_test.py EXACTLY.
    """
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
        n_k = min(n_centroids_per_class, len(cls_images))
        flat = cls_images.reshape(len(cls_images), -1)

        # CRITICAL: seed=cls to match cluster_submission_test.py
        rng = np.random.RandomState(seed=cls)
        init_idx = rng.choice(len(flat), n_k, replace=False)
        centers = flat[init_idx].copy()

        for _ in range(30):
            dists = np.zeros((len(flat), n_k), dtype=np.float32)
            for k in range(n_k):
                diff = flat - centers[k]
                dists[:, k] = np.sum(diff ** 2, axis=1)
            assignments = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(n_k):
                mask = assignments == k
                if np.any(mask):
                    new_centers[k] = flat[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]

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
def find_medoids(class_images, assignments, k):
    """
    Find medoids: the actual auxiliary image closest to each cluster center.
    Returns list of (medoid_image, cluster_index, mse_to_center).
    """
    flat = class_images.reshape(len(class_images), -1)
    medoids = []
    for ki in range(k):
        mask = assignments == ki
        if not np.any(mask):
            continue
        cluster_flat = flat[mask]
        cluster_imgs = class_images[mask]
        center = cluster_flat.mean(axis=0)
        dists = np.sum((cluster_flat - center) ** 2, axis=1)
        best_idx = int(np.argmin(dists))
        medoids.append((cluster_imgs[best_idx].copy(), ki, float(dists[best_idx])))
    return medoids


# ================================================================
# MSE UTILITIES
# ================================================================
def compute_mse(img1, img2):
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def compute_proxy_metrics(submission, submission_labels, aux_images, aux_labels):
    """
    Compute local proxy metrics for MSE quality assessment.

    Returns dict with:
      - avg_mse_to_nearest_same_class: for each submission img, MSE to nearest
        same-class auxiliary image (proxy for actual metric)
      - avg_mse_to_class_mean: MSE to class mean (lower = more central)
      - avg_intra_class_diversity: avg pairwise MSE within each class's selected images
      - per_class_stats: detailed per-class breakdown
    """
    per_class_stats = {}
    all_nearest_mses = []
    all_mean_mses = []
    all_diversities = []

    for cls in range(10):
        # Submission images for this class
        sub_mask = submission_labels == cls
        sub_imgs = submission[sub_mask]

        # Auxiliary images for this class
        aux_mask = aux_labels == cls
        cls_aux = aux_images[aux_mask]
        cls_aux_flat = cls_aux.reshape(len(cls_aux), -1)

        # Class mean
        class_mean = cls_aux_flat.mean(axis=0).reshape(1, 3, 32, 32)

        nearest_mses = []
        mean_mses = []
        for img in sub_imgs:
            flat_img = img.reshape(1, -1)
            # MSE to nearest aux image
            mses = np.mean((cls_aux_flat - flat_img) ** 2, axis=1)
            nearest_mses.append(float(np.min(mses)))
            # MSE to class mean
            mean_mses.append(compute_mse(img, class_mean[0]))

        # Intra-class diversity (avg pairwise MSE)
        diversity = 0.0
        count = 0
        for i in range(len(sub_imgs)):
            for j in range(i + 1, len(sub_imgs)):
                diversity += compute_mse(sub_imgs[i], sub_imgs[j])
                count += 1
        diversity = diversity / max(count, 1)

        per_class_stats[cls] = {
            "n_images": int(sub_mask.sum()),
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
        "per_class": per_class_stats,
    }


# ================================================================
# VARIANT BUILDERS
# ================================================================
def build_variant_a(centroids, centroid_labels):
    """
    Variant A: Pure centroids only.
    Exact reproduction of cluster_submission_test.py (10 centroids per class).
    """
    images = []
    labels = []
    for cls in range(10):
        cls_centroids = centroids[centroid_labels == cls]
        for c in cls_centroids[:10]:
            images.append(c)
            labels.append(cls)

    images = np.clip(np.array(images[:100], dtype=np.float32), 0.0, 1.0)
    labels = np.array(labels[:100])
    return images, labels


def build_variant_b(centroids, centroid_labels, opt_imgs, opt_labels, opt_scores,
                    class_images, n_per_class=10):
    """
    Variant B: Centroids + optimized candidates (no medoids).

    Strategy: For each class, start with all centroids. If an optimized candidate
    is closer to the class centroid mean than the worst centroid, swap it in.
    This tests whether optimized images help when used selectively.
    """
    images = []
    labels = []
    source_log = []

    for cls in range(10):
        cls_centroids = centroids[centroid_labels == cls]

        # Build pool: centroids + optimized for this class
        pool = []
        for i, c in enumerate(cls_centroids):
            pool.append((c, f"centroid_{i}"))

        if opt_imgs is not None and opt_labels is not None:
            opt_mask = opt_labels == cls
            cls_opt = opt_imgs[opt_mask]
            cls_opt_scores = opt_scores[opt_mask] if opt_scores is not None else np.zeros(opt_mask.sum())

            for i, (opt_img, score) in enumerate(zip(cls_opt, cls_opt_scores)):
                pool.append((opt_img, f"optimized_{i}"))

        # Score each image by MSE to class mean (lower = better representative)
        cls_aux = class_images.get(cls, np.array([]))
        if len(cls_aux) > 0:
            class_mean = cls_aux.reshape(len(cls_aux), -1).mean(axis=0).reshape(3, 32, 32)
        else:
            class_mean = cls_centroids.mean(axis=0) if len(cls_centroids) > 0 else np.zeros((3, 32, 32))

        scored_pool = []
        for img, src in pool:
            mse_to_mean = compute_mse(img, class_mean)
            scored_pool.append((img, src, mse_to_mean))

        # Sort by MSE to class mean (ascending = more central)
        scored_pool.sort(key=lambda x: x[2])

        # Take top n_per_class
        selected = scored_pool[:n_per_class]
        for img, src, _ in selected:
            images.append(img)
            labels.append(cls)
            source_log.append(src)

        sources = Counter(s.split("_")[0] for _, s, _ in selected)
        print(f"  Class {cls}: {dict(sources)}")

    images = np.clip(np.array(images[:100], dtype=np.float32), 0.0, 1.0)
    labels = np.array(labels[:100])
    return images, labels, source_log


def build_variant_c(centroids, centroid_labels, class_images, assignments,
                    n_per_class=10):
    """
    Variant C: Centroids + medoids (no optimized candidates).

    Strategy: For each class, compute medoids (actual images nearest to cluster
    centers). Replace centroids with medoids where the medoid is closer to the
    class mean than the centroid. This tests whether using real images instead
    of synthetic centroids helps.
    """
    images = []
    labels = []
    source_log = []

    for cls in range(10):
        cls_centroids = centroids[centroid_labels == cls]
        cls_aux = class_images.get(cls, np.array([]))

        # Build pool: centroids + medoids
        pool = []
        for i, c in enumerate(cls_centroids):
            pool.append((c, f"centroid_{i}"))

        if cls in assignments and len(cls_aux) > 0:
            k = len(cls_centroids)
            medoids = find_medoids(cls_aux, assignments[cls], k)
            for med_img, ki, med_mse in medoids:
                pool.append((med_img, f"medoid_{ki}"))

        # Score by MSE to class mean
        if len(cls_aux) > 0:
            class_mean = cls_aux.reshape(len(cls_aux), -1).mean(axis=0).reshape(3, 32, 32)
        else:
            class_mean = cls_centroids.mean(axis=0) if len(cls_centroids) > 0 else np.zeros((3, 32, 32))

        scored_pool = []
        for img, src in pool:
            mse_to_mean = compute_mse(img, class_mean)
            scored_pool.append((img, src, mse_to_mean))

        scored_pool.sort(key=lambda x: x[2])
        selected = scored_pool[:n_per_class]

        for img, src, _ in selected:
            images.append(img)
            labels.append(cls)
            source_log.append(src)

        sources = Counter(s.split("_")[0] for _, s, _ in selected)
        print(f"  Class {cls}: {dict(sources)}")

    images = np.clip(np.array(images[:100], dtype=np.float32), 0.0, 1.0)
    labels = np.array(labels[:100])
    return images, labels, source_log


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
# VERIFICATION: Compare Variant A to existing cluster_submission_test.npz
# ================================================================
def verify_variant_a(variant_a_images):
    """Check if Variant A matches the existing cluster_submission_test.npz exactly."""
    existing_path = os.path.join(WORK_DIR, "submission_cluster_test.npz")
    if not os.path.exists(existing_path):
        print("  [SKIP] No existing submission_cluster_test.npz to compare against.")
        return False

    existing = np.load(existing_path)["images"]
    mse = float(np.mean((variant_a_images.astype(np.float64) - existing.astype(np.float64)) ** 2))
    max_diff = float(np.max(np.abs(variant_a_images.astype(np.float64) - existing.astype(np.float64))))

    if mse < 1e-10:
        print(f"  [MATCH] Variant A matches cluster_submission_test.npz exactly (MSE={mse:.2e})")
        return True
    else:
        print(f"  [MISMATCH] Variant A differs from cluster_submission_test.npz:")
        print(f"    MSE={mse:.6f}, max_diff={max_diff:.6f}")
        # Find which images differ
        per_img_mse = np.mean((variant_a_images.astype(np.float64) - existing.astype(np.float64)) ** 2,
                               axis=(1, 2, 3))
        n_diff = int(np.sum(per_img_mse > 1e-10))
        print(f"    {n_diff}/100 images differ")
        return False


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Controlled ablation variants")
    parser.add_argument("--submit", type=str, choices=["A", "B", "C"],
                        help="Submit variant A, B, or C")
    args = parser.parse_args()

    print("=" * 70)
    print("ABLATION STUDY: Identifying which solve_v2 additions hurt MSE")
    print("=" * 70)

    # Load data
    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images ({images.shape})")

    # Compute centroids (matching cluster_submission_test.py exactly)
    print(f"\nComputing k-means centroids (seed=cls, k=10)...")
    centroids, centroid_labels, assignments, class_images = compute_kmeans_centroids(
        images, labels, n_centroids_per_class=10
    )
    print(f"  Total centroids: {len(centroids)}")

    # Load optimized candidates
    opt_data = load_optimized_candidates()
    if opt_data is not None:
        opt_imgs, opt_labels, opt_scores = opt_data
        print(f"  Loaded {len(opt_imgs)} optimized candidates (avg score: {opt_scores.mean():.4f})")
    else:
        opt_imgs = opt_labels = opt_scores = None
        print("  No optimized candidates found.")

    # ============================================================
    # BUILD VARIANTS
    # ============================================================

    # --- Variant A: Pure centroids ---
    print(f"\n{'='*70}")
    print("VARIANT A: Pure centroids only (cluster_submission_test reproduction)")
    print("=" * 70)
    va_images, va_labels = build_variant_a(centroids, centroid_labels)
    print(f"  Shape: {va_images.shape}, classes: {dict(sorted(Counter(va_labels.tolist()).items()))}")
    print(f"  Source: 100% centroids")
    verify_variant_a(va_images)
    va_path = save_submission(va_images, "submission_ablation_A.npz")

    # --- Variant B: Centroids + optimized ---
    print(f"\n{'='*70}")
    print("VARIANT B: Centroids + optimized candidates (no medoids)")
    print("=" * 70)
    vb_images, vb_labels, vb_sources = build_variant_b(
        centroids, centroid_labels, opt_imgs, opt_labels, opt_scores,
        class_images, n_per_class=10
    )
    print(f"  Shape: {vb_images.shape}, classes: {dict(sorted(Counter(vb_labels.tolist()).items()))}")
    vb_source_counts = Counter(s.split("_")[0] for s in vb_sources)
    print(f"  Source composition: {dict(vb_source_counts)}")
    vb_path = save_submission(vb_images, "submission_ablation_B.npz")

    # --- Variant C: Centroids + medoids ---
    print(f"\n{'='*70}")
    print("VARIANT C: Centroids + medoids (no optimized candidates)")
    print("=" * 70)
    vc_images, vc_labels, vc_sources = build_variant_c(
        centroids, centroid_labels, class_images, assignments,
        n_per_class=10
    )
    print(f"  Shape: {vc_images.shape}, classes: {dict(sorted(Counter(vc_labels.tolist()).items()))}")
    vc_source_counts = Counter(s.split("_")[0] for s in vc_sources)
    print(f"  Source composition: {dict(vc_source_counts)}")
    vc_path = save_submission(vc_images, "submission_ablation_C.npz")

    # ============================================================
    # COMPARISON METRICS
    # ============================================================
    print(f"\n{'='*70}")
    print("PROXY METRIC COMPARISON")
    print("=" * 70)

    variants = {
        "A (centroids only)": (va_images, va_labels),
        "B (centroids + optimized)": (vb_images, vb_labels),
        "C (centroids + medoids)": (vc_images, vc_labels),
    }

    # Also compare against existing solve_v2 submission if it exists
    v2_path = os.path.join(WORK_DIR, "submission_v2.npz")
    if os.path.exists(v2_path):
        v2_data = np.load(v2_path)
        v2_images = v2_data["images"]
        # We don't know the labels for v2, so estimate from nearest centroid
        v2_labels = np.zeros(100, dtype=int)
        for i, img in enumerate(v2_images):
            mses = [compute_mse(img, centroids[j]) for j in range(len(centroids))]
            v2_labels[i] = centroid_labels[np.argmin(mses)]
        variants["V2 (solve_v2.py)"] = (v2_images, v2_labels)

    all_metrics = {}
    for name, (sub_imgs, sub_labels) in variants.items():
        metrics = compute_proxy_metrics(sub_imgs, sub_labels, images, labels)
        all_metrics[name] = metrics

    # Print comparison table
    print(f"\n{'Variant':<30} {'MSE→NearestAux':>15} {'MSE→ClassMean':>15} {'IntraDiversity':>15}")
    print("-" * 77)
    for name, m in all_metrics.items():
        print(f"{name:<30} {m['avg_mse_to_nearest_aux']:>15.6f} "
              f"{m['avg_mse_to_class_mean']:>15.6f} {m['avg_intra_diversity']:>15.6f}")

    # Per-class comparison
    print(f"\n--- Per-class MSE to nearest aux image ---")
    print(f"{'Class':<8}", end="")
    for name in all_metrics:
        short_name = name.split("(")[0].strip()
        print(f"{short_name:>12}", end="")
    print()
    print("-" * (8 + 12 * len(all_metrics)))
    for cls in range(10):
        print(f"{cls:<8}", end="")
        for name, m in all_metrics.items():
            val = m["per_class"].get(cls, {}).get("avg_mse_nearest_aux", 0.0)
            print(f"{val:>12.6f}", end="")
        print()

    # Pairwise MSE between variants
    print(f"\n--- Pairwise MSE between variants ---")
    variant_names = list(variants.keys())
    for i in range(len(variant_names)):
        for j in range(i + 1, len(variant_names)):
            name_i = variant_names[i]
            name_j = variant_names[j]
            imgs_i = variants[name_i][0]
            imgs_j = variants[name_j][0]
            mse = float(np.mean((imgs_i.astype(np.float64) - imgs_j.astype(np.float64)) ** 2))
            print(f"  {name_i} vs {name_j}: MSE={mse:.6f}")

    # ============================================================
    # ANALYSIS & RECOMMENDATIONS
    # ============================================================
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("=" * 70)

    ma = all_metrics["A (centroids only)"]
    mb = all_metrics["B (centroids + optimized)"]
    mc = all_metrics["C (centroids + medoids)"]

    b_delta = mb["avg_mse_to_nearest_aux"] - ma["avg_mse_to_nearest_aux"]
    c_delta = mc["avg_mse_to_nearest_aux"] - ma["avg_mse_to_nearest_aux"]

    print(f"\nVariant B (+ optimized) changes MSE→NearestAux by: {b_delta:+.6f}")
    if b_delta > 0:
        print(f"  -> WORSE: Optimized candidates increase proxy MSE (pixel drift)")
    elif b_delta < -0.0001:
        print(f"  -> BETTER: Optimized candidates decrease proxy MSE")
    else:
        print(f"  -> NEUTRAL: Negligible difference")

    print(f"\nVariant C (+ medoids) changes MSE→NearestAux by: {c_delta:+.6f}")
    if c_delta > 0:
        print(f"  -> WORSE: Medoids increase proxy MSE")
    elif c_delta < -0.0001:
        print(f"  -> BETTER: Medoids decrease proxy MSE")
    else:
        print(f"  -> NEUTRAL: Negligible difference")

    if "V2 (solve_v2.py)" in all_metrics:
        mv2 = all_metrics["V2 (solve_v2.py)"]
        v2_delta = mv2["avg_mse_to_nearest_aux"] - ma["avg_mse_to_nearest_aux"]
        print(f"\nsolve_v2.py (different seeds + all additions) vs pure centroids: {v2_delta:+.6f}")

    print(f"\nRECOMMENDATION:")
    if b_delta > 0 and c_delta > 0:
        print(f"  Both additions hurt. Stick with pure centroids (Variant A).")
        print(f"  Do NOT submit B or C.")
    elif b_delta < c_delta:
        print(f"  Optimized candidates help more (or hurt less) than medoids.")
        print(f"  Consider submitting B if delta is meaningfully negative.")
    elif c_delta < b_delta:
        print(f"  Medoids help more (or hurt less) than optimized candidates.")
        print(f"  Consider submitting C if delta is meaningfully negative.")
    else:
        print(f"  Both changes are similar. Pure centroids remains safest bet.")

    # ============================================================
    # SUBMIT IF REQUESTED
    # ============================================================
    if args.submit:
        paths = {"A": va_path, "B": vb_path, "C": vc_path}
        submit_solution(paths[args.submit])


if __name__ == "__main__":
    main()
