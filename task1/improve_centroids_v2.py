"""
improve_centroids_v2.py — Multi-seed centroid improvement (coverage-aware).

LESSON LEARNED: Proxy MSE (nearest aux) is misleading. Centroids that are
close to aux images are NOT better — they lose the generalization benefit of
being class-mean representatives. Variant C beat D and E because centroids
as synthetic averages are closer to unseen training images.

CORRECT STRATEGY:
  1. Run k-means with many seeds
  2. Score centroids by CLUSTER SIZE (larger cluster = more representative)
     and INERTIA (lower within-cluster variance = tighter fit)
  3. Pick centroids that represent the largest/tightest clusters
  4. Add medoids only for clusters where the centroid is far from all aux images
     (i.e., the centroid is a poor representative)
  5. Ensure diversity via min pairwise distance

NO API calls. Output-only.

USAGE:
  python task1/improve_centroids_v2.py              # Generate + compare
  python task1/improve_centroids_v2.py --submit     # Generate and submit
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
    """Run k-means, return centroids, assignments, and per-cluster stats."""
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

    # Compute per-cluster stats
    cluster_stats = []
    for ki in range(n_k):
        mask = assignments == ki
        size = int(np.sum(mask))
        if size > 0:
            cluster_flat = flat[mask]
            inertia = float(np.mean(np.sum((cluster_flat - centers[ki]) ** 2, axis=1)))
        else:
            inertia = float('inf')
        cluster_stats.append({"size": size, "inertia": inertia})

    centroids = np.clip(centers.reshape(n_k, 3, 32, 32), 0.0, 1.0).astype(np.float32)
    return centroids, assignments, cluster_stats


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


def is_near_dup(img, selected, threshold=0.001):
    for s in selected:
        if compute_mse(img, s) < threshold:
            return True
    return False


def farthest_point_insert(candidate, selected_flat):
    """Compute min distance from candidate to all already-selected images."""
    if len(selected_flat) == 0:
        return float('inf')
    cand_flat = candidate.reshape(1, -1).astype(np.float64)
    dists = np.sum((selected_flat - cand_flat) ** 2, axis=1)
    return float(np.min(dists))


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


def proxy_mse_to_nearest_aux(img, cls_aux_flat):
    """MSE to nearest same-class aux image."""
    flat = img.reshape(1, -1)
    mses = np.mean((cls_aux_flat - flat) ** 2, axis=1)
    return float(np.min(mses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-SEED CENTROID IMPROVEMENT v2 (coverage-aware)")
    print(f"  Seeds: {args.n_seeds}, k={args.k}")
    print("=" * 70)

    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images")

    N_PER_CLASS = 10
    # Target: ~6 centroids + ~4 medoids (matching C's winning ratio)
    TARGET_CENTROIDS = 6
    TARGET_MEDOIDS = 4

    final_images = []
    final_labels = []
    final_sources = []

    for cls in range(10):
        cls_mask = labels == cls
        cls_images = images[cls_mask]
        cls_aux_flat = cls_images.reshape(len(cls_images), -1)
        n_cls = len(cls_images)

        # Collect centroids from all seeds with their cluster quality score
        # Score = cluster_size * (1 / (1 + inertia)) — big tight clusters are best
        centroid_candidates = []  # (image, quality_score, seed, ki)
        medoid_candidates = []

        for seed in range(args.n_seeds):
            centroids, assignments, stats = kmeans_one_class(cls_images, args.k, seed)

            for ki in range(len(centroids)):
                size = stats[ki]["size"]
                inertia = stats[ki]["inertia"]

                # Skip tiny clusters (likely noise/outliers)
                if size < 3:
                    continue

                # Quality = cluster size proportion * tightness
                # Bigger cluster = more representative of the distribution
                quality = (size / n_cls) / (1.0 + inertia)
                centroid_candidates.append((centroids[ki], quality, seed, ki))

                medoid = get_medoid(cls_images, assignments, ki)
                if medoid is not None:
                    medoid_candidates.append((medoid, quality, seed, ki))

        # Sort centroids by quality (descending = best first)
        centroid_candidates.sort(key=lambda x: -x[1])
        medoid_candidates.sort(key=lambda x: -x[1])

        # Select centroids: greedy coverage-aware
        # Pick top-quality centroids that are also diverse (min pairwise distance)
        selected = []
        selected_sources = []

        # Pick centroids with diversity enforcement
        for img, qual, sd, ki in centroid_candidates:
            if len([s for s in selected_sources if s == "centroid"]) >= TARGET_CENTROIDS:
                break
            if not is_near_dup(img, selected, threshold=0.002):
                selected.append(img)
                selected_sources.append("centroid")

        # Pick medoids with diversity enforcement
        for img, qual, sd, ki in medoid_candidates:
            if len([s for s in selected_sources if s == "medoid"]) >= TARGET_MEDOIDS:
                break
            if not is_near_dup(img, selected, threshold=0.002):
                selected.append(img)
                selected_sources.append("medoid")

        # Fill any remaining slots (prefer centroids for coverage)
        if len(selected) < N_PER_CLASS:
            for img, qual, sd, ki in centroid_candidates:
                if len(selected) >= N_PER_CLASS:
                    break
                if not is_near_dup(img, selected, threshold=0.002):
                    selected.append(img)
                    selected_sources.append("centroid")

        if len(selected) < N_PER_CLASS:
            for img, qual, sd, ki in medoid_candidates:
                if len(selected) >= N_PER_CLASS:
                    break
                if not is_near_dup(img, selected, threshold=0.002):
                    selected.append(img)
                    selected_sources.append("medoid")

        sources_count = Counter(selected_sources)
        # Compute proxy MSE for reporting
        scores = [proxy_mse_to_nearest_aux(img, cls_aux_flat) for img in selected]
        avg_proxy = np.mean(scores) if scores else 0
        print(f"  Class {cls}: {dict(sources_count)}, "
              f"avg proxy MSE: {avg_proxy:.6f}, n_candidates tried: {len(centroid_candidates)}")

        for img, src in zip(selected, selected_sources):
            final_images.append(img)
            final_labels.append(cls)
            final_sources.append(src)

    final_images = np.clip(np.array(final_images[:100], dtype=np.float32), 0.0, 1.0)
    final_labels = np.array(final_labels[:100])

    total_sources = Counter(final_sources)
    print(f"\n  Total composition: {dict(total_sources)}")

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print("=" * 70)

    # Compute full proxy metrics
    all_nearest = []
    all_div = []
    for cls in range(10):
        cls_imgs = final_images[final_labels == cls]
        cls_aux_flat = images[labels == cls].reshape(-1, 3072)
        for img in cls_imgs:
            all_nearest.append(proxy_mse_to_nearest_aux(img, cls_aux_flat))
        div = 0.0
        cnt = 0
        for i in range(len(cls_imgs)):
            for j in range(i + 1, len(cls_imgs)):
                div += compute_mse(cls_imgs[i], cls_imgs[j])
                cnt += 1
        all_div.append(div / max(cnt, 1))

    new_mse = np.mean(all_nearest)
    new_div = np.mean(all_div)

    # Load comparisons
    comparisons = {}
    for name, fname in [("A (pure centroids)", "submission_ablation_A.npz"),
                         ("C (57c+43m, leaderboard best)", "submission_ablation_C.npz")]:
        path = os.path.join(WORK_DIR, fname)
        if not os.path.exists(path):
            continue
        data = np.load(path)["images"]
        c_nearest = []
        c_div_list = []
        for c in range(10):
            c_imgs = data[c*10:(c+1)*10]
            c_aux_flat = images[labels == c].reshape(-1, 3072)
            for img in c_imgs:
                c_nearest.append(proxy_mse_to_nearest_aux(img, c_aux_flat))
            d = 0.0
            cnt = 0
            for i in range(len(c_imgs)):
                for j in range(i+1, len(c_imgs)):
                    d += compute_mse(c_imgs[i], c_imgs[j])
                    cnt += 1
            c_div_list.append(d / max(cnt, 1))
        comparisons[name] = (np.mean(c_nearest), np.mean(c_div_list))

    print(f"\n{'Variant':<40} {'MSE→NearAux':>12} {'Diversity':>12}")
    print("-" * 66)
    for name, (mse, div) in comparisons.items():
        print(f"{name:<40} {mse:>12.6f} {div:>12.6f}")
    print(f"{'Multi-seed v2 (new)':<40} {new_mse:>12.6f} {new_div:>12.6f}")

    # Per-class
    print(f"\n  Per-class MSE→NearAux comparison:")
    print(f"  {'Cls':<5} {'Var C':>10} {'New':>10} {'Delta':>10}")
    print(f"  {'-'*37}")

    vc_path = os.path.join(WORK_DIR, "submission_ablation_C.npz")
    if os.path.exists(vc_path):
        vc_data = np.load(vc_path)["images"]
        for cls in range(10):
            vc_imgs = vc_data[cls*10:(cls+1)*10]
            new_imgs = final_images[final_labels == cls]
            cls_aux_flat = images[labels == cls].reshape(-1, 3072)

            vc_mse = np.mean([proxy_mse_to_nearest_aux(img, cls_aux_flat) for img in vc_imgs])
            new_mse_cls = np.mean([proxy_mse_to_nearest_aux(img, cls_aux_flat) for img in new_imgs])
            marker = " <-- worse" if new_mse_cls > vc_mse + 0.001 else ""
            print(f"  {cls:<5} {vc_mse:>10.6f} {new_mse_cls:>10.6f} {new_mse_cls-vc_mse:>+10.6f}{marker}")

    path = save_submission(final_images, "submission_multiseed_v2.npz")

    if args.submit:
        submit_solution(path)


if __name__ == "__main__":
    main()
