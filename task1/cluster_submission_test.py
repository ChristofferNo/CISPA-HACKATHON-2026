"""
cluster_submission_test.py — Standalone clustering-based submission test.

Creates a submission from k-means centroids + top auxiliary images.
Does NOT touch any optimization files (optimize_candidates.npz, optimize_progress.json, etc).

USAGE:
  python task1/cluster_submission_test.py                  # Create submission only
  python task1/cluster_submission_test.py --submit         # Create and submit
"""

import torch
import numpy as np
import requests
import os
import sys
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


def compute_kmeans_centroids(images, labels, n_centroids_per_class=10):
    """Pure numpy k-means: n centroids per class."""
    all_centroids = []
    all_labels = []
    for cls in range(10):
        cls_mask = labels == cls
        cls_images = images[cls_mask]
        if len(cls_images) == 0:
            continue
        n_k = min(n_centroids_per_class, len(cls_images))
        flat = cls_images.reshape(len(cls_images), -1)

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

    centroids = np.clip(np.concatenate(all_centroids, axis=0).astype(np.float32), 0.0, 1.0)
    centroid_labels = np.array(all_labels)
    return centroids, centroid_labels


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


def main():
    parser = argparse.ArgumentParser(description="Clustering-based submission test")
    parser.add_argument("--submit", action="store_true", help="Submit to server")
    parser.add_argument("--centroids-per-class", type=int, default=10,
                        help="Number of k-means centroids per class")
    args = parser.parse_args()

    print("=" * 60)
    print("Cluster Submission Test")
    print("=" * 60)

    # Load data
    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images ({images.shape})")

    # Compute k-means centroids
    n_per_class = args.centroids_per_class
    print(f"\nComputing {n_per_class} k-means centroids per class...")
    centroids, centroid_labels = compute_kmeans_centroids(images, labels, n_per_class)
    print(f"  Total centroids: {len(centroids)}")

    # Build submission: fill up to 100 images
    selected_images = []
    selected_labels = []
    target_per_class = 10  # 10 per class = 100 total

    # Try to load stored confidence scores for supplementing
    scores_path = os.path.join(WORK_DIR, "all_scores.json")
    has_scores = os.path.exists(scores_path)
    if has_scores:
        with open(scores_path) as f:
            all_scores = json.load(f)
        # Compute max_conf per image
        scores_by_class = {c: [] for c in range(10)}
        for s in all_scores:
            scores_by_class[s["label"]].append(s)
        for c in scores_by_class:
            scores_by_class[c].sort(key=lambda x: -x["max_conf"])
        print("  Loaded stored confidence scores for supplementing")

    for cls in range(10):
        cls_centroids = centroids[centroid_labels == cls]
        n_centroids = min(target_per_class, len(cls_centroids))

        # Add centroids first
        for i in range(n_centroids):
            selected_images.append(cls_centroids[i])
            selected_labels.append(cls)

        # Fill remaining slots with top-confidence aux images or evenly sampled
        remaining = target_per_class - n_centroids
        if remaining > 0:
            if has_scores and cls in scores_by_class and len(scores_by_class[cls]) > 0:
                for s in scores_by_class[cls][:remaining]:
                    selected_images.append(images[s["global_index"]].copy())
                    selected_labels.append(cls)
            else:
                cls_mask = labels == cls
                cls_images = images[cls_mask]
                step = max(1, len(cls_images) // remaining)
                for i in range(0, min(remaining * step, len(cls_images)), step):
                    if len(selected_images) < (cls + 1) * target_per_class:
                        selected_images.append(cls_images[i])
                        selected_labels.append(cls)

    # Trim to exactly 100
    selected_images = selected_images[:100]
    selected_labels = selected_labels[:100]

    # Pad if needed (shouldn't happen with 10 classes x 10)
    while len(selected_images) < 100:
        idx = len(selected_images) % len(images)
        selected_images.append(images[idx].copy())
        selected_labels.append(int(labels[idx]))

    submission = np.clip(np.array(selected_images, dtype=np.float32), 0.0, 1.0)
    assert submission.shape == (100, 3, 32, 32), f"Bad shape: {submission.shape}"

    # Print summary
    dist = Counter(selected_labels)
    print(f"\nSubmission summary:")
    print(f"  Shape: {submission.shape}, dtype: {submission.dtype}")
    print(f"  Value range: [{submission.min():.3f}, {submission.max():.3f}]")
    print(f"  Class distribution: {dict(sorted(dist.items()))}")

    # Save
    out_path = os.path.join(WORK_DIR, "submission_cluster_test.npz")
    np.savez_compressed(out_path, images=submission)
    print(f"\n  Saved: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

    # Submit if requested
    if args.submit:
        submit_solution(out_path)
    else:
        print("\n  Use --submit to submit this file to the server.")


if __name__ == "__main__":
    main()
