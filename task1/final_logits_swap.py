"""
final_logits_swap.py — One logits call, swap weak images, submit.

1. Query logits on Variant C's 100 images
2. Find images with lowest confidence for their class
3. Replace bottom ~20 with best alternatives from multi-seed pool
4. Submit immediately
"""

import torch
import numpy as np
import requests
import os
import re
import time
import tempfile
import json
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


def query_logits(images_np):
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
        np.savez_compressed(tmp_path, images=images_np.astype(np.float32))
    try:
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
            detail = resp.json().get("detail", "")
            match = re.search(r"(\d+)\s*seconds", detail)
            wait = int(match.group(1)) + 5 if match else 65
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            with open(tmp_path, "rb") as f:
                resp = requests.post(
                    f"{BASE_URL}/{TASK_ID}/logits",
                    files={"npz": (tmp_path, f, "application/octet-stream")},
                    headers={"X-API-Key": API_KEY},
                    timeout=(10, 300),
                )
            if resp.status_code == 200:
                return resp.json()
        print(f"  API error {resp.status_code}: {resp.text[:200]}")
        return None
    finally:
        os.unlink(tmp_path)


def submit_solution(npz_path):
    print(f"\nSubmitting {npz_path}...")
    with open(npz_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (os.path.basename(npz_path), f, "application/octet-stream")},
            timeout=(10, 300),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    print(f"  Status {resp.status_code}: {body}")
    return body


def kmeans_one_class(cls_images, k, seed, max_iter=30):
    n_k = min(k, len(cls_images))
    flat = cls_images.reshape(len(cls_images), -1)
    rng = np.random.RandomState(seed=seed)
    init_idx = rng.choice(len(flat), n_k, replace=False)
    centers = flat[init_idx].copy()
    for _ in range(max_iter):
        dists = np.zeros((len(flat), n_k), dtype=np.float32)
        for ki in range(n_k):
            dists[:, ki] = np.sum((flat - centers[ki]) ** 2, axis=1)
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
    return np.clip(centers.reshape(n_k, 3, 32, 32), 0.0, 1.0).astype(np.float32), assignments


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


def compute_mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def min_dist_to_set(img, imgs):
    if len(imgs) == 0:
        return float('inf')
    flat = img.reshape(1, -1).astype(np.float64)
    s = np.array(imgs).reshape(len(imgs), -1).astype(np.float64)
    return float(np.min(np.mean((s - flat) ** 2, axis=1)))


def main():
    print("=" * 60)
    print("FINAL PUSH: Logits-guided swap + submit")
    print("=" * 60)

    # Load Variant C
    vc_path = os.path.join(WORK_DIR, "submission_ablation_C.npz")
    vc_images = np.load(vc_path)["images"].copy()
    vc_labels = np.repeat(np.arange(10), 10)  # C has 10 per class in order
    print(f"Loaded Variant C: {vc_images.shape}")

    # Step 1: Query logits
    print("\n--- Querying logits API ---")
    result = query_logits(vc_images)
    if result is None:
        print("FATAL: API call failed. Submitting Variant C as-is.")
        submit_solution(vc_path)
        return

    # Extract per-image class confidence
    scores = []
    for i, entry in enumerate(result.get("results", [])):
        logits = np.array(entry["logits"], dtype=np.float64)
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
        target_cls = vc_labels[i]
        target_conf = float(probs[target_cls])
        max_cls = int(np.argmax(probs))
        max_conf = float(probs[max_cls])
        scores.append({
            "idx": i, "cls": int(target_cls),
            "target_conf": target_conf, "max_conf": max_conf,
            "max_cls": max_cls, "correct": max_cls == target_cls,
        })

    # Print summary
    n_correct = sum(1 for s in scores if s["correct"])
    avg_conf = np.mean([s["target_conf"] for s in scores])
    print(f"  {n_correct}/100 correctly classified, avg target conf: {avg_conf:.4f}")

    # Sort by target confidence (ascending = weakest first)
    scores.sort(key=lambda x: x["target_conf"])

    print(f"\n  Bottom 25 weakest images:")
    for s in scores[:25]:
        print(f"    idx={s['idx']:3d} cls={s['cls']} target_conf={s['target_conf']:.4f} "
              f"pred={s['max_cls']} {'WRONG' if not s['correct'] else ''}")

    # Step 2: Build replacement pool from many seeds
    print("\n--- Building replacement pool ---")
    aux_images, aux_labels = load_auxiliary()

    # Pre-build replacement candidates per class
    class_candidates = {c: [] for c in range(10)}
    for cls in range(10):
        cls_aux = aux_images[aux_labels == cls]
        for seed in range(200):
            for k in [8, 10, 12, 15]:
                centroids, assignments = kmeans_one_class(cls_aux, k, seed=seed)
                for ki in range(len(centroids)):
                    class_candidates[cls].append(centroids[ki])
                    med = get_medoid(cls_aux, assignments, ki)
                    if med is not None:
                        class_candidates[cls].append(med)
        print(f"  Class {cls}: {len(class_candidates[cls])} candidates")

    # Step 3: Replace bottom 20 weakest with best alternatives
    N_REPLACE = 20
    new_images = vc_images.copy()
    replaced = 0

    to_replace = scores[:N_REPLACE]
    # Process replacements
    for s in to_replace:
        idx = s["idx"]
        cls = s["cls"]
        cls_start = cls * 10

        # Get current class images (excluding this slot)
        remaining = [new_images[cls_start + j] for j in range(10)
                     if cls_start + j != idx]

        # Find candidate that is most different from remaining
        # (exploring new territory since current image is weak)
        best_cand = None
        best_dist = -1.0
        for cand in class_candidates[cls]:
            d = min_dist_to_set(cand, remaining)
            if d < 0.002:  # skip near-duplicates
                continue
            if d > best_dist:
                best_dist = d
                best_cand = cand

        if best_cand is not None:
            new_images[idx] = best_cand
            replaced += 1
            print(f"  Replaced idx={idx} cls={cls} (conf={s['target_conf']:.4f}) "
                  f"-> novelty={best_dist:.4f}")

    print(f"\n  Total replaced: {replaced}/{N_REPLACE}")

    # Step 4: Save and submit
    path = os.path.join(WORK_DIR, "submission_final_logits.npz")
    new_images = np.clip(new_images, 0.0, 1.0).astype(np.float32)
    assert new_images.shape == (100, 3, 32, 32)
    np.savez_compressed(path, images=new_images)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")

    submit_solution(path)


if __name__ == "__main__":
    main()
