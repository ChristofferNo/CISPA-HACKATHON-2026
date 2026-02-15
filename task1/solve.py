"""
solve.py — Data reconstruction attack.

TASK: Reconstruct 100 images close (MSE) to the classifier's 5000 training images.

WHAT WE HAVE:
  - 1000 auxiliary images (same distribution as training data)
  - Logits API (100 images per call, 10-min rate limit)
  - Evaluation: each submitted image matched to closest training image by MSE

STRATEGY:
  Phase 1: Quick baseline — pick 100 diverse auxiliary images (1 API call)
  Phase 2: Score all 1000 auxiliary images, pick best 100 (10 API calls)
  Phase 3: Optimize candidates via perturbation (many API calls)
  Phase 4: Reseed weak candidates + reheat annealing (class coverage + k-means)

USAGE:
  python task1/solve.py                    # Phase 1 baseline (instant)
  python task1/solve.py --phase 2          # Score all auxiliary images
  python task1/solve.py --phase 3          # Optimization loop
  python task1/solve.py --phase 4          # Reseed + reheat
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
# API HELPER
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
    """
    Extract per-image scores from API response.
    Returns list of dicts with max_conf, max_class, entropy, logits.

    Key insight: we use MAX confidence (any class), not true-class confidence,
    because images the model is confident about are likely close to training data.
    """
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
            "max_conf": max_conf,
            "max_class": max_class,
            "entropy": entropy,
            "max_logit": max_logit,
            "logits": logits.tolist(),
        })
    return scores


def save_submission(images_np, filename="submission.npz"):
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


# ================================
# LOAD DATA
# ================================
def load_auxiliary():
    dataset = torch.load(os.path.join(WORK_DIR, "auxiliary_dataset.pt"), weights_only=False)
    images = dataset["images"].numpy().astype(np.float32)  # (1000, 3, 32, 32)
    labels = dataset["labels"].numpy()                      # (1000,)
    return images, labels


# ================================
# PHASE 1: Quick baseline (no API needed)
# ================================
def phase1_baseline(images, labels, do_submit=False):
    """
    Pick 100 diverse auxiliary images (10 per class).
    This is instant and gives a reasonable baseline.
    """
    print("=" * 60)
    print("PHASE 1: Quick baseline — 10 images per class")
    print("=" * 60)

    selected = []
    for cls in range(10):
        cls_indices = np.where(labels == cls)[0]
        # Pick 10 evenly spaced images from this class for diversity
        step = max(1, len(cls_indices) // 10)
        chosen = cls_indices[:10 * step:step][:10]
        selected.extend(chosen.tolist())
        print(f"  Class {cls}: picked {len(chosen)} images")

    selected = selected[:100]
    submission = images[selected]
    print(f"\n  Total: {len(selected)} images")

    path = save_submission(submission)

    if do_submit:
        submit_solution(path)

    return selected


# ================================
# PHASE 2: Score all 1000, pick best 100 (10 API calls)
# ================================
def phase2_score_all(images, labels, do_submit=False):
    """
    Send all 1000 auxiliary images in 10 batches.
    Score each by model confidence (max logit).
    Pick the 100 the model is most confident about.
    """
    print("=" * 60)
    print("PHASE 2: Score all 1000 auxiliary images")
    print("=" * 60)

    all_scores = []
    for batch_idx in range(10):
        start = batch_idx * 100
        end = start + 100
        cache_file = os.path.join(WORK_DIR, f"raw_api_response_batch{batch_idx}.json")

        if os.path.exists(cache_file):
            print(f"  Batch {batch_idx} (images {start}-{end-1}): cached")
            with open(cache_file) as f:
                result = json.load(f)
        else:
            print(f"  Batch {batch_idx} (images {start}-{end-1}): querying API...")
            result = query_logits(images[start:end])
            if result is None:
                print(f"  FAILED on batch {batch_idx}. Use cached batches so far.")
                break
            with open(cache_file, "w") as f:
                json.dump(result, f)

        batch_scores = extract_scores(result)
        for i, s in enumerate(batch_scores):
            s["global_index"] = start + i
            s["label"] = int(labels[start + i])
        all_scores.extend(batch_scores)

    print(f"\n  Scored {len(all_scores)} images total")

    if len(all_scores) == 0:
        print("  No scores available! Falling back to Phase 1.")
        return phase1_baseline(images, labels, do_submit)

    # Save all scores
    scores_path = os.path.join(WORK_DIR, "all_scores.json")
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"  Scores saved to: {scores_path}")

    # Pick top 100 by max confidence, ensuring class diversity
    # Sort by max_conf descending
    all_scores.sort(key=lambda x: -x["max_conf"])

    # Greedy selection: pick top per class to ensure diversity
    selected_indices = []
    class_counts = {c: 0 for c in range(10)}
    max_per_class = 15  # Allow some imbalance but not too much

    for s in all_scores:
        if len(selected_indices) >= 100:
            break
        cls = s["label"]
        if class_counts[cls] < max_per_class:
            selected_indices.append(s["global_index"])
            class_counts[cls] += 1

    # If we don't have 100 yet, fill with remaining highest-confidence
    if len(selected_indices) < 100:
        remaining = [s["global_index"] for s in all_scores
                     if s["global_index"] not in set(selected_indices)]
        selected_indices.extend(remaining[:100 - len(selected_indices)])

    print(f"\n  Selected {len(selected_indices)} images")
    print(f"  Class distribution: {dict(sorted(class_counts.items()))}")

    # Show top 10
    print("\n  Top 10 by confidence:")
    for s in all_scores[:10]:
        print(f"    Image {s['global_index']:4d} | class {s['label']} | "
              f"max_conf={s['max_conf']:.4f} | pred_class={s['max_class']} | "
              f"entropy={s['entropy']:.4f}")

    submission = images[selected_indices]
    path = save_submission(submission)

    if do_submit:
        submit_solution(path)

    return selected_indices


# ================================
# SCORING HELPER
# ================================
MAX_LOGIT_SCALE = 7.25   # observed max logit from data analysis
MAX_ENTROPY = np.log(10)  # max entropy for 10 classes

def composite_score(score_dict):
    """
    Composite score combining max_conf, max_logit, and entropy.
    Richer signal than max_conf alone.
    """
    mc = score_dict["max_conf"]
    ml = min(score_dict["max_logit"] / MAX_LOGIT_SCALE, 1.0)
    ent = 1.0 - min(score_dict["entropy"] / MAX_ENTROPY, 1.0)
    return 0.5 * mc + 0.3 * ml + 0.2 * ent


# ================================
# PHASE 4 HELPERS
# ================================
def ensure_all_scored(images, labels):
    """Score any missing batches (0-9) and rebuild all_scores.json with all 1000 images."""
    all_scores = []
    for batch_idx in range(10):
        start = batch_idx * 100
        end = start + 100
        cache_file = os.path.join(WORK_DIR, f"raw_api_response_batch{batch_idx}.json")

        if os.path.exists(cache_file):
            print(f"  Batch {batch_idx} (images {start}-{end-1}): cached")
            with open(cache_file) as f:
                result = json.load(f)
        else:
            print(f"  Batch {batch_idx} (images {start}-{end-1}): querying API...")
            result = query_logits(images[start:end])
            if result is None:
                print(f"  FAILED on batch {batch_idx}. Continuing with what we have.")
                continue
            with open(cache_file, "w") as f:
                json.dump(result, f)

        batch_scores = extract_scores(result)
        for i, s in enumerate(batch_scores):
            s["global_index"] = start + i
            s["label"] = int(labels[start + i])
        all_scores.extend(batch_scores)

    scores_path = os.path.join(WORK_DIR, "all_scores.json")
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"  Total scored: {len(all_scores)} images")
    return all_scores


def compute_kmeans_centroids(images, labels, n_centroids_per_class=3):
    """Compute k-means centroids per class using pure numpy. Returns (centroids, centroid_labels)."""
    all_centroids = []
    all_labels = []
    for cls in range(10):
        cls_mask = labels == cls
        cls_images = images[cls_mask]
        if len(cls_images) == 0:
            continue
        n_k = min(n_centroids_per_class, len(cls_images))
        flat = cls_images.reshape(len(cls_images), -1)  # (N_cls, 3072)

        # Initialize centers with evenly-spaced images
        rng = np.random.RandomState(seed=cls)
        init_idx = rng.choice(len(flat), n_k, replace=False)
        centers = flat[init_idx].copy()

        for _ in range(20):
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
    print(f"  Computed {len(centroids)} k-means centroids ({n_centroids_per_class} per class)")
    return centroids, centroid_labels


def reseed_candidates(candidates, candidate_labels, best_scores, momentum,
                      stagnation, noise_mults, all_scores, images, labels,
                      n_reseed=20, n_centroids_per_class=3):
    """Replace the worst n_reseed candidates with seeds from k-means centroids and top aux images."""
    centroids, centroid_labels = compute_kmeans_centroids(images, labels, n_centroids_per_class)

    # Find worst candidates
    indexed_scores = sorted(range(len(best_scores)), key=lambda i: best_scores[i])
    worst_indices = indexed_scores[:n_reseed]

    # Build seed pool — prioritize under-represented classes
    seed_pool = []
    current_class_counts = Counter(candidate_labels.tolist())

    # Score all_scores entries with composite
    all_scores_by_class = {c: [] for c in range(10)}
    for s in all_scores:
        s["composite"] = composite_score(s)
        all_scores_by_class[s["label"]].append(s)
    for c in all_scores_by_class:
        all_scores_by_class[c].sort(key=lambda x: -x["composite"])

    # Source 1: Centroids + top aux images for under-represented classes (count < 5)
    for cls in range(10):
        if current_class_counts.get(cls, 0) < 5:
            cls_centroid_mask = centroid_labels == cls
            for img in centroids[cls_centroid_mask]:
                seed_pool.append((img, cls, f"centroid_class{cls}"))
            for s in all_scores_by_class.get(cls, [])[:10]:
                seed_pool.append((images[s["global_index"]].copy(), cls,
                                  f"top_aux_class{cls}_idx{s['global_index']}"))

    # Source 2: Remaining centroids for all classes
    for i in range(len(centroids)):
        seed_pool.append((centroids[i], int(centroid_labels[i]), f"centroid_{i}"))

    # Deduplicate
    seen = set()
    unique_pool = []
    for img, lbl, desc in seed_pool:
        key = (lbl, desc)
        if key not in seen:
            seen.add(key)
            unique_pool.append((img, lbl, desc))
    seed_pool = unique_pool

    print(f"  Seed pool size: {len(seed_pool)}")
    print(f"  Replacing {len(worst_indices)} worst candidates")

    seed_idx = 0
    for worst_i in worst_indices:
        if seed_idx >= len(seed_pool):
            seed_idx = seed_idx % max(1, len(seed_pool))

        new_img, new_lbl, desc = seed_pool[seed_idx]
        old_score = best_scores[worst_i]
        old_label = candidate_labels[worst_i]

        candidates[worst_i] = new_img.copy()
        candidate_labels[worst_i] = new_lbl
        best_scores[worst_i] = -999.0
        momentum[worst_i] = np.zeros_like(momentum[worst_i])
        stagnation[worst_i] = 0
        noise_mults[worst_i] = 1.0

        print(f"    Slot {worst_i}: class {old_label} (score {old_score:.4f}) -> "
              f"class {new_lbl} ({desc})")
        seed_idx += 1

    return candidates, candidate_labels, best_scores, momentum, stagnation, noise_mults


# ================================
# PHASE 3: Optimized optimization loop
# ================================
# Key improvements over naive hill-climbing:
#   1. Batch splitting: 5 variants per image (20 images/group, rotating)
#   2. Simulated annealing: escape local optima
#   3. Adaptive noise: stagnant images get more noise
#   4. Momentum: track successful perturbation directions
#   5. Composite scoring: use full logit signal
#   6. Smarter perturbations: 5 strategies including channel shift & interpolation
#   7. Auto-submit on improvement

N_GROUPS = 5            # split 100 candidates into 5 groups of 20
N_VARIANTS = 5          # test 5 perturbations per image per turn
GROUP_SIZE = 20         # 20 images per group
INIT_TEMPERATURE = 0.1  # simulated annealing start temperature
MIN_TEMPERATURE = 0.005
TEMP_DECAY = 0.97
BASE_NOISE = 0.03
STAGNATION_BOOST = 3.0  # noise multiplier for stagnant images
CONVERGED_REDUCE = 0.5  # noise multiplier for high-scoring images
STAGNATION_THRESHOLD = 3  # rounds without improvement before boosting
MOMENTUM_DECAY = 0.7
MOMENTUM_MIX = 0.3


def generate_perturbation(image, noise_std, momentum, aux_images, aux_labels, label):
    """Generate one perturbed variant using one of 5 strategies."""
    perturbed = image.copy()
    strategy = np.random.choice(
        ["noise", "channel_shift", "interpolate", "block_swap", "momentum_noise"],
        p=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    if strategy == "noise":
        # Gaussian noise over entire image
        perturbed += np.random.randn(*perturbed.shape).astype(np.float32) * noise_std

    elif strategy == "channel_shift":
        # Shift entire R, G, or B channel — color matters for CIFAR
        ch = np.random.randint(0, 3)
        perturbed[ch] += np.random.randn() * noise_std * 2

    elif strategy == "interpolate":
        # Blend with a random same-class auxiliary image
        same_class = np.where(aux_labels == label)[0]
        if len(same_class) > 0:
            donor_idx = np.random.choice(same_class)
            weight = 0.05 + np.random.rand() * 0.15  # 5-20% blend
            perturbed = (1 - weight) * perturbed + weight * aux_images[donor_idx]

    elif strategy == "block_swap":
        # Swap a 4x4 block from a random auxiliary image
        donor = aux_images[np.random.randint(0, len(aux_images))]
        y = np.random.randint(0, 28)
        x = np.random.randint(0, 28)
        w = 0.3 + np.random.rand() * 0.4
        perturbed[:, y:y+4, x:x+4] = (
            (1 - w) * perturbed[:, y:y+4, x:x+4] +
            w * donor[:, y:y+4, x:x+4]
        )

    elif strategy == "momentum_noise":
        # Noise biased toward momentum direction (if available)
        random_part = np.random.randn(*perturbed.shape).astype(np.float32) * noise_std
        if momentum is not None and np.linalg.norm(momentum) > 1e-8:
            norm_momentum = momentum / (np.linalg.norm(momentum) + 1e-8) * noise_std
            perturbed += 0.5 * random_part + 0.5 * norm_momentum
        else:
            perturbed += random_part

    return np.clip(perturbed, 0.0, 1.0).astype(np.float32)


def phase3_optimize(images, labels, max_iters=50, do_submit=False):
    """
    Optimized iterative reconstruction with batch splitting, simulated
    annealing, adaptive noise, momentum, and auto-submit.
    """
    print("=" * 60)
    print("PHASE 3: Optimized reconstruction")
    print("=" * 60)

    progress_file = os.path.join(WORK_DIR, "optimize_progress.json")
    candidates_file = os.path.join(WORK_DIR, "optimize_candidates.npz")

    # --- Load or initialize state ---
    if os.path.exists(progress_file) and os.path.exists(candidates_file):
        print("  Resuming from saved progress...")
        with open(progress_file) as f:
            progress = json.load(f)
        data = np.load(candidates_file)
        candidates = data["images"]
        best_scores = np.array(progress["best_scores"])
        start_iter = progress["iteration"] + 1
        temperature = progress.get("temperature", INIT_TEMPERATURE)
        stagnation = np.array(progress.get("stagnation", [0] * 100))
        noise_mults = np.array(progress.get("noise_multipliers", [1.0] * 100))
        # Load momentum if available
        momentum_file = os.path.join(WORK_DIR, "optimize_momentum.npy")
        if os.path.exists(momentum_file):
            momentum = np.load(momentum_file)
        else:
            momentum = np.zeros_like(candidates)
        last_submit_score = progress.get("last_submit_score", 0.0)
        candidate_labels = np.array(progress.get("candidate_labels", [0] * 100))
        print(f"  Iter {start_iter}, avg={np.mean(best_scores):.4f}, temp={temperature:.4f}")
    else:
        # Seed from Phase 2 scores or Phase 1 baseline
        scores_path = os.path.join(WORK_DIR, "all_scores.json")
        if os.path.exists(scores_path):
            print("  Seeding from Phase 2 scores...")
            with open(scores_path) as f:
                all_scores = json.load(f)
            all_scores.sort(key=lambda x: -x["max_conf"])
            selected = []
            selected_labels = []
            class_counts = {c: 0 for c in range(10)}
            for s in all_scores:
                if len(selected) >= 100:
                    break
                if class_counts[s["label"]] < 15:
                    selected.append(s["global_index"])
                    selected_labels.append(s["label"])
                    class_counts[s["label"]] += 1
            if len(selected) < 100:
                remaining = [s for s in all_scores if s["global_index"] not in set(selected)]
                for s in remaining[:100 - len(selected)]:
                    selected.append(s["global_index"])
                    selected_labels.append(s["label"])
        else:
            print("  No Phase 2 scores. Seeding 10 per class...")
            selected = []
            selected_labels = []
            for cls in range(10):
                cls_idx = np.where(labels == cls)[0][:10]
                selected.extend(cls_idx.tolist())
                selected_labels.extend([cls] * len(cls_idx))

        candidates = images[selected[:100]].copy()
        candidate_labels = np.array(selected_labels[:100])
        best_scores = np.full(100, -999.0)
        momentum = np.zeros_like(candidates)
        stagnation = np.zeros(100, dtype=np.int32)
        noise_mults = np.ones(100, dtype=np.float32)
        temperature = INIT_TEMPERATURE
        start_iter = 0
        last_submit_score = 0.0

    print(f"\n  Optimization config:")
    print(f"    Groups: {N_GROUPS} x {GROUP_SIZE} images, {N_VARIANTS} variants each")
    print(f"    Temperature: {temperature:.4f} (annealing)")
    print(f"    Base noise: {BASE_NOISE}")
    print(f"    Rate limit: ~10 min between API calls")
    print()

    for iteration in range(start_iter, max_iters):
        iter_start = time.time()

        # --- Determine which group to optimize this iteration ---
        group_idx = iteration % N_GROUPS
        img_start = group_idx * GROUP_SIZE
        img_end = img_start + GROUP_SIZE
        group_indices = list(range(img_start, img_end))

        print(f"--- Iter {iteration} | Group {group_idx} (images {img_start}-{img_end-1}) | temp={temperature:.4f} ---")

        if iteration == start_iter and np.all(best_scores == -999.0):
            # First call: get baseline scores for ALL 100 candidates
            print("  Baseline scoring all 100 candidates...")
            batch = candidates.copy().astype(np.float32)
            result = query_logits(batch)
            if result is None:
                print("  API failed!")
                break
            scores = extract_scores(result)
            for i in range(100):
                best_scores[i] = composite_score(scores[i])
            print(f"  Baseline avg={np.mean(best_scores):.4f}, "
                  f"min={np.min(best_scores):.4f}, max={np.max(best_scores):.4f}")
            # Save and continue to next iteration (which will start perturbations)
        else:
            # --- Generate N_VARIANTS perturbations per image in this group ---
            batch = np.zeros((GROUP_SIZE * N_VARIANTS, 3, 32, 32), dtype=np.float32)
            for local_i, global_i in enumerate(group_indices):
                img_noise = BASE_NOISE * noise_mults[global_i]
                lbl = int(candidate_labels[global_i])
                for v in range(N_VARIANTS):
                    batch[local_i * N_VARIANTS + v] = generate_perturbation(
                        candidates[global_i], img_noise,
                        momentum[global_i], images, labels, lbl
                    )

            # --- Query API (sends GROUP_SIZE * N_VARIANTS = 100 images) ---
            result = query_logits(batch)
            if result is None:
                print("  API failed! Saving and stopping.")
                break

            scores = extract_scores(result)

            # --- Pick best variant per image, apply acceptance rule ---
            improved = 0
            annealing_accepts = 0
            for local_i, global_i in enumerate(group_indices):
                # Find best variant for this image
                variant_scores = []
                for v in range(N_VARIANTS):
                    s = scores[local_i * N_VARIANTS + v]
                    variant_scores.append(composite_score(s))

                best_variant_idx = int(np.argmax(variant_scores))
                best_variant_score = variant_scores[best_variant_idx]
                best_variant_img = batch[local_i * N_VARIANTS + best_variant_idx]

                delta = best_variant_score - best_scores[global_i]

                if delta > 0:
                    # Improvement — always accept
                    direction = best_variant_img - candidates[global_i]
                    momentum[global_i] = MOMENTUM_DECAY * momentum[global_i] + MOMENTUM_MIX * direction
                    candidates[global_i] = best_variant_img
                    best_scores[global_i] = best_variant_score
                    stagnation[global_i] = 0
                    noise_mults[global_i] = 1.0
                    improved += 1
                elif temperature > 0 and np.random.rand() < np.exp(delta / temperature):
                    # Simulated annealing — accept worse with probability
                    candidates[global_i] = best_variant_img
                    best_scores[global_i] = best_variant_score
                    stagnation[global_i] = 0
                    annealing_accepts += 1
                else:
                    # Reject — track stagnation
                    stagnation[global_i] += 1

                # Adaptive noise
                if stagnation[global_i] >= STAGNATION_THRESHOLD:
                    noise_mults[global_i] = STAGNATION_BOOST
                elif best_scores[global_i] > 0.95:
                    noise_mults[global_i] = CONVERGED_REDUCE

            # Decay temperature
            temperature = max(MIN_TEMPERATURE, temperature * TEMP_DECAY)

            elapsed = time.time() - iter_start
            avg_score = float(np.mean(best_scores))
            print(f"  Improved: {improved}/{GROUP_SIZE} | SA accepts: {annealing_accepts} | "
                  f"Avg: {avg_score:.4f} | Min: {np.min(best_scores):.4f} | Max: {np.max(best_scores):.4f} | "
                  f"{elapsed:.0f}s")

        # --- Save progress ---
        np.savez_compressed(candidates_file, images=candidates.astype(np.float32))
        np.save(os.path.join(WORK_DIR, "optimize_momentum.npy"), momentum)
        avg_score = float(np.mean(best_scores))
        with open(progress_file, "w") as f:
            json.dump({
                "iteration": iteration,
                "temperature": float(temperature),
                "best_scores": best_scores.tolist(),
                "stagnation": stagnation.tolist(),
                "noise_multipliers": noise_mults.tolist(),
                "candidate_labels": candidate_labels.tolist(),
                "last_submit_score": last_submit_score,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

        path = save_submission(candidates)

        # --- Auto-submit on significant improvement ---
        if avg_score > last_submit_score + 0.005:
            print(f"  ** Score improved {last_submit_score:.4f} -> {avg_score:.4f}, auto-submitting!")
            submit_solution(path)
            last_submit_score = avg_score

        print()

    # Final submit
    if do_submit:
        submit_solution(os.path.join(WORK_DIR, "submission.npz"))


# ================================
# PHASE 4: Reseed + Reheat
# ================================
PHASE4_TEMPERATURE = 0.08
PHASE4_N_RESEED = 20
PHASE4_N_CENTROIDS = 3

def phase4_reseed_and_reheat(images, labels, max_iters=50, n_reseed=PHASE4_N_RESEED, do_submit=False):
    """
    Phase 4: Complete scoring, reseed weak candidates, reheat, and optimize.
    Sub-steps tracked in phase4_state.json for resumability.
    """
    print("=" * 60)
    print("PHASE 4: Reseed + Reheat")
    print("=" * 60)

    progress_file = os.path.join(WORK_DIR, "optimize_progress.json")
    candidates_file = os.path.join(WORK_DIR, "optimize_candidates.npz")
    phase4_state_file = os.path.join(WORK_DIR, "phase4_state.json")

    # --- Check if Phase 4 is resuming ---
    if os.path.exists(phase4_state_file):
        with open(phase4_state_file) as f:
            p4_state = json.load(f)
        phase4_step = p4_state.get("completed_step", 0)
        print(f"  Resuming Phase 4 from step {phase4_step + 1}")
    else:
        phase4_step = 0
        p4_state = {}

    # ========== STEP 1: Score remaining images ==========
    if phase4_step < 1:
        print("\n--- Step 1: Score remaining auxiliary images ---")
        all_scores = ensure_all_scored(images, labels)
        p4_state["completed_step"] = 1
        p4_state["total_scored"] = len(all_scores)
        with open(phase4_state_file, "w") as f:
            json.dump(p4_state, f, indent=2)
    else:
        scores_path = os.path.join(WORK_DIR, "all_scores.json")
        with open(scores_path) as f:
            all_scores = json.load(f)

    # ========== STEPS 2-4: Reseed candidates ==========
    if phase4_step < 4:
        print("\n--- Steps 2-4: Reseed candidates ---")

        # Load current Phase 3 state
        if os.path.exists(progress_file) and os.path.exists(candidates_file):
            with open(progress_file) as f:
                progress = json.load(f)
            data = np.load(candidates_file)
            candidates = data["images"].copy()
            best_scores = np.array(progress["best_scores"])
            candidate_labels = np.array(progress.get("candidate_labels", [0] * 100))
            stagnation = np.array(progress.get("stagnation", [0] * 100))
            noise_mults = np.array(progress.get("noise_multipliers", [1.0] * 100))
            momentum_file = os.path.join(WORK_DIR, "optimize_momentum.npy")
            momentum = np.load(momentum_file) if os.path.exists(momentum_file) else np.zeros_like(candidates)
            last_submit_score = progress.get("last_submit_score", 0.0)
        else:
            # No Phase 3 state — build from scratch
            print("  No Phase 3 state found. Building from Phase 2 scores.")
            all_scores.sort(key=lambda x: -x["max_conf"])
            selected = []
            selected_labels = []
            class_counts = {c: 0 for c in range(10)}
            for s in all_scores:
                if len(selected) >= 100:
                    break
                if class_counts.get(s["label"], 0) < 15:
                    selected.append(s["global_index"])
                    selected_labels.append(s["label"])
                    class_counts[s["label"]] = class_counts.get(s["label"], 0) + 1
            if len(selected) < 100:
                remaining = [s["global_index"] for s in all_scores if s["global_index"] not in set(selected)]
                for idx in remaining[:100 - len(selected)]:
                    selected.append(idx)
                    selected_labels.append(int(labels[idx]))
            candidates = images[selected[:100]].copy()
            candidate_labels = np.array(selected_labels[:100])
            best_scores = np.full(100, -999.0)
            momentum = np.zeros_like(candidates)
            stagnation = np.zeros(100, dtype=np.int32)
            noise_mults = np.ones(100, dtype=np.float32)
            last_submit_score = 0.0

        pre_dist = Counter(candidate_labels.tolist())
        print(f"  Pre-reseed class distribution: {dict(sorted(pre_dist.items()))}")

        candidates, candidate_labels, best_scores, momentum, stagnation, noise_mults = \
            reseed_candidates(
                candidates, candidate_labels, best_scores, momentum,
                stagnation, noise_mults, all_scores, images, labels,
                n_reseed=n_reseed, n_centroids_per_class=PHASE4_N_CENTROIDS,
            )

        post_dist = Counter(candidate_labels.tolist())
        print(f"  Post-reseed class distribution: {dict(sorted(post_dist.items()))}")

        # Save reseeded state
        np.savez_compressed(candidates_file, images=candidates.astype(np.float32))
        np.save(os.path.join(WORK_DIR, "optimize_momentum.npy"), momentum)

        p4_state["completed_step"] = 4
        p4_state["last_submit_score"] = last_submit_score
        with open(phase4_state_file, "w") as f:
            json.dump(p4_state, f, indent=2)
    else:
        # Resuming step 5 — reload state
        with open(progress_file) as f:
            progress = json.load(f)
        data = np.load(candidates_file)
        candidates = data["images"].copy()
        best_scores = np.array(progress["best_scores"])
        candidate_labels = np.array(progress.get("candidate_labels", [0] * 100))
        stagnation = np.array(progress.get("stagnation", [0] * 100))
        noise_mults = np.array(progress.get("noise_multipliers", [1.0] * 100))
        momentum_file = os.path.join(WORK_DIR, "optimize_momentum.npy")
        momentum = np.load(momentum_file) if os.path.exists(momentum_file) else np.zeros_like(candidates)
        last_submit_score = p4_state.get("last_submit_score", 0.0)

    # ========== STEP 5: Reheat and optimize ==========
    print("\n--- Step 5: Reheat annealing and optimize ---")

    if "phase4_iteration" in p4_state:
        start_iter = p4_state["phase4_iteration"] + 1
        temperature = p4_state.get("phase4_temperature", PHASE4_TEMPERATURE)
        last_submit_score = p4_state.get("last_submit_score", last_submit_score)
    else:
        start_iter = 0
        temperature = PHASE4_TEMPERATURE
        stagnation[:] = 0

    print(f"  Reheated temperature: {temperature:.4f}")
    print(f"  Starting iteration: {start_iter}, max: {max_iters}")
    print()

    for iteration in range(start_iter, max_iters):
        iter_start = time.time()

        group_idx = iteration % N_GROUPS
        img_start = group_idx * GROUP_SIZE
        img_end = img_start + GROUP_SIZE
        group_indices = list(range(img_start, img_end))

        print(f"--- Phase4 Iter {iteration} | Group {group_idx} "
              f"(images {img_start}-{img_end-1}) | temp={temperature:.4f} ---")

        # Check if any candidates need baseline scoring (reseeded = -999)
        needs_baseline = any(best_scores[i] == -999.0 for i in range(100))

        if needs_baseline:
            print("  Scoring all 100 candidates (includes reseeded)...")
            result = query_logits(candidates.astype(np.float32))
            if result is None:
                print("  API failed!")
                break
            scores = extract_scores(result)
            for i in range(100):
                new_score = composite_score(scores[i])
                if best_scores[i] == -999.0 or new_score > best_scores[i]:
                    best_scores[i] = new_score
            print(f"  Baseline avg={np.mean(best_scores):.4f}, "
                  f"min={np.min(best_scores):.4f}, max={np.max(best_scores):.4f}")
        else:
            # Normal perturbation cycle
            batch = np.zeros((GROUP_SIZE * N_VARIANTS, 3, 32, 32), dtype=np.float32)
            for local_i, global_i in enumerate(group_indices):
                img_noise = BASE_NOISE * noise_mults[global_i]
                lbl = int(candidate_labels[global_i])
                for v in range(N_VARIANTS):
                    batch[local_i * N_VARIANTS + v] = generate_perturbation(
                        candidates[global_i], img_noise,
                        momentum[global_i], images, labels, lbl
                    )

            result = query_logits(batch)
            if result is None:
                print("  API failed! Saving and stopping.")
                break

            scores = extract_scores(result)

            improved = 0
            annealing_accepts = 0
            for local_i, global_i in enumerate(group_indices):
                variant_scores = []
                for v in range(N_VARIANTS):
                    s = scores[local_i * N_VARIANTS + v]
                    variant_scores.append(composite_score(s))

                best_variant_idx = int(np.argmax(variant_scores))
                best_variant_score = variant_scores[best_variant_idx]
                best_variant_img = batch[local_i * N_VARIANTS + best_variant_idx]

                delta = best_variant_score - best_scores[global_i]

                if delta > 0:
                    direction = best_variant_img - candidates[global_i]
                    momentum[global_i] = MOMENTUM_DECAY * momentum[global_i] + MOMENTUM_MIX * direction
                    candidates[global_i] = best_variant_img
                    best_scores[global_i] = best_variant_score
                    stagnation[global_i] = 0
                    noise_mults[global_i] = 1.0
                    improved += 1
                elif temperature > 0 and np.random.rand() < np.exp(delta / temperature):
                    candidates[global_i] = best_variant_img
                    best_scores[global_i] = best_variant_score
                    stagnation[global_i] = 0
                    annealing_accepts += 1
                else:
                    stagnation[global_i] += 1

                if stagnation[global_i] >= STAGNATION_THRESHOLD:
                    noise_mults[global_i] = STAGNATION_BOOST
                elif best_scores[global_i] > 0.95:
                    noise_mults[global_i] = CONVERGED_REDUCE

            temperature = max(MIN_TEMPERATURE, temperature * TEMP_DECAY)

            elapsed = time.time() - iter_start
            avg_score = float(np.mean(best_scores))
            print(f"  Improved: {improved}/{GROUP_SIZE} | SA accepts: {annealing_accepts} | "
                  f"Avg: {avg_score:.4f} | Min: {np.min(best_scores):.4f} | "
                  f"Max: {np.max(best_scores):.4f} | {elapsed:.0f}s")

        # --- Save progress ---
        np.savez_compressed(candidates_file, images=candidates.astype(np.float32))
        np.save(os.path.join(WORK_DIR, "optimize_momentum.npy"), momentum)
        avg_score = float(np.mean(best_scores))
        with open(progress_file, "w") as f:
            json.dump({
                "iteration": iteration + 1000,
                "temperature": float(temperature),
                "best_scores": best_scores.tolist(),
                "stagnation": stagnation.tolist(),
                "noise_multipliers": noise_mults.tolist(),
                "candidate_labels": candidate_labels.tolist(),
                "last_submit_score": last_submit_score,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

        p4_state["phase4_iteration"] = iteration
        p4_state["phase4_temperature"] = float(temperature)
        p4_state["last_submit_score"] = last_submit_score
        with open(phase4_state_file, "w") as f:
            json.dump(p4_state, f, indent=2)

        path = save_submission(candidates)

        if avg_score > last_submit_score + 0.005:
            print(f"  ** Score improved {last_submit_score:.4f} -> {avg_score:.4f}, auto-submitting!")
            submit_solution(path)
            last_submit_score = avg_score

        print()

    # Final submit
    if do_submit:
        submit_solution(os.path.join(WORK_DIR, "submission.npz"))


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data reconstruction attack")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="1=baseline, 2=score all, 3=optimize, 4=reseed+reheat")
    parser.add_argument("--submit", action="store_true", help="Auto-submit after creating")
    parser.add_argument("--max-iters", type=int, default=50, help="Max iterations for Phase 3/4")
    parser.add_argument("--n-reseed", type=int, default=20, help="Number of candidates to reseed (Phase 4)")
    args = parser.parse_args()

    images, labels = load_auxiliary()
    print(f"Loaded {len(images)} auxiliary images ({images.shape})")
    print()

    if args.phase == 1:
        phase1_baseline(images, labels, do_submit=args.submit)
    elif args.phase == 2:
        phase2_score_all(images, labels, do_submit=args.submit)
    elif args.phase == 3:
        phase3_optimize(images, labels, max_iters=args.max_iters, do_submit=args.submit)
    elif args.phase == 4:
        phase4_reseed_and_reheat(images, labels, max_iters=args.max_iters,
                                 n_reseed=args.n_reseed, do_submit=args.submit)
