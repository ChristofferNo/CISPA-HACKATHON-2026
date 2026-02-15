"""
diversity_cleanup.py — Detect near-duplicate candidates and suggest replacements.

Offline analysis tool. Does NOT modify optimize_candidates.npz or any running
optimization state. No API calls. Safe to run between optimization rounds.

Usage:
    python task1/diversity_cleanup.py
    python task1/diversity_cleanup.py --threshold 0.003
    python task1/diversity_cleanup.py --top 30
"""

import numpy as np
import json
import os
import sys

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def load_candidates():
    path = os.path.join(WORK_DIR, "optimize_candidates.npz")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    data = np.load(path)
    return data["images"]  # (100, 3, 32, 32)


def load_progress():
    path = os.path.join(WORK_DIR, "optimize_progress.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Scores unavailable.")
        return None
    with open(path) as f:
        return json.load(f)


def load_auxiliary():
    """Load auxiliary dataset if available (for replacement suggestions)."""
    path = os.path.join(WORK_DIR, "auxiliary_dataset.pt")
    if not os.path.exists(path):
        return None, None
    try:
        import torch
        dataset = torch.load(path, weights_only=False)
        images = dataset["images"].numpy().astype(np.float32)
        labels = dataset["labels"].numpy()
        return images, labels
    except Exception as e:
        print(f"WARNING: Could not load auxiliary dataset: {e}")
        return None, None


def compute_pairwise_mse(candidates):
    """Compute full 100x100 pairwise MSE matrix."""
    flat = candidates.reshape(100, -1)  # (100, 3072)
    dists = ((flat[:, None, :] - flat[None, :, :]) ** 2).mean(axis=2)
    return dists


def find_duplicate_pairs(dists, threshold=0.002):
    """Find all (i, j) pairs where MSE < threshold, i < j."""
    pairs = []
    n = dists.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < threshold:
                pairs.append((i, j, float(dists[i, j])))
    pairs.sort(key=lambda x: x[2])
    return pairs


def pick_weaker(i, j, scores):
    """Return the index of the weaker candidate (lower score). Tie: pick j."""
    if scores is None:
        return j
    if scores[i] <= scores[j]:
        return i
    return j


def compute_kmeans_centroids(images, labels, n_per_class=3):
    """Compute k-means centroids per class from auxiliary images."""
    all_centroids = []
    all_labels = []
    for cls in range(10):
        cls_images = images[labels == cls]
        if len(cls_images) == 0:
            continue
        n_k = min(n_per_class, len(cls_images))
        flat = cls_images.reshape(len(cls_images), -1)

        rng = np.random.RandomState(seed=cls)
        idx = rng.choice(len(flat), n_k, replace=False)
        centers = flat[idx].copy()

        for _ in range(20):
            d = np.zeros((len(flat), n_k), dtype=np.float32)
            for k in range(n_k):
                d[:, k] = ((flat - centers[k]) ** 2).sum(axis=1)
            assign = np.argmin(d, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(n_k):
                mask = assign == k
                new_centers[k] = flat[mask].mean(axis=0) if mask.any() else centers[k]
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        all_centroids.append(centers.reshape(n_k, 3, 32, 32))
        all_labels.extend([cls] * n_k)

    centroids = np.clip(np.concatenate(all_centroids, axis=0).astype(np.float32), 0.0, 1.0)
    return centroids, np.array(all_labels)


def find_unused_auxiliaries(candidates, aux_images, candidate_labels, aux_labels):
    """Find auxiliary images not already close to any candidate (same class)."""
    flat_cand = candidates.reshape(100, -1)
    unused = []
    for idx in range(len(aux_images)):
        cls = int(aux_labels[idx])
        same_class_cand = [i for i in range(100) if candidate_labels[i] == cls]
        if not same_class_cand:
            unused.append(idx)
            continue
        flat_aux = aux_images[idx].reshape(1, -1)
        flat_sc = flat_cand[same_class_cand]
        mse_to_cands = ((flat_aux - flat_sc) ** 2).mean(axis=1)
        if mse_to_cands.min() > 0.01:
            unused.append(idx)
    return unused


def suggest_replacements(to_replace, candidate_labels, candidates, aux_images, aux_labels):
    """
    For each slot to replace, suggest the best unused auxiliary image or
    k-means centroid of the same class that is maximally distant from
    remaining candidates.
    """
    suggestions = []
    flat_cand = candidates.reshape(100, -1)

    # Build pool: unused aux images + centroids
    pool = []  # (image, label, source_desc)

    if aux_images is not None:
        unused_idx = find_unused_auxiliaries(candidates, aux_images, candidate_labels, aux_labels)
        for idx in unused_idx:
            pool.append((aux_images[idx], int(aux_labels[idx]), f"aux_image_{idx}"))

        centroids, centroid_labels = compute_kmeans_centroids(aux_images, aux_labels)
        for i in range(len(centroids)):
            pool.append((centroids[i], int(centroid_labels[i]), f"kmeans_centroid_{i}"))

    if not pool:
        return suggestions

    # For each slot, find best replacement from pool matching the class
    for slot in to_replace:
        cls = int(candidate_labels[slot])
        class_pool = [(img, lbl, desc) for img, lbl, desc in pool if lbl == cls]
        if not class_pool:
            # Fall back to any class
            class_pool = pool

        # Pick the pool entry most distant from all current candidates
        best_dist = -1
        best_entry = None
        for img, lbl, desc in class_pool:
            flat_img = img.reshape(1, -1)
            min_dist = ((flat_img - flat_cand) ** 2).mean(axis=1).min()
            if min_dist > best_dist:
                best_dist = min_dist
                best_entry = (desc, lbl, float(min_dist))

        if best_entry:
            suggestions.append({
                "slot": int(slot),
                "current_class": cls,
                "replacement_source": best_entry[0],
                "replacement_class": best_entry[1],
                "min_mse_to_candidates": best_entry[2],
            })

    return suggestions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect near-duplicate candidates")
    parser.add_argument("--threshold", type=float, default=0.002,
                        help="MSE threshold for near-duplicate detection (default: 0.002)")
    parser.add_argument("--top", type=int, default=0,
                        help="Also show the N weakest candidates (default: 0 = off)")
    parser.add_argument("--save", action="store_true",
                        help="Save diversity_report.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Diversity Cleanup — Near-Duplicate Detection")
    print("=" * 60)

    # Load data
    candidates = load_candidates()
    progress = load_progress()
    scores = progress["best_scores"] if progress else None
    candidate_labels = progress.get("candidate_labels") if progress else None

    print(f"  Candidates: {candidates.shape}")
    if scores:
        print(f"  Score range: {min(scores):.4f} — {max(scores):.4f} (avg {np.mean(scores):.4f})")
    print()

    # Compute pairwise MSE
    print("Computing pairwise MSE...")
    dists = compute_pairwise_mse(candidates)
    print(f"  Distance matrix: {dists.shape}")
    print(f"  MSE range (off-diagonal): {dists[dists > 0].min():.6f} — {dists.max():.6f}")
    print()

    # Find near-duplicates
    pairs = find_duplicate_pairs(dists, threshold=args.threshold)
    print(f"Near-duplicate pairs (MSE < {args.threshold}):")
    if not pairs:
        print("  None found. Candidates are sufficiently diverse.")
    else:
        print(f"  Found {len(pairs)} pair(s):\n")
        to_replace = set()
        for i, j, mse in pairs:
            weaker = pick_weaker(i, j, scores)
            stronger = i if weaker == j else j
            score_i = f"{scores[i]:.4f}" if scores else "?"
            score_j = f"{scores[j]:.4f}" if scores else "?"
            label_i = candidate_labels[i] if candidate_labels else "?"
            label_j = candidate_labels[j] if candidate_labels else "?"
            print(f"    [{i}] (class {label_i}, score {score_i}) <-> "
                  f"[{j}] (class {label_j}, score {score_j})  MSE={mse:.6f}")
            print(f"      -> Mark [{weaker}] for replacement (weaker)")
            to_replace.add(weaker)
        print(f"\n  Candidates marked for replacement: {sorted(to_replace)}")

    # Optionally show weakest candidates
    if args.top > 0 and scores:
        print(f"\n{'=' * 60}")
        print(f"Bottom {args.top} candidates by score:")
        print("=" * 60)
        ranked = sorted(range(100), key=lambda i: scores[i])
        for rank, idx in enumerate(ranked[:args.top]):
            lbl = candidate_labels[idx] if candidate_labels else "?"
            print(f"  #{rank+1:2d}  slot [{idx:2d}]  class {lbl}  score {scores[idx]:.4f}")

    # Suggest replacements
    to_replace_list = sorted(to_replace) if pairs else []
    if to_replace_list:
        print(f"\n{'=' * 60}")
        print("Replacement Suggestions")
        print("=" * 60)

        aux_images, aux_labels = load_auxiliary()
        if aux_images is not None:
            suggestions = suggest_replacements(
                to_replace_list, candidate_labels, candidates,
                aux_images, aux_labels,
            )
            if suggestions:
                for s in suggestions:
                    print(f"  Slot [{s['slot']}] (class {s['current_class']}) -> "
                          f"{s['replacement_source']} (class {s['replacement_class']}, "
                          f"min_mse_to_cands={s['min_mse_to_candidates']:.6f})")
            else:
                print("  No suitable replacements found in pool.")
        else:
            print("  Auxiliary dataset not available — cannot suggest replacements.")
            suggestions = []
    else:
        suggestions = []

    # Class distribution summary
    if candidate_labels:
        print(f"\n{'=' * 60}")
        print("Class Distribution")
        print("=" * 60)
        from collections import Counter
        dist = Counter(candidate_labels)
        for cls in range(10):
            count = dist.get(cls, 0)
            bar = "#" * count
            print(f"  Class {cls}: {count:2d}  {bar}")

    # Save report
    if args.save:
        report = {
            "threshold": args.threshold,
            "num_duplicate_pairs": len(pairs),
            "duplicate_pairs": [
                {"i": i, "j": j, "mse": mse,
                 "weaker": pick_weaker(i, j, scores),
                 "score_i": scores[i] if scores else None,
                 "score_j": scores[j] if scores else None}
                for i, j, mse in pairs
            ],
            "candidates_to_replace": to_replace_list,
            "suggestions": suggestions,
            "score_stats": {
                "min": float(min(scores)),
                "max": float(max(scores)),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
            } if scores else None,
            "class_distribution": dict(Counter(candidate_labels)) if candidate_labels else None,
        }
        report_path = os.path.join(WORK_DIR, "diversity_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

    print(f"\n{'=' * 60}")
    print("Done. No files were modified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
