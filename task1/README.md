# Task 1: Data Reconstruction Challenge

## Overview

This task involved reconstructing training images from a CIFAR-10-like dataset using only black-box classifier access. The goal was to generate 100 images (10 per class) that minimize the Mean Squared Error (MSE) to the nearest image in the hidden training dataset.

## Challenge Description

### Constraints
- **Black-box classifier**: We could query a pre-trained neural network to get logits for submitted images
- **Auxiliary dataset**: 1000 labeled images from the same distribution (but NOT in the training set)
- **API rate limits**: Limited queries per time period
- **Evaluation metric**: MSE matching — for each reconstructed image, find its nearest neighbor in the training set and compute MSE. Lower is better.

### Resources Available
1. `auxiliary_dataset.pt`: 1000 labeled images (100 per class) from the same distribution
2. Logits API: Query up to 100 images at a time to get classifier predictions
3. Submission API: Submit 100 reconstructed images for evaluation

## Development Journey

Our approach evolved through several distinct phases:

### Phase 1: Auxiliary Image Baseline
**Files**: `basic.py`, `visualize.py`

Started with the simplest possible approach: select the best 10 images per class from the auxiliary dataset. This established a baseline score but was limited by the quality of the auxiliary images.

**Key insight**: The auxiliary dataset provides good coverage but isn't optimized for MSE to the training set.

### Phase 2: Logit-Guided Optimization
**Files**: `reconstruct.py`, `solve.py`

Attempted to improve images through gradient-free optimization:
- Started with auxiliary images
- Applied random perturbations (Gaussian noise, block swaps, color shifts)
- Used classifier logits as a proxy for image quality
- Accepted perturbations that increased confidence for the target class

**Challenge encountered**: Optimizing for classifier confidence doesn't correlate well with MSE to training images. The classifier is invariant to many pixel-level variations, so maximizing logits often moves images away from the pixel-space centers we're trying to match.

### Phase 3: Clustering-Based Pivot
**Files**: `cluster_submission_test.py`, `strategy_pivot_plan.txt`

**Critical discovery**: A pure clustering approach (k-means centroids) outperformed our heavily optimized candidates!

**Root cause analysis**:
- The MSE metric measures pixel-space distance to the nearest training image
- K-means centroids minimize average squared distance to cluster members
- This directly aligns with the MSE evaluation metric
- Logit-based optimization was pushing images toward decision boundaries rather than pixel-space centers

**New strategy**: Cluster-first pipeline
1. Run k-means clustering per class on auxiliary images
2. Use cluster centroids as primary candidates
3. Also extract medoids (actual images closest to centroids)
4. Apply only minimal, drift-bounded optimization

### Phase 4: Centroid vs Medoid Experiments
**Files**: `improve_centroids.py`, `improve_centroids_v2.py`, `ablation_variants.py`, `ablation_medoid_ratio.py`

Systematically tested different clustering configurations:
- **Centroids**: Averaged cluster centers (blurry but well-centered)
- **Medoids**: Actual images closest to centroids (sharp but potentially off-center)
- **Multi-seed clustering**: Run k-means with different random seeds, merge best results
- **Varying k**: Experimented with different numbers of clusters per class (k=5, 10, 15, 20)

**Finding**: A blend of centroids and medoids worked best, with multi-seed clustering providing better coverage.

### Phase 5: Diversity Expansion
**Files**: `expand_coverage.py`, `diversity_cleanup.py`, `final_safe_variant.py`

Focused on maximizing coverage of the image space:
- Identified and removed duplicate or highly similar images
- Replaced redundant images with alternatives that maximize pairwise distance
- Ensured each class had diverse representatives
- Used MSE-based diversity metrics to guide selection

**Strategy**:
1. Within each class, compute pairwise MSE between all candidates
2. Find the most similar pairs
3. Replace the weaker member with the best alternative from the auxiliary pool
4. "Best alternative" = image farthest from all currently selected images

### Phase 6: Final Logits-Based Refinement
**Files**: `final_logits_swap.py`, `solve_v2.py`

Final optimization pass with minimal API usage:
1. Query logits for the full 100-image submission
2. Identify images with lowest confidence for their target class
3. Swap out the weakest ~10-20 images with better alternatives
4. Better alternatives = images with higher confidence AND good MSE coverage

**Key principle**: Use logits for final validation and targeted fixes, not for primary optimization.

## Final Solution

Our final pipeline combines all insights:

1. **Cluster-first approach**: Generate candidates via multi-seed k-means clustering
   - Run k-means with k=10 clusters per class
   - Use multiple random seeds to get diverse clusterings
   - Extract both centroids and medoids from each clustering

2. **MSE-based selection**: Build initial 100-image set by selecting candidates with best proxy MSE
   - Proxy MSE = MSE to nearest auxiliary image (since we can't access training data)
   - Prioritize centroids over medoids for stability

3. **Diversity maximization**: Replace redundant images to maximize coverage
   - Remove near-duplicates (MSE < threshold within same class)
   - Replace with images that maximize minimum distance to selected set

4. **Minimal logits optimization**: One-shot logits query for validation
   - Identify low-confidence images
   - Swap out bottom performers if better alternatives exist
   - Preserve high-diversity, low-MSE candidates even if confidence is moderate

## Repository Structure

### Core Pipeline Files
- **`solve_v2.py`**: Main cluster-first pipeline (final approach)
- **`final_safe_variant.py`**: Diversity-focused variant with safety checks
- **`final_logits_swap.py`**: Logits-based final refinement

### Development Evolution Files
- **`basic.py`**: Initial auxiliary-image baseline
- **`reconstruct.py`**: Early reconstruction experiments
- **`solve.py`**: Original logit-guided optimization pipeline
- **`cluster_submission_test.py`**: Clustering approach prototype

### Utility Files
- **`task_template.py`**: Original competition template with API usage examples
- **`visualize.py`**: Visualization utilities for exploring the dataset
- **`infinity.py`** / **`generate_infinity_images.py`**: Helper utilities

### Data Files
- **`auxiliary_dataset.pt`**: 1000 labeled images (100 per class)
- **`strategy_pivot_plan.txt`**: Strategic document outlining the cluster-first pivot

### Archived Files
See `archive_unused/` for intermediate experiments, ablation studies, cached API responses, and intermediate submissions.

## How to Run

### Generate a submission using the final pipeline:
```bash
python task1/solve_v2.py
```

### Options:
```bash
# Clustering only (no API calls)
python task1/solve_v2.py --skip-optimize

# Use different number of clusters
python task1/solve_v2.py --k 15

# Use medoids instead of centroids
python task1/solve_v2.py --use-medoids

# Submit after generation
python task1/solve_v2.py --submit
```

### Run the diversity-focused variant:
```bash
python task1/final_safe_variant.py
python task1/final_safe_variant.py --submit
```

### Run logits-based refinement:
```bash
python task1/final_logits_swap.py
```

## Key Lessons Learned

1. **Evaluation metric matters**: Always align your optimization objective with the actual evaluation metric. Optimizing classifier confidence (logits) doesn't necessarily improve MSE to training data.

2. **Domain-specific approaches win**: For MSE-based reconstruction, clustering methods that explicitly minimize squared distances outperform generic optimization.

3. **Diversity is critical**: When reconstructing multiple images, coverage matters as much as individual quality. Removing redundancy and maximizing inter-image distance improved scores.

4. **Gradient-free can beat gradient-based**: Even with black-box access to a neural network, classical unsupervised methods (k-means) can outperform guided optimization when the objectives misalign.

5. **API efficiency matters**: Under rate limits, it's better to use clustering (no API) for 90% of the work and save API calls for validation and targeted refinement.

6. **Centroids vs Medoids trade-off**:
   - Centroids are optimal centers but can be blurry
   - Medoids are sharp but may not perfectly represent the cluster
   - A blend often works best

7. **Multi-seed clustering helps**: Running k-means with different initializations and merging results provides better coverage than a single run.

## Reproduction Notes

To fully reproduce the development process:

1. Start with the auxiliary dataset baseline (`basic.py`)
2. Explore logit-guided optimization (`solve.py`)
3. Pivot to clustering when optimization underperforms (`cluster_submission_test.py`)
4. Refine with diversity expansion (`expand_coverage.py`, `final_safe_variant.py`)
5. Add final logits-based validation (`final_logits_swap.py`)

The `archive_unused/` directory contains all intermediate experiments, ablation studies, and cached responses from the development process.

## Competition Details

- **Task ID**: 12-data-reconstruction
- **Metric**: Mean Squared Error (MSE) to nearest training image
- **Submission format**: `.npz` file containing 100 images of shape (100, 3, 32, 32), dtype float32
- **Image range**: [0, 1]

## API Usage

See `task_template.py` for complete API documentation. Key endpoints:

- **Logits API**: `POST /{task_id}/logits` - Get classifier predictions for up to 100 images
- **Submission API**: `POST /submit/{task_id}` - Submit final reconstruction for evaluation

Rate limits apply - use API calls sparingly and cache responses when possible.
