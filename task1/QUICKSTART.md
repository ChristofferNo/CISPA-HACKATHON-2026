# Quick Start Guide

New to this repository? Start here!

## What is this?

This is our solution to a **data reconstruction challenge**: given only black-box access to a trained classifier, reconstruct 100 training images (10 per class) that minimize Mean Squared Error (MSE) to the actual training data.

## TL;DR - What worked?

**K-means clustering beat gradient-free optimization.**

Why? The evaluation metric (MSE to nearest training image) directly aligns with what k-means optimizes for (minimizing squared distance within clusters), but doesn't align well with classifier confidence.

## Quick File Guide

### Start Here
1. **`README.md`** - Full story of our development journey ← READ THIS FIRST
2. **`strategy_pivot_plan.txt`** - The "aha moment" when we discovered clustering beats optimization

### Run the Solution
- **`solve_v2.py`** - Main pipeline (cluster-first approach)
  ```bash
  python task1/solve_v2.py
  ```

### Explore the Evolution
Read these in order to understand how we got here:

1. **`basic.py`** - Baseline: just pick best auxiliary images
2. **`reconstruct.py`** - Attempt 1: optimize using logits feedback
3. **`solve.py`** - Attempt 2: more sophisticated logit-guided optimization
4. **`cluster_submission_test.py`** - Breakthrough: clustering outperforms optimization!
5. **`improve_centroids.py`** / **`improve_centroids_v2.py`** - Refining the clustering approach
6. **`expand_coverage.py`** - Adding diversity maximization
7. **`final_safe_variant.py`** - Safe diversity-focused final variant
8. **`final_logits_swap.py`** - Minimal logits-based final refinement

### Utilities
- **`task_template.py`** - Original competition template with API docs
- **`visualize.py`** - Explore the auxiliary dataset
- **`infinity.py`** / **`generate_infinity_images.py`** - Helper utilities

### Data
- **`auxiliary_dataset.pt`** - 1000 labeled images (100 per class)
- **`submission_final_logits.npz`** - Final submission with logits refinement
- **`submission_final_safe.npz`** - Final submission (diversity-focused)

## The Key Insight (in 3 steps)

1. **Initial approach**: Optimize images to maximize classifier confidence
   - Result: High confidence, but poor MSE (images drift to decision boundaries)

2. **Breakthrough discovery**: Pure k-means centroids scored better than heavily optimized images
   - Why? K-means minimizes average squared distance = exactly what MSE measures!

3. **Final solution**: Cluster-first pipeline
   - Use k-means to find pixel-space centers
   - Add diversity expansion to maximize coverage
   - Use logits only for final validation (not primary optimization)

## Common Questions

**Q: Why not just use the auxiliary images?**
A: They're not optimal for the training set. Clustering finds better representatives by averaging similar images.

**Q: Why not optimize with gradient descent?**
A: We only have black-box classifier access (no gradients), and the classifier confidence doesn't correlate well with MSE anyway.

**Q: What's the difference between centroids and medoids?**
A: Centroids are cluster centers (averaged, can be blurry). Medoids are actual images closest to centers (sharp but possibly off-center). We blend both.

**Q: How many API calls did the final approach use?**
A: Very few! Clustering is done locally (0 API calls), and we only use 1-2 API calls for final validation. Compare to 50+ calls in the optimization approach.

## Want to Learn More?

1. Read the full **`README.md`** for the complete development story
2. Check **`archive_unused/README_ARCHIVE.md`** to see all the experiments we tried
3. Run the code yourself to see it in action
4. Explore visualizations with `visualize.py`

## File Size Reference

- Main directory: ~28 MB (mostly the 12 MB dataset)
- Archive directory: ~15 MB (intermediate experiments and submissions)
- Total: ~43 MB

The repository is intentionally kept lightweight - no large model checkpoints or excessive artifacts.
