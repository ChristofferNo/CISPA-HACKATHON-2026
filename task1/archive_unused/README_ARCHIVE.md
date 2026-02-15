# Archive: Intermediate Experiments and Artifacts

This directory contains files from the development process that were important for exploration but are not needed to understand or reproduce the final solution.

## Contents

### Ablation Studies
- **`ablation_variants.py`**: Systematic testing of different clustering configurations (Variant A-F)
  - Variant A: Pure centroids
  - Variant B: Pure medoids
  - Variant C: 50/50 centroid-medoid blend
  - Variant D-F: Different ratios and multi-seed variants
- **`ablation_medoid_ratio.py`**: Testing optimal centroid/medoid mixing ratios

### Intermediate Experiments
- **`diversity_cleanup.py`**: Early diversity improvement experiments
  - Tested removing near-duplicates and replacing with diverse alternatives
  - Led to the final diversity expansion approach

### Cached API Responses
- **`raw_api_response_batch0.json` through `raw_api_response_batch9.json`**:
  - Cached logits queries to avoid redundant API calls during development
  - Each batch contains logits for up to 100 images
  - Total: ~10 batches, representing different experimental runs

### Optimization State Files
- **`optimize_candidates.npz`**: Candidate images from logit-guided optimization runs
- **`optimize_momentum.npy`**: Momentum state for optimization algorithm
- **`optimize_progress.json`**: Progress tracking for optimization runs
- **`phase4_state.json`**: State checkpoint from Phase 4 optimization

These files represent the logit-based optimization approach that was later replaced by the cluster-first pipeline.

### Intermediate Submissions
All `.npz` submission files except the final variants:
- **`submission.npz`**: Early optimized submission
- **`submission_v2.npz`**: Refined optimization
- **`submission_cluster_test.npz`**: Pure clustering test (this was the breakthrough!)
- **`submission_kmeans.npz`**: K-means clustering variant
- **`submission_kmeans_global.npz`**: Global k-means approach
- **`submission_kmeans_blend.npz`**: Blended centroid-medoid approach
- **`submission_multiseed.npz`**: Multi-seed clustering v1
- **`submission_multiseed_v2.npz`**: Multi-seed clustering v2
- **`submission_ablation_A.npz` through `submission_ablation_F.npz`**: Ablation study results
- **`submission_expanded.npz`**: After diversity expansion

The final submissions kept in the main directory:
- `submission_final_logits.npz`
- `submission_final_safe.npz`

### Intermediate Reports
- **`all_scores.json`**: Comprehensive scores from all optimization runs
- **`diversity_report.json`**: Diversity metrics for candidate sets

### Visualizations
- **`class_distribution.png`**: Distribution of classes in auxiliary dataset
- **`class_samples.png`**: Sample images from each class
- **`sample_grid.png`**: Grid visualization of auxiliary images

These can be regenerated using `visualize.py` in the main directory.

### Build Artifacts
- **`__pycache__/`**: Python bytecode cache
- **`.ipynb_checkpoints/`**: Jupyter notebook checkpoints (if any)
- **`.claude/`**: Claude Code local settings

## Why These Were Archived

These files were crucial during development but don't need to be in the main directory because:

1. **Ablation studies**: Important for understanding what works, but the final approach is documented in the main README and implemented in the core scripts
2. **Cached API responses**: Useful during development to avoid rate limits, but not needed for reproduction (can be regenerated)
3. **Optimization state**: Represents the old logit-optimization approach that was superseded by clustering
4. **Intermediate submissions**: Useful for tracking progress, but only the final submissions matter for reproduction
5. **Visualizations**: Can be regenerated from the auxiliary dataset using `visualize.py`
6. **Build artifacts**: Standard development artifacts that should not be version-controlled

## Historical Notes

The most important archived file is **`submission_cluster_test.npz`** — this was the submission that outperformed all our logit-optimized candidates and triggered the strategic pivot to the cluster-first approach. This moment is documented in `strategy_pivot_plan.txt` in the main directory.

The ablation studies systematically confirmed that:
- Pure centroids work well but can be blurry
- Pure medoids are sharp but less centered
- 50/50 blends provided the best balance
- Multi-seed clustering improved coverage
- Diversity expansion gave small but consistent improvements

All these insights are incorporated into the final `solve_v2.py` pipeline.
