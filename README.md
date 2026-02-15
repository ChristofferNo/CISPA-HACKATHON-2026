## Part 3: Chimera Generation Challenge

### The Challenge

**Goal**: Generate "chimera" images that produce **different predictions** when run on different linear algebra backends (Apple Accelerate, Intel MKL, Nvidia CUDA, BLIS).

**The Concept**: Due to floating-point arithmetic differences across hardware/software implementations, there exist specific images that sit on decision boundaries where tiny numerical differences cause the model to predict different classes. These are called "chimeras."

**Submission Format**: 
- Submit 1,000 images (200 unique + replicates)
- Score = percentage of images that are chimeras (conflicting predictions across backends)
- Target: Find as many chimeras as possible

**The Challenge**: The paper mentions chimera "pockets" are extremely rare (1-10,000 ULP wide) and finding them is like searching for needles in a haystack.

![Chimera Concept](task3/images/chimera.png)
*Figure: Conflicting predictions on different backends due to numerical instabilities*

---

### My Approach

I developed an **iterative optimization strategy** to find decision boundaries where chimeras are most likely to exist:

#### Strategy 1: Decision Boundary Optimization

**Core Insight**: Chimeras occur where the model is maximally uncertain between two classes.
```python
def get_to_perfect_boundary(x_base, max_iters=1500):
    """
    Optimize image to reach PERFECT decision boundary
    Goal: Top 2 classes have EXACTLY equal probability
    """
    # Use tanh parameterization for unbounded optimization
    w = torch.atanh(2 * x_base - 1)
    w.requires_grad = True
    
    optimizer = torch.optim.Adam([w], lr=0.015)
    
    for i in range(max_iters):
        x = 0.5 * (torch.tanh(w) + 1)
        x_q = quantize(x)  # Round to 8-bit
        
        logits = model(x_q)
        probs = F.softmax(logits, dim=1)
        
        # Get top 2 class probabilities
        top2_probs, _ = torch.topk(probs, 2, dim=1)
        gap = torch.abs(top2_probs[0, 0] - top2_probs[0, 1])
        
        # Minimize gap (make classes equally likely)
        boundary_loss = gap
        ce_loss = F.cross_entropy(logits, target_class)
        
        loss = 0.8 * boundary_loss + 0.2 * ce_loss
        loss.backward()
        optimizer.step()
```

**Key Parameters**:
- 1500 iterations for aggressive boundary search
- Learning rate 0.015 for fine-grained control
- 80% weight on boundary loss (minimize gap between top-2 classes)

#### Strategy 2: Dense Variation Sampling

Once at the boundary, create **dense variations** with multiple noise scales to cover the "pocket":
```python
def create_dense_variations(x_boundary, num_variations=50):
    """
    Create MANY variations around the boundary point
    Different noise scales to find the chimera pocket
    """
    variations = [x_boundary]  # Include exact boundary
    
    # Multiple noise scales (1-10000 ULP range)
    noise_scales = [0.0005, 0.001, 0.002, 0.003, 
                    0.005, 0.008, 0.01, 0.015, 0.02]
    
    for noise_scale in noise_scales:
        for _ in range(variations_per_scale):
            noise = torch.randn_like(x_boundary) * noise_scale
            x_var = torch.clamp(x_boundary + noise, 0, 1)
            x_var_q = quantize(x_var)
            variations.append(x_var_q)
    
    return variations
```

**Why this works**: The paper states chimera pockets can range from 1 to 10,000 ULP (Units in Last Place). By using 9 different noise scales, we cover this range systematically.

#### Strategy 3: Uncertainty-Based Replication

After generating candidates, I analyzed which images were most uncertain:
```python
# Calculate confidence gap for each image
for img in all_images:
    logits = model(img)
    probs = F.softmax(logits, dim=1)
    top2_probs, _ = torch.topk(probs, 2, dim=1)
    
    gap = (top2_probs[0, 0] - top2_probs[0, 1]).item()
    
    # Smaller gap = more uncertain = more likely chimera
    uncertainties.append({'path': img, 'gap': gap})

# Sort by smallest gap
uncertainties.sort(key=lambda x: x['gap'])

# Replicate top 10 most uncertain images
# Each replicated 100x to reach 1000 total
```

---

### Pipeline Overview
```
1. Load base images (test.png + random samples)
   ↓
2. Optimize to decision boundary (1500 iterations)
   → Minimize gap between top-2 class probabilities
   ↓
3. Generate 50 dense variations per boundary
   → 9 different noise scales (0.0005 to 0.02)
   ↓
4. Analyze all candidates for uncertainty
   → Measure confidence gap
   ↓
5. Replicate top-10 most uncertain images
   → 100 copies each = 1000 images total
   ↓
6. Submit to evaluation server
```

---

### Key Files

- **`find_and_replicate.py`**: Analyzes generated images for uncertainty, replicates most likely chimeras
- **`generate_submissions.py`**: Main chimera generation pipeline with boundary optimization
- **`deep_analysis.py`**: Multi-criteria analysis (gap, entropy, top-3 spread) to identify chimeras
- **`visualize_boundary_v2.py`**: Visualization of decision boundaries and chimera locations

---

### Results & Insights

#### Scores Achieved:
- **Initial random baseline**: ~0.000 (no chimeras found)
- **First boundary approach**: 0.004631 (~0.5% chimera rate - 1 chimera found!)
- **Refined dense variations**: Target 5-10% chimera rate

#### What Worked:
1. **Boundary optimization**: Getting gap < 0.01 was critical
2. **Multiple noise scales**: Single scale missed many pockets
3. **Replication strategy**: Amplifying confirmed chimeras improved score
4. **Quantization awareness**: Always round to 8-bit (0-255) to match submission format

#### What Didn't Work:
1. **Pure random noise**: Success rate ~0%
2. **Single noise scale**: Missed chimera pockets at different ULP ranges  
3. **Too few iterations**: <1000 iterations didn't reach tight boundaries
4. **Ignoring quantization**: Floating-point boundaries ≠ quantized boundaries

#### Key Challenge:
The fundamental difficulty is that chimera pockets are **extremely narrow** in the input space. Even after optimizing to gap < 0.01, you still need to explore the local neighborhood densely because:
- Quantization (8-bit) discretizes the space
- Backend differences are subtle (10^-6 to 10^-8 range)
- The "pocket" might be 1-10,000 ULP wide but in a 32×32×3 = 3,072-dimensional space

---

### Lessons Learned

1. **Optimization is necessary but not sufficient**: Getting to the boundary is step 1, but you still need dense sampling to find the actual chimera pocket.

2. **Multi-scale exploration is critical**: A single noise level will miss pockets that exist at different scales.

3. **Quantization matters**: Always work with quantized images (`torch.round(x * 255) / 255.0`) since that's what the evaluation server uses.

4. **Uncertainty is a good proxy**: Images with smallest confidence gap are statistically more likely to be chimeras, making replication an effective strategy.

5. **Rare but real**: The challenge demonstrates that numerical instability in deep learning is a **real problem** - these chimeras aren't theoretical, they actually exist and can be found with the right approach.

---

### Future Improvements

If continuing this work, I would try:

1. **Adaptive noise scaling**: Start with large noise, gradually refine around promising regions
2. **Ensemble boundary search**: Optimize for boundaries between multiple class pairs simultaneously  
3. **Gradient-free methods**: Basin-hopping or genetic algorithms might find pockets that gradient descent misses
4. **Exploit known chimeras**: Once you find one, the local neighborhood likely contains more
5. **Hardware-in-the-loop**: If possible, test candidates on actual different backends during generation

---

**TL;DR**: Generated chimera images by optimizing to decision boundaries (where top-2 classes are equally likely), then densely sampling around those boundaries with multiple noise scales. Successfully found chimeras with ~0.5% rate, demonstrating that numerical instabilities in neural networks are a real, measurable phenomenon.

