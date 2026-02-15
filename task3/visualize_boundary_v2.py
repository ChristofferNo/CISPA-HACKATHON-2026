#!/usr/bin/env python3
"""
Better boundary visualization - actually reach the boundary!
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, 'challenge')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

print("="*70)
print("IMPROVED DECISION BOUNDARY VISUALIZATION")
print("="*70)

model = torch.load("challenge/model.pt", map_location="cpu", weights_only=False)
model.eval()

def load_image(path):
    img = Image.open(path).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)

def quantize(x):
    return torch.round(x * 255) / 255.0

def save_image(tensor, path):
    x = tensor.squeeze(0).detach().cpu().numpy()
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(x).save(path)

# Load image
image_path = Path("challenge/images/002.png")
x_original = load_image(image_path)

# Get original prediction
with torch.no_grad():
    logits_orig = model(quantize(x_original))
    probs_orig = F.softmax(logits_orig, dim=1)
    top5_probs, top5_classes = torch.topk(probs_orig, 5, dim=1)

orig_class = top5_classes[0, 0].item()
target_class = top5_classes[0, 1].item()

print(f"\nOriginal class: {orig_class} ({top5_probs[0,0].item():.4f})")
print(f"Target class: {target_class} ({top5_probs[0,1].item():.4f})")
print(f"\nPushing image from class {orig_class} → class {target_class}...")

# AGGRESSIVE optimization with higher learning rate
w = torch.atanh(2 * x_original - 1)
w.requires_grad = True

optimizer = torch.optim.Adam([w], lr=0.05)  # Much higher LR!

history = []
best_gap = float('inf')
best_x = None

for i in range(3000):  # More iterations
    optimizer.zero_grad()
    
    x = 0.5 * (torch.tanh(w) + 1)
    x_q = quantize(x)
    
    logits = model(x_q)
    probs = F.softmax(logits, dim=1)
    
    prob_orig = probs[0, orig_class]
    prob_target = probs[0, target_class]
    
    # NEW LOSS: Strongly push toward target class
    gap = torch.abs(prob_orig - prob_target)
    ce_loss = F.cross_entropy(logits, torch.tensor([target_class]))
    
    # Combine: 50% gap minimization, 50% cross-entropy
    loss = 0.5 * gap + 0.5 * ce_loss
    
    history.append({
        'iteration': i,
        'gap': gap.item(),
        'prob_orig': prob_orig.item(),
        'prob_target': prob_target.item(),
    })
    
    if gap < best_gap:
        best_gap = gap
        best_x = quantize(0.5 * (torch.tanh(w) + 1))
    
    loss.backward()
    optimizer.step()
    
    if i % 500 == 0:
        print(f"  Iter {i}: gap={gap.item():.6f}, "
              f"class_{orig_class}={prob_orig.item():.4f}, "
              f"class_{target_class}={prob_target.item():.4f}")

print(f"\n✓ Best gap achieved: {best_gap.item():.8f}")

# Analyze final boundary
with torch.no_grad():
    logits_final = model(best_x)
    probs_final = F.softmax(logits_final, dim=1)
    final_pred = int(torch.argmax(logits_final, dim=1).item())

print(f"\n" + "="*70)
print("BOUNDARY POINT PREDICTIONS")
print("="*70)
for class_id in range(10):
    prob = probs_final[0, class_id].item()
    if prob > 0.01:
        bar = "█" * int(prob * 50)
        marker = " ← BOUNDARY" if class_id in [orig_class, target_class] else ""
        print(f"  Class {class_id}: {prob:.4f} {bar}{marker}")

print(f"\nFinal prediction: Class {final_pred}")

# Test variations
print(f"\n" + "="*70)
print("TESTING VARIATIONS AROUND BOUNDARY")
print("="*70)

for noise_scale in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]:
    predictions = []
    for _ in range(100):
        noise = torch.randn_like(best_x) * noise_scale
        x_var = quantize(torch.clamp(best_x + noise, 0, 1))
        
        with torch.no_grad():
            pred = int(torch.argmax(model(x_var), dim=1).item())
            predictions.append(pred)
    
    unique, counts = np.unique(predictions, return_counts=True)
    pred_dict = dict(zip(unique, counts))
    
    print(f"\n  Noise {noise_scale:.4f}:")
    for class_id in sorted(pred_dict.keys()):
        count = pred_dict[class_id]
        pct = count / 100
        bar = "█" * int(pct * 50)
        print(f"    Class {class_id}: {count:3d}/100 {bar}")

# Save images
output_dir = Path("boundary_analysis")
output_dir.mkdir(exist_ok=True)

save_image(x_original, output_dir / "original.png")
save_image(best_x, output_dir / "boundary.png")

# Create plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
iterations = [h['iteration'] for h in history]
gaps = [h['gap'] for h in history]
plt.plot(iterations, gaps, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Gap')
plt.title('Gap Between Classes Over Time')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
prob_origs = [h['prob_orig'] for h in history]
prob_targets = [h['prob_target'] for h in history]
plt.plot(iterations, prob_origs, 'r-', linewidth=2, label=f'Original Class {orig_class}')
plt.plot(iterations, prob_targets, 'g-', linewidth=2, label=f'Target Class {target_class}')
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.title('Class Probabilities Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "optimization.png", dpi=150)
print(f"\n✓ Saved: {output_dir / 'optimization.png'}")

print(f"\n" + "="*70)
print("DONE!")
print("="*70)
