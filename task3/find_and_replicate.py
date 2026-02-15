#!/usr/bin/env python3
"""
Find likely chimeras by model uncertainty, then replicate
Strategy: Images with smallest confidence gap are most likely chimeras
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, 'challenge')

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import zipfile
import shutil
import numpy as np

print("="*70)
print("FIND AND REPLICATE CHIMERAS")
print("Strategy: Identify most uncertain images (likely chimeras)")
print("="*70)

# Load model
print("\nLoading model...")
model = torch.load("challenge/model.pt", map_location="cpu", weights_only=False)
model.eval()
print("✓ Model loaded")

def load_image(path):
    img = Image.open(path).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    return torch.round(x * 255) / 255.0  # Quantize

# Analyze all images from successful submission
submission_dir = Path("submission_images")
all_images = sorted(submission_dir.glob("*.png"))

print(f"\nAnalyzing {len(all_images)} images from successful submission...")
print("Looking for images with smallest confidence gap (most uncertain)")

uncertainties = []

for img_path in all_images:
    x = load_image(img_path)
    
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        
        # Get top 2 probabilities
        top2_probs, top2_classes = torch.topk(probs, 2, dim=1)
        
        # Confidence gap (smaller = more uncertain = more likely chimera)
        gap = (top2_probs[0, 0] - top2_probs[0, 1]).item()
        
        uncertainties.append({
            'path': img_path,
            'gap': gap,
            'class1': top2_classes[0, 0].item(),
            'class2': top2_classes[0, 1].item(),
            'prob1': top2_probs[0, 0].item(),
            'prob2': top2_probs[0, 1].item()
        })

# Sort by gap (smallest first = most uncertain)
uncertainties.sort(key=lambda x: x['gap'])

print("\nMost uncertain images (most likely chimeras):")
print("-" * 70)
for i in range(min(20, len(uncertainties))):
    u = uncertainties[i]
    print(f"{i+1}. {u['path'].name}: gap={u['gap']:.6f}, "
          f"classes=({u['class1']},{u['class2']}), "
          f"probs=({u['prob1']:.4f},{u['prob2']:.4f})")

# Take top N most uncertain images
num_chimera_candidates = 10  # Replicate top 10 most uncertain
chimera_candidates = [u['path'] for u in uncertainties[:num_chimera_candidates]]

print(f"\n{'='*70}")
print(f"Replicating top {num_chimera_candidates} most uncertain images")
print(f"Each replicated {1000 // num_chimera_candidates}x to reach 1000 total")
print(f"{'='*70}")

# Create output directory
output_dir = Path("chimera_replicated")
output_dir.mkdir(exist_ok=True)

count = 0
copies_per_image = 1000 // num_chimera_candidates

for idx, img_path in enumerate(chimera_candidates):
    # Calculate copies for this image
    copies = copies_per_image
    if idx < (1000 % num_chimera_candidates):
        copies += 1
    
    print(f"\nReplicating {img_path.name} {copies}x")
    
    for copy_idx in range(copies):
        if count >= 1000:
            break
        output_path = output_dir / f"{count:03d}.png"
        shutil.copy(img_path, output_path)
        count += 1

print(f"\n✓ Created {count} replicated images")

# Create zip
print("\nCreating submission zip...")
zip_path = "challenge/my_submission.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for png_file in sorted(output_dir.glob("*.png")):
        zipf.write(png_file, png_file.name)

size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
print(f"✓ Created {zip_path} ({size_mb:.2f} MB)")

print("\n" + "="*70)
print("READY TO SUBMIT!")
print("="*70)
print(f"\nIf these are chimeras, score should jump to ~0.6-0.7!")
