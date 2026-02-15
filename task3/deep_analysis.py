#!/usr/bin/env python3

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
print("DEEP ANALYSIS - Find Chimeras by Multiple Criteria")
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
    return torch.round(x * 255) / 255.0

# Check which submission had the best score
# Let's analyze chimera_replicated if it exists, otherwise submission_images
if Path("chimera_replicated").exists():
    submission_dir = Path("chimera_replicated")
    print("\nAnalyzing: chimera_replicated/ (from 0.067 score)")
else:
    submission_dir = Path("submission_images")
    print("\nAnalyzing: submission_images/")

all_images = sorted(submission_dir.glob("*.png"))[:200]  # Analyze first 200
print(f"Analyzing {len(all_images)} images...\n")

scores = []

for img_path in all_images:
    x = load_image(img_path)
    
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        
        # Multiple criteria for chimera likelihood
        
        # 1. Confidence gap (smaller = more likely chimera)
        top2_probs, top2_classes = torch.topk(probs, 2, dim=1)
        gap = (top2_probs[0, 0] - top2_probs[0, 1]).item()
        
        # 2. Entropy (higher = more uncertain)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        # 3. Top-3 spread (smaller = more clustered = more uncertain)
        top3_probs, _ = torch.topk(probs, 3, dim=1)
        top3_spread = (top3_probs[0, 0] - top3_probs[0, 2]).item()
        
        # 4. Combined score (lower = more likely chimera)
        # Weight: gap is most important
        combined_score = gap * 2.0 + (1 - entropy) + top3_spread
        
        scores.append({
            'path': img_path,
            'gap': gap,
            'entropy': entropy,
            'top3_spread': top3_spread,
            'combined': combined_score,
            'class1': top2_classes[0, 0].item(),
            'class2': top2_classes[0, 1].item(),
        })

# Sort by combined score (lowest = most likely chimeras)
scores.sort(key=lambda x: x['combined'])

print("TOP 20 MOST LIKELY CHIMERAS:")
print("-" * 90)
print(f"{'Rank':<6} {'Image':<20} {'Gap':<10} {'Entropy':<10} {'Spread':<10} {'Classes':<15}")
print("-" * 90)

for i in range(min(20, len(scores))):
    s = scores[i]
    print(f"{i+1:<6} {s['path'].name:<20} {s['gap']:<10.6f} {s['entropy']:<10.4f} "
          f"{s['top3_spread']:<10.4f} ({s['class1']},{s['class2']})")

# Create submission with top chimera candidates
print(f"\n{'='*70}")
print("Creating submission with TOP 20 chimera candidates")
print("Each replicated 50x = 1000 images")
print(f"{'='*70}")

output_dir = Path("top_chimeras")
output_dir.mkdir(exist_ok=True)

top_candidates = [s['path'] for s in scores[:20]]

count = 0
for img_path in top_candidates:
    for copy_idx in range(50):
        if count >= 1000:
            break
        output_path = output_dir / f"{count:03d}.png"
        shutil.copy(img_path, output_path)
        count += 1

print(f"\n✓ Created {count} images")

# Create zip
zip_path = "challenge/my_submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for png_file in sorted(output_dir.glob("*.png")):
        zipf.write(png_file, png_file.name)

size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
print(f"✓ Created {zip_path} ({size_mb:.2f} MB)")

print("\n" + "="*70)
print("READY TO SUBMIT!")
print("="*70)
print("\nStrategy: Top 20 most uncertain images (by multiple criteria)")
print("If these are chimeras, score should jump to ~1.0!")
