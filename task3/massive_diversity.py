#!/usr/bin/env python3
"""
Generate 1000 HIGHLY DIVERSE boundary images
Strategy: Use MANY different base images and class pairs
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
import zipfile

print("="*70)
print("MASSIVE DIVERSITY APPROACH")
print("Generate 1000 unique boundary images from different sources")
print("="*70)

model = torch.load("challenge/model.pt", map_location="cpu", weights_only=False)
model.eval()

def load_image(path):
    img = Image.open(path).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)

def save_image(tensor, path):
    x = tensor.squeeze(0).detach().cpu().numpy()
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(x).save(path)

def quantize(x):
    return torch.round(x * 255) / 255.0

def create_boundary(x_base, class_a, class_b, iters=1000):
    """Quick boundary creation"""
    w = torch.atanh(2 * x_base - 1)
    w.requires_grad = True
    
    optimizer = torch.optim.Adam([w], lr=0.02)
    
    for i in range(iters):
        optimizer.zero_grad()
        
        x = 0.5 * (torch.tanh(w) + 1)
        x_q = quantize(x)
        
        logits = model(x_q)
        probs = F.softmax(logits, dim=1)
        
        prob_a = probs[0, class_a]
        prob_b = probs[0, class_b]
        gap = torch.abs(prob_a - prob_b)
        
        loss = gap
        loss.backward()
        optimizer.step()
    
    return quantize(0.5 * (torch.tanh(w.detach()) + 1))

# Use MANY different base images
all_base_images = sorted(Path("challenge/images").glob("*.png"))[:100]

print(f"\nUsing {len(all_base_images)} different base images")
print("Creating 10 boundaries per base image = 1000 total\n")

output_dir = Path("diverse_boundaries")
output_dir.mkdir(exist_ok=True)

count = 0

for img_idx, img_path in enumerate(all_base_images):
    if count >= 1000:
        break
    
    x_base = load_image(img_path)
    
    # Get top classes
    with torch.no_grad():
        logits = model(quantize(x_base))
        probs = F.softmax(logits, dim=1)
        top5_probs, top5_classes = torch.topk(probs, 5, dim=1)
    
    top_classes = top5_classes[0].tolist()
    
    # Create boundaries for DIFFERENT class pairs
    pairs = []
    for i in range(min(5, len(top_classes))):
        for j in range(i+1, min(5, len(top_classes))):
            pairs.append((top_classes[i], top_classes[j]))
    
    # Take 10 random pairs
    np.random.shuffle(pairs)
    for class_a, class_b in pairs[:10]:
        if count >= 1000:
            break
        
        x_boundary = create_boundary(x_base, class_a, class_b, iters=800)
        
        output_path = output_dir / f"{count:03d}.png"
        save_image(x_boundary, output_path)
        count += 1
    
    if (img_idx + 1) % 10 == 0:
        print(f"  Processed {img_idx+1}/100 base images ({count}/1000 total)")

print(f"\n✓ Generated {count} diverse boundary images")

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
print("\nMaximum diversity: 100 base images × 10 class pairs each")
