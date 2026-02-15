#!/usr/bin/env python3
"""
REFINED Chimera Generator - v2
Improvements based on 0.004631 score success:
1. MORE AGGRESSIVE boundary search (gap → 0)
2. DENSER variations around boundary (50 instead of 20)
3. MORE noise scales for better coverage
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

class RefinedChimeraGenerator:
    def __init__(self):
        print("Loading model...")
        self.model = torch.load("challenge/model.pt", map_location="cpu", weights_only=False)
        self.model.eval()
        print("✓ Model loaded")
        
        self.test_image = Path("challenge/test.png")
        self.base_images = list(Path("challenge/images").glob("*.png"))[:100]
        print(f"✓ Found {len(self.base_images)} base images")
    
    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        x = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    
    def save_image(self, tensor, path):
        x = tensor.squeeze(0).detach().cpu().numpy()
        x = np.clip(x, 0, 1)
        x = (x * 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(x).save(path)
    
    def quantize(self, x):
        return torch.round(x * 255) / 255.0
    
    def get_to_perfect_boundary(self, x_base, max_iters=1500):
        """
        REFINED: Get to PERFECT decision boundary
        Goal: Top 2 classes have EXACTLY equal probability
        This is where backend differences are MOST likely
        """
        w = torch.atanh(2 * x_base - 1)
        w.requires_grad = True
        
        optimizer = torch.optim.Adam([w], lr=0.015)  # Slightly higher LR
        
        # Get original class
        with torch.no_grad():
            x_q = self.quantize(x_base)
            logits = self.model(x_q)
            orig_class = int(torch.argmax(logits, dim=1).item())
        
        num_classes = 10
        target_class = (orig_class + np.random.randint(1, num_classes)) % num_classes
        
        best_gap = float('inf')
        best_x = None
        
        for i in range(max_iters):
            optimizer.zero_grad()
            
            x = 0.5 * (torch.tanh(w) + 1)
            x_q = self.quantize(x)
            
            logits = self.model(x_q)
            probs = F.softmax(logits, dim=1)
            
            # Get top 2 probabilities
            top2_probs, top2_classes = torch.topk(probs, 2, dim=1)
            
            # Calculate gap
            gap = torch.abs(top2_probs[0, 0] - top2_probs[0, 1])
            
            # REFINED LOSS: Heavily penalize gap, moderately push toward target
            boundary_loss = gap  # Want this to be 0!
            ce_loss = F.cross_entropy(logits, torch.tensor([target_class]))
            
            # Weight boundary loss MORE heavily
            loss = 0.8 * boundary_loss + 0.2 * ce_loss
            
            loss.backward()
            optimizer.step()
            
            # Track best (smallest gap)
            with torch.no_grad():
                if gap < best_gap:
                    best_gap = gap
                    best_x = self.quantize(0.5 * (torch.tanh(w) + 1))
            
            # Check progress
            if i % 300 == 0:
                with torch.no_grad():
                    print(f"    Iter {i}: gap={gap.item():.6f}, best_gap={best_gap.item():.6f}")
                    
                    # If we get VERY close, we can stop early
                    if best_gap < 0.01:
                        print(f"    ✓ Excellent boundary found! (gap={best_gap.item():.6f})")
                        break
        
        print(f"    Final best gap: {best_gap.item():.6f}")
        return best_x if best_x is not None else self.quantize(0.5 * (torch.tanh(w.detach()) + 1))
    
    def create_dense_variations(self, x_boundary, num_variations=50):
        """
        REFINED: Create MANY more variations with MORE noise scales
        Cover the "pocket" around the boundary more densely
        """
        variations = []
        
        # Always include the exact boundary point
        variations.append(x_boundary)
        
        # REFINED: More noise scales for better coverage
        # Paper says pockets can be 1-10000 ULP, so we need variety
        noise_scales = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
        
        variations_per_scale = (num_variations - 1) // len(noise_scales)
        
        for noise_scale in noise_scales:
            for _ in range(variations_per_scale):
                # Random noise
                noise = torch.randn_like(x_boundary) * noise_scale
                x_var = torch.clamp(x_boundary + noise, 0, 1)
                x_var_q = self.quantize(x_var)
                variations.append(x_var_q)
        
        # Fill remaining slots with very tiny noise (most likely to be chimeras)
        while len(variations) < num_variations:
            noise = torch.randn_like(x_boundary) * 0.001
            x_var = torch.clamp(x_boundary + noise, 0, 1)
            x_var_q = self.quantize(x_var)
            variations.append(x_var_q)
        
        return variations
    
    def generate_dataset(self, num_images=200):
        print(f"\nGenerating {num_images} REFINED boundary-focused images...")
        print("Improvements:")
        print("  - MORE aggressive boundary search (1500 iters)")
        print("  - DENSER variations (50 per boundary)")
        print("  - MORE noise scales (9 different scales)")
        print()
        
        output_dir = Path("submission_images")
        output_dir.mkdir(exist_ok=True)
        
        images_generated = 0
        
        # ~50 variations per base = 4 bases for 200 images
        num_bases = max(1, num_images // 50)
        
        for base_idx in range(num_bases):
            if images_generated >= num_images:
                break
            
            # Start with test.png, then random
            if base_idx == 0:
                base_img_path = self.test_image
                print(f"Base 1/{num_bases}: test.png (known to work!)")
            else:
                base_img_path = np.random.choice(self.base_images)
                print(f"\nBase {base_idx + 1}/{num_bases}: {base_img_path.name}")
            
            x_base = self.load_image(base_img_path)
            
            # Get to PERFECT boundary
            x_boundary = self.get_to_perfect_boundary(x_base, max_iters=1500)
            
            # Create DENSE variations
            variations_needed = min(50, num_images - images_generated)
            variations = self.create_dense_variations(x_boundary, variations_needed)
            
            # Save all variations
            for var in variations:
                if images_generated >= num_images:
                    break
                
                output_path = output_dir / f"{images_generated:03d}.png"
                self.save_image(var, output_path)
                images_generated += 1
            
            print(f"  → Generated {len(variations)} variations (total: {images_generated}/{num_images})")
        
        print(f"\n✓ Generated {images_generated} REFINED images!")
        print("Expected improvement: 0.5% → 5-10% chimera rate")
        
        return output_dir

def create_zip(images_dir):
    print("\nCreating submission zip...")
    zip_path = "challenge/my_submission.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for png_file in sorted(Path(images_dir).glob("*.png")):
            zipf.write(png_file, png_file.name)
    
    size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
    print(f"✓ Created {zip_path} ({size_mb:.2f} MB)")

def main():
    print("="*70)
    print("REFINED CHIMERA GENERATOR v2")
    print("Based on successful 0.004631 score!")
    print("="*70)
    
    gen = RefinedChimeraGenerator()
    images_dir = gen.generate_dataset(num_images=200)
    create_zip(images_dir)
    
    print("\n" + "="*70)
    print("READY TO SUBMIT!")
    print("cd challenge && python sample_submission.py")
    print("="*70)
    print("\nExpecting: 10-20 chimeras (5-10% rate) 🚀")

if __name__ == "__main__":
    main()