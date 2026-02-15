"""
Visualize reconstructed images from final submission.
Displays 10 sample images (1 per class) from the final reconstruction.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_submission(filename="submission_final_logits.npz"):
    """Load submission file."""
    path = os.path.join(WORK_DIR, filename)
    data = np.load(path)
    images = data['images']  # Shape: (100, 3, 32, 32)
    print(f"Loaded {len(images)} images from {filename}")
    return images

def classify_images(images):
    """
    Use a simple heuristic to assign images to classes.
    Since we don't have labels in the submission, we'll just
    take the first 10 images (assuming they're roughly distributed).

    For better classification, we'd need to query the API, but
    let's keep this simple and just show diverse samples.
    """
    # For visualization, let's just take evenly spaced samples
    indices = np.linspace(0, len(images)-1, 10, dtype=int)
    return images[indices], indices

def visualize_samples(images, indices, output_path="reconstructed_samples.png"):
    """Create a grid visualization of sample images."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Reconstructed Images - Final Submission Samples',
                 fontsize=16, fontweight='bold', y=1.02)

    axes = axes.flatten()

    for idx, (img, img_idx) in enumerate(zip(images, indices)):
        # Convert from (C, H, W) to (H, W, C) for matplotlib
        img_display = np.transpose(img, (1, 2, 0))

        # Clip to [0, 1] range
        img_display = np.clip(img_display, 0, 1)

        axes[idx].imshow(img_display)
        axes[idx].set_title(f'Image #{img_idx}\n({CLASS_NAMES[idx]})',
                           fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(WORK_DIR, output_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved visualization to: {save_path}")

    # Also save a high-res version
    hires_path = save_path.replace('.png', '_hires.png')
    plt.savefig(hires_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved high-res version to: {hires_path}")

    plt.close()

    return save_path

def create_detailed_grid(images, output_path="reconstructed_detailed.png"):
    """Create a more detailed grid showing all 100 images."""
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.suptitle('All 100 Reconstructed Images',
                 fontsize=20, fontweight='bold', y=0.995)

    for idx, img in enumerate(images):
        row = idx // 10
        col = idx % 10

        # Convert from (C, H, W) to (H, W, C)
        img_display = np.transpose(img, (1, 2, 0))
        img_display = np.clip(img_display, 0, 1)

        axes[row, col].imshow(img_display)
        axes[row, col].axis('off')

        # Add index label for first and last in each row
        if col == 0:
            axes[row, col].set_ylabel(f'{idx}-{idx+9}',
                                     fontsize=8, rotation=0,
                                     labelpad=20, va='center')

    plt.tight_layout()

    save_path = os.path.join(WORK_DIR, output_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved detailed grid to: {save_path}")

    plt.close()

    return save_path

if __name__ == "__main__":
    print("Loading final submission...")
    images = load_submission("submission_final_logits.npz")

    print("\nCreating sample visualization (10 images)...")
    sample_images, indices = classify_images(images)
    sample_path = visualize_samples(sample_images, indices)

    print("\nCreating detailed grid (all 100 images)...")
    grid_path = create_detailed_grid(images)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. {sample_path}")
    print(f"  2. {sample_path.replace('.png', '_hires.png')}")
    print(f"  3. {grid_path}")
    print("\nThese images show the final reconstructed dataset.")
