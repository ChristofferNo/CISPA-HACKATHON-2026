"""
VAR White-Box Image Generation and Likelihood Analysis
USAGE
-----
This script generates class-conditional images using a pretrained VAR + VQ-VAE model
and computes token-level cross-entropy and token probabilities.


Run:
    python generate_var_images.py

Outputs (saved in ./outputs/):
- sample_<id>_class_<label>.png   : generated images
- summary.csv                     : per-image loss/probability statistics
- details.npz                     : per-token losses/probabilities and tokens (NumPy arrays)
- metadata.json                   : run configuration
"""

import os
import os.path as osp
import random
import json
import csv
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

from models import build_vae_var

MODEL_DEPTH = 16  # must match checkpoint
CHECKPOINT_DIR = "checkpoints"
OUT_DIR = "outputs"
ENC_NAME = "orig_enc" 
FT_VAE_CKPT = "checkpoints/var_ae_ft.pth"
os.makedirs(OUT_DIR, exist_ok=True)

if ENC_NAME == "orig_enc":
    VAE_CKPT = osp.join(CHECKPOINT_DIR, "vae_ch160v4096z32.pth")
elif ENC_NAME == "ft_enc":
    VAE_CKPT = FT_VAE_CKPT
else:
    raise ValueError(f"Unknown encoder name: {ENC_NAME}")
VAR_CKPT = osp.join(CHECKPOINT_DIR, f"var_d{MODEL_DEPTH}.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet class labels to generate
class_labels = [12, 12, 14]  # Example: 'golden retriever'

# Sampling parameters
cfg_scale = 3.0
top_k = 900
top_p = 0.95
more_smooth = False

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

vae, var = build_vae_var(
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    device=device,
    patch_nums=patch_nums,
    num_classes=1000,
    depth=MODEL_DEPTH,
    shared_aln=False,
)

vae.load_state_dict(torch.load(VAE_CKPT, map_location="cpu"), strict=True)
var.load_state_dict(torch.load(VAR_CKPT, map_location="cpu"), strict=True)

vae.eval()
var.eval()
for p in vae.parameters():
    p.requires_grad_(False)
for p in var.parameters():
    p.requires_grad_(False)

print("Models loaded.")

def get_token_list(images):
    return vae.img_to_idxBl(images, v_patch_nums=patch_nums)

label_B = torch.tensor(class_labels, device=device)
B = len(class_labels)

with torch.inference_mode():
    with torch.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
        images = var.autoregressive_infer_cfg(
            B=B,
            label_B=label_B,
            cfg=cfg_scale,
            top_k=top_k,
            top_p=top_p,
            g_seed=seed,
            more_smooth=more_smooth,
        )

for i, img in enumerate(images):
    img = (
        img.permute(1, 2, 0)
        .mul(255)
        .clamp(0, 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    Image.fromarray(img).save(
        osp.join(OUT_DIR, f"sample_{i}_class_{class_labels[i]}.png")
    )

print("Images saved.")

with torch.inference_mode():
    # VQ-VAE expects float inputs in [-1, 1]
    images_for_loss = images.float().mul(2.0).sub(1.0)

    token_list = get_token_list(images_for_loss)
    gt_BL = torch.cat(token_list, dim=1)
    var_input = vae.quantize.idxBl_to_var_input(token_list)
    logits = var(label_B, var_input)

    loss_per_token_ce = F.cross_entropy(
        logits.permute(0, 2, 1),
        gt_BL,
        reduction="none",
    )

    token_probs = torch.exp(-loss_per_token_ce)

    mean_token_ce = loss_per_token_ce.mean(dim=1)
    mean_token_prob = token_probs.mean(dim=1)

csv_path = osp.join(OUT_DIR, "summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id",
        "class_label",
        "mean_token_ce",
        "mean_token_prob",
    ])

    for i in range(B):
        writer.writerow([
            i,
            class_labels[i],
            mean_token_ce[i].item(),
            mean_token_prob[i].item(),
        ])

print("CSV summary saved.")


np.savez(
    osp.join(OUT_DIR, "details.npz"),
    tokens=gt_BL.cpu().numpy(),
    loss_per_token_ce=loss_per_token_ce.cpu().numpy(),
    token_probs=token_probs.cpu().numpy(),
)


with open(osp.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(
        {
            "model_depth": MODEL_DEPTH,
            "cfg_scale": cfg_scale,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
        },
        f,
        indent=2,
    )

print("Generation complete. All outputs saved.")
