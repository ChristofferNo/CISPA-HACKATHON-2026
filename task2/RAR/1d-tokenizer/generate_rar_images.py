"""
RAR IMAGE GENERATION + LIKELIHOOD EVALUATION
===========================================

This script generates class-conditional images using a pretrained RAR
(Reconstruction-Aware Autoregressive) generator with a MaskGit-VQ tokenizer,
then computes token-level loss and probability on the generated images.

Key steps:
1. Generate images with the RAR sampler (no logits returned during sampling).
2. Re-tokenize the generated images using the tokenizer.
3. Run the generator on those tokens to get logits and labels.
4. Compute per-token NLL and mean token probability.

Run from /RAR/1d-tokenizer as -
    python generate_rar_images.py
------------------------------------------------
WHAT THE SCRIPT OUTPUTS
------------------------------------------------
1. Generated images:
   - outputs_rar/rar_sample_<id>_class_<label>.png

2. CSV summary (human-readable):
   - outputs_rar/summary.csv

   Columns:
   - image_id
   - class_label
   - mean_token_nll
   - mean_token_prob

3. NumPy archive (framework-agnostic):
   - outputs_rar/details.npz
     Contains:
       - tokens
       - loss_per_token
       - token_probs

4. Metadata:
   - outputs_rar/metadata.json
     Records model size, checkpoint, sampling parameters, and seed.

------------------------------------------------
NOTES
------------------------------------------------
- Token-level losses are computed on the generated images, not on ground-truth.
- No retraining or weight modification is performed.
"""

import os
import csv
import json
import torch
import numpy as np
from PIL import Image

import demo_util
from utils.train_utils import create_pretrained_tokenizer

def update_weights(model, ckpt_path, delta=True):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if delta:
        state_dict_to_apply = model.state_dict().copy()
        for key in state_dict:
            if key in state_dict_to_apply:
                state_dict_to_apply[key] = state_dict_to_apply[key] + state_dict[key].to(
                    state_dict_to_apply[key].device
                )
            else:
                state_dict_to_apply[key] = state_dict[key]
    else:
        state_dict_to_apply = state_dict

    missing, unexpected = model.load_state_dict(state_dict_to_apply, strict=False)
    print(f"Missing: {missing}")
    print(f"Unexpected: {unexpected}")

CHECKPOINT_DIR = "checkpoints"
OUT_DIR = "outputs_rar"
enc_name = "orig_enc"
ft_enc_path = "checkpoints/rar_ae_ft_delta.pth"
os.makedirs(OUT_DIR, exist_ok=True)

MASKGIT_CKPT = os.path.join(
    CHECKPOINT_DIR, "maskgit-vqgan-imagenet-f16-256.bin"
)

# CHANGE ONLY THIS LINE to switch RAR model size
RAR_CKPT = os.path.join(CHECKPOINT_DIR, "rar_b.bin")
# supported: rar_b.bin, rar_l.bin, rar_xl.bin, rar_xxl.bin

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# ImageNet class labels to generate
class_labels = [980, 437, 22, 562]
B = len(class_labels)

RAR_CONFIGS = {
    "rar_b": {
        "hidden_size": 768,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 3072,
    },
    "rar_l": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
    "rar_xl": {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "intermediate_size": 5120,
    },
    "rar_xxl": {
        "hidden_size": 1408,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "intermediate_size": 6144,
    },
}

ckpt_name = os.path.basename(RAR_CKPT)
ckpt_key = ckpt_name.replace(".bin", "")

if ckpt_key not in RAR_CONFIGS:
    raise ValueError(
        f"Unknown RAR checkpoint '{ckpt_name}'. "
        f"Expected one of: {list(RAR_CONFIGS.keys())}"
    )

rar_cfg = RAR_CONFIGS[ckpt_key]

print(f"Using RAR checkpoint: {ckpt_name}")
print(f"Auto-configured architecture: {ckpt_key}")

config = demo_util.get_config("configs/training/generator/rar.yaml")
config.experiment.generator_checkpoint = RAR_CKPT
config.model.vq_model.pretrained_tokenizer_weight = MASKGIT_CKPT
config.model.generator.hidden_size = rar_cfg["hidden_size"]
config.model.generator.num_hidden_layers = rar_cfg["num_hidden_layers"]
config.model.generator.num_attention_heads = rar_cfg["num_attention_heads"]
config.model.generator.intermediate_size = rar_cfg["intermediate_size"]

tokenizer = create_pretrained_tokenizer(config).to(device)

match enc_name:
    case "orig_enc":
        pass
    case "ft_enc":
        update_weights(tokenizer.encoder, ft_enc_path)
        print("Loaded finetuned tokenizer encoder (delta).")
    case _:
        raise ValueError(f"Unknown encoder name: {enc_name}")
generator = demo_util.get_rar_generator(config).to(device)
generator.eval()

print("RAR tokenizer and generator loaded.")

labels = torch.tensor(class_labels, device=device)

with torch.no_grad():
    images = demo_util.sample_fn(
        generator=generator,
        tokenizer=tokenizer,
        labels=labels,
        guidance_scale=4.0,
        guidance_scale_pow=0.0,
        randomize_temperature=1.0,
        device=device,
    )

for i, sample in enumerate(images):
    Image.fromarray(sample).save(
        f"{OUT_DIR}/rar_sample_{i}_class_{class_labels[i]}.png"
    )

print("Images saved.")

# Tokenize generated images for loss computation
with torch.no_grad():
    tokens = tokenizer.encode(
        torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    )

with torch.no_grad():
    cond = generator.preprocess_condition(labels)
    logits, labels_tf = generator(tokens, cond, return_labels=True)

    # logits: (B, N_tokens + 1, V), labels_tf: (B, N_tokens)
    logits = logits[:, :-1]

    loss_per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels_tf.reshape(-1),
        reduction="none",
    ).reshape(labels_tf.shape)

    mean_token_nll = loss_per_token.mean(dim=1)
    token_probs = torch.exp(-loss_per_token)
    mean_token_prob = token_probs.mean(dim=1)

csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id",
        "class_label",
        "mean_token_nll",
        "mean_token_prob",
    ])

    for i in range(B):
        writer.writerow([
            i,
            class_labels[i],
            mean_token_nll[i].item(),
            mean_token_prob[i].item(),
        ])

print("CSV summary saved.")

np.savez(
    os.path.join(OUT_DIR, "details.npz"),
    tokens=tokens.cpu().numpy(),
    loss_per_token=loss_per_token.cpu().numpy(),
    token_probs=token_probs.cpu().numpy(),
)

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(
        {
            "rar_checkpoint": ckpt_name,
            "model_size": ckpt_key,
            "cfg_scale": 4.0,
            "guidance_scale_pow": 0.0,
            "randomize_temperature": 1.0,
            "seed": seed,
        },
        f,
        indent=2,
    )

print("RAR generation + likelihood computation complete.")
