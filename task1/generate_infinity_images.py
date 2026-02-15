import random
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath("."))
from tools.run_infinity import *

import infinity.models.basic as basic

import csv
from torch.utils.data import Dataset

basic.flash_attn_func = None
basic.flash_attn_varlen_kvpacked_func = None
basic.flash_attn_varlen_qkvpacked_func = None
basic.flash_attn_varlen_func = None
basic.flash_fused_op_installed = False

model_path='weights/infinity_2b_reg.pth'
vae_path='weights/infinity_vae_d32reg.pth'
text_encoder_ckpt = 'google/flan-t5-xl'
args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    enable_model_cache=0,
)

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)

# PROMPT
prompts = {
    "stockholm": "A panorama photo of the beautiful city of Stockholm.",
    "hackathon": "A photorealistic image of a room full of energetic and motivated people working on programming tasks."   
}

# OUTPUT
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# GEN IMG
for category, prompt in prompts.items():
    cfg = 3
    tau = 0.5
    h_div_w = 1/1 # Aspect Ratio
    seed = random.randint(0, 10000)
    enable_positive_prompt = 0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # GEN
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
    )

    # SAVE
    save_path = osp.join(output_dir, f"{category}.jpg")
    cv2.imwrite(save_path, generated_image.cpu().numpy())
    print(f"{category} image saved to {save_path}")