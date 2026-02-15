import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(CKPT_DIR, exist_ok=True)

urls = [
    # RAR models
    "https://huggingface.co/yucornetto/RAR/resolve/main/rar_b.bin",
    "https://huggingface.co/yucornetto/RAR/resolve/main/rar_l.bin",
    "https://huggingface.co/yucornetto/RAR/resolve/main/rar_xl.bin",
    "https://huggingface.co/yucornetto/RAR/resolve/main/rar_xxl.bin",
    "https://huggingface.co/fun-research/TiTok/resolve/main/maskgit-vqgan-imagenet-f16-256.bin",
]

for url in urls:
    path = os.path.join(CKPT_DIR, os.path.basename(url))
    if os.path.exists(path):
        print(f"Skipping {path}, already exists")
        continue

    print(f"Downloading {path}")
    urllib.request.urlretrieve(url, path)

print("All RAR + tokenizer checkpoints downloaded into RAR/1d-tokenizer/checkpoints")
