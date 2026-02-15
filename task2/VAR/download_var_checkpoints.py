import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(CKPT_DIR, exist_ok=True)

urls = [
    "https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth",
    "https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth",
    "https://huggingface.co/FoundationVision/var/resolve/main/var_d24.pth",
    "https://huggingface.co/FoundationVision/var/resolve/main/var_d30.pth",
    "https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth",
]

for url in urls:
    filename = os.path.join(CKPT_DIR, os.path.basename(url))
    if os.path.exists(filename):
        print(f"Skipping {filename}, already exists")
        continue

    print(f"Downloading {filename}")
    urllib.request.urlretrieve(url, filename)

print("All VAR checkpoints downloaded into VAR/checkpoints")
