import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

def load_image(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x).unsqueeze(0)

def main():
    if len(sys.argv) != 3:
        print("usage: infer_one.py <model.pt> <image.png>", file=sys.stderr)
        sys.exit(2)

    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    x = load_image(image_path)

    with torch.no_grad():
        pred = int(torch.argmax(model(x), dim=1).item())

    print(pred)

if __name__ == "__main__":
    main()