import torch
import requests
import sys
import os
import numpy as np

# --------------------------------
# AUXILIARY DATASET
# --------------------------------

"""
Auxiliary dataset contents:

-"images": Tensor containing the 1000 images, has shape (1000, 3, 32, 32)
-"labels": List of true labels for the images, has shape (1000)
"""

# Load the dataset
dataset = torch.load("auxiliary_dataset.pt", weights_only=False)

print("Images shape:", dataset["images"].shape)
print("Labels shape:", dataset["labels"].shape)
print("First 10 labels:", dataset["labels"][:10])
print("First image tensor:", dataset["images"][:1])

# --------------------------------
# SUBMISSION FORMAT
# --------------------------------

"""
The submission must be a .npz file of the following format:

-"images": Tensor containing the reconstructed images with shape (100, 3, 32, 32).
    These images do not have to be in any particular order.
    The dtype must be float32, with values in [0, 1] range.
"""

# Example Submission:

reconstructed_images = torch.randint(0, 256, size=(100, 3, 32, 32), dtype=torch.uint8)

reconstructed_images = reconstructed_images.float() / 255.0  # normalize to [0, 1] range

images_np = reconstructed_images.detach().cpu().numpy()

np.savez_compressed("example_submission.npz", images=images_np)

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

"""
Data Reconstruction Task — Participant Submission Guide
========================================================

You will upload a single **.npz** file that contains ONLY an array named **'images'**.
The evaluator will load your file, run shape/dtype checks,
and then score it by comparing to the training dataset.

Follow these rules carefully to avoid automatic rejection.

1) File format
--------------
- **Extension:** `.npz` (NumPy compressed archive)
- **Content:** must contain exactly one required key: `'images'`
- **Max file size:** 200 MB (hard limit). Larger files are rejected.

2) Array requirements
---------------------
Let `G` be the ground-truth tensor loaded:

- **Shape:** `images.shape` must match `G["images"].shape` **exactly**.
  - If `G["images"]` is `(N, 3, H, W)`, your array must also be `(N, 3, H, W)`.
  - No extra samples; no fewer; no different dimensions.
- **Dtype:** `images.dtype` must match `G["images"].dtype` **exactly**.
  - If the GT uses `float32`, you must submit `float32`.
  - Safe cast example: `images = np.asarray(images, dtype=np.float32)`
- **Contiguity:** The server will convert to a contiguous Torch tensor; standard NumPy arrays are fine.


3) Typical failure messages & what they mean
--------------------------------------------
- "File must be .npz and contain an 'images' array."
  → Wrong extension or missing `'images'` key.
- "File too large: X bytes (limit 209715200)."
  → Your file exceeds 200 MB.
- "Failed to read .npz: ..."
  → The file is corrupted or not a valid `.npz` created with `allow_pickle=False`.
- "Failed to convert 'images' to torch tensor: ..."
  → Your `'images'` array has an unsupported dtype or structure (e.g., object array).
- "Submitted images must have shape (N, C, H, W), but got (...)."
  → Shape mismatch with the ground-truth images.
- "Submitted images must be of type torch.float32, but got torch.float64."
  → Dtype mismatch with the ground-truth images.
"""

BASE_URL  = "http://35.192.205.84:80"
API_KEY  = "YOUR_API_KEY_HERE"

TASK_ID = "12-data-reconstruction"

# Path to the .npz file containing the images you want to get logits for

QUERY_PATH = "/PATH/FILE.npz"

# Path to the .npz file you want to send

FILE_PATH = "/PATH/FILE.npz"


GET_LOGITS = False      # set True to get logits from the API
SUBMIT     = False      # set True to submit your solution


def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if GET_LOGITS:

    with open(QUERY_PATH, "rb") as f:
        files = {"npz": (QUERY_PATH, f, "application/octet-stream")}
        response = requests.post(
            f"{BASE_URL}/{TASK_ID}/logits",
            files=files,
            headers={"X-API-Key": API_KEY},
        )

    if response.status_code == 200:
        data = response.json()
        print("Request successful")
        print(data)

    else:
        print("Request failed")
        print("Status code:", response.status_code)
        print("Detail:", response.text)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                "file": (os.path.basename(FILE_PATH), f, "application/octet-stream"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
