"""
Chimera - Participant Submission Guide
============================================

Please read the following instructions carefully before preparing your 
submission. Submissions that do not strictly follow the formatting 
requirements will be automatically rejected.

You need to upload a single **.zip** file containing the chimera PNG images.

1) File format
--------------
- **Extension:** `.zip`
- **Max file size:** 150 MB (hard limit). Larger files are rejected.
- **Contents:** up to 1000 PNG files named `000.png` through `999.png`
- Each PNG must be a valid **32 × 32 RGB** image.
- No subdirectories, no extra files inside the zip.

2) Image requirements
---------------------
- **Resolution:** 32 × 32 pixels.
- **Colour mode:** RGB (3 channels).

3) Notes on submission
----------------------
- Remember that you can test your images locally before submitting.
- Submission evaluation is performed on a different server with different hardware. 
  This simulates a more realistic threat model, but may introduce additional 
  numerical variation. Expect some difference between your local results and the 
  evaluation on the submission server and try to account for these during generation.
- Searching for backend-dependent differences may be easier in the model's internal 
  input space rather than directly in pixel space. However, submissions must be valid 
  images in pixel space; that is, PNG files.
- The more images you submit, the longer processing and evaluation will take (expect 
  approx. 5 min for all 1000 images).
"""

import os
import sys
import requests

BASE_URL  = "http://35.192.205.84:80"
API_KEY  = "c8286483e3f08d5579bea4e972a7d21b"

TASK_ID   = "16-hack-chimera"
FILE_PATH = "my_submission.zip"



def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if not os.path.isfile(FILE_PATH):
    die(f"File not found: {FILE_PATH}")

try:
    with open(FILE_PATH, "rb") as f:
        files = {
            "file": (os.path.basename(FILE_PATH), f, "application/zip"),
        }
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files=files,
            timeout=(10, 600),  
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

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)
