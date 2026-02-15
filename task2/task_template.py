import csv
import random
import zipfile
import requests
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from PIL import Image


# ----------------------------
# CONFIG
# ----------------------------
ZIP_FILE = "Dataset.zip"     # Path to dataset zip
DATASET_DIR = Path("dataset")  # Unzipped folder
SUBMISSION_FILE = "submission.csv"
LABELS = ["var16", "var20", "var24", "var30", "rarb", "rarl", "rarxl", "rarxxl", "outlier"] # Donot change this

# Leaderboard submission
SERVER_URL = "http://35.192.205.84:80"
API_KEY = "c8286483e3f08d5579bea4e972a7d21b"  # teams insert their assigned token here
TASK_ID = "15-model-tracer"

# ----------------------------
# UNZIP DATASET
# ----------------------------
if not DATASET_DIR.exists():
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)
else:
    print("Dataset already extracted.")


# ----------------------------
# TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ----------------------------s
# DATASETS & DATALOADERS
# ----------------------------
print("Loading datasets...")

train_dataset = datasets.ImageFolder(root=DATASET_DIR / "train", transform=transform)
val_dataset   = datasets.ImageFolder(root=DATASET_DIR / "val", transform=transform)

# Custom dataset for unlabeled test images
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.files = sorted(list(self.root.glob("*.*")))  # all files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "image_name": img_path.name}

test_dataset = TestDataset(DATASET_DIR / "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Print classes and per-class counts for train/val
def _print_class_stats(name: str, ds):
    counts = Counter(getattr(ds, "targets", []))
    print(f"{name} classes: {ds.classes}")
    for cls, idx in ds.class_to_idx.items():
        print(f"  {cls}: {counts.get(idx, 0)}")

_print_class_stats("Train", train_dataset)
_print_class_stats("Val", val_dataset)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")


# ----------------------------
# EXAMPLE MODEL (ResNet18)
# ----------------------------
print("Building dummy model...")
model = models.resnet18(weights=None, num_classes=len(LABELS))  # untrained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# ----------------------------
# DUMMY INFERENCE ON TEST / DUMMY SUBMISSION
# ----------------------------
print("Generating random predictions for submission...")
preds = []
for batch in test_loader:
    for fname in batch["image_name"]:
        label = random.choice(LABELS)  # random baseline
        preds.append([fname, label])

# ----------------------------
# SAVE SUBMISSION
# ----------------------------
with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(preds)

print(f"Saved submission file to {SUBMISSION_FILE}")
print("Format: image_name,label")


# ----------------------------
# SUBMIT TO LEADERBOARD SERVER
# ----------------------------
if API_KEY is None:
    print("No TOKEN provided. Please set your team TOKEN in this script to submit.")
else:
    print("Submitting to leaderboard server...")

    response = requests.post(
        f"{SERVER_URL}/submit/{TASK_ID}",
        files={"file": open(SUBMISSION_FILE, "rb")},
        headers={"X-API-Key": API_KEY},
    )
    print("Server response:", response.json())
