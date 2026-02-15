import csv
import zipfile
import requests
import pickle
import sys
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CONFIGURATION
# ============================================================

FORCE_REEXTRACT = False        # Set to True to re-extract features
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_INFERENCE = 4
NUM_WORKERS = 4

# Which classifier to use
CLASSIFIER_TYPE = "gradient_boost"   # Options: "logistic", "random_forest", "svm", "gradient_boost"

# Outlier detection
OUTLIER_CONF_THRESHOLD = 0.5
USE_ADAPTIVE_THRESHOLD = True

# Model selection
USE_VAR_MODELS = True
USE_RAR_MODELS = True

# Submission
API_KEY = "c8286483e3f08d5579bea4e972a7d21b"

# ============================================================
# PATHS
# ============================================================

ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("dataset")
SUBMISSION_FILE = "submission.csv"
LABELS = ["var16", "var20", "var24", "var30", "rarb", "rarl", "rarxl", "rarxxl", "outlier"]

SERVER_URL = "http://35.192.205.84:80"
TASK_ID = "15-model-tracer"

VAR_ROOT = Path("VAR")
RAR_ROOT = Path("RAR/1d-tokenizer")
CACHE_DIR = Path(".cache_scores")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if FORCE_REEXTRACT:
    print("="*60)
    print("CLEARING OLD CACHE")
    print("="*60)
    for cache_file in CACHE_DIR.glob("*_provenance_features.pkl"):
        cache_file.unlink()
        print(f"  Deleted {cache_file}")

# ============================================================
# DATASET SETUP
# ============================================================

if not DATASET_DIR.exists():
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)
else:
    print("Dataset already extracted.")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(root=DATASET_DIR / "train", transform=transform)
val_dataset = datasets.ImageFolder(root=DATASET_DIR / "val", transform=transform)

class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.files = sorted(list(self.root.glob("*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path.name

test_dataset = TestDataset(DATASET_DIR / "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=NUM_WORKERS)

def _print_class_stats(name: str, ds):
    counts = Counter(getattr(ds, "targets", []))
    print(f"{name} classes: {ds.classes}")
    for cls, idx in ds.class_to_idx.items():
        print(f"  {cls}: {counts.get(idx, 0)}")

_print_class_stats("Train", train_dataset)
_print_class_stats("Val", val_dataset)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize torch.distributed for VAR models
import torch.distributed as dist
if not dist.is_initialized():
    try:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:12355', 
                                world_size=1, rank=0)
        print("✓ Initialized torch.distributed")
    except:
        try:
            dist.init_process_group(backend='gloo', init_method='tcp://localhost:12355',
                                    world_size=1, rank=0)
            print("✓ Initialized torch.distributed (gloo)")
        except Exception as e:
            def mock_get_world_size(group=None): return 1
            def mock_get_rank(group=None): return 0
            dist.get_world_size = mock_get_world_size
            dist.get_rank = mock_get_rank
            print("✓ Patched torch.distributed")

# ============================================================
# MODEL LOADING
# ============================================================

def load_var_vae_direct(depth):
    """Load VAR VAE"""
    vae_ckpt = VAR_ROOT / "checkpoints" / "vae_ch160v4096z32.pth"
    if not vae_ckpt.exists():
        return None
    
    var_path = str(VAR_ROOT.absolute())
    original_path = sys.path.copy()
    sys.path.insert(0, var_path)
    
    try:
        import models
        
        vae, _ = models.build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4, device=device,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            num_classes=1000, depth=depth, shared_aln=False,
        )
        
        state_dict = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(state_dict, strict=True)
        
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        
        return vae
        
    except Exception as e:
        print(f"Error loading VAR: {e}")
        return None
    finally:
        sys.path = original_path

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_for_all_models(loader, split_name, is_test=False):
    """Extract provenance features"""
    cache_file = CACHE_DIR / f"{split_name}_provenance_features.pkl"
    
    if cache_file.exists() and not FORCE_REEXTRACT:
        print(f"Loading cached features for {split_name}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES FOR {split_name.upper()}")
    print(f"{'='*60}")
    
    # Load all images
    all_images = []
    all_labels = []
    all_names = []
    
    for batch in tqdm(loader, desc=f"Loading {split_name}"):
        if is_test:
            images, names = batch
            all_images.append(images)
            all_names.extend(names)
        else:
            images, labels = batch
            all_images.append(images)
            all_labels.extend(labels)
    
    all_images = torch.cat(all_images, dim=0)
    if all_labels:
        all_labels = torch.tensor(all_labels)
    
    print(f"Loaded {all_images.shape[0]} images")
    
    all_features = []
    
    # Extract VAR features
    if USE_VAR_MODELS:
        print(f"\nExtracting VAR features...")
        
        for depth in [16, 20, 24, 30]:
            print(f"  VAR depth={depth}...")
            vae = load_var_vae_direct(depth)
            
            if vae is None:
                print(f"    Using dummy features")
                features = torch.zeros(all_images.shape[0], 2)
            else:
                vae = vae.to(device)
                features_list = []
                
                for i in tqdm(range(0, all_images.shape[0], BATCH_SIZE_INFERENCE), 
                             desc=f"    Processing", leave=False):
                    batch = all_images[i:i+BATCH_SIZE_INFERENCE].to(device)
                    
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        images_norm = batch * 2.0 - 1.0
                        f = vae.encoder(images_norm)
                        
                        quant_output = vae.quantize(f)
                        if isinstance(quant_output, tuple):
                            f_quant = quant_output[0]
                        else:
                            f_quant = quant_output
                        
                        quant_loss = F.mse_loss(f, f_quant, reduction='none').mean(dim=[1, 2, 3])
                        recon = vae.decoder(f_quant)
                        enc_loss = F.mse_loss(recon, images_norm, reduction='none').mean(dim=[1, 2, 3])
                        
                        batch_features = torch.stack([quant_loss, enc_loss], dim=1).cpu()
                    
                    features_list.append(batch_features)
                
                features = torch.cat(features_list)
                print(f"    ✓ QuantLoss: {features[:, 0].mean():.6f}, EncLoss: {features[:, 1].mean():.6f}")
                
                del vae
                torch.cuda.empty_cache()
            
            all_features.append(features)
    else:
        for _ in range(4):
            all_features.append(torch.zeros(all_images.shape[0], 2))
    
    # RAR features (simplified - using zeros for now)
    print(f"\nSkipping RAR features (using zeros)...")
    for _ in range(4):
        all_features.append(torch.zeros(all_images.shape[0], 2))
    
    # Concatenate
    all_features = torch.cat(all_features, dim=1)
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Mean: {all_features.mean():.6f}, Std: {all_features.std():.6f}")
    
    result = {
        'features': all_features,
        'labels': all_labels if len(all_labels) > 0 else None,
        'names': all_names if len(all_names) > 0 else None,
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"✓ Cached to {cache_file}\n")
    return result

# ============================================================
# EXTRACT FEATURES
# ============================================================

train_data = extract_features_for_all_models(train_loader, 'train', is_test=False)
val_data = extract_features_for_all_models(val_loader, 'val', is_test=False)
test_data = extract_features_for_all_models(test_loader, 'test', is_test=True)

# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_data['features'].numpy()
y_train = train_data['labels'].numpy()
X_val = val_data['features'].numpy()
y_val = val_data['labels'].numpy()
X_test = test_data['features'].numpy()

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"\n{'='*60}")
print("TRAINING SKLEARN CLASSIFIER")
print(f"{'='*60}")
print(f"Classifier type: {CLASSIFIER_TYPE}")
print(f"Train samples: {X_train.shape[0]}")
print(f"Val samples: {X_val.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# ============================================================
# TRAIN CLASSIFIER
# ============================================================

if CLASSIFIER_TYPE == "logistic":
    # Simple logistic regression (linear classifier)
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0,  # Regularization (higher = less regularization)
        random_state=42
    )
    
elif CLASSIFIER_TYPE == "random_forest":
    # Random Forest (ensemble of decision trees)
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
elif CLASSIFIER_TYPE == "svm":
    # Support Vector Machine
    print("\nTraining SVM...")
    clf = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,  # Enable probability estimates for outlier detection
        random_state=42
    )
    
elif CLASSIFIER_TYPE == "gradient_boost":
    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
else:
    raise ValueError(f"Unknown classifier type: {CLASSIFIER_TYPE}")

# Train
clf.fit(X_train, y_train)

# Evaluate
train_preds = clf.predict(X_train)
val_preds = clf.predict(X_val)

train_acc = accuracy_score(y_train, train_preds)
val_acc = accuracy_score(y_val, val_preds)

print(f"\n✓ Training complete!")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy: {val_acc:.4f}")

print(f"\nValidation Classification Report:")
print(classification_report(y_val, val_preds, target_names=LABELS, zero_division=0))

# ============================================================
# OUTLIER DETECTION
# ============================================================

print(f"\n{'='*60}")
print("OUTLIER DETECTION")
print(f"{'='*60}")

val_probs = clf.predict_proba(X_val)
outlier_idx = LABELS.index("outlier")

max_probs = val_probs.max(axis=1)
is_outlier = (y_val == outlier_idx)

outlier_probs = max_probs[is_outlier]
normal_probs = max_probs[~is_outlier]

print(f"Outlier confidence: mean={outlier_probs.mean():.4f}, std={outlier_probs.std():.4f}")
print(f"Normal confidence: mean={normal_probs.mean():.4f}, std={normal_probs.std():.4f}")

if USE_ADAPTIVE_THRESHOLD:
    tau = max(OUTLIER_CONF_THRESHOLD, np.percentile(outlier_probs, 75))
else:
    tau = OUTLIER_CONF_THRESHOLD

print(f"Threshold: tau={tau:.4f}")

# ============================================================
# INFERENCE
# ============================================================

print(f"\n{'='*60}")
print("GENERATING PREDICTIONS")
print(f"{'='*60}")

test_probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)

max_probs = test_probs.max(axis=1)
outlier_mask = (max_probs < tau)
preds[outlier_mask] = outlier_idx

# ============================================================
# SAVE SUBMISSION
# ============================================================

submission_data = []
for i, name in enumerate(test_data['names']):
    label = LABELS[preds[i]]
    submission_data.append([name, label])

with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(submission_data)

print(f"\n✓ Saved to {SUBMISSION_FILE}")
print(f"Outliers: {outlier_mask.sum()} ({100*outlier_mask.sum()/len(submission_data):.1f}%)")

label_counts = Counter([row[1] for row in submission_data])
print("\nLabel distribution:")
for label in LABELS:
    count = label_counts[label]
    percentage = 100 * count / len(submission_data)
    print(f"  {label:8s}: {count:5d} ({percentage:5.1f}%)")

# ============================================================
# SUBMIT
# ============================================================

if API_KEY != "IWILL_TYPE":
    print("\nSubmitting...")
    response = requests.post(
        f"{SERVER_URL}/submit/{TASK_ID}",
        files={"file": open(SUBMISSION_FILE, "rb")},
        headers={"X-API-Key": API_KEY},
    )
    print("Response:", response.json())
else:
    print("\nSet API_KEY to submit")