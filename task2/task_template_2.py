import csv
import zipfile
import requests
import pickle
import sys
import os
import shutil
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

# ============================================================
# CONFIGURATION PARAMETERS - ADJUST THESE TO TUNE PERFORMANCE
# ============================================================

# --- Data Loading ---
BATCH_SIZE_TRAIN = 16          # Batch size for loading training data (larger = faster but more memory)
BATCH_SIZE_INFERENCE = 8       # Batch size for feature extraction (reduce if OOM)
NUM_WORKERS = 4                # Number of parallel data loading workers

# --- Feature Extraction ---
FORCE_REEXTRACT = True         # Set to True to delete cache and re-extract features
                               # Set to False to use cached features (much faster)

# --- Classifier Architecture ---
HIDDEN_DIM_1 = 512            # First hidden layer size (larger = more capacity)
HIDDEN_DIM_2 = 256            # Second hidden layer size
DROPOUT_RATE = 0.3            # Dropout probability (0.0-0.5, higher = more regularization)

# --- Training Hyperparameters ---
LEARNING_RATE = 0.1         # Initial learning rate (0.0001-0.01, lower = more stable)
WEIGHT_DECAY = 0.01           # L2 regularization strength (0.0-0.1, higher = more regularization)
BATCH_SIZE_CLASSIFIER = 64    # Batch size for classifier training
NUM_EPOCHS = 100              # Maximum number of training epochs
PATIENCE = 15                 # Early stopping patience (stop if no improvement for N epochs)
SCHEDULER_T_MAX = 50          # Cosine annealing schedule period

# --- Outlier Detection Thresholds ---
OUTLIER_CONF_THRESHOLD = 0.1  # Base confidence threshold for outlier detection (0.1-0.5)
                               # Lower = more aggressive outlier detection
OUTLIER_MARGIN_THRESHOLD = 0.01  # Margin between top-2 predictions (0.01-0.2)
                                  # Lower = more aggressive outlier detection
USE_ADAPTIVE_THRESHOLD = True    # Use validation set to calibrate threshold automatically

# --- Model Selection ---
# Enable/disable specific model families for feature extraction
USE_VAR_MODELS = True         # Extract features from VAR models
USE_RAR_MODELS = True         # Extract features from RAR models

# --- Feature Engineering ---
USE_DOUBLE_RECON_CALIBRATION = True  # Use double reconstruction for EncLoss calibration
                                      # Following the paper's methodology

# --- Submission ---
API_KEY = "c8286483e3f08d5579bea4e972a7d21b"        # Your team API key for leaderboard submission

# ============================================================
# END OF CONFIGURATION
# ============================================================

# Fixed paths and labels
ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("dataset")
SUBMISSION_FILE = "submission.csv"
LABELS = ["var16", "var20", "var24", "var30", "rarb", "rarl", "rarxl", "rarxxl", "outlier"]

SERVER_URL = "http://35.192.205.84:80"
TASK_ID = "15-model-tracer"

VAR_ROOT = Path("task2/VAR")
RAR_ROOT = Path("task2/RAR/1d-tokenizer")
CACHE_DIR = Path("task2/.cache_scores")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# DELETE OLD CACHE IF REQUESTED
if FORCE_REEXTRACT:
    print("="*60)
    print("CLEARING OLD CACHE (FORCE_REEXTRACT=True)")
    print("="*60)
    for cache_file in CACHE_DIR.glob("*_provenance_features.pkl"):
        cache_file.unlink()
        print(f"  Deleted {cache_file}")
    print()

# Model configurations
VAR_MODELS = {
    "var16": {"depth": 16},
    "var20": {"depth": 20},
    "var24": {"depth": 24},
    "var30": {"depth": 30},
}

RAR_MODELS = {
    "rarb": {"ckpt": "rar_b.bin"},
    "rarl": {"ckpt": "rar_l.bin"},
    "rarxl": {"ckpt": "rar_xl.bin"},
    "rarxxl": {"ckpt": "rar_xxl.bin"},
}

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
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ----------------------------
# DATASETS & DATALOADERS
# ----------------------------
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

# ----------------------------
# CHECK WHAT FILES EXIST
# ----------------------------
print("\n" + "="*60)
print("CHECKING CHECKPOINT FILES")
print("="*60)

var_vae_ckpt = VAR_ROOT / "checkpoints" / "vae_ch160v4096z32.pth"
print(f"VAR VAE checkpoint: {'✓ EXISTS' if var_vae_ckpt.exists() else '✗ NOT FOUND'}")
print(f"  Path: {var_vae_ckpt}")

rar_tokenizer_ckpt = RAR_ROOT / "checkpoints" / "maskgit-vqgan-imagenet-f16-256.bin"
print(f"RAR tokenizer checkpoint: {'✓ EXISTS' if rar_tokenizer_ckpt.exists() else '✗ NOT FOUND'}")
print(f"  Path: {rar_tokenizer_ckpt}")

print(f"\nFiles in {VAR_ROOT / 'checkpoints'}:")
for f in sorted((VAR_ROOT / "checkpoints").glob("*")):
    print(f"  {f.name}")

print(f"\nFiles in {RAR_ROOT / 'checkpoints'}:")
for f in sorted((RAR_ROOT / "checkpoints").glob("*")):
    print(f"  {f.name}")

# ----------------------------
# MODEL LOADING FUNCTIONS
# ----------------------------

def load_var_model(model_name, depth):
    """Load VAR VAE for provenance feature extraction"""
    if not USE_VAR_MODELS:
        return None
        
    print(f"\n  Loading VAR model {model_name} (depth={depth})...")
    
    vae_ckpt = VAR_ROOT / "checkpoints" / "vae_ch160v4096z32.pth"
    if not vae_ckpt.exists():
        print(f"    ✗ VAE checkpoint not found at {vae_ckpt}")
        return None
    
    try:
        var_path = str(VAR_ROOT)
        if var_path not in sys.path:
            sys.path.insert(0, var_path)
        
        from models import build_vae_var
        
        print(f"    Building VAE model...")
        vae, _ = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,
            device=device,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            num_classes=1000,
            depth=depth,
            shared_aln=False,
        )
        
        print(f"    Loading weights from {vae_ckpt.name}...")
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device), strict=True)
        
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        
        print(f"    ✓ VAR model loaded successfully")
        
        if var_path in sys.path:
            sys.path.remove(var_path)
        
        return vae
        
    except Exception as e:
        print(f"    ✗ ERROR loading VAR model: {e}")
        import traceback
        traceback.print_exc()
        if var_path in sys.path:
            sys.path.remove(var_path)
        return None

def load_rar_tokenizer():
    """Load RAR tokenizer for provenance feature extraction"""
    if not USE_RAR_MODELS:
        return None
        
    print(f"\n  Loading RAR tokenizer...")
    
    tokenizer_ckpt = RAR_ROOT / "checkpoints" / "maskgit-vqgan-imagenet-f16-256.bin"
    if not tokenizer_ckpt.exists():
        print(f"    ✗ Tokenizer checkpoint not found at {tokenizer_ckpt}")
        return None
    
    try:
        rar_path = str(RAR_ROOT)
        if rar_path not in sys.path:
            sys.path.insert(0, rar_path)
        
        from utils.train_utils import create_pretrained_tokenizer
        import demo_util
        
        print(f"    Loading config...")
        config = demo_util.get_config(str(RAR_ROOT / "configs/training/generator/rar.yaml"))
        config.model.vq_model.pretrained_tokenizer_weight = str(tokenizer_ckpt)
        
        print(f"    Creating tokenizer...")
        tokenizer = create_pretrained_tokenizer(config).to(device)
        tokenizer.eval()
        for p in tokenizer.parameters():
            p.requires_grad_(False)
        
        print(f"    ✓ RAR tokenizer loaded successfully")
        
        if rar_path in sys.path:
            sys.path.remove(rar_path)
        
        return tokenizer
        
    except Exception as e:
        print(f"    ✗ ERROR loading RAR tokenizer: {e}")
        import traceback
        traceback.print_exc()
        if rar_path in sys.path:
            sys.path.remove(rar_path)
        return None

# ----------------------------
# PROVENANCE FEATURE EXTRACTION
# ----------------------------

def compute_var_provenance_features(vae, images):
    """Compute provenance features using VAR VAE"""
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Normalize to [-1, 1] for VAR
            images_norm = images * 2.0 - 1.0
            
            # Encode
            f = vae.encoder(images_norm)
            
            # Quantize
            f_quant = vae.quantize(f)
            
            # QuantLoss: distance between pre-quant and post-quant features
            quant_loss = F.mse_loss(f, f_quant, reduction='none').mean(dim=[1, 2, 3])
            
            # Decode
            recon = vae.decoder(f_quant)
            
            # EncLoss: reconstruction error
            enc_loss = F.mse_loss(recon, images_norm, reduction='none').mean(dim=[1, 2, 3])
            
            if USE_DOUBLE_RECON_CALIBRATION:
                # Double reconstruction for calibration (from paper)
                f2 = vae.encoder(recon)
                f2_quant = vae.quantize(f2)
                recon2 = vae.decoder(f2_quant)
                enc_loss2 = F.mse_loss(recon2, recon, reduction='none').mean(dim=[1, 2, 3])
                
                # Calibrated EncLoss
                enc_loss_cal = enc_loss / (enc_loss2 + 1e-8)
            else:
                enc_loss_cal = enc_loss
            
            return torch.stack([quant_loss, enc_loss_cal], dim=1).cpu()
            
    except Exception as e:
        print(f"      ✗ Error in compute_var_provenance_features: {e}")
        return torch.zeros(images.shape[0], 2)

def compute_rar_provenance_features(tokenizer, images):
    """Compute provenance features using RAR tokenizer"""
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Encode to tokens
            tokens = tokenizer.encode(images)
            
            # Decode back
            recon = tokenizer.decode(tokens)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, images, reduction='none').mean(dim=[1, 2, 3])
            
            if USE_DOUBLE_RECON_CALIBRATION:
                # Double reconstruction for calibration
                tokens2 = tokenizer.encode(recon)
                recon2 = tokenizer.decode(tokens2)
                recon_loss2 = F.mse_loss(recon2, recon, reduction='none').mean(dim=[1, 2, 3])
                
                # Calibrated loss
                recon_loss_cal = recon_loss / (recon_loss2 + 1e-8)
            else:
                recon_loss_cal = recon_loss
            
            return torch.stack([recon_loss, recon_loss_cal], dim=1).cpu()
            
    except Exception as e:
        print(f"      ✗ Error in compute_rar_provenance_features: {e}")
        return torch.zeros(images.shape[0], 2)

def extract_features_for_all_models(loader, split_name, is_test=False):
    """Extract provenance features for all source models"""
    cache_file = CACHE_DIR / f"{split_name}_provenance_features.pkl"
    
    if cache_file.exists() and not FORCE_REEXTRACT:
        print(f"Loading cached features for {split_name}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES FOR {split_name.upper()}")
    print(f"{'='*60}")
    
    # Collect all images
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
    
    # Process VAR models
    if USE_VAR_MODELS:
        print(f"\n{'-'*60}")
        print("EXTRACTING VAR FEATURES")
        print(f"{'-'*60}")
        
        for model_name in ["var16", "var20", "var24", "var30"]:
            print(f"\nProcessing {model_name}...")
            depth = VAR_MODELS[model_name]["depth"]
            vae = load_var_model(model_name, depth)
            
            if vae is None:
                print(f"  ✗ Failed to load {model_name}, using dummy features")
                features = torch.zeros(all_images.shape[0], 2)
            else:
                vae = vae.to(device)
                
                features_list = []
                
                print(f"  Extracting features...")
                for i in tqdm(range(0, all_images.shape[0], BATCH_SIZE_INFERENCE), desc=f"    {model_name}", leave=False):
                    batch = all_images[i:i+BATCH_SIZE_INFERENCE].to(device)
                    batch_features = compute_var_provenance_features(vae, batch)
                    features_list.append(batch_features)
                
                features = torch.cat(features_list)
                
                print(f"  ✓ QuantLoss mean={features[:, 0].mean():.6f} std={features[:, 0].std():.6f}")
                print(f"  ✓ EncLoss mean={features[:, 1].mean():.6f} std={features[:, 1].std():.6f}")
                
                del vae
                torch.cuda.empty_cache()
            
            all_features.append(features)
    else:
        print("\nSkipping VAR models (USE_VAR_MODELS=False)")
        for model_name in ["var16", "var20", "var24", "var30"]:
            all_features.append(torch.zeros(all_images.shape[0], 2))
    
    # Process RAR models
    if USE_RAR_MODELS:
        print(f"\n{'-'*60}")
        print("EXTRACTING RAR FEATURES")
        print(f"{'-'*60}")
        
        rar_tokenizer = load_rar_tokenizer()
        
        if rar_tokenizer is None:
            print("  ✗ Failed to load RAR tokenizer, using dummy features")
            for model_name in ["rarb", "rarl", "rarxl", "rarxxl"]:
                all_features.append(torch.zeros(all_images.shape[0], 2))
        else:
            rar_tokenizer = rar_tokenizer.to(device)
            
            print("  Extracting tokenizer features...")
            features_list = []
            
            for i in tqdm(range(0, all_images.shape[0], BATCH_SIZE_INFERENCE), desc="    RAR", leave=False):
                batch = all_images[i:i+BATCH_SIZE_INFERENCE].to(device)
                batch_features = compute_rar_provenance_features(rar_tokenizer, batch)
                features_list.append(batch_features)
            
            rar_features = torch.cat(features_list)
            print(f"  ✓ ReconLoss mean={rar_features[:, 0].mean():.6f} std={rar_features[:, 0].std():.6f}")
            print(f"  ✓ CalibLoss mean={rar_features[:, 1].mean():.6f} std={rar_features[:, 1].std():.6f}")
            
            # Use same features for all RAR models
            for model_name in ["rarb", "rarl", "rarxl", "rarxxl"]:
                all_features.append(rar_features.clone())
            
            del rar_tokenizer
            torch.cuda.empty_cache()
    else:
        print("\nSkipping RAR models (USE_RAR_MODELS=False)")
        for model_name in ["rarb", "rarl", "rarxl", "rarxxl"]:
            all_features.append(torch.zeros(all_images.shape[0], 2))
    
    # Concatenate all features
    all_features = torch.cat(all_features, dim=1)
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETE FOR {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Total feature dimension: {all_features.shape[1]}")
    print(f"Feature statistics:")
    print(f"  Mean: {all_features.mean():.6f}")
    print(f"  Std:  {all_features.std():.6f}")
    print(f"  Min:  {all_features.min():.6f}")
    print(f"  Max:  {all_features.max():.6f}")
    
    result = {
        'features': all_features,
        'labels': all_labels if len(all_labels) > 0 else None,
        'names': all_names if len(all_names) > 0 else None,
    }
    
    # Cache
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"✓ Cached features to {cache_file}\n")
    return result

# ----------------------------
# EXTRACT FEATURES
# ----------------------------
train_data = extract_features_for_all_models(train_loader, 'train', is_test=False)
val_data = extract_features_for_all_models(val_loader, 'val', is_test=False)
test_data = extract_features_for_all_models(test_loader, 'test', is_test=True)

# ----------------------------
# TRAIN CLASSIFIER
# ----------------------------

class ProvenanceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.fc3 = nn.Linear(HIDDEN_DIM_2, num_classes)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM_1)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM_2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

print(f"\n{'='*60}")
print("TRAINING CLASSIFIER")
print(f"{'='*60}")
print(f"Configuration:")
print(f"  Hidden dims: {HIDDEN_DIM_1} → {HIDDEN_DIM_2}")
print(f"  Dropout: {DROPOUT_RATE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Batch size: {BATCH_SIZE_CLASSIFIER}")

input_dim = train_data['features'].shape[1]
num_classes = len(LABELS)
model = ProvenanceClassifier(input_dim, num_classes).to(device)

print(f"\nModel architecture:")
print(f"  Input features: {input_dim}")
print(f"  Output classes: {num_classes}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

X_train = train_data['features']
y_train = train_data['labels']
X_val = val_data['features']
y_val = val_data['labels']

print(f"\nDataset sizes:")
print(f"  Train samples: {X_train.shape[0]}")
print(f"  Val samples: {X_val.shape[0]}")

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
patience_counter = 0

print(f"\nTraining for up to {NUM_EPOCHS} epochs (patience={PATIENCE})...")

for epoch in range(NUM_EPOCHS):
    model.train()
    
    perm = torch.randperm(X_train.shape[0])
    train_loss = 0
    train_correct = 0
    
    for i in range(0, X_train.shape[0], BATCH_SIZE_CLASSIFIER):
        indices = perm[i:i+BATCH_SIZE_CLASSIFIER]
        batch_X = X_train[indices].to(device)
        batch_y = y_train[indices].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        val_preds = val_outputs.argmax(dim=1).cpu()
        val_acc = (val_preds == y_val).float().mean().item()
        train_acc = train_correct / X_train.shape[0]
    
    print(f"Epoch {epoch+1:3d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Loss={train_loss/len(train_loader):.4f}", end="")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), CACHE_DIR / "best_model.pth")
        print(" → ✓ New best!")
    else:
        patience_counter += 1
        print()
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered (no improvement for {PATIENCE} epochs)")
            break

model.load_state_dict(torch.load(CACHE_DIR / "best_model.pth"))
print(f"\n✓ Best validation accuracy: {best_val_acc:.4f}")

# ----------------------------
# OUTLIER CALIBRATION
# ----------------------------
print(f"\n{'='*60}")
print("CALIBRATING OUTLIER DETECTION")
print(f"{'='*60}")

model.eval()
with torch.no_grad():
    val_outputs = model(X_val.to(device))
    val_probs = F.softmax(val_outputs, dim=1).cpu()

outlier_idx = LABELS.index("outlier")
is_outlier = (y_val == outlier_idx)

max_probs, _ = val_probs.max(dim=1)
outlier_probs = max_probs[is_outlier]
normal_probs = max_probs[~is_outlier]

print(f"Confidence distribution:")
print(f"  Outlier - Mean: {outlier_probs.mean():.4f}, Std: {outlier_probs.std():.4f}")
print(f"  Normal  - Mean: {normal_probs.mean():.4f}, Std: {normal_probs.std():.4f}")

if USE_ADAPTIVE_THRESHOLD:
    tau = max(OUTLIER_CONF_THRESHOLD, outlier_probs.quantile(0.75).item())
    print(f"\nUsing adaptive threshold (75th percentile of outlier confidences)")
else:
    tau = OUTLIER_CONF_THRESHOLD
    print(f"\nUsing fixed threshold")

m = OUTLIER_MARGIN_THRESHOLD

print(f"Final thresholds:")
print(f"  Confidence threshold (tau): {tau:.4f}")
print(f"  Margin threshold (m): {m:.4f}")

# ----------------------------
# INFERENCE
# ----------------------------
print(f"\n{'='*60}")
print("GENERATING PREDICTIONS")
print(f"{'='*60}")

model.eval()
with torch.no_grad():
    test_outputs = model(test_data['features'].to(device))
    test_probs = F.softmax(test_outputs, dim=1).cpu()

max_probs, preds = test_probs.max(dim=1)
top2_probs = test_probs.topk(2, dim=1)[0]
margins = top2_probs[:, 0] - top2_probs[:, 1]

outlier_mask = (max_probs < tau) | (margins < m)
preds[outlier_mask] = outlier_idx

# ----------------------------
# SAVE SUBMISSION
# ----------------------------
submission_data = []
for i, name in enumerate(test_data['names']):
    label = LABELS[preds[i].item()]
    submission_data.append([name, label])

with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(submission_data)

print(f"\n✓ Saved submission to {SUBMISSION_FILE}")
print(f"Total predictions: {len(submission_data)}")
print(f"Outliers detected: {outlier_mask.sum().item()} ({100*outlier_mask.sum().item()/len(submission_data):.1f}%)")

label_counts = Counter([row[1] for row in submission_data])
print("\nLabel distribution:")
for label in LABELS:
    count = label_counts[label]
    percentage = 100 * count / len(submission_data)
    print(f"  {label:8s}: {count:5d} ({percentage:5.1f}%)")

# ----------------------------
# SUBMIT
# ----------------------------
if API_KEY is None or API_KEY == "IWILL_TYPE":
    print("\n" + "="*60)
    print("No API_KEY provided - skipping submission")
    print("Set API_KEY at the top of the script to submit")
    print("="*60)
else:
    print("\n" + "="*60)
    print("SUBMITTING TO LEADERBOARD")
    print("="*60)
    response = requests.post(
        f"{SERVER_URL}/submit/{TASK_ID}",
        files={"file": open(SUBMISSION_FILE, "rb")},
        headers={"X-API-Key": API_KEY},
    )
    print("Server response:", response.json())