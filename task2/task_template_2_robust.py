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

# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

FORCE_REEXTRACT = True         # Force re-extraction of features
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_INFERENCE = 4       # Smaller batch for feature extraction
NUM_WORKERS = 4

# Classifier
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
DROPOUT_RATE = 0.3

# Training
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
BATCH_SIZE_CLASSIFIER = 64
NUM_EPOCHS = 100
PATIENCE = 15
SCHEDULER_T_MAX = 50

# Outlier detection
OUTLIER_CONF_THRESHOLD = 0.3
OUTLIER_MARGIN_THRESHOLD = 0.05
USE_ADAPTIVE_THRESHOLD = True

# Model selection
USE_VAR_MODELS = True
USE_RAR_MODELS = True
USE_DOUBLE_RECON_CALIBRATION = True

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

# Clear cache if requested
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

# Initialize torch.distributed for VAR models (they expect it)
# This is a single-process setup, so we just initialize with nccl backend
import torch.distributed as dist
if not dist.is_initialized():
    try:
        # Try to initialize distributed
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:12355', 
                                world_size=1, rank=0)
        print("✓ Initialized torch.distributed (single process)")
    except Exception as e:
        # If that fails, try gloo backend
        try:
            dist.init_process_group(backend='gloo', init_method='tcp://localhost:12355',
                                    world_size=1, rank=0)
            print("✓ Initialized torch.distributed with gloo backend")
        except Exception as e2:
            print(f"⚠️  Could not initialize torch.distributed: {e2}")
            print("   Attempting to patch torch.distributed functions...")
            
            # Monkey-patch the distributed functions to work in single-process mode
            original_get_world_size = dist.get_world_size
            original_get_rank = dist.get_rank
            
            def mock_get_world_size(group=None):
                return 1
            
            def mock_get_rank(group=None):
                return 0
            
            dist.get_world_size = mock_get_world_size
            dist.get_rank = mock_get_rank
            print("   ✓ Patched torch.distributed functions")

# ============================================================
# CHECKPOINT VERIFICATION
# ============================================================

print("\n" + "="*60)
print("VERIFYING CHECKPOINTS")
print("="*60)

# Check VAR
var_ckpt_dir = VAR_ROOT / "checkpoints"
print(f"\nVAR checkpoint directory: {var_ckpt_dir}")
print(f"Exists: {var_ckpt_dir.exists()}")
if var_ckpt_dir.exists():
    print("Files found:")
    for f in sorted(var_ckpt_dir.glob("*.pth")):
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name} ({size_mb:.1f} MB)")

# Check RAR
rar_ckpt_dir = RAR_ROOT / "checkpoints"
print(f"\nRAR checkpoint directory: {rar_ckpt_dir}")
print(f"Exists: {rar_ckpt_dir.exists()}")
if rar_ckpt_dir.exists():
    print("Files found:")
    for f in sorted(rar_ckpt_dir.glob("*.bin")):
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name} ({size_mb:.1f} MB)")

# ============================================================
# MODEL LOADING - OPTION 1: DIRECT IMPORT APPROACH
# ============================================================

def load_var_vae_direct(depth):
    """
    Try to load VAR VAE using direct import after adding path
    """
    print(f"  Attempting to load VAR VAE (depth={depth})...")
    
    # Check if checkpoint exists
    vae_ckpt = VAR_ROOT / "checkpoints" / "vae_ch160v4096z32.pth"
    if not vae_ckpt.exists():
        print(f"    ✗ Checkpoint not found: {vae_ckpt}")
        return None
    
    # Add VAR path
    var_path = str(VAR_ROOT.absolute())
    print(f"    Adding to sys.path: {var_path}")
    
    original_path = sys.path.copy()
    sys.path.insert(0, var_path)
    
    try:
        # Try import
        import models
        print(f"    ✓ Successfully imported models from {models.__file__}")
        
        # Build model
        vae, _ = models.build_vae_var(
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
        
        # Load weights
        print(f"    Loading weights from {vae_ckpt.name}...")
        state_dict = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(state_dict, strict=True)
        
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        
        print(f"    ✓ VAR VAE loaded successfully")
        return vae
        
    except Exception as e:
        print(f"    ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Restore original path
        sys.path = original_path

# ============================================================
# MODEL LOADING - OPTION 2: SUBPROCESS APPROACH
# ============================================================

def test_var_loading_subprocess():
    """
    Test if VAR can be loaded via subprocess
    This helps diagnose import issues
    """
    import subprocess
    
    test_script = f"""
import sys
sys.path.insert(0, '{VAR_ROOT.absolute()}')

try:
    from models import build_vae_var
    print("SUCCESS: models imported")
    print("build_vae_var:", build_vae_var)
except Exception as e:
    print("FAILED:", e)
    import traceback
    traceback.print_exc()
"""
    
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        cwd=str(VAR_ROOT)
    )
    
    print("\n  Subprocess test output:")
    print("  STDOUT:", result.stdout)
    if result.stderr:
        print("  STDERR:", result.stderr)
    print("  Return code:", result.returncode)
    
    return "SUCCESS" in result.stdout

# ============================================================
# MODEL LOADING - OPTION 3: MANUAL FEATURE EXTRACTION
# ============================================================

def compute_simple_features(images):
    """
    Fallback: Extract simple statistical features if models won't load
    These are NOT provenance features but can help debug
    """
    with torch.no_grad():
        # Compute various statistical features per image
        features = []
        
        # Color statistics
        mean_rgb = images.mean(dim=[2, 3])  # [B, 3]
        std_rgb = images.std(dim=[2, 3])    # [B, 3]
        
        # Frequency features (simple)
        # FFT magnitude statistics
        fft = torch.fft.fft2(images)
        fft_mag = torch.abs(fft).mean(dim=[1, 2, 3])  # [B]
        
        # Edge statistics (simple gradient)
        dx = images[:, :, :, 1:] - images[:, :, :, :-1]
        dy = images[:, :, 1:, :] - images[:, :, :-1, :]
        edge_mag = (dx.pow(2).mean(dim=[1, 2, 3]) + dy.pow(2).mean(dim=[1, 2, 3])).sqrt()
        
        features = torch.stack([
            mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2],
            std_rgb[:, 0], std_rgb[:, 1], std_rgb[:, 2],
            fft_mag,
            edge_mag,
        ], dim=1)
        
        return features.cpu()

# ============================================================
# FEATURE EXTRACTION WITH DIAGNOSTIC
# ============================================================

def extract_features_with_diagnostics(loader, split_name, is_test=False):
    """Extract features with extensive diagnostics"""
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
    
    # Test if we can load VAR
    print("\n" + "-"*60)
    print("DIAGNOSTIC: Testing VAR model loading")
    print("-"*60)
    
    success = test_var_loading_subprocess()
    print(f"Subprocess test: {'✓ PASSED' if success else '✗ FAILED'}")
    
    if success:
        print("\nSubprocess can import models, trying direct load...")
        test_vae = load_var_vae_direct(16)
        if test_vae is not None:
            print("✓ Direct load successful!")
            del test_vae
            torch.cuda.empty_cache()
            use_real_models = True
        else:
            print("✗ Direct load failed, falling back to simple features")
            use_real_models = False
    else:
        print("✗ Cannot import models even in subprocess")
        print("This suggests missing dependencies or corrupted VAR code")
        use_real_models = False
    
    # Extract features
    all_features = []
    
    if use_real_models and USE_VAR_MODELS:
        print(f"\n{'='*60}")
        print("EXTRACTING REAL VAR FEATURES")
        print(f"{'='*60}")
        
        for depth in [16, 20, 24, 30]:
            print(f"\nProcessing VAR depth={depth}...")
            vae = load_var_vae_direct(depth)
            
            if vae is None:
                print(f"  Using fallback features for depth={depth}")
                features_list = []
                for i in range(0, all_images.shape[0], BATCH_SIZE_INFERENCE):
                    batch = all_images[i:i+BATCH_SIZE_INFERENCE]
                    batch_features = compute_simple_features(batch)[:, :2]  # Take first 2
                    features_list.append(batch_features)
                features = torch.cat(features_list)
            else:
                vae = vae.to(device)
                features_list = []
                
                print(f"  Extracting features...")
                for i in tqdm(range(0, all_images.shape[0], BATCH_SIZE_INFERENCE), 
                             desc=f"    VAR-{depth}", leave=False):
                    batch = all_images[i:i+BATCH_SIZE_INFERENCE].to(device)
                    
                    # Compute provenance features
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        images_norm = batch * 2.0 - 1.0
                        f = vae.encoder(images_norm)
                        
                        # vae.quantize returns (f_quant, other_info) as a tuple
                        quant_output = vae.quantize(f)
                        if isinstance(quant_output, tuple):
                            f_quant = quant_output[0]  # Take the quantized features
                        else:
                            f_quant = quant_output
                        
                        quant_loss = F.mse_loss(f, f_quant, reduction='none').mean(dim=[1, 2, 3])
                        
                        recon = vae.decoder(f_quant)
                        enc_loss = F.mse_loss(recon, images_norm, reduction='none').mean(dim=[1, 2, 3])
                        
                        batch_features = torch.stack([quant_loss, enc_loss], dim=1).cpu()
                    
                    features_list.append(batch_features)
                
                features = torch.cat(features_list)
                
                print(f"  ✓ QuantLoss: mean={features[:, 0].mean():.6f} std={features[:, 0].std():.6f}")
                print(f"  ✓ EncLoss: mean={features[:, 1].mean():.6f} std={features[:, 1].std():.6f}")
                
                del vae
                torch.cuda.empty_cache()
            
            all_features.append(features)
    else:
        print(f"\n{'='*60}")
        print("USING FALLBACK SIMPLE FEATURES")
        print(f"{'='*60}")
        
        # Extract simple features for all models
        print("  Extracting statistical features...")
        simple_feats_list = []
        for i in tqdm(range(0, all_images.shape[0], BATCH_SIZE_INFERENCE), desc="  Computing"):
            batch = all_images[i:i+BATCH_SIZE_INFERENCE]
            batch_features = compute_simple_features(batch)
            simple_feats_list.append(batch_features)
        
        simple_feats = torch.cat(simple_feats_list)
        
        # Use different subsets for different "models"
        for idx in range(8):  # 8 models total
            # Use different feature combinations
            start = idx % 6
            feats = simple_feats[:, start:start+2]
            all_features.append(feats)
    
    # Concatenate
    all_features = torch.cat(all_features, dim=1)
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Statistics:")
    print(f"  Mean: {all_features.mean():.6f}")
    print(f"  Std:  {all_features.std():.6f}")
    print(f"  Min:  {all_features.min():.6f}")
    print(f"  Max:  {all_features.max():.6f}")
    
    # Check if features are all zeros (bad sign)
    if all_features.abs().max() < 1e-6:
        print("\n  ⚠️  WARNING: All features are near zero!")
        print("  This suggests models failed to load properly")
    
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

train_data = extract_features_with_diagnostics(train_loader, 'train', is_test=False)
val_data = extract_features_with_diagnostics(val_loader, 'val', is_test=False)
test_data = extract_features_with_diagnostics(test_loader, 'test', is_test=True)

# ============================================================
# TRAIN CLASSIFIER
# ============================================================

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

input_dim = train_data['features'].shape[1]
num_classes = len(LABELS)
model = ProvenanceClassifier(input_dim, num_classes).to(device)

X_train = train_data['features']
y_train = train_data['labels']
X_val = val_data['features']
y_val = val_data['labels']

print(f"Input features: {input_dim}")
print(f"Output classes: {num_classes}")
print(f"Train samples: {X_train.shape[0]}")
print(f"Val samples: {X_val.shape[0]}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
patience_counter = 0

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
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        val_preds = val_outputs.argmax(dim=1).cpu()
        val_acc = (val_preds == y_val).float().mean().item()
        train_acc = train_correct / X_train.shape[0]
    
    print(f"Epoch {epoch+1:3d}: Train={train_acc:.4f}, Val={val_acc:.4f}, Loss={train_loss/len(train_loader):.4f}", end="")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), CACHE_DIR / "best_model.pth")
        print(" ✓")
    else:
        patience_counter += 1
        print()
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

model.load_state_dict(torch.load(CACHE_DIR / "best_model.pth"))
print(f"\n✓ Best validation accuracy: {best_val_acc:.4f}")

# ============================================================
# OUTLIER DETECTION
# ============================================================

print(f"\n{'='*60}")
print("OUTLIER DETECTION")
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

print(f"Outlier confidence: mean={outlier_probs.mean():.4f}, std={outlier_probs.std():.4f}")
print(f"Normal confidence: mean={normal_probs.mean():.4f}, std={normal_probs.std():.4f}")

if USE_ADAPTIVE_THRESHOLD:
    tau = max(OUTLIER_CONF_THRESHOLD, outlier_probs.quantile(0.75).item())
else:
    tau = OUTLIER_CONF_THRESHOLD

m = OUTLIER_MARGIN_THRESHOLD

print(f"Thresholds: tau={tau:.4f}, margin={m:.4f}")

# ============================================================
# INFERENCE
# ============================================================

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

# ============================================================
# SAVE SUBMISSION
# ============================================================

submission_data = []
for i, name in enumerate(test_data['names']):
    label = LABELS[preds[i].item()]
    submission_data.append([name, label])

with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(submission_data)

print(f"\n✓ Saved to {SUBMISSION_FILE}")
print(f"Outliers: {outlier_mask.sum().item()} ({100*outlier_mask.sum().item()/len(submission_data):.1f}%)")

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