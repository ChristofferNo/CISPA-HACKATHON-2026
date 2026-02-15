#!/usr/bin/env python3
"""
Data Provenance Detection for Image Autoregressive Models
CISPA European Championship 2026 - Stockholm

Task: Identify which generative model (VAR or RAR) produced each image,
or classify as outlier if from an unknown source.

Methodology:
1. Extract provenance features from VAR and RAR models
2. Train classifier on concatenated features
3. Apply confidence-based outlier detection

Models:
- VAR (Visual AutoRegressive): 4 depths [16, 20, 24, 30]
- RAR (Randomized AutoRegressive): 4 sizes [B, L, XL, XXL]
"""

import csv
import zipfile
import requests
import pickle
import sys
import traceback
import importlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

# Feature extraction
FORCE_REEXTRACT = False  # Set True to recompute features (slow)
ALLOW_DUMMY_FEATURES = False  # If True: use zeros on model load failure

# Data loading
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_INFERENCE = 8
NUM_WORKERS = 4

# Classifier
CLASSIFIER_TYPE = "logistic"  # Options: logistic, random_forest, gradient_boost

# Outlier detection
OUTLIER_CONF_THRESHOLD = 0.5
USE_ADAPTIVE_THRESHOLD = True

# Model selection
USE_VAR_MODELS = True
USE_RAR_MODELS = True
RAR_TOPK_LABELS = 2  # Number of ImageNet labels for RAR marginalization

# Submission
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your team API key
SERVER_URL = "http://35.192.205.84:80"
TASK_ID = "15-model-tracer"

# Paths
ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("dataset")
SUBMISSION_FILE = "submission.csv"
LABELS = [
    "var16", "var20", "var24", "var30",
    "rarb", "rarl", "rarxl", "rarxxl",
    "outlier"
]

REPO_ROOT = Path(__file__).resolve().parent
VAR_ROOT = (REPO_ROOT / "VAR").resolve()
RAR_ROOT = (REPO_ROOT / "RAR" / "1d-tokenizer").resolve()
CACHE_DIR = (REPO_ROOT / ".cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear cache if requested
if FORCE_REEXTRACT:
    print("Clearing feature cache...")
    for cache_file in CACHE_DIR.glob("*_features.pkl"):
        cache_file.unlink()

# Initialize torch.distributed (required by VAR models)
import torch.distributed as dist
if not dist.is_initialized():
    try:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://localhost:12355",
            world_size=1,
            rank=0,
        )
    except Exception:
        dist.get_world_size = lambda group=None: 1
        dist.get_rank = lambda group=None: 0


# ============================================================
# IMPORT HELPERS
# ============================================================

def safe_import(root: Path, module: str, name: str):
    """Safely import module from path with error handling."""
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing {name} directory: {root}")
    
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    
    try:
        return importlib.import_module(module)
    except Exception as e:
        tb = traceback.format_exc()
        raise ImportError(f"Failed importing '{module}' from '{root}'.\n{tb}") from e


def handle_error(err: Exception, context: str):
    """Handle errors based on ALLOW_DUMMY_FEATURES setting."""
    if ALLOW_DUMMY_FEATURES:
        print(f"[WARNING] {context}: {err}")
        print("[WARNING] Using dummy features (ALLOW_DUMMY_FEATURES=True)")
        return None
    raise RuntimeError(f"{context}: {err}") from err


# ============================================================
# DATASET
# ============================================================

if not DATASET_DIR.exists():
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=DATASET_DIR / "train", transform=transform)
val_dataset = datasets.ImageFolder(root=DATASET_DIR / "val", transform=transform)


class TestDataset(Dataset):
    """Dataset for unlabeled test images."""
    
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, 
                          shuffle=False, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_TRAIN,
                        shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TRAIN,
                         shuffle=False, num_workers=NUM_WORKERS)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_dataset)}")
print(f"  Val: {len(val_dataset)}")
print(f"  Test: {len(test_dataset)}")


# ============================================================
# VAR FEATURE EXTRACTION
# ============================================================

def load_var_vae(depth: int):
    """Load VAR VAE model for given depth."""
    vae_ckpt = VAR_ROOT / "checkpoints" / "vae_ch160v4096z32.pth"
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"Missing VAR checkpoint: {vae_ckpt}")

    models_mod = safe_import(VAR_ROOT, "models", "VAR")
    vae, _ = models_mod.build_vae_var(
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


@torch.no_grad()
def compute_var_features(vae, images: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute VAR provenance features.
    
    Features:
        - QuantLoss: ||f - Q(f)||^2 (quantization error)
        - ReconLoss: Calibrated reconstruction error
    
    Returns:
        [batch_size, 2] tensor
    """
    x = images.to(device)
    x_norm = x * 2.0 - 1.0  # Normalize to [-1, 1]

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        # Encode and quantize
        f = vae.encoder(x_norm)
        q = vae.quantize(f)
        f_q = q[0] if isinstance(q, tuple) else q

        # Quantization loss
        quant_loss = F.mse_loss(f, f_q, reduction="none").mean(dim=[1, 2, 3])

        # First reconstruction
        recon1 = vae.decoder(f_q)
        enc_loss1 = F.mse_loss(recon1, x_norm, reduction="none").mean(dim=[1, 2, 3])

        # Second reconstruction (for calibration)
        f2 = vae.encoder(recon1)
        q2 = vae.quantize(f2)
        f2_q = q2[0] if isinstance(q2, tuple) else q2
        recon2 = vae.decoder(f2_q)
        enc_loss2 = F.mse_loss(recon2, recon1, reduction="none").mean(dim=[1, 2, 3])

        # Calibrated reconstruction loss
        enc_loss_cal = enc_loss1 / (enc_loss2 + eps)

    return torch.stack([quant_loss, enc_loss_cal], dim=1).cpu()


def extract_var_features(images: torch.Tensor) -> list[torch.Tensor]:
    """Extract VAR features for all depth configurations."""
    features = []
    
    for depth in [16, 20, 24, 30]:
        print(f"  Processing VAR-{depth}...")
        
        try:
            vae = load_var_vae(depth).to(device)
        except Exception as e:
            vae = handle_error(e, f"Failed to load VAR-{depth}")
            if vae is None:
                features.append(torch.zeros(images.shape[0], 2))
                continue

        batch_features = []
        for i in tqdm(range(0, images.shape[0], BATCH_SIZE_INFERENCE),
                     desc=f"    Extracting", leave=False):
            batch = images[i:i + BATCH_SIZE_INFERENCE]
            batch_features.append(compute_var_features(vae, batch))
        
        feats = torch.cat(batch_features, dim=0)
        print(f"    QuantLoss: μ={feats[:, 0].mean():.6f} σ={feats[:, 0].std():.6f}")
        print(f"    ReconLoss: μ={feats[:, 1].mean():.6f} σ={feats[:, 1].std():.6f}")
        
        features.append(feats)
        
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return features


# ============================================================
# RAR FEATURE EXTRACTION
# ============================================================

RAR_CKPT_DIR = RAR_ROOT / "checkpoints"
MASKGIT_CKPT = RAR_CKPT_DIR / "maskgit-vqgan-imagenet-f16-256.bin"

RAR_CKPTS = {
    "rarb": RAR_CKPT_DIR / "rar_b.bin",
    "rarl": RAR_CKPT_DIR / "rar_l.bin",
    "rarxl": RAR_CKPT_DIR / "rar_xl.bin",
    "rarxxl": RAR_CKPT_DIR / "rar_xxl.bin",
}

RAR_ARCH = {
    "rar_b": dict(hidden_size=768, num_hidden_layers=24, 
                  num_attention_heads=16, intermediate_size=3072),
    "rar_l": dict(hidden_size=1024, num_hidden_layers=24,
                  num_attention_heads=16, intermediate_size=4096),
    "rar_xl": dict(hidden_size=1280, num_hidden_layers=32,
                   num_attention_heads=16, intermediate_size=5120),
    "rar_xxl": dict(hidden_size=1408, num_hidden_layers=40,
                    num_attention_heads=16, intermediate_size=6144),
}

_rar_modules_loaded = False
_demo_util = None
_train_utils = None


def load_rar_modules():
    """Load RAR modules (lazy loading)."""
    global _rar_modules_loaded, _demo_util, _train_utils
    
    if _rar_modules_loaded:
        return
    
    _demo_util = safe_import(RAR_ROOT, "demo_util", "RAR")
    _train_utils = safe_import(RAR_ROOT, "utils.train_utils", "RAR")
    _rar_modules_loaded = True


def load_rar_model(size_key: str):
    """Load RAR tokenizer and generator for given size."""
    ckpt = RAR_CKPTS[size_key]
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing RAR checkpoint: {ckpt}")
    if not MASKGIT_CKPT.exists():
        raise FileNotFoundError(f"Missing RAR tokenizer: {MASKGIT_CKPT}")

    load_rar_modules()

    # Configure model
    arch_key = ckpt.name.replace(".bin", "")
    config = _demo_util.get_config("configs/training/generator/rar.yaml")
    config.experiment.generator_checkpoint = str(ckpt)
    config.model.vq_model.pretrained_tokenizer_weight = str(MASKGIT_CKPT)

    arch = RAR_ARCH[arch_key]
    for key, value in arch.items():
        setattr(config.model.generator, key, value)

    # Create models
    tokenizer = _train_utils.create_pretrained_tokenizer(config).to(device)
    generator = _demo_util.get_rar_generator(config).to(device)

    tokenizer.eval()
    generator.eval()
    
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    for p in generator.parameters():
        p.requires_grad_(False)

    return tokenizer, generator


# Label proposal for RAR likelihood marginalization
_resnet18 = None


@torch.no_grad()
def get_candidate_labels(images: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Get top-k ImageNet label candidates using ResNet18."""
    global _resnet18
    
    try:
        import torchvision.models as tvm
        
        if _resnet18 is None:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            _resnet18 = tvm.resnet18(weights=weights).to(device).eval()
            _resnet18._weights = weights
        
        weights = _resnet18._weights
        x = images.to(device)
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = weights.transforms()(x)
        logits = _resnet18(x)
        
        return torch.topk(logits, k=k, dim=1).indices
        
    except Exception as e:
        print(f"[WARNING] Label proposal failed: {e}")
        # Fallback to fixed labels
        B = images.shape[0]
        fixed = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)[:k]
        return fixed.unsqueeze(0).repeat(B, 1)


@torch.no_grad()
def compute_rar_features(tokenizer, generator, images: torch.Tensor,
                         k_labels: int = 3) -> torch.Tensor:
    """
    Compute RAR provenance features with label marginalization.
    
    Features:
        - NLL: Negative log-likelihood (marginalized over labels)
        - Prob: exp(-NLL) as confidence proxy
    
    Returns:
        [batch_size, 2] tensor
    """
    x = images.to(device)
    
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        tokens = tokenizer.encode(x)

    # Get candidate labels
    candidates = get_candidate_labels(images, k=k_labels)  # [B, k]
    
    logp_list = []
    for j in range(candidates.shape[1]):
        labels = candidates[:, j]
        cond = generator.preprocess_condition(labels)
        logits, tokens_tf = generator(tokens, cond, return_labels=True)

        # Shift logits and reshape
        logits = logits[:, :-1]
        tokens_tf = tokens_tf.reshape(tokens_tf.shape[0], -1)
        logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])

        # Compute token log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, tokens_tf.unsqueeze(-1)).squeeze(-1)
        
        # Sum over sequence
        logp = token_logp.sum(dim=1)
        logp_list.append(logp)

    # Marginalize over labels (log-mean-exp)
    logp_stack = torch.stack(logp_list, dim=1)  # [B, k]
    logp_marg = torch.logsumexp(logp_stack, dim=1) - np.log(candidates.shape[1])

    # Compute features
    seq_len = tokens_tf.shape[1]
    nll = (-logp_marg / max(seq_len, 1)).cpu()
    prob = torch.exp(-nll).cpu()

    return torch.stack([nll, prob], dim=1)


def extract_rar_features(images: torch.Tensor) -> list[torch.Tensor]:
    """Extract RAR features for all size configurations."""
    features = []
    
    for size_key in ["rarb", "rarl", "rarxl", "rarxxl"]:
        print(f"  Processing RAR-{size_key.upper()}...")
        
        try:
            tokenizer, generator = load_rar_model(size_key)
        except Exception as e:
            pair = handle_error(e, f"Failed to load RAR-{size_key}")
            if pair is None:
                features.append(torch.zeros(images.shape[0], 2))
                continue

        batch_features = []
        for i in tqdm(range(0, images.shape[0], BATCH_SIZE_INFERENCE),
                     desc=f"    Extracting", leave=False):
            batch = images[i:i + BATCH_SIZE_INFERENCE]
            batch_features.append(
                compute_rar_features(tokenizer, generator, batch, k_labels=RAR_TOPK_LABELS)
            )
        
        feats = torch.cat(batch_features, dim=0)
        print(f"    NLL: μ={feats[:, 0].mean():.6f} σ={feats[:, 0].std():.6f}")
        print(f"    Prob: μ={feats[:, 1].mean():.6f} σ={feats[:, 1].std():.6f}")
        
        features.append(feats)
        
        del tokenizer, generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return features


# ============================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================

def extract_all_features(loader, split_name: str, is_test: bool = False):
    """Extract and cache provenance features for a dataset split."""
    cache_file = CACHE_DIR / f"{split_name}_features.pkl"
    
    if cache_file.exists() and not FORCE_REEXTRACT:
        print(f"Loading cached features: {split_name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES: {split_name.upper()}")
    print(f"{'='*60}")

    # Load all images
    all_images = []
    all_labels = []
    all_names = []

    for batch in tqdm(loader, desc="Loading images"):
        if is_test:
            images, names = batch
            all_images.append(images)
            all_names.extend(names)
        else:
            images, labels = batch
            all_images.append(images)
            all_labels.extend(labels)

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels) if all_labels else None

    print(f"Loaded {all_images.shape[0]} images")

    # Extract features
    feature_blocks = []

    if USE_VAR_MODELS:
        print("\nExtracting VAR features...")
        feature_blocks.extend(extract_var_features(all_images))
    else:
        print("\nSkipping VAR features")
        feature_blocks.extend([torch.zeros(all_images.shape[0], 2) for _ in range(4)])

    if USE_RAR_MODELS:
        print("\nExtracting RAR features...")
        feature_blocks.extend(extract_rar_features(all_images))
    else:
        print("\nSkipping RAR features")
        feature_blocks.extend([torch.zeros(all_images.shape[0], 2) for _ in range(4)])

    # Concatenate all features
    all_features = torch.cat(feature_blocks, dim=1)

    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Statistics: μ={all_features.mean():.6f} σ={all_features.std():.6f}")

    # Sanity checks
    if torch.isnan(all_features).any() or torch.isinf(all_features).any():
        raise ValueError("Found NaN/Inf in features")
    if all_features.std().item() < 1e-10:
        raise ValueError("Feature std ~0 (possible model load failure)")

    result = {
        "features": all_features,
        "labels": all_labels,
        "names": all_names if all_names else None,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    
    print(f"Cached to: {cache_file}\n")
    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

print("\n" + "="*60)
print("STEP 1: FEATURE EXTRACTION")
print("="*60)

train_data = extract_all_features(train_loader, "train", is_test=False)
val_data = extract_all_features(val_loader, "val", is_test=False)
test_data = extract_all_features(test_loader, "test", is_test=True)

# Prepare data
X_train = train_data["features"].numpy()
y_train = train_data["labels"].numpy()
X_val = val_data["features"].numpy()
y_val = val_data["labels"].numpy()
X_test = test_data["features"].numpy()

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\n" + "="*60)
print("STEP 2: TRAIN CLASSIFIER")
print("="*60)
print(f"Classifier: {CLASSIFIER_TYPE}")
print(f"Features: {X_train.shape[1]}")
print(f"Train samples: {X_train.shape[0]}")
print(f"Val samples: {X_val.shape[0]}")

# Create classifier
if CLASSIFIER_TYPE == "logistic":
    clf = LogisticRegression(
        max_iter=1000, multi_class="multinomial",
        solver="lbfgs", C=1.0, random_state=42
    )
elif CLASSIFIER_TYPE == "random_forest":
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        random_state=42, n_jobs=-1
    )
elif CLASSIFIER_TYPE == "gradient_boost":
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, random_state=42
    )
else:
    raise ValueError(f"Unknown classifier: {CLASSIFIER_TYPE}")

# Train
print("\nTraining...")
clf.fit(X_train, y_train)

# Evaluate
train_preds = clf.predict(X_train)
val_preds = clf.predict(X_val)
train_acc = accuracy_score(y_train, train_preds)
val_acc = accuracy_score(y_val, val_preds)

print(f"\nResults:")
print(f"  Train accuracy: {train_acc:.4f}")
print(f"  Val accuracy: {val_acc:.4f}")

print("\nValidation Report:")
print(classification_report(y_val, val_preds, target_names=LABELS, zero_division=0))

print("\n" + "="*60)
print("STEP 3: OUTLIER DETECTION")
print("="*60)

# Calibrate threshold
val_probs = clf.predict_proba(X_val)
outlier_idx = LABELS.index("outlier")

max_probs = val_probs.max(axis=1)
is_outlier = (y_val == outlier_idx)

outlier_probs = max_probs[is_outlier] if np.any(is_outlier) else np.array([])
normal_probs = max_probs[~is_outlier] if np.any(~is_outlier) else np.array([])

if outlier_probs.size > 0:
    print(f"Outlier confidence: μ={outlier_probs.mean():.4f} σ={outlier_probs.std():.4f}")
if normal_probs.size > 0:
    print(f"Normal confidence: μ={normal_probs.mean():.4f} σ={normal_probs.std():.4f}")

# Set threshold
if USE_ADAPTIVE_THRESHOLD and outlier_probs.size > 0:
    tau = max(OUTLIER_CONF_THRESHOLD, np.percentile(outlier_probs, 75))
else:
    tau = OUTLIER_CONF_THRESHOLD

print(f"Threshold: τ={tau:.4f}")

print("\n" + "="*60)
print("STEP 4: GENERATE PREDICTIONS")
print("="*60)

test_probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)

# Apply outlier detection
max_probs_test = test_probs.max(axis=1)
outlier_mask = (max_probs_test < tau)
preds[outlier_mask] = outlier_idx

# Create submission
submission_data = []
for i, name in enumerate(test_data["names"]):
    submission_data.append([name, LABELS[int(preds[i])]])

with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(submission_data)

print(f"\nSaved: {SUBMISSION_FILE}")
print(f"Total predictions: {len(submission_data)}")
print(f"Outliers: {outlier_mask.sum()} ({100*outlier_mask.sum()/len(submission_data):.1f}%)")

# Show distribution
label_counts = Counter([row[1] for row in submission_data])
print("\nPrediction distribution:")
for label in LABELS:
    count = label_counts[label]
    pct = 100 * count / len(submission_data)
    print(f"  {label:8s}: {count:5d} ({pct:5.1f}%)")

# Submit
if API_KEY and API_KEY not in ["YOUR_API_KEY_HERE", "IWILL_TYPE"]:
    print("\n" + "="*60)
    print("SUBMITTING TO LEADERBOARD")
    print("="*60)
    
    try:
        response = requests.post(
            f"{SERVER_URL}/submit/{TASK_ID}",
            files={"file": open(SUBMISSION_FILE, "rb")},
            headers={"X-API-Key": API_KEY},
            timeout=120,
        )
        print("Response:", response.json())
    except Exception as e:
        print(f"Submission failed: {e}")
else:
    print("\nSet API_KEY to submit to leaderboard")

print("\n" + "="*60)
print("DONE")
print("="*60)