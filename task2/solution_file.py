#!/usr/bin/env python3
"""
solution_file.py — Model Tracer (ICLR 2026-style provenance features)

Pipeline:
1) Load dataset from Dataset.zip into ./dataset (train/val/test)
2) Extract provenance features:
   - VAR depths [16,20,24,30]: [QuantLoss, ReconLoss] per depth
   - RAR sizes [b,l,xl,xxl]: [mean_token_nll_best, mean_token_prob_best]
     using teacher-forcing token likelihood, with label marginalization
     approximated by trying top-k ImageNet labels from a cheap ResNet18.
3) Train simple sklearn classifier on concatenated features
4) Outlier detection by max class probability thresholding
5) Write submission.csv and optionally submit

Notes:
- This file does NOT paste VAR/RAR source code. Instead it robustly imports from:
  ./VAR and ./RAR/1d-tokenizer (absolute paths inserted into sys.path).
- By default, model load failures are fatal to avoid silently training on dummy features.
"""

import csv
import zipfile
import requests
import pickle
import sys
import os
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

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier



# ============================================================
# CONFIGURATION
# ============================================================

FORCE_REEXTRACT = False
ALLOW_DUMMY_FEATURES = False   # If True: fall back to zeros instead of raising on model-load errors

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_INFERENCE = 8
NUM_WORKERS = 4

# Which classifier to use
CLASSIFIER_TYPE = "logistic"   # Options: "logistic", "random_forest", "svm", "gradient_boost"

# Outlier detection
OUTLIER_CONF_THRESHOLD = 0.5
USE_ADAPTIVE_THRESHOLD = True

# Model selection
USE_VAR_MODELS = True
USE_RAR_MODELS = True

# RAR label marginalization approximation
RAR_TOPK_LABELS = 2  # try top-k labels from a cheap ImageNet classifier (resnet18)

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

# Resolve repo root from this file location (important for robust imports)
REPO_ROOT = Path(__file__).resolve().parent

VAR_ROOT = (REPO_ROOT / "VAR").resolve()
RAR_ROOT = (REPO_ROOT / "RAR" / "1d-tokenizer").resolve()

CACHE_DIR = (REPO_ROOT / ".cache_scores").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if FORCE_REEXTRACT:
    print("=" * 60)
    print("CLEARING OLD CACHE")
    print("=" * 60)
    for cache_file in CACHE_DIR.glob("*_provenance_features.pkl"):
        cache_file.unlink()
        print(f"  Deleted {cache_file}")


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# TORCH.DISTRIBUTED PATCH (VAR sometimes expects it)
# ============================================================

import torch.distributed as dist

if not dist.is_initialized():
    try:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://localhost:12355",
            world_size=1,
            rank=0,
        )
        print("✓ Initialized torch.distributed")
    except Exception:
        # Patch minimal functions so code that checks world size doesn't crash
        dist.get_world_size = lambda group=None: 1
        dist.get_rank = lambda group=None: 0
        print("✓ Patched torch.distributed (fallback)")


# ============================================================
# ROBUST IMPORT HELPERS
# ============================================================

def _add_sys_path(p: Path, name: str) -> None:
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"[PATH] Missing {name} path: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def _import_from(root: Path, module: str, name: str):
    _add_sys_path(root, name)
    try:
        return importlib.import_module(module)
    except Exception as e:
        tb = traceback.format_exc()
        raise ImportError(f"[IMPORT] Failed importing '{module}' from '{root}'.\n{tb}") from e

def _maybe_or_raise(err: Exception, context: str):
    if ALLOW_DUMMY_FEATURES:
        print(f"[WARN] {context}: {err}\n[WARN] Using dummy zeros because ALLOW_DUMMY_FEATURES=True")
        return None
    raise RuntimeError(f"{context}: {err}") from err


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
    transforms.ToTensor(),  # [0,1]
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


# ============================================================
# VAR MODEL LOADING + FEATURES
# ============================================================

def load_var_vae(depth: int):
    """Load VAR VAE from ./VAR (fail-loud unless ALLOW_DUMMY_FEATURES=True)."""
    vae_ckpt = VAR_ROOT / "checkpoints" / "vae_ch160v4096z32.pth"
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"[VAR] Missing checkpoint: {vae_ckpt}")

    models_mod = _import_from(VAR_ROOT, "models", "VAR_ROOT")  # VAR/models/__init__.py exposes build_vae_var
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
def var_batch_features(vae, batch_01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Paper-style VAR provenance features:
      QuantLoss = || f - Q^{-1}(Q(f)) ||^2
      EncLoss (calibrated) = recon_err(x -> recon1) / (recon_err(recon1 -> recon2) + eps)
    """
    x = batch_01.to(device)
    x_norm = x * 2.0 - 1.0  # [-1,1]

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        # Encode + quantize
        f = vae.encoder(x_norm)
        q = vae.quantize(f)
        f_q = q[0] if isinstance(q, tuple) else q

        # QuantLoss
        quant_loss = F.mse_loss(f, f_q, reduction="none").mean(dim=[1, 2, 3])

        # First reconstruction
        recon1 = vae.decoder(f_q)
        enc_loss1 = F.mse_loss(recon1, x_norm, reduction="none").mean(dim=[1, 2, 3])

        # Second reconstruction (calibration)
        f2 = vae.encoder(recon1)
        q2 = vae.quantize(f2)
        f2_q = q2[0] if isinstance(q2, tuple) else q2
        recon2 = vae.decoder(f2_q)
        enc_loss2 = F.mse_loss(recon2, recon1, reduction="none").mean(dim=[1, 2, 3])

        enc_loss_cal = enc_loss1 / (enc_loss2 + eps)

        return torch.stack([quant_loss, enc_loss_cal], dim=1).detach().cpu()


def extract_var_features(all_images: torch.Tensor) -> list[torch.Tensor]:
    feats = []
    for depth in [16, 20, 24, 30]:
        print(f"  VAR depth={depth}...")
        try:
            vae = load_var_vae(depth).to(device)
        except Exception as e:
            vae = _maybe_or_raise(e, f"[VAR] Failed to load depth={depth}")
            if vae is None:
                feats.append(torch.zeros(all_images.shape[0], 2))
                continue

        out = []
        for i in tqdm(range(0, all_images.shape[0], BATCH_SIZE_INFERENCE), desc="    Processing", leave=False):
            batch = all_images[i:i+BATCH_SIZE_INFERENCE]
            out.append(var_batch_features(vae, batch))
        f = torch.cat(out, dim=0)

        print(f"    ✓ QuantLoss mean={f[:,0].mean():.6f} std={f[:,0].std():.6f}")
        print(f"    ✓ ReconLoss mean={f[:,1].mean():.6f} std={f[:,1].std():.6f}")

        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        feats.append(f)
    return feats


# ============================================================
# RAR MODEL LOADING + FEATURES
# ============================================================

# RAR checkpoints (expected names based on provided repo)
RAR_CKPT_DIR = RAR_ROOT / "checkpoints"
MASKGIT_CKPT = RAR_CKPT_DIR / "maskgit-vqgan-imagenet-f16-256.bin"

RAR_CKPTS = {
    "rarb":   RAR_CKPT_DIR / "rar_b.bin",
    "rarl":   RAR_CKPT_DIR / "rar_l.bin",
    "rarxl":  RAR_CKPT_DIR / "rar_xl.bin",
    "rarxxl": RAR_CKPT_DIR / "rar_xxl.bin",
}

# Architecture mapping from the provided generate_rar_images.py style
RAR_ARCH = {
    "rar_b":  dict(hidden_size=768,  num_hidden_layers=24, num_attention_heads=16, intermediate_size=3072),
    "rar_l":  dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
    "rar_xl": dict(hidden_size=1280, num_hidden_layers=32, num_attention_heads=16, intermediate_size=5120),
    "rar_xxl":dict(hidden_size=1408, num_hidden_layers=40, num_attention_heads=16, intermediate_size=6144),
}

_rar_loaded = False
_demo_util = None
_train_utils = None

def _load_rar_modules_once():
    global _rar_loaded, _demo_util, _train_utils
    if _rar_loaded:
        return
    _demo_util = _import_from(RAR_ROOT, "demo_util", "RAR_ROOT")
    _train_utils = _import_from(RAR_ROOT, "utils.train_utils", "RAR_ROOT")
    _rar_loaded = True

def load_rar_pair(size_key: str):
    """
    size_key in {"rarb","rarl","rarxl","rarxxl"}.
    Returns (tokenizer, generator).
    """
    ckpt = RAR_CKPTS[size_key]
    if not ckpt.exists():
        raise FileNotFoundError(f"[RAR] Missing generator checkpoint: {ckpt}")
    if not MASKGIT_CKPT.exists():
        raise FileNotFoundError(f"[RAR] Missing tokenizer checkpoint: {MASKGIT_CKPT}")

    _load_rar_modules_once()

    ckpt_name = ckpt.name  # rar_b.bin, ...
    arch_key = ckpt_name.replace(".bin", "")  # rar_b, rar_l, ...
    if arch_key not in RAR_ARCH:
        raise ValueError(f"[RAR] Unknown arch key from checkpoint: {arch_key}")

    # Load config from provided helper
    config = _demo_util.get_config("configs/training/generator/rar.yaml")
    config.experiment.generator_checkpoint = str(ckpt)
    config.model.vq_model.pretrained_tokenizer_weight = str(MASKGIT_CKPT)

    arch = RAR_ARCH[arch_key]
    config.model.generator.hidden_size = arch["hidden_size"]
    config.model.generator.num_hidden_layers = arch["num_hidden_layers"]
    config.model.generator.num_attention_heads = arch["num_attention_heads"]
    config.model.generator.intermediate_size = arch["intermediate_size"]

    # Create tokenizer + generator
    tokenizer = _train_utils.create_pretrained_tokenizer(config).to(device)
    generator = _demo_util.get_rar_generator(config).to(device)

    tokenizer.eval()
    generator.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    for p in generator.parameters():
        p.requires_grad_(False)

    return tokenizer, generator

# Cheap label proposals for RAR marginalization
_resnet18 = None

@torch.no_grad()
def topk_imagenet_labels(batch_01: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Returns [B,k] int64 candidate labels.
    Uses torchvision ResNet18 weights if available.
    If loading weights fails, falls back to a fixed set of labels.
    """
    global _resnet18
    try:
        import torchvision.models as tvm
        if _resnet18 is None:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            _resnet18 = tvm.resnet18(weights=weights).to(device).eval()
            _resnet18._weights = weights  # stash transforms
        weights = _resnet18._weights

        x = batch_01.to(device)
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = weights.transforms()(x)  # includes normalization
        logits = _resnet18(x)
        return torch.topk(logits, k=k, dim=1).indices
    except Exception as e:
        # fallback: deterministic label list (still gives some marginalization)
        print(f"[WARN] ResNet18 label proposal failed: {e}")
        B = batch_01.shape[0]
        fixed = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)[:k]
        return fixed.unsqueeze(0).repeat(B, 1)

@torch.no_grad()
def rar_batch_features(tokenizer, generator, batch_01: torch.Tensor, k_labels: int = 3) -> torch.Tensor:
    """
    Paper-ish RAR features with label marginalization (approx via top-k labels):
      logp_marg ≈ logsumexp_y log p(tokens | y) - log(k)
      mean_nll = -logp_marg / T
      mean_prob = exp(-mean_nll)   (a simple monotone proxy; optional)
    """
    x = batch_01.to(device)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        tokens = tokenizer.encode(x)

    cand = topk_imagenet_labels(batch_01, k=k_labels)  # [B,k]
    B = batch_01.shape[0]

    logp_list = []

    for j in range(cand.shape[1]):
        labels = cand[:, j]
        cond = generator.preprocess_condition(labels)
        logits, labels_tf = generator(tokens, cond, return_labels=True)

        logits = logits[:, :-1]  # shift
        labels_tf = labels_tf.reshape(labels_tf.shape[0], -1)  # [B,T]
        logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])  # [B,T,V]

        # token logprobs
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, labels_tf.unsqueeze(-1)).squeeze(-1)  # [B,T]

        # total log p(tokens | y)
        logp = token_logp.sum(dim=1)  # [B]
        logp_list.append(logp)

    logp_stack = torch.stack(logp_list, dim=1)  # [B,k]

    # log-mean-exp ≈ marginal with uniform prior over top-k
    logp_marg = torch.logsumexp(logp_stack, dim=1) - np.log(cand.shape[1])  # [B]

    T = labels_tf.shape[1]
    mean_nll = (-logp_marg / max(T, 1)).detach().cpu()
    mean_prob = torch.exp(-mean_nll).detach().cpu()

    return torch.stack([mean_nll, mean_prob], dim=1)


def extract_rar_features(all_images: torch.Tensor) -> list[torch.Tensor]:
    feats = []
    for size_key in ["rarb", "rarl", "rarxl", "rarxxl"]:
        print(f"  RAR {size_key}...")
        try:
            tokenizer, generator = load_rar_pair(size_key)
        except Exception as e:
            pair = _maybe_or_raise(e, f"[RAR] Failed to load {size_key}")
            if pair is None:
                feats.append(torch.zeros(all_images.shape[0], 2))
                continue

        out = []
        for i in tqdm(range(0, all_images.shape[0], BATCH_SIZE_INFERENCE), desc="    Processing", leave=False):
            batch = all_images[i:i+BATCH_SIZE_INFERENCE]
            out.append(rar_batch_features(tokenizer, generator, batch, k_labels=RAR_TOPK_LABELS))
        f = torch.cat(out, dim=0)

        print(f"    ✓ NLL mean={f[:,0].mean():.6f} std={f[:,0].std():.6f}")
        print(f"    ✓ Prob mean={f[:,1].mean():.6f} std={f[:,1].std():.6f}")

        del tokenizer, generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        feats.append(f)
    return feats


# ============================================================
# FEATURE EXTRACTION (WITH CACHING)
# ============================================================

def extract_features_for_all_models(loader, split_name: str, is_test: bool = False):
    cache_file = CACHE_DIR / f"{split_name}_provenance_features.pkl"
    if cache_file.exists() and not FORCE_REEXTRACT:
        print(f"Loading cached features for {split_name}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES FOR {split_name.upper()}")
    print(f"{'='*60}")

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

    all_images = torch.cat(all_images, dim=0)  # [N,3,256,256]
    all_labels_t = torch.tensor(all_labels) if len(all_labels) > 0 else None

    print(f"Loaded {all_images.shape[0]} images")

    feature_blocks = []

    # VAR
    if USE_VAR_MODELS:
        print("\nExtracting VAR features...")
        feature_blocks.extend(extract_var_features(all_images))
    else:
        feature_blocks.extend([torch.zeros(all_images.shape[0], 2) for _ in range(4)])

    # RAR
    if USE_RAR_MODELS:
        print("\nExtracting RAR features...")
        feature_blocks.extend(extract_rar_features(all_images))
    else:
        feature_blocks.extend([torch.zeros(all_images.shape[0], 2) for _ in range(4)])

    all_features = torch.cat(feature_blocks, dim=1)  # [N, 16]

    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Mean: {all_features.mean():.6f}, Std: {all_features.std():.6f}")

    # Basic sanity checks (to catch “all zeros” situations early)
    if torch.isnan(all_features).any() or torch.isinf(all_features).any():
        raise ValueError("[SANITY] Found NaN/Inf in features.")
    if all_features.std().item() < 1e-10:
        raise ValueError("[SANITY] Feature std is ~0. Something is wrong (likely model load failure).")

    result = {
        "features": all_features,
        "labels": all_labels_t,
        "names": all_names if len(all_names) > 0 else None,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    print(f"✓ Cached to {cache_file}\n")
    return result


# ============================================================
# RUN: EXTRACT FEATURES
# ============================================================

train_data = extract_features_for_all_models(train_loader, "train", is_test=False)
val_data = extract_features_for_all_models(val_loader, "val", is_test=False)
test_data = extract_features_for_all_models(test_loader, "test", is_test=True)


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_data["features"].numpy()
y_train = train_data["labels"].numpy()
X_val = val_data["features"].numpy()
y_val = val_data["labels"].numpy()
X_test = test_data["features"].numpy()

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
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )

elif CLASSIFIER_TYPE == "random_forest":
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

elif CLASSIFIER_TYPE == "svm":
    print("\nTraining SVM...")
    clf = SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        random_state=42,
    )

elif CLASSIFIER_TYPE == "gradient_boost":
    print("\nTraining Gradient Boosting...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

elif CLASSIFIER_TYPE == "tree":
    print("\nTraining Decision Tree...")
    clf = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )

elif CLASSIFIER_TYPE == "bagging":
    print("\nTraining Bagging (trees)...")
    base = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf = BaggingClassifier(
        estimator=base,
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )


else:
    raise ValueError(f"Unknown classifier type: {CLASSIFIER_TYPE}")

clf.fit(X_train, y_train)

train_preds = clf.predict(X_train)
val_preds = clf.predict(X_val)

train_acc = accuracy_score(y_train, train_preds)
val_acc = accuracy_score(y_val, val_preds)

print(f"\n✓ Training complete!")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy: {val_acc:.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, val_preds, target_names=LABELS, zero_division=0))


# ============================================================
# OUTLIER DETECTION
# ============================================================

print(f"\n{'='*60}")
print("OUTLIER DETECTION")
print(f"{'='*60}")

val_probs = clf.predict_proba(X_val)
outlier_idx = LABELS.index("outlier")

max_probs_val = val_probs.max(axis=1)
is_outlier = (y_val == outlier_idx)

outlier_probs = max_probs_val[is_outlier] if np.any(is_outlier) else np.array([])
normal_probs = max_probs_val[~is_outlier] if np.any(~is_outlier) else np.array([])

if outlier_probs.size > 0:
    print(f"Outlier confidence: mean={outlier_probs.mean():.4f}, std={outlier_probs.std():.4f}")
else:
    print("Outlier confidence: (no outliers in val?)")
if normal_probs.size > 0:
    print(f"Normal confidence: mean={normal_probs.mean():.4f}, std={normal_probs.std():.4f}")
else:
    print("Normal confidence: (no normals in val?)")

if USE_ADAPTIVE_THRESHOLD and outlier_probs.size > 0:
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

max_probs_test = test_probs.max(axis=1)
outlier_mask = (max_probs_test < tau)
preds[outlier_mask] = outlier_idx


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission_data = []
for i, name in enumerate(test_data["names"]):
    submission_data.append([name, LABELS[int(preds[i])]])

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

if API_KEY != "IWILL_TYPE" and API_KEY != "I_WILL_TYPE":
    print("\nSubmitting...")
    response = requests.post(
        f"{SERVER_URL}/submit/{TASK_ID}",
        files={"file": open(SUBMISSION_FILE, "rb")},
        headers={"X-API-Key": API_KEY},
        timeout=120,
    )
    try:
        print("Response:", response.json())
    except Exception:
        print("Response (text):", response.text)
else:
    print("\nSet API_KEY to submit")
