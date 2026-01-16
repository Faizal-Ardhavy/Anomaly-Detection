#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POST-PROCESSING PIPELINE (FINAL)
================================
✔ Supports BOTH:
  - Small .npy embeddings (np.load)
  - Ultra-large RAW memmap embeddings (np.memmap)

✔ Streaming normalization + Incremental PCA
✔ Safe for 200M+ rows / 600GB
✔ No allow_pickle
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
import joblib
from tqdm import tqdm
import time
import gc
import json

# =============================================================================
# CONFIG
# =============================================================================

INPUT_DIR = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector")
OUT_NORM = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_normalized")
OUT_PCA256 = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_pca256")
OUT_PCA128 = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_pca128")

EMBEDDING_DIMS = 768
DTYPE = np.float32

PCA_256_DIMS = 256
PCA_128_DIMS = 128

BATCH_SIZE = 50_000
LARGE_FILE_THRESHOLD = 10_000_000  # rows

OUT_NORM.mkdir(parents=True, exist_ok=True)
OUT_PCA256.mkdir(parents=True, exist_ok=True)
OUT_PCA128.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def infer_num_rows(path: Path) -> int:
    size = path.stat().st_size
    return size // (EMBEDDING_DIMS * np.dtype(DTYPE).itemsize)

def load_embeddings_auto(path: Path, num_rows: int):
    """
    Auto-detect loader:
    - Try .npy
    - Fallback to RAW memmap
    """
    try:
        arr = np.load(path, mmap_mode="r")
        return arr, False
    except Exception:
        return np.memmap(
            path,
            dtype=DTYPE,
            mode="r",
            shape=(num_rows, EMBEDDING_DIMS)
        ), True

def save_memmap(path, shape):
    return np.memmap(path, dtype=DTYPE, mode="w+", shape=shape)

# =============================================================================
# STEP 1 — SCAN FILES
# =============================================================================

print("\n🔍 Scanning input files...")
files = sorted(INPUT_DIR.glob("*_embeddings.npy"))
if not files:
    raise RuntimeError("No *_embeddings.npy found")

file_info = []
total_samples = 0

for f in tqdm(files):
    try:
        arr = np.load(f, mmap_mode="r")
        n = arr.shape[0]
        is_raw = False
    except Exception:
        n = infer_num_rows(f)
        is_raw = True

    file_info.append({
        "path": f,
        "name": f.name,
        "rows": n,
        "is_raw": is_raw
    })
    total_samples += n

print(f"✓ Files: {len(file_info)}")
print(f"✓ Total samples: {total_samples:,}")

# =============================================================================
# STEP 2 — FIT PCA (INCREMENTAL)
# =============================================================================

print("\n🧠 Fitting Incremental PCA...")
pca256 = IncrementalPCA(n_components=PCA_256_DIMS, batch_size=BATCH_SIZE)
pca128 = IncrementalPCA(n_components=PCA_128_DIMS, batch_size=BATCH_SIZE)

with tqdm(total=total_samples, desc="PCA fit", unit="rows") as pbar:
    for info in file_info:
        emb, _ = load_embeddings_auto(info["path"], info["rows"])

        for start in range(0, info["rows"], BATCH_SIZE):
            end = min(start + BATCH_SIZE, info["rows"])
            batch = emb[start:end]
            batch = normalize(batch, axis=1)

            pca256.partial_fit(batch)
            pca128.partial_fit(batch)

            pbar.update(end - start)
            del batch
            gc.collect()

print("✓ PCA fitting done")

joblib.dump(pca256, OUT_PCA256 / "pca_model_256.pkl")
joblib.dump(pca128, OUT_PCA128 / "pca_model_128.pkl")

# =============================================================================
# STEP 3 — TRANSFORM FILES
# =============================================================================

print("\n🔄 Transforming files...")

for info in file_info:
    print(f"\n➡ {info['name']} ({info['rows']:,} rows)")
    emb, is_raw = load_embeddings_auto(info["path"], info["rows"])
    is_large = info["rows"] > LARGE_FILE_THRESHOLD

    # Output paths
    norm_path = OUT_NORM / info["name"].replace("_embeddings.npy", "_normalized_embeddings.npy")
    pca256_path = OUT_PCA256 / info["name"].replace("_embeddings.npy", "_pca256_embeddings.npy")
    pca128_path = OUT_PCA128 / info["name"].replace("_embeddings.npy", "_pca128_embeddings.npy")

    if is_large:
        out_norm = save_memmap(norm_path, (info["rows"], EMBEDDING_DIMS))
        out_256 = save_memmap(pca256_path, (info["rows"], PCA_256_DIMS))
        out_128 = save_memmap(pca128_path, (info["rows"], PCA_128_DIMS))
    else:
        norm_all = np.zeros((info["rows"], EMBEDDING_DIMS), dtype=DTYPE)
        pca256_all = np.zeros((info["rows"], PCA_256_DIMS), dtype=DTYPE)
        pca128_all = np.zeros((info["rows"], PCA_128_DIMS), dtype=DTYPE)

    for start in tqdm(range(0, info["rows"], BATCH_SIZE), desc="  batches"):
        end = min(start + BATCH_SIZE, info["rows"])
        batch = emb[start:end]
        batch = normalize(batch, axis=1)

        if is_large:
            out_norm[start:end] = batch
            out_256[start:end] = pca256.transform(batch)
            out_128[start:end] = pca128.transform(batch)
        else:
            norm_all[start:end] = batch
            pca256_all[start:end] = pca256.transform(batch)
            pca128_all[start:end] = pca128.transform(batch)

        del batch
        gc.collect()

    if not is_large:
        np.save(norm_path, norm_all)
        np.save(pca256_path, pca256_all)
        np.save(pca128_path, pca128_all)

    print("   ✓ done")

# =============================================================================
# METADATA
# =============================================================================

meta = {
    "total_samples": total_samples,
    "pca256_variance": float(pca256.explained_variance_ratio_.sum()),
    "pca128_variance": float(pca128.explained_variance_ratio_.sum()),
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(OUT_PCA256.parent / "pca_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ PIPELINE COMPLETE (SAFE FOR 200M+ ROWS)")
