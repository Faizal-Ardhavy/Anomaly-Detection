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
import pickle

# =============================================================================
# CONFIG
# =============================================================================

INPUT_DIR = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_testing")
# Normalized files NOT saved (only processed in-memory for PCA)
OUT_PCA256 = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_pca256_testing")
OUT_PCA128 = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_pca128_testing")
CHECKPOINT_FILE = Path("/media/bioinfo04/Expansion/2427051003_checkpoint.json")

EMBEDDING_DIMS = 768
DTYPE = np.float32

PCA_256_DIMS = 256
PCA_128_DIMS = 128

BATCH_SIZE = 50_000
LARGE_FILE_THRESHOLD = 10_000_000  # rows

OUT_PCA256.mkdir(parents=True, exist_ok=True)
OUT_PCA128.mkdir(parents=True, exist_ok=True)

print("⚙️  Configuration:")
print(f"   • Normalization: IN-MEMORY ONLY (not saved)")
print(f"   • PCA-256 output: {OUT_PCA256}")
print(f"   • PCA-128 output: {OUT_PCA128}")

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

def load_checkpoint():
    """Load checkpoint jika ada"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"pca_fitted": False, "completed_files": []}

def save_checkpoint(checkpoint):
    """Save checkpoint progress"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"💾 Checkpoint saved: {len(checkpoint['completed_files'])} files done")

# =============================================================================
# STEP 0 — LOAD CHECKPOINT
# =============================================================================

checkpoint = load_checkpoint()
print(f"\n📋 Checkpoint loaded: PCA fitted={checkpoint['pca_fitted']}, Completed={len(checkpoint['completed_files'])}")

# =============================================================================
# STEP 1 — SCAN FILES
# =============================================================================

print("\n🔍 Scanning input files...")
print(f"   Input directory: {INPUT_DIR}")
print(f"   Looking for pattern: *_embeddings.npy")

files = sorted(INPUT_DIR.glob("*_embeddings.npy"))
if not files:
    # Try alternative patterns
    print("   ⚠️ No *_embeddings.npy found, trying alternative patterns...")
    alt_patterns = ["*.npy", "*embeddings*.npy", "*_emb.npy"]
    for pattern in alt_patterns:
        files = sorted(INPUT_DIR.glob(pattern))
        if files:
            print(f"   ✓ Found {len(files)} files with pattern: {pattern}")
            break
    
    if not files:
        raise RuntimeError(f"No .npy files found in {INPUT_DIR}")

file_info = []
total_samples = 0

print(f"\n📂 Processing {len(files)} files:")
for f in tqdm(files, desc="Scanning files"):
    try:
        arr = np.load(f, mmap_mode="r")
        n = arr.shape[0]
        is_raw = False
        load_method = "np.load"
    except Exception:
        n = infer_num_rows(f)
        is_raw = True
        load_method = "memmap"

    file_size_gb = f.stat().st_size / (1024**3)
    print(f"   • {f.name}: {n:,} rows, {file_size_gb:.2f} GB ({load_method})")
    
    file_info.append({
        "path": f,
        "name": f.name,
        "rows": n,
        "is_raw": is_raw
    })
    total_samples += n

print(f"\n✓ Files: {len(file_info)}")
print(f"✓ Total samples BEFORE processing: {total_samples:,}")
print(f"✓ Expected output: {total_samples:,} rows (SAME number, different dimensions)")

# =============================================================================
# STEP 2 — FIT PCA (INCREMENTAL)
# =============================================================================

pca256_model_path = OUT_PCA256 / "pca_model_256.pkl"
pca128_model_path = OUT_PCA128 / "pca_model_128.pkl"

if checkpoint["pca_fitted"] and pca256_model_path.exists() and pca128_model_path.exists():
    print("\n♻️ Loading existing PCA models...")
    pca256 = joblib.load(pca256_model_path)
    pca128 = joblib.load(pca128_model_path)
    print("✓ PCA models loaded from checkpoint")
else:
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

    joblib.dump(pca256, pca256_model_path)
    joblib.dump(pca128, pca128_model_path)
    
    checkpoint["pca_fitted"] = True
    save_checkpoint(checkpoint)

# =============================================================================
# STEP 3 — TRANSFORM FILES
# =============================================================================

print("\n🔄 Transforming files...")
print(f"   Files to process: {len(file_info)}")
print(f"   Already completed: {len(checkpoint['completed_files'])}")

files_actually_processed = 0

for info in file_info:
    # Skip jika sudah selesai
    if info["name"] in checkpoint["completed_files"]:
        print(f"\n⏭️ SKIP {info['name']} (already completed)")
        continue
    
    files_actually_processed += 1
    
    print(f"\n➡ {info['name']} ({info['rows']:,} rows)")
    emb, is_raw = load_embeddings_auto(info["path"], info["rows"])
    is_large = info["rows"] > LARGE_FILE_THRESHOLD

    # Output paths (NO normalized file saved)
    pca256_path = OUT_PCA256 / info["name"].replace("_embeddings.npy", "_pca256_embeddings.npy")
    pca128_path = OUT_PCA128 / info["name"].replace("_embeddings.npy", "_pca128_embeddings.npy")
    
    # Cek apakah output sudah ada dan valid
    if pca256_path.exists() and pca128_path.exists():
        try:
            # Verifikasi ukuran file output
            if is_large:
                test_256 = np.memmap(pca256_path, dtype=DTYPE, mode='r', shape=(info["rows"], PCA_256_DIMS))
                test_128 = np.memmap(pca128_path, dtype=DTYPE, mode='r', shape=(info["rows"], PCA_128_DIMS))
                del test_256, test_128
                print("   ✓ Valid PCA output files exist, marking as complete")
                checkpoint["completed_files"].append(info["name"])
                save_checkpoint(checkpoint)
                continue
            else:
                test_256 = np.load(pca256_path, mmap_mode='r')
                test_128 = np.load(pca128_path, mmap_mode='r')
                if test_256.shape[0] == info["rows"] and test_128.shape[0] == info["rows"]:
                    print(f"   ✓ Valid PCA output files exist ({test_256.shape[0]:,} rows), marking as complete")
                    checkpoint["completed_files"].append(info["name"])
                    save_checkpoint(checkpoint)
                    continue
        except Exception as e:
            print(f"   ⚠️ Invalid output files, reprocessing: {e}")

    # Allocate output arrays (NO normalized storage)
    if is_large:
        out_256 = save_memmap(pca256_path, (info["rows"], PCA_256_DIMS))
        out_128 = save_memmap(pca128_path, (info["rows"], PCA_128_DIMS))
    else:
        pca256_all = np.zeros((info["rows"], PCA_256_DIMS), dtype=DTYPE)
        pca128_all = np.zeros((info["rows"], PCA_128_DIMS), dtype=DTYPE)
    
    processed_rows = 0  # Track untuk verify no data loss

    for start in tqdm(range(0, info["rows"], BATCH_SIZE), desc="  batches"):
        end = min(start + BATCH_SIZE, info["rows"])
        batch = emb[start:end]
        
        # Normalize in-memory (not saved to disk)
        batch_normalized = normalize(batch, axis=1)
        
        # Verify normalization didn't drop rows
        assert batch_normalized.shape[0] == (end - start), f"Normalization dropped rows! Expected {end-start}, got {batch_normalized.shape[0]}"

        # Apply PCA transformations
        pca256_batch = pca256.transform(batch_normalized)
        pca128_batch = pca128.transform(batch_normalized)
        
        # Verify PCA didn't drop rows
        assert pca256_batch.shape[0] == (end - start), f"PCA-256 dropped rows! Expected {end-start}, got {pca256_batch.shape[0]}"
        assert pca128_batch.shape[0] == (end - start), f"PCA-128 dropped rows! Expected {end-start}, got {pca128_batch.shape[0]}"

        if is_large:
            out_256[start:end] = pca256_batch
            out_128[start:end] = pca128_batch
        else:
            pca256_all[start:end] = pca256_batch
            pca128_all[start:end] = pca128_batch
        
        processed_rows += (end - start)

        del batch, batch_normalized, pca256_batch, pca128_batch
        gc.collect()

    # Verify ALL rows processed
    assert processed_rows == info["rows"], f"❌ DATA LOSS! Expected {info['rows']:,} rows, processed {processed_rows:,}"

    if not is_large:
        np.save(pca256_path, pca256_all)
        np.save(pca128_path, pca128_all)

    print(f"   ✓ done - Verified {processed_rows:,} rows processed (no data loss)")
    
    # Mark file as completed
    checkpoint["completed_files"].append(info["name"])
    save_checkpoint(checkpoint)

print(f"\n📊 TRANSFORMATION SUMMARY:")
print(f"   • Total input rows: {total_samples:,}")
print(f"   • Files in checkpoint: {len(checkpoint['completed_files'])}")
print(f"   • Files processed this run: {files_actually_processed}")
print(f"   • Expected output rows: {total_samples:,} (SAME as input)")

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

# Cleanup checkpoint file after successful completion
if CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()
    print("🗑️ Checkpoint file removed (pipeline completed)")

# =============================================================================
# FINAL VERIFICATION
# =============================================================================
print("\n" + "="*80)
print("🔍 FINAL VERIFICATION - Checking output file dimensions")
print("="*80)

output_total_rows_256 = 0
output_total_rows_128 = 0

for info in file_info:
    pca256_path = OUT_PCA256 / info["name"].replace("_embeddings.npy", "_pca256_embeddings.npy")
    pca128_path = OUT_PCA128 / info["name"].replace("_embeddings.npy", "_pca128_embeddings.npy")
    
    if pca256_path.exists() and pca128_path.exists():
        try:
            if info["rows"] > LARGE_FILE_THRESHOLD:
                test_256 = np.memmap(pca256_path, dtype=DTYPE, mode='r', shape=(info["rows"], PCA_256_DIMS))
                test_128 = np.memmap(pca128_path, dtype=DTYPE, mode='r', shape=(info["rows"], PCA_128_DIMS))
                shape_256 = (info["rows"], PCA_256_DIMS)
                shape_128 = (info["rows"], PCA_128_DIMS)
                del test_256, test_128
            else:
                test_256 = np.load(pca256_path, mmap_mode='r')
                test_128 = np.load(pca128_path, mmap_mode='r')
                shape_256 = test_256.shape
                shape_128 = test_128.shape
            
            output_total_rows_256 += shape_256[0]
            output_total_rows_128 += shape_128[0]
            print(f"✓ {info['name']}: PCA-256={shape_256[0]:,} rows, PCA-128={shape_128[0]:,} rows")
        except Exception as e:
            print(f"⚠️ {info['name']}: ERROR - {e}")
    else:
        print(f"❌ {info['name']}: OUTPUT NOT FOUND")

print(f"\n{'='*80}")
print(f"📊 FINAL RESULT:")
print(f"   • INPUT total:      {total_samples:,} rows × 768 dims")
print(f"   • PCA-256 output:   {output_total_rows_256:,} rows × 256 dims")
print(f"   • PCA-128 output:   {output_total_rows_128:,} rows × 128 dims")

if output_total_rows_256 == total_samples and output_total_rows_128 == total_samples:
    print(f"   ✅ SUCCESS: No data loss! All {total_samples:,} rows preserved")
else:
    print(f"   ❌ WARNING: Data loss detected!")
    if output_total_rows_256 != total_samples:
        print(f"   Missing in PCA-256: {total_samples - output_total_rows_256:,} rows ({((total_samples - output_total_rows_256) / total_samples * 100):.2f}%)")
    if output_total_rows_128 != total_samples:
        print(f"   Missing in PCA-128: {total_samples - output_total_rows_128:,} rows ({((total_samples - output_total_rows_128) / total_samples * 100):.2f}%)")
print(f"{'='*80}")

print("\n✅ PIPELINE COMPLETE (SAFE FOR 200M+ ROWS)")
