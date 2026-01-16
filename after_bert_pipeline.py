"""
POST-PROCESSING PIPELINE untuk BERT Embeddings
================================================

Pipeline ini melakukan normalisasi dan dimensi reduction pada BERT embeddings
untuk persiapan clustering yang lebih optimal.

EKSPERIMEN DESIGN:
------------------
1. Baseline: Raw BERT embeddings (768 dims, no normalization)
2. Variant 1: Normalized only (768 dims, L2 normalized)
3. Variant 2: Normalized + PCA-256 (256 dims, ~95-98% variance)
4. Variant 3: Normalized + PCA-128 (128 dims, ~90-95% variance)

KEPUTUSAN DESIGN & ALASANNYA:
-----------------------------

1. NORMALISASI: L2 Normalization (Unit Vectors)
   Alasan:
   - Mengubah embeddings menjadi unit vectors (magnitude=1)
   - Membuat distance measure fokus ke DIRECTION, bukan magnitude
   - Essential untuk cosine similarity (yang dipakai banyak clustering algorithms)
   - Mengurangi bias dari sentence length (log panjang vs pendek)
   - Membuat PCA lebih stabil (variance lebih merata)
   
   Formula: x_norm = x / ||x||_2
   
   Trade-off:
   ✅ Better untuk similarity-based clustering
   ✅ Lebih robust terhadap outliers
   ❌ Hilang informasi magnitude (tapi untuk log, magnitude kurang penting)

2. DIMENSI REDUCTION: PCA (Principal Component Analysis)
   Alasan memilih PCA:
   - Linear transformation, preserves global structure
   - Deterministik (hasil konsisten untuk same data)
   - Fast untuk inference (matrix multiplication)
   - Interpretable (PC1 = most variance direction)
   - Mudah di-save dan di-apply ke new data
   
   Alternatif yang TIDAK dipakai:
   - UMAP: Non-linear, bagus untuk viz tapi slower dan stochastic
   - t-SNE: Hanya untuk visualisasi, tidak untuk clustering
   - Autoencoder: Overkill, perlu training, GPU intensive

3. TARGET DIMENSI: 256 dan 128
   
   PCA-256 (RECOMMENDED untuk production):
   Alasan:
   - Preserves 95-98% variance (berdasarkan research BERT embeddings)
   - 3x lebih cepat clustering dibanding 768 dims
   - 3x lebih kecil storage (700GB → 233GB)
   - Sweet spot antara speed vs quality   
   - Masih cukup ekspresif untuk diverse log patterns

   
   PCA-128 (untuk extreme speed):
   Alasan:
   - Preserves 90-95% variance
   - 6x lebih cepat clustering
   - 6x lebih kecil storage (700GB → 117GB)
   - Cukup untuk well-separated clusters
   - Trade-off: bisa lose subtle differences
   
   Kenapa TIDAK 64 atau 512?
   - 64: Terlalu kecil, variance < 85%, hilang banyak informasi
   - 512: Gain marginal (< 2% variance) tapi 2x lebih lambat dari 256

EXPECTED IMPACT:
----------------
Metric          | Baseline | Norm-768 | PCA-256 | PCA-128
----------------|----------|----------|---------|----------
Storage         | 100%     | 100%     | 33%     | 17%
Clustering Time | 100%     | 100%     | 30%     | 17%
Memory Usage    | 100%     | 100%     | 33%     | 17%
Silhouette      | baseline | ≈same    | -2~5%   | -5~10%
Inertia         | baseline | ↓better  | ↓better | ↓better

Variance Preserved: 100% | 100% | 95-98% | 90-95%

USAGE:
------
1. Baseline (no processing):
   embeddings = np.load('file_embeddings.npy')  # Direct dari bert.py
   
2. Normalized only:
   embeddings = np.load('file_normalized_embeddings.npy')
   
3. PCA-256:
   embeddings = np.load('file_pca256_embeddings.npy')
   pca_model = joblib.load('pca_model_256.pkl')  # Untuk transform new data
   
4. PCA-128:
   embeddings = np.load('file_pca128_embeddings.npy')
   pca_model = joblib.load('pca_model_128.pkl')

RECOMMENDATION:
---------------
Untuk dataset BESAR (>10M logs):
1. Start dengan PCA-256 untuk balance speed vs quality
2. Jika perlu lebih cepat, coba PCA-128
3. Jika quality drops, fallback ke Normalized-768

Untuk dataset KECIL (<1M logs):
1. Langsung pakai Baseline atau Normalized-768
2. PCA tidak perlu karena clustering sudah cukup cepat
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import normalize
import joblib
from tqdm import tqdm
import time
import gc
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output directories
INPUT_DIR = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector")
OUTPUT_DIR_NORMALIZED = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_normalized")
OUTPUT_DIR_PCA256 = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_pca256")
OUTPUT_DIR_PCA128 = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_pca128")

# PCA target dimensions
# 256: Recommended - preserves ~95-98% variance, 3x speed improvement
# 128: Aggressive - preserves ~90-95% variance, 6x speed improvement
PCA_256_DIMS = 256
PCA_128_DIMS = 128

# Memory management
# Untuk dataset besar, gunakan IncrementalPCA (streaming)
USE_INCREMENTAL_PCA = True  # Set False jika RAM cukup (>64GB)
INCREMENTAL_BATCH_SIZE = 50000  # Process 50k samples at a time

# Create output directories
OUTPUT_DIR_NORMALIZED.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PCA256.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PCA128.mkdir(parents=True, exist_ok=True)

print("="*100)
print("🔧 POST-PROCESSING PIPELINE - BERT EMBEDDINGS")
print("="*100)
print(f"\n📂 Input directory: {INPUT_DIR}")
print(f"📂 Output directories:")
print(f"   - Normalized (768 dims): {OUTPUT_DIR_NORMALIZED}")
print(f"   - PCA-256 (256 dims): {OUTPUT_DIR_PCA256}")
print(f"   - PCA-128 (128 dims): {OUTPUT_DIR_PCA128}")

# ============================================================================
# STEP 1: SCAN INPUT FILES
# ============================================================================
print("\n" + "="*100)
print("STEP 1: SCANNING INPUT FILES")
print("="*100)

npy_files = sorted(list(INPUT_DIR.glob("*_embeddings.npy")))
print(f"\n✓ Found {len(npy_files)} embedding files to process")

if len(npy_files) == 0:
    print("\n⚠️  No embedding files found!")
    print("Expected files: *_embeddings.npy from bert.py output")
    exit(1)

# Calculate total samples (for PCA fitting)
print("\n📊 Analyzing dataset size...")
total_samples = 0
file_info = []

for npy_file in tqdm(npy_files, desc="Scanning files"):
    try:
        # Use memory mapping to avoid loading into RAM
        embeddings = np.load(npy_file, mmap_mode='r')
        num_samples = embeddings.shape[0]
        total_samples += num_samples
        
        file_size_mb = npy_file.stat().st_size / (1024 * 1024)
        file_info.append({
            'filename': npy_file.name,
            'samples': num_samples,
            'size_mb': file_size_mb,
            'path': npy_file
        })
    except Exception as e:
        print(f"\n⚠️  Error reading {npy_file.name}: {e}")
        continue

print(f"\n✓ Total samples: {total_samples:,}")
print(f"✓ Total input size: {sum(f['size_mb'] for f in file_info):.2f} MB")

# Estimate output sizes
normalized_size_mb = sum(f['size_mb'] for f in file_info)  # Same as input
pca256_size_mb = normalized_size_mb * (256 / 768)
pca128_size_mb = normalized_size_mb * (128 / 768)

print(f"\n📊 Expected output sizes:")
print(f"   - Normalized (768 dims): {normalized_size_mb:.2f} MB (same as input)")
print(f"   - PCA-256 (256 dims): {pca256_size_mb:.2f} MB ({pca256_size_mb/normalized_size_mb*100:.1f}% of original)")
print(f"   - PCA-128 (128 dims): {pca128_size_mb:.2f} MB ({pca128_size_mb/normalized_size_mb*100:.1f}% of original)")

# ============================================================================
# STEP 2: FIT PCA MODELS
# ============================================================================
print("\n" + "="*100)
print("STEP 2: FITTING PCA MODELS")
print("="*100)

print(f"\n⚙️  Strategy: {'Incremental PCA (streaming)' if USE_INCREMENTAL_PCA else 'Standard PCA (in-memory)'}")

if USE_INCREMENTAL_PCA:
    print(f"✓ Batch size: {INCREMENTAL_BATCH_SIZE:,} samples")
    print(f"✓ Total batches: {(total_samples + INCREMENTAL_BATCH_SIZE - 1) // INCREMENTAL_BATCH_SIZE:,}")

# Initialize PCA models
print("\n🔧 Initializing PCA models...")
if USE_INCREMENTAL_PCA:
    pca_256 = IncrementalPCA(n_components=PCA_256_DIMS, batch_size=INCREMENTAL_BATCH_SIZE)
    pca_128 = IncrementalPCA(n_components=PCA_128_DIMS, batch_size=INCREMENTAL_BATCH_SIZE)
else:
    pca_256 = PCA(n_components=PCA_256_DIMS)
    pca_128 = PCA(n_components=PCA_128_DIMS)

print("✓ PCA-256 initialized")
print("✓ PCA-128 initialized")

# Fit PCA models
print("\n🔄 Fitting PCA models on data...")
print("📝 Note: PCA fitting uses NORMALIZED embeddings (L2 norm)")

fit_start_time = time.time()
samples_processed = 0

with tqdm(total=total_samples, desc="Fitting PCA", unit="samples") as pbar:
    for file_idx, file_data in enumerate(file_info):
        npy_file = file_data['path']
        
        # Load embeddings (memory mapped)
        embeddings = np.load(npy_file, mmap_mode='r')
        
        if USE_INCREMENTAL_PCA:
            # Process in batches
            num_samples = embeddings.shape[0]
            for start_idx in range(0, num_samples, INCREMENTAL_BATCH_SIZE):
                end_idx = min(start_idx + INCREMENTAL_BATCH_SIZE, num_samples)
                batch = embeddings[start_idx:end_idx]
                
                # Normalize batch
                batch_normalized = normalize(batch, norm='l2', axis=1)
                
                # Partial fit
                pca_256.partial_fit(batch_normalized)
                pca_128.partial_fit(batch_normalized)
                
                samples_processed += (end_idx - start_idx)
                pbar.update(end_idx - start_idx)
                
                # Memory cleanup
                del batch, batch_normalized
                gc.collect()
        else:
            # Load all into memory (not recommended for large datasets)
            embeddings_array = np.array(embeddings)
            embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
            
            pca_256.partial_fit(embeddings_normalized)
            pca_128.partial_fit(embeddings_normalized)
            
            samples_processed += embeddings_array.shape[0]
            pbar.update(embeddings_array.shape[0])
            
            del embeddings_array, embeddings_normalized
            gc.collect()

fit_time = time.time() - fit_start_time

print(f"\n✓ PCA fitting complete!")
print(f"✓ Samples processed: {samples_processed:,}")
print(f"✓ Time: {fit_time:.2f} seconds ({fit_time/60:.2f} minutes)")

# Analyze PCA models
print("\n📊 PCA Model Statistics:")
print("\n--- PCA-256 ---")
variance_256 = pca_256.explained_variance_ratio_.sum()
print(f"✓ Explained variance: {variance_256*100:.2f}%")
print(f"✓ Components: {pca_256.n_components}")
print(f"✓ Top 5 components variance: {pca_256.explained_variance_ratio_[:5]}")

print("\n--- PCA-128 ---")
variance_128 = pca_128.explained_variance_ratio_.sum()
print(f"✓ Explained variance: {variance_128*100:.2f}%")
print(f"✓ Components: {pca_128.n_components}")
print(f"✓ Top 5 components variance: {pca_128.explained_variance_ratio_[:5]}")

# Save PCA models
print("\n💾 Saving PCA models...")
pca_256_path = OUTPUT_DIR_PCA256 / "pca_model_256.pkl"
pca_128_path = OUTPUT_DIR_PCA128 / "pca_model_128.pkl"

joblib.dump(pca_256, pca_256_path)
joblib.dump(pca_128, pca_128_path)

print(f"✓ PCA-256 model saved: {pca_256_path}")
print(f"✓ PCA-128 model saved: {pca_128_path}")

# Save metadata
metadata = {
    'total_samples': total_samples,
    'pca_256': {
        'n_components': PCA_256_DIMS,
        'explained_variance': float(variance_256),
        'explained_variance_per_component': pca_256.explained_variance_ratio_.tolist()
    },
    'pca_128': {
        'n_components': PCA_128_DIMS,
        'explained_variance': float(variance_128),
        'explained_variance_per_component': pca_128.explained_variance_ratio_.tolist()
    },
    'fit_time_seconds': fit_time,
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
}

metadata_path = OUTPUT_DIR_PCA256.parent / "pca_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved: {metadata_path}")

# ============================================================================
# STEP 3: TRANSFORM ALL FILES
# ============================================================================
print("\n" + "="*100)
print("STEP 3: TRANSFORMING ALL FILES")
print("="*100)

transform_start_time = time.time()
files_processed = 0
failed_files = []

for file_idx, file_data in enumerate(file_info, 1):
    npy_file = file_data['path']
    filename = file_data['filename']
    num_samples = file_data['samples']
    
    try:
        print(f"\n[{file_idx}/{len(file_info)}] Processing: {filename}")
        print(f"    Samples: {num_samples:,}")
        
        # Load embeddings
        embeddings = np.load(npy_file)
        print(f"    ✓ Loaded: shape={embeddings.shape}")
        
        # ===== VARIANT 1: NORMALIZED ONLY (768 dims) =====
        print(f"    🔄 Variant 1: L2 Normalization...")
        embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
        
        # Save normalized
        output_filename_normalized = filename.replace('_embeddings.npy', '_normalized_embeddings.npy')
        output_path_normalized = OUTPUT_DIR_NORMALIZED / output_filename_normalized
        
        np.save(output_path_normalized, embeddings_normalized)
        size_mb = output_path_normalized.stat().st_size / (1024 * 1024)
        print(f"    ✓ Saved normalized: {output_filename_normalized} ({size_mb:.2f} MB)")
        
        # ===== VARIANT 2: PCA-256 =====
        print(f"    🔄 Variant 2: PCA-256 transformation...")
        embeddings_pca256 = pca_256.transform(embeddings_normalized)
        
        # Save PCA-256
        output_filename_pca256 = filename.replace('_embeddings.npy', '_pca256_embeddings.npy')
        output_path_pca256 = OUTPUT_DIR_PCA256 / output_filename_pca256
        
        np.save(output_path_pca256, embeddings_pca256)
        size_mb = output_path_pca256.stat().st_size / (1024 * 1024)
        print(f"    ✓ Saved PCA-256: {output_filename_pca256} ({size_mb:.2f} MB)")
        
        # ===== VARIANT 3: PCA-128 =====
        print(f"    🔄 Variant 3: PCA-128 transformation...")
        embeddings_pca128 = pca_128.transform(embeddings_normalized)
        
        # Save PCA-128
        output_filename_pca128 = filename.replace('_embeddings.npy', '_pca128_embeddings.npy')
        output_path_pca128 = OUTPUT_DIR_PCA128 / output_filename_pca128
        
        np.save(output_path_pca128, embeddings_pca128)
        size_mb = output_path_pca128.stat().st_size / (1024 * 1024)
        print(f"    ✓ Saved PCA-128: {output_filename_pca128} ({size_mb:.2f} MB)")
        
        files_processed += 1
        
        # Cleanup
        del embeddings, embeddings_normalized, embeddings_pca256, embeddings_pca128
        gc.collect()
        
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        failed_files.append((filename, str(e)))
        continue

transform_time = time.time() - transform_start_time
total_time = time.time() - fit_start_time

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("📊 FINAL SUMMARY")
print("="*100)

print(f"\n✓ Files processed: {files_processed}/{len(file_info)}")
print(f"✓ Total samples: {total_samples:,}")

print(f"\n⏱️  Performance:")
print(f"   - PCA fitting time: {fit_time:.2f}s ({fit_time/60:.2f} min)")
print(f"   - Transform time: {transform_time:.2f}s ({transform_time/60:.2f} min)")
print(f"   - Total time: {total_time:.2f}s ({total_time/60:.2f} min)")

print(f"\n📊 Variance Preserved:")
print(f"   - PCA-256: {variance_256*100:.2f}%")
print(f"   - PCA-128: {variance_128*100:.2f}%")

print(f"\n💾 Output Sizes:")
actual_norm_size = sum((OUTPUT_DIR_NORMALIZED / f['filename'].replace('_embeddings.npy', '_normalized_embeddings.npy')).stat().st_size 
                       for f in file_info if (OUTPUT_DIR_NORMALIZED / f['filename'].replace('_embeddings.npy', '_normalized_embeddings.npy')).exists()) / (1024**2)
actual_pca256_size = sum((OUTPUT_DIR_PCA256 / f['filename'].replace('_embeddings.npy', '_pca256_embeddings.npy')).stat().st_size 
                         for f in file_info if (OUTPUT_DIR_PCA256 / f['filename'].replace('_embeddings.npy', '_pca256_embeddings.npy')).exists()) / (1024**2)
actual_pca128_size = sum((OUTPUT_DIR_PCA128 / f['filename'].replace('_embeddings.npy', '_pca128_embeddings.npy')).stat().st_size 
                         for f in file_info if (OUTPUT_DIR_PCA128 / f['filename'].replace('_embeddings.npy', '_pca128_embeddings.npy')).exists()) / (1024**2)

print(f"   - Normalized (768): {actual_norm_size:.2f} MB")
print(f"   - PCA-256: {actual_pca256_size:.2f} MB ({actual_pca256_size/actual_norm_size*100:.1f}% of normalized)")
print(f"   - PCA-128: {actual_pca128_size:.2f} MB ({actual_pca128_size/actual_norm_size*100:.1f}% of normalized)")

print(f"\n📂 Output Locations:")
print(f"   - Baseline (raw): {INPUT_DIR}")
print(f"   - Normalized: {OUTPUT_DIR_NORMALIZED}")
print(f"   - PCA-256: {OUTPUT_DIR_PCA256}")
print(f"   - PCA-128: {OUTPUT_DIR_PCA128}")

print(f"\n📝 PCA Models:")
print(f"   - {pca_256_path}")
print(f"   - {pca_128_path}")
print(f"   - {metadata_path}")

if failed_files:
    print(f"\n⚠️  Failed files: {len(failed_files)}")
    for filename, error in failed_files:
        print(f"   - {filename}: {error}")

print("\n" + "="*100)
print("🎯 NEXT STEPS - EKSPERIMEN CLUSTERING")
print("="*100)
print("""
Sekarang Anda punya 4 variant untuk clustering experiment:

1️⃣  BASELINE (Raw BERT - 768 dims):
   embeddings = np.load('INPUT_DIR/file_embeddings.npy')
   
2️⃣  NORMALIZED (768 dims):
   embeddings = np.load('OUTPUT_DIR_NORMALIZED/file_normalized_embeddings.npy')
   
3️⃣  PCA-256 (256 dims - RECOMMENDED):
   embeddings = np.load('OUTPUT_DIR_PCA256/file_pca256_embeddings.npy')
   
4️⃣  PCA-128 (128 dims - Fastest):
   embeddings = np.load('OUTPUT_DIR_PCA128/file_pca128_embeddings.npy')

RECOMMENDED EXPERIMENT SEQUENCE:
--------------------------------
A. Quick Test (1 file):
   1. Baseline vs PCA-256 → Compare silhouette score
   2. Jika quality drop < 5% → pakai PCA-256
   3. Jika quality drop > 5% → pakai Normalized-768

B. Full Clustering:
   1. Start dengan PCA-256 (best balance)
   2. Measure: silhouette, Davies-Bouldin, clustering time
   3. If quality OK → production ready!
   4. If need faster → try PCA-128
   5. If need better quality → fallback to Normalized-768

C. Production Inference:
   new_embeddings = bert_encode(new_logs)  # (M, 768)
   new_normalized = normalize(new_embeddings, norm='l2')
   new_reduced = pca_256.transform(new_normalized)  # (M, 256)
   cluster_labels = kmeans.predict(new_reduced)

EXPECTED RESULTS:
-----------------
Variant         | Speed    | Memory  | Quality  | Best For
----------------|----------|---------|----------|------------------
Baseline (768)  | 1x       | 1x      | ★★★★★   | Small datasets
Normalized (768)| 1x       | 1x      | ★★★★★   | Better distances
PCA-256         | 3-4x     | 0.33x   | ★★★★☆   | Large datasets ✅
PCA-128         | 5-6x     | 0.17x   | ★★★☆☆   | Extreme scale
""")

print("\n✅ POST-PROCESSING PIPELINE COMPLETE!")
print("="*100)
