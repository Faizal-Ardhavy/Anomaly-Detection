# DBSCAN Testing Script - GPU Accelerated
# Testing data baru menggunakan hasil clustering dari training

import numpy as np
from pathlib import Path
import gc
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - EDIT PATHS DI SINI!
# ============================================================================

# Path ke hasil training DBSCAN
TRAINING_LABELS_PATH = Path("dbscan_labels.npy")  # Hasil dari training
TRAINING_CONFIG_PATH = Path("dbscan_config.npy")  # Config dari training
TRAINING_EMBEDDINGS_PATH = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector/after_preprocessed_bgl_embeddings.npy")  # Training embeddings

# Path ke testing data - EDIT DI SINI!
# Untuk BGL:
TESTING_EMBEDDINGS_PATHS = [
    Path("testing_error.npy"),
    Path("testing_warning.npy"),
    Path("testing_info.npy"),
]

# Untuk Thunderbird (PCA128):
# TESTING_EMBEDDINGS_PATHS = [
#     Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_pca128/testing_thunderbird_pca128_embeddings.npy"),
# ]

# Output paths
OUTPUT_KNN_LABELS = Path("dbscan_labels_test_knn.npy")
OUTPUT_INDEPENDENT_LABELS = Path("dbscan_labels_test_independent.npy")

# Testing mode
RUN_KNN_ASSIGNMENT = True  # Assign testing data ke training clusters
RUN_INDEPENDENT_CLUSTERING = True  # Run DBSCAN independen pada testing data

RANDOM_STATE = 42

# ============================================================================
# GPU DETECTION
# ============================================================================
USE_GPU = False
try:
    import cupy as cp
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    
    gpu_count = cp.cuda.runtime.getDeviceCount()
    if gpu_count > 0:
        USE_GPU = True
        print("="*70)
        print("🎮 GPU ACCELERATION ENABLED")
        print("="*70)
except ImportError:
    print("="*70)
    print("ℹ️ GPU NOT AVAILABLE - Using CPU Mode")
    print("="*70)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_embeddings_from_files(files):
    """Load embeddings from multiple files"""
    if len(files) == 0:
        raise FileNotFoundError('No embedding files provided')
    
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f'File not found: {f}')
    
    if len(files) == 1:
        print(f"Loading single file: {files[0].name}")
        return np.load(files[0], mmap_mode='r')
    
    print(f"Loading {len(files)} files...")
    arrays = []
    for f in files:
        arr = np.load(f, mmap_mode='r')
        arrays.append(arr)
        print(f"  - {f.name}: {arr.shape[0]:,} rows")
    
    return np.vstack(arrays)

# ============================================================================
# LOAD TRAINING RESULTS
# ============================================================================

print("\n" + "="*70)
print("LOADING TRAINING RESULTS")
print("="*70)

# Load training config
if not TRAINING_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Training config not found: {TRAINING_CONFIG_PATH}")

config = np.load(TRAINING_CONFIG_PATH)
CHOSEN_EPS = float(config[0])
CHOSEN_MIN_SAMPLES = int(config[1])
USE_COSINE_DISTANCE = bool(config[2])
TRAINED_WITH_GPU = bool(config[3])

print(f"\nTraining Configuration:")
print(f"  eps: {CHOSEN_EPS:.6f}")
print(f"  min_samples: {CHOSEN_MIN_SAMPLES}")
print(f"  Cosine distance: {'Yes' if USE_COSINE_DISTANCE else 'No'}")
print(f"  Trained with: {'GPU' if TRAINED_WITH_GPU else 'CPU'}")

# Load training labels
if not TRAINING_LABELS_PATH.exists():
    raise FileNotFoundError(f"Training labels not found: {TRAINING_LABELS_PATH}")

labels_train = np.load(TRAINING_LABELS_PATH)
print(f"\nTraining labels loaded: {len(labels_train):,} samples")

n_clusters_train = len(set(labels_train) - {-1})
n_noise_train = np.sum(labels_train == -1)
noise_pct_train = (n_noise_train / len(labels_train)) * 100

print(f"  Clusters: {n_clusters_train}")
print(f"  Noise: {n_noise_train:,} ({noise_pct_train:.1f}%)")

# ============================================================================
# LOAD TESTING DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING TESTING DATA")
print("="*70)

emb_test = load_embeddings_from_files(TESTING_EMBEDDINGS_PATHS)
print(f"\nTesting data shape: {emb_test.shape}")

# Normalize if training used cosine distance
if USE_COSINE_DISTANCE:
    print('🔄 Normalizing testing data for cosine distance...')
    emb_test = normalize(emb_test, norm='l2')
    print('✓ Normalized')

# ============================================================================
# METHOD 1: k-NN ASSIGNMENT
# ============================================================================

if RUN_KNN_ASSIGNMENT:
    print("\n" + "="*70)
    print("METHOD 1: k-NN ASSIGNMENT")
    print("="*70)
    print("Assigning testing data to training clusters via k-NN")
    
    # Load training embeddings
    print(f"\nLoading training embeddings from: {TRAINING_EMBEDDINGS_PATH.name}")
    if not TRAINING_EMBEDDINGS_PATH.exists():
        print(f"⚠️ WARNING: Training embeddings not found!")
        print(f"   Skipping k-NN assignment")
        RUN_KNN_ASSIGNMENT = False
    else:
        emb_train = np.load(TRAINING_EMBEDDINGS_PATH, mmap_mode='r')
        
        # Normalize training data if needed
        if USE_COSINE_DISTANCE:
            print('🔄 Normalizing training data...')
            emb_train = normalize(emb_train, norm='l2')
        
        # Filter out noise points from training
        non_noise_mask = labels_train != -1
        emb_train_clustered = emb_train[non_noise_mask]
        labels_train_clustered = labels_train[non_noise_mask]
        
        print(f'Using {len(emb_train_clustered):,} non-noise training points')
        
        # k-NN Assignment
        if USE_GPU:
            print('\n🎮 GPU k-NN Assignment...')
            
            emb_train_gpu = cp.asarray(emb_train_clustered, dtype=cp.float32)
            knn = cuNearestNeighbors(n_neighbors=5)
            knn.fit(emb_train_gpu)
            
            BATCH_SIZE = 500000
            n_test = len(emb_test)
            labels_test_knn = np.zeros(n_test, dtype=np.int32)
            
            for i in tqdm(range(0, n_test, BATCH_SIZE), desc='GPU k-NN', unit='batch'):
                batch_end = min(i + BATCH_SIZE, n_test)
                emb_batch_gpu = cp.asarray(emb_test[i:batch_end], dtype=cp.float32)
                
                distances, indices = knn.kneighbors(emb_batch_gpu)
                indices_cpu = cp.asnumpy(indices)
                
                for j in range(len(indices_cpu)):
                    neighbor_labels = labels_train_clustered[indices_cpu[j]]
                    labels_test_knn[i + j] = np.bincount(neighbor_labels).argmax()
                
                del emb_batch_gpu
                cp.get_default_memory_pool().free_all_blocks()
            
            del emb_train_gpu, knn
            cp.get_default_memory_pool().free_all_blocks()
            
        else:
            print('\n💻 CPU k-NN Assignment...')
            knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
            knn.fit(emb_train_clustered)
            
            labels_test_knn = np.zeros(len(emb_test), dtype=np.int32)
            
            print('Computing k-NN...')
            distances, indices = knn.kneighbors(emb_test)
            
            for i in tqdm(range(len(emb_test)), desc='Assigning labels'):
                neighbor_labels = labels_train_clustered[indices[i]]
                labels_test_knn[i] = np.bincount(neighbor_labels).argmax()
            
            del knn
        
        gc.collect()
        
        # Save results
        np.save(OUTPUT_KNN_LABELS, labels_test_knn)
        print(f'\n✅ Saved: {OUTPUT_KNN_LABELS} ({len(labels_test_knn):,} labels)')
        
        # Analysis
        n_clusters_assigned = len(set(labels_test_knn))
        cluster_dist = Counter(labels_test_knn)
        
        print(f'\n📊 k-NN Assignment Results:')
        print(f'   Test samples: {len(labels_test_knn):,}')
        print(f'   Assigned to {n_clusters_assigned} unique clusters')
        print(f'   Top 5 clusters: {cluster_dist.most_common(5)}')
        
        # Cluster distribution
        cluster_pcts = [(lbl, cnt, cnt/len(labels_test_knn)*100) 
                       for lbl, cnt in cluster_dist.most_common(10)]
        print(f'\n   Top 10 cluster distribution:')
        for lbl, cnt, pct in cluster_pcts:
            print(f'     Cluster {lbl:3d}: {cnt:8,} samples ({pct:5.2f}%)')

# ============================================================================
# METHOD 2: INDEPENDENT CLUSTERING
# ============================================================================

if RUN_INDEPENDENT_CLUSTERING:
    print("\n" + "="*70)
    print("METHOD 2: INDEPENDENT DBSCAN CLUSTERING")
    print("="*70)
    print("Running DBSCAN on testing data with training parameters")
    
    print(f'\nUsing training config:')
    print(f'  eps: {CHOSEN_EPS:.6f}')
    print(f'  min_samples: {CHOSEN_MIN_SAMPLES}')
    
    # Run DBSCAN
    if USE_GPU:
        print('\n🎮 GPU DBSCAN on testing data...')
        
        emb_test_gpu = cp.asarray(emb_test, dtype=cp.float32)
        
        model_test = cuDBSCAN(eps=CHOSEN_EPS, min_samples=CHOSEN_MIN_SAMPLES)
        labels_test_independent_gpu = model_test.fit_predict(emb_test_gpu)
        labels_test_independent = cp.asnumpy(labels_test_independent_gpu)
        
        del emb_test_gpu, labels_test_independent_gpu, model_test
        cp.get_default_memory_pool().free_all_blocks()
        
    else:
        print('\n💻 CPU DBSCAN on testing data...')
        from sklearn.cluster import DBSCAN
        
        model_test = DBSCAN(eps=CHOSEN_EPS, min_samples=CHOSEN_MIN_SAMPLES,
                           algorithm='ball_tree', n_jobs=-1)
        labels_test_independent = model_test.fit_predict(emb_test)
        del model_test
    
    gc.collect()
    
    # Save results
    np.save(OUTPUT_INDEPENDENT_LABELS, labels_test_independent)
    print(f'\n✅ Saved: {OUTPUT_INDEPENDENT_LABELS}')
    
    # Analysis
    n_clusters_test = len(set(labels_test_independent) - {-1})
    n_noise_test = np.sum(labels_test_independent == -1)
    noise_pct_test = (n_noise_test / len(labels_test_independent)) * 100
    
    print(f'\n📊 Independent Clustering Results:')
    print(f'   Test samples: {len(labels_test_independent):,}')
    print(f'   Clusters found: {n_clusters_test}')
    print(f'   Noise: {n_noise_test:,} ({noise_pct_test:.1f}%)')
    
    # Comparison with training
    print(f'\n📊 Training vs Testing Comparison:')
    print(f'\n  {"Metric":<20s} {"Training":>15s} {"Testing":>15s} {"Difference":>15s}')
    print(f'  {"-"*20} {"-"*15} {"-"*15} {"-"*15}')
    print(f'  {"Clusters":<20s} {n_clusters_train:>15,} {n_clusters_test:>15,} {n_clusters_test-n_clusters_train:>+15,}')
    print(f'  {"Noise %":<20s} {noise_pct_train:>15.2f} {noise_pct_test:>15.2f} {noise_pct_test-noise_pct_train:>+15.2f}')
    
    # Stability assessment
    cluster_diff = abs(n_clusters_train - n_clusters_test)
    noise_diff = abs(noise_pct_train - noise_pct_test)
    
    if cluster_diff <= 5 and noise_diff <= 5.0:
        stability = "✅ HIGH (clusters and noise similar)"
    elif cluster_diff <= 10 and noise_diff <= 10.0:
        stability = "⚠️ MODERATE (some differences)"
    else:
        stability = "❌ LOW (significant differences)"
    
    print(f'\n  Cluster Stability: {stability}')

# ============================================================================
# COMPARISON OF BOTH METHODS (if both ran)
# ============================================================================

if RUN_KNN_ASSIGNMENT and RUN_INDEPENDENT_CLUSTERING:
    print("\n" + "="*70)
    print("COMPARISON: k-NN vs Independent Clustering")
    print("="*70)
    
    # Agreement between methods
    agreement = np.sum(labels_test_knn == labels_test_independent) / len(labels_test_knn) * 100
    print(f'\nLabel Agreement: {agreement:.2f}%')
    
    # Cluster overlap
    clusters_knn = set(labels_test_knn)
    clusters_independent = set(labels_test_independent) - {-1}
    
    print(f'\nUnique clusters:')
    print(f'  k-NN: {len(clusters_knn)}')
    print(f'  Independent: {len(clusters_independent)}')
    
    if agreement >= 80:
        print('\n✅ HIGH agreement between methods')
    elif agreement >= 60:
        print('\n⚠️ MODERATE agreement between methods')
    else:
        print('\n❌ LOW agreement between methods')

print("\n" + "="*70)
print("✅ TESTING COMPLETE!")
print("="*70)
