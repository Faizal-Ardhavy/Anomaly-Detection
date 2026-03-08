"""
Comprehensive Cluster Testing Pipeline for Log Anomaly Detection
3-WAY CLASSIFICATION (Normal / Non-Normal / Anomaly)

Strategy for MANY clusters (dozens to hundreds):
1. Load 3-way ground truth labels from template files:
   - Normal template → NORMAL (0)
   - Non-Normal template → NON-NORMAL (1)
   - No template match → ANOMALY (2)

2. Analyze cluster characteristics (purity, size, distribution)

3. Hybrid prediction strategy:
   - Noise points (DBSCAN) → ANOMALY
   - Very small clusters (<50) → ANOMALY
   - Small clusters (50-200) → NON-NORMAL
   - Pure clusters (>95%) → Trust dominant label
   - Medium purity (70-95%) → NON-NORMAL
   - Low purity (<70%) → k-NN vote (3-way)

4. Calculate 3x3 metrics: Per-class precision/recall/F1

5. Visualize: Cluster purity, 3x3 confusion matrix, statistics

Supports:
- K-Means and DBSCAN
- BGL and Thunderbird datasets  
- Base/PCA256/PCA128 embeddings
- Template-based ground truth
- Full testing dataset (no sampling)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import csv
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU libraries (optional - will fallback to CPU if not available)
try:
    import faiss
    FAISS_AVAILABLE = True
    # Check if GPU is available
    try:
        faiss.StandardGpuResources()
        FAISS_GPU_AVAILABLE = True
    except:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# ============================================================================
# CONFIGURATION - EDIT PATHS HERE!
# ============================================================================

# Dataset & Algorithm Selection
DATASET = "BGL"  # "BGL" or "Thunderbird"
ALGORITHM = "dbscan"  # "kmeans" or "dbscan"
EMBEDDING_TYPE = "base"  # "base", "pca256", or "pca128"

# Template paths for 3-way ground truth classification
if DATASET == "BGL":
    NORMAL_TEMPLATE_PATH = Path("log_processing/bgl/bgl_normal_template.txt")
    NONNORMAL_TEMPLATE_PATH = Path("log_processing/bgl/bgl_nonNormal_template.txt")
    METADATA_TSV_PATH = Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_bgl_meta.tsv")
else:  # Thunderbird
    NORMAL_TEMPLATE_PATH = Path("log_processing/thunderbird/thunderbird_normal_template.txt")
    NONNORMAL_TEMPLATE_PATH = Path("log_processing/thunderbird/thunderbird_nonNormal_template.txt")
    METADATA_TSV_PATH = Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_thunderbird_meta.tsv")

# Path to metadata TSV (for EventId extraction)
# Format: Contains EventId column for template matching

# Path to training results
if ALGORITHM == "kmeans":
    TRAINED_MODEL_PATH = Path("model_kmeans_log.pkl")
    TRAINING_LABELS_PATH = Path("cluster_labels.npy")
    TRAINING_EMBEDDINGS_PATH = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector/after_preprocessed_bgl_embeddings.npy")
else:  # dbscan
    TRAINING_LABELS_PATH = Path("dbscan/bgl_base_model/dbscan_labels.npy")
    TRAINING_CONFIG_PATH = Path("dbscan/bgl_base_model/dbscan_config.npy")
    TRAINING_EMBEDDINGS_PATH = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector/after_preprocessed_bgl_embeddings.npy")

# Path to testing data - MULTIPLE SETS WITH SEPARATE METADATA
# Each testing set should have its own embeddings file and metadata TSV
TESTING_SETS = [
    {
        'name': 'normal',
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_testing/after_preprocessed_bgl_normal_embeddings.npy"),
        'metadata': Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_bgl_normal_meta.tsv")
    },
    {
        'name': 'nonnormal',
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_testing/after_preprocessed_bgl_non_normal_embeddings.npy"),
        'metadata': Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_bgl_non_normal_meta.tsv")
    }
]

# LEGACY: Single testing file support (if you still use old format)
# Uncomment below and comment TESTING_SETS if you want old behavior
# TESTING_EMBEDDINGS_PATHS = [
#     Path("testing_error.npy"),
#     Path("testing_warning.npy"),
#     Path("testing_info.npy"),
# ]
# TESTING_METADATA_TSV = Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/testing_bgl_meta.tsv")

# Hybrid prediction parameters (3-way classification)
PURITY_THRESHOLD_HIGH = 0.95       # Pure cluster (trust dominant label)
PURITY_THRESHOLD_MEDIUM = 0.70     # Medium purity → NON-NORMAL

VERY_SMALL_CLUSTER_THRESHOLD = 50   # < 50 samples → ANOMALY
SMALL_CLUSTER_THRESHOLD = 200       # 50-200 samples → NON-NORMAL

KNN_NEIGHBORS = 10                  # For k-NN vote in low purity clusters
KNN_HIGH_CONFIDENCE = 0.80          # 8/10 vote = high confidence
KNN_MEDIUM_CONFIDENCE = 0.60        # 6/10 vote = medium confidence

USE_COSINE_DISTANCE = True          # Normalize embeddings (recommended for BERT)

# Large dataset optimization parameters
SUBSAMPLE_KNN_TRAINING = True       # Subsample training data for k-NN (for huge datasets)
KNN_SUBSAMPLE_SIZE = 1_000_000      # Max training samples for k-NN (1M samples)
NORMALIZE_INPLACE = True            # Use copy=False to save memory during normalization

# GPU acceleration parameters
USE_GPU = True                      # Enable GPU acceleration (auto-detect, fallback to CPU)
GPU_KNN_BATCH_SIZE = 10_000         # Process k-NN queries in batches (prevent GPU OOM)
GPU_MEMORY_FRACTION = 0.9           # Fraction of GPU memory to use (0.0-1.0)

# Output paths
OUTPUT_DIR = Path("testing_results") / f"{DATASET.lower()}_{ALGORITHM}_{EMBEDDING_TYPE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREDICTIONS = OUTPUT_DIR / "predictions.npy"
OUTPUT_CLUSTER_ANALYSIS = OUTPUT_DIR / "cluster_analysis.csv"
OUTPUT_METRICS = OUTPUT_DIR / "metrics.txt"
OUTPUT_CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"
OUTPUT_PURITY_DISTRIBUTION = OUTPUT_DIR / "purity_distribution.png"
OUTPUT_DETAILED_RESULTS = OUTPUT_DIR / "detailed_results.csv"
OUTPUT_PER_SET_METRICS = OUTPUT_DIR / "per_set_metrics.csv"

# ============================================================================
# GPU HELPER FUNCTIONS
# ============================================================================

def detect_gpu_capabilities():
    """
    Detect available GPU capabilities and print info
    
    Returns: dict with GPU availability info
    """
    gpu_info = {
        'faiss_available': FAISS_AVAILABLE,
        'faiss_gpu_available': FAISS_GPU_AVAILABLE,
        'cupy_available': CUPY_AVAILABLE,
        'use_gpu': False
    }
    
    if USE_GPU:
        if FAISS_GPU_AVAILABLE:
            try:
                res = faiss.StandardGpuResources()
                gpu_info['use_gpu'] = True
                gpu_info['gpu_name'] = 'NVIDIA GPU'
                print(f"\n🚀 GPU ACCELERATION ENABLED")
                print(f"   ✓ FAISS-GPU available")
                if CUPY_AVAILABLE:
                    print(f"   ✓ CuPy available (GPU normalization)")
                    gpu_info['gpu_name'] = cp.cuda.Device(0).name.decode()
                    total_mem = cp.cuda.Device(0).mem_info[1] / (1024**3)
                    print(f"   ✓ GPU: {gpu_info['gpu_name']}")
                    print(f"   ✓ GPU Memory: {total_mem:.1f} GB")
                else:
                    print(f"   ⚠️ CuPy not available (CPU normalization)")
            except Exception as e:
                print(f"\n⚠️ GPU detected but initialization failed: {e}")
                print(f"   Falling back to CPU mode")
                gpu_info['use_gpu'] = False
        elif FAISS_AVAILABLE:
            print(f"\n💻 FAISS available but no GPU detected")
            print(f"   Using FAISS-CPU (still faster than sklearn)")
        else:
            print(f"\n💻 CPU MODE (GPU libraries not installed)")
            print(f"   Install: conda install -c pytorch faiss-gpu cupy")
    else:
        print(f"\n💻 CPU MODE (USE_GPU=False)")
    
    return gpu_info


def normalize_embeddings_gpu(embeddings, use_gpu=True, inplace=False):
    """
    Normalize embeddings using GPU (CuPy) if available, otherwise CPU
    
    Args:
        embeddings: numpy array of embeddings
        use_gpu: whether to use GPU (if available)
        inplace: whether to modify array in-place (memory efficient)
    
    Returns:
        normalized embeddings (numpy array)
    """
    if use_gpu and CUPY_AVAILABLE:
        # GPU normalization using CuPy (10-30x faster)
        try:
            # Transfer to GPU
            embeddings_gpu = cp.asarray(embeddings)
            
            # Normalize (L2 norm)
            norms = cp.linalg.norm(embeddings_gpu, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_gpu = embeddings_gpu / norms
            
            # Transfer back to CPU
            if inplace:
                # Copy back to original array
                cp.asnumpy(embeddings_gpu, out=embeddings)
                return embeddings
            else:
                return cp.asnumpy(embeddings_gpu)
        except Exception as e:
            print(f"      ⚠️ GPU normalization failed: {e}")
            print(f"      Falling back to CPU normalization")
    
    # CPU normalization (sklearn)
    return normalize(embeddings, norm='l2', copy=(not inplace))


def build_faiss_knn_index(embeddings, use_gpu=True, use_cosine=True):
    """
    Build FAISS k-NN index (GPU or CPU)
    
    Args:
        embeddings: normalized numpy array (n_samples, n_features)
        use_gpu: whether to use GPU
        use_cosine: whether to use cosine distance (Inner Product for normalized vectors)
    
    Returns:
        FAISS index object
    """
    n_samples, dim = embeddings.shape
    
    # For normalized vectors, Inner Product = Cosine Similarity
    if use_cosine:
        # IndexFlatIP = Inner Product (cosine for normalized vectors)
        index = faiss.IndexFlatIP(dim)
    else:
        # IndexFlatL2 = Euclidean distance
        index = faiss.IndexFlatL2(dim)
    
    # Move to GPU if available
    if use_gpu and FAISS_GPU_AVAILABLE:
        try:
            res = faiss.StandardGpuResources()
            # Configure memory
            res.setTempMemory(int(GPU_MEMORY_FRACTION * 1024 * 1024 * 1024))  # Convert to bytes
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"      ✓ Index moved to GPU")
        except Exception as e:
            print(f"      ⚠️ Failed to move index to GPU: {e}")
            print(f"      Using CPU index")
    
    # Add vectors to index
    # FAISS requires float32
    embeddings_f32 = embeddings.astype(np.float32)
    index.add(embeddings_f32)
    
    return index


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_template_events(template_path: Path) -> set:
    """
    Load Label set from template TSV file
    
    Returns: set of Labels (e.g., {'-', 'APPREAD', 'KERNDTLB', ...})
    """
    print(f"   Loading template: {template_path.name}")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    df = pd.read_csv(template_path, sep='\t')
    
    if 'Label' not in df.columns:
        raise ValueError(f"Label column not found in {template_path}")
    
    label_set = set(df['Label'].unique())
    print(f"   ✓ Found {len(label_set)} unique Labels")
    
    return label_set


def load_metadata_labels_3way(tsv_path: Path, 
                                normal_template_path: Path,
                                nonnormal_template_path: Path,
                                use_chunking: bool = True,
                                chunksize: int = 3_000_000) -> np.ndarray:
    """
    Load ground truth labels using template-based 3-way classification
    
    Args:
        tsv_path: Path to metadata TSV file
        normal_template_path: Path to normal template
        nonnormal_template_path: Path to non-normal template
        use_chunking: If True, use streaming/chunking for large files (default: True)
        chunksize: Number of rows per chunk (default: 3M rows)
    
    Returns: 
        numpy array with:
        - 0 = NORMAL (Label in normal template)
        - 1 = NON-NORMAL (Label in nonNormal template)  
        - 2 = ANOMALY (Label not in either template)
    """
    print(f"\n📖 Loading 3-way ground truth labels from: {tsv_path}")
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {tsv_path}")
    
    # Load templates
    print("\n   Loading template files...")
    normal_events = load_template_events(normal_template_path)
    nonnormal_events = load_template_events(nonnormal_template_path)
    
    # Check overlap (should be 0)
    overlap = normal_events & nonnormal_events
    if overlap:
        print(f"   ⚠️ WARNING: {len(overlap)} Labels in both templates!")
        print(f"      Examples: {list(overlap)[:5]}")
    
    # Determine file size to decide strategy
    file_size_mb = tsv_path.stat().st_size / (1024 * 1024)
    print(f"\n   Metadata file size: {file_size_mb:.1f} MB")
    
    # Use chunking for large files (>2 GB)
    if use_chunking and file_size_mb > 2000:
        print(f"   Using CHUNKED STREAMING (chunksize={chunksize:,} rows) for large file...")
        return _load_metadata_chunked(tsv_path, normal_events, nonnormal_events, chunksize)
    
    # Standard loading for smaller files
    print(f"\n   Loading metadata TSV (standard mode)...")
    df = pd.read_csv(tsv_path, sep='\t')
    
    if 'label' not in df.columns:
        raise ValueError(f"label column not found in {tsv_path}. Available columns: {df.columns.tolist()}")
    
    # Assign 3-way labels
    labels = []
    stats = {'normal': 0, 'nonnormal': 0, 'anomaly': 0, 'unknown': 0}
    
    for label_val in df['label']:
        if pd.isna(label_val) or label_val == '':
            labels.append(2)  # Missing/empty label = ANOMALY
            stats['unknown'] += 1
        elif label_val in normal_events:
            labels.append(0)  # NORMAL
            stats['normal'] += 1
        elif label_val in nonnormal_events:
            labels.append(1)  # NON-NORMAL
            stats['nonnormal'] += 1
        else:
            labels.append(2)  # ANOMALY (novel pattern)
            stats['anomaly'] += 1
    
    labels_array = np.array(labels, dtype=np.int32)
    
    # Print statistics
    total = len(labels_array)
    print(f"\n   ✓ Loaded {total:,} labels")
    print(f"\n   Class Distribution:")
    print(f"      NORMAL (0):     {stats['normal']:,} ({stats['normal']/total*100:.2f}%)")
    print(f"      NON-NORMAL (1): {stats['nonnormal']:,} ({stats['nonnormal']/total*100:.2f}%)")
    print(f"      ANOMALY (2):    {stats['anomaly']:,} ({stats['anomaly']/total*100:.2f}%)")
    if stats['unknown'] > 0:
        print(f"      Unknown/Empty Label: {stats['unknown']:,}")
    
    return labels_array


def _load_metadata_chunked(tsv_path: Path, normal_events: set, nonnormal_events: set, chunksize: int) -> np.ndarray:
    """
    Load large metadata TSV using streaming/chunking to avoid memory overflow
    
    Memory-efficient for files >20 GB
    """
    labels = []
    stats = {'normal': 0, 'nonnormal': 0, 'anomaly': 0, 'unknown': 0}
    total_rows = 0
    
    # Stream read in chunks
    chunk_iterator = pd.read_csv(tsv_path, sep='\t', chunksize=chunksize)
    
    for chunk_num, chunk_df in enumerate(chunk_iterator, 1):
        if 'label' not in chunk_df.columns:
            raise ValueError(f"label column not found in chunk {chunk_num}")
        
        # Process chunk
        for label_val in chunk_df['label']:
            if pd.isna(label_val) or label_val == '':
                labels.append(2)
                stats['unknown'] += 1
            elif label_val in normal_events:
                labels.append(0)
                stats['normal'] += 1
            elif label_val in nonnormal_events:
                labels.append(1)
                stats['nonnormal'] += 1
            else:
                labels.append(2)
                stats['anomaly'] += 1
        
        total_rows += len(chunk_df)
        
        # Progress indicator every 10 chunks
        if chunk_num % 10 == 0:
            print(f"      Processed {total_rows:,} rows ({chunk_num} chunks)...")
    
    labels_array = np.array(labels, dtype=np.int32)
    
    # Print statistics
    print(f"\n   ✓ Loaded {len(labels_array):,} labels via chunked streaming")
    print(f"\n   Class Distribution:")
    print(f"      NORMAL (0):     {stats['normal']:,} ({stats['normal']/len(labels_array)*100:.2f}%)")
    print(f"      NON-NORMAL (1): {stats['nonnormal']:,} ({stats['nonnormal']/len(labels_array)*100:.2f}%)")
    print(f"      ANOMALY (2):    {stats['anomaly']:,} ({stats['anomaly']/len(labels_array)*100:.2f}%)")
    if stats['unknown'] > 0:
        print(f"      Unknown/Empty Label: {stats['unknown']:,}")
    
    return labels_array


def load_multiple_testing_sets(testing_sets, normal_template_path, nonnormal_template_path):
    """
    Load multiple testing sets with separate metadata files
    
    Args:
        testing_sets: List of dicts with keys: 'name', 'embeddings', 'metadata'
        normal_template_path: Path to normal template file
        nonnormal_template_path: Path to non-normal template file
    
    Returns:
        combined_embeddings: numpy array of all embeddings concatenated
        combined_labels: numpy array of all ground truth labels
        test_set_info: list of dicts with per-set statistics
    """
    print(f"\n📦 Loading {len(testing_sets)} testing sets...")
    
    all_embeddings = []
    all_labels = []
    test_set_info = []
    
    for test_set in testing_sets:
        name = test_set['name']
        embeddings_path = test_set['embeddings']
        metadata_path = test_set['metadata']
        
        print(f"\n  Loading test set: {name}")
        print(f"    Embeddings: {embeddings_path.name}")
        print(f"    Metadata:   {metadata_path.name}")
        
        # Load embeddings
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        embeddings = np.load(embeddings_path, mmap_mode='r')
        print(f"    ✓ Loaded embeddings: {embeddings.shape}")
        
        # Load ground truth labels from metadata
        labels = load_metadata_labels_3way(
            metadata_path,
            normal_template_path,
            nonnormal_template_path
        )
        
        # Verify length match
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Mismatch for set '{name}': "
                f"embeddings={len(embeddings)}, labels={len(labels)}"
            )
        
        # Store info
        test_set_info.append({
            'name': name,
            'start_idx': len(all_labels),
            'end_idx': len(all_labels) + len(labels),
            'n_samples': len(labels)
        })
        
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    
    # Combine all sets
    combined_embeddings = np.vstack(all_embeddings)
    combined_labels = np.concatenate(all_labels)
    
    print(f"\n✓ Combined testing data:")
    print(f"  Total samples: {len(combined_labels):,}")
    print(f"  Embeddings shape: {combined_embeddings.shape}")
    
    return combined_embeddings, combined_labels, test_set_info


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


def determine_cluster_type_3way(cluster_id, size, purity):
    """
    Classify cluster type for 3-way classification strategy
    
    Returns: "noise", "very_small", "small", "pure", "medium_purity", or "low_purity"
    """
    if cluster_id == -1:
        return "noise"
    elif size < VERY_SMALL_CLUSTER_THRESHOLD:
        return "very_small"
    elif size < SMALL_CLUSTER_THRESHOLD:
        return "small"
    elif purity > PURITY_THRESHOLD_HIGH:
        return "pure"
    elif purity > PURITY_THRESHOLD_MEDIUM:
        return "medium_purity"
    else:
        return "low_purity"


def analyze_cluster_characteristics(cluster_labels, ground_truth_labels):
    """
    Analyze each cluster: purity, size, dominant label (3-way classification)
    
    Returns:
    - DataFrame with cluster statistics
    - Dict with cluster_id → cluster_info
    """
    print("\n🔍 Analyzing cluster characteristics...")
    
    unique_clusters = sorted(set(cluster_labels))
    cluster_info = []
    
    for cluster_id in tqdm(unique_clusters, desc="Analyzing clusters"):
        mask = cluster_labels == cluster_id
        n_samples = np.sum(mask)
        
        # Get ground truth labels for this cluster (3 classes)
        labels_in_cluster = ground_truth_labels[mask]
        n_normal = np.sum(labels_in_cluster == 0)
        n_nonnormal = np.sum(labels_in_cluster == 1)
        n_anomaly = np.sum(labels_in_cluster == 2)
        
        # Calculate purity (max class ratio)
        total = n_normal + n_nonnormal + n_anomaly
        purity = max(n_normal, n_nonnormal, n_anomaly) / total if total > 0 else 0
        
        # Dominant label = most frequent class
        class_counts = [(n_normal, 0), (n_nonnormal, 1), (n_anomaly, 2)]
        dominant_label = max(class_counts)[1]
        
        # Classify cluster type using 3-way strategy
        cluster_type = determine_cluster_type_3way(cluster_id, n_samples, purity)
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'n_samples': n_samples,
            'n_normal': n_normal,
            'n_nonnormal': n_nonnormal,
            'n_anomaly': n_anomaly,
            'pct_normal': (n_normal / n_samples) * 100,
            'pct_nonnormal': (n_nonnormal / n_samples) * 100,
            'pct_anomaly': (n_anomaly / n_samples) * 100,
            'purity': purity,
            'dominant_label': dominant_label,
            'cluster_type': cluster_type
        })
    
    df = pd.DataFrame(cluster_info)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("CLUSTER CHARACTERISTICS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nTotal clusters: {len(df)}")
    
    # Count by type
    type_counts = df['cluster_type'].value_counts()
    print(f"\nCluster Types:")
    for ctype, count in type_counts.items():
        samples = df[df['cluster_type'] == ctype]['n_samples'].sum()
        print(f"  {ctype:10s}: {count:4d} clusters, {samples:,} samples")
    
    # Purity statistics
    print(f"\nPurity Statistics:")
    print(f"  Mean:   {df['purity'].mean():.4f}")
    print(f"  Median: {df['purity'].median():.4f}")
    print(f"  Std:    {df['purity'].std():.4f}")
    print(f"  Min:    {df['purity'].min():.4f}")
    print(f"  Max:    {df['purity'].max():.4f}")
    
    # Size statistics
    print(f"\nCluster Size Statistics:")
    print(f"  Mean:   {df['n_samples'].mean():.0f}")
    print(f"  Median: {df['n_samples'].median():.0f}")
    print(f"  Min:    {df['n_samples'].min()}")
    print(f"  Max:    {df['n_samples'].max():,}")
    
    # Create lookup dict
    cluster_dict = df.set_index('cluster_id').to_dict('index')
    
    return df, cluster_dict


def hybrid_predict(test_cluster_labels, cluster_dict, 
                   training_embeddings, training_labels, 
                   test_embeddings, use_knn=True, gpu_info=None):
    """
    Hybrid prediction strategy (3-way classification):
    1. Noise points → ANOMALY (label=2)
    2. Very small clusters (<50) → ANOMALY (label=2)
    3. Small clusters (50-200) → NON-NORMAL (label=1)
    4. Pure clusters (>95%) → Use cluster's dominant label (0/1/2)
    5. Medium purity (70-95%) → NON-NORMAL (label=1)
    6. Low purity (<70%) → Use k-NN vote for 3-way classification
    
    Returns:
    - predictions (numpy array: 0/1/2)
    - confidence scores (numpy array)
    - prediction_method (list of strings)
    """
    print("\n🎯 Performing hybrid prediction (3-way classification)...")
    
    n_test = len(test_cluster_labels)
    predictions = np.zeros(n_test, dtype=np.int32)  # Will store 0, 1, or 2
    confidence = np.zeros(n_test, dtype=np.float32)
    methods = []
    
    # Determine GPU usage
    use_gpu = gpu_info['use_gpu'] if gpu_info else False
    
    # Pre-normalize embeddings if needed
    if USE_COSINE_DISTANCE:
        print("   Normalizing embeddings for cosine distance...")
        if use_gpu and CUPY_AVAILABLE:
            print("      Using GPU normalization (CuPy)...")
        elif NORMALIZE_INPLACE:
            print("      Using in-place normalization (memory efficient)")
        
        training_embeddings = normalize_embeddings_gpu(
            training_embeddings, 
            use_gpu=use_gpu, 
            inplace=NORMALIZE_INPLACE
        )
        test_embeddings = normalize_embeddings_gpu(
            test_embeddings, 
            use_gpu=use_gpu, 
            inplace=NORMALIZE_INPLACE
        )
    
    # Build k-NN model for mixed clusters (if needed)
    knn_index = None
    knn_train_labels = None
    use_faiss = FAISS_AVAILABLE and use_knn
    
    if use_knn:
        # For HUGE datasets, subsample training data for k-NN
        if SUBSAMPLE_KNN_TRAINING and len(training_embeddings) > KNN_SUBSAMPLE_SIZE:
            print(f"   ⚠️ Training set too large ({len(training_embeddings):,} samples)")
            print(f"   Subsampling to {KNN_SUBSAMPLE_SIZE:,} samples for k-NN model...")
            
            # Random subsample
            subsample_indices = np.random.choice(
                len(training_embeddings), 
                size=KNN_SUBSAMPLE_SIZE, 
                replace=False
            )
            knn_train_embeddings = training_embeddings[subsample_indices]
            knn_train_labels = training_labels[subsample_indices]
            
            print(f"      Subsampled embeddings shape: {knn_train_embeddings.shape}")
        else:
            knn_train_embeddings = training_embeddings
            knn_train_labels = training_labels
        
        # Build FAISS index (GPU or CPU) or fallback to sklearn
        if use_faiss:
            print(f"   Building FAISS k-NN index (samples={len(knn_train_embeddings):,})...")
            if use_gpu and FAISS_GPU_AVAILABLE:
                print(f"      Using FAISS-GPU (30-50x faster!)")
            else:
                print(f"      Using FAISS-CPU (still 3-5x faster than sklearn)")
            
            knn_index = build_faiss_knn_index(
                knn_train_embeddings, 
                use_gpu=use_gpu, 
                use_cosine=USE_COSINE_DISTANCE
            )
        else:
            # Fallback to sklearn
            print(f"   Building sklearn k-NN model (k={KNN_NEIGHBORS}, samples={len(knn_train_embeddings):,})...")
            knn_model = NearestNeighbors(
                n_neighbors=KNN_NEIGHBORS,
                metric='cosine' if USE_COSINE_DISTANCE else 'euclidean',
                algorithm='auto',
                n_jobs=-1
            )
            knn_model.fit(knn_train_embeddings)
            knn_model.train_labels = knn_train_labels
    
    # Group samples by cluster for efficient processing
    unique_clusters = np.unique(test_cluster_labels)
    
    for cluster_id in tqdm(unique_clusters, desc="Predicting"):
        mask = test_cluster_labels == cluster_id
        indices = np.where(mask)[0]
        
        cluster_info = cluster_dict.get(cluster_id, None)
        
        if cluster_info is None:
            # Unknown cluster (shouldn't happen, but handle gracefully)
            predictions[indices] = 1  # Treat as anomaly
            confidence[indices] = 0.5
            methods.extend(["unknown"] * len(indices))
            continue
        
        # Decision based on cluster type (3-way classification)
        cluster_type = cluster_info['cluster_type']
        
        if cluster_type == "noise":
            # Noise points → ANOMALY (high confidence)
            predictions[indices] = 2
            confidence[indices] = 0.85
            methods.extend(["noise"] * len(indices))
        
        elif cluster_type == "very_small":
            # Very rare patterns → ANOMALY
            predictions[indices] = 2
            confidence[indices] = 0.75
            methods.extend(["very_small"] * len(indices))
        
        elif cluster_type == "small":
            # Unusual patterns → NON-NORMAL
            predictions[indices] = 1
            confidence[indices] = 0.65
            methods.extend(["small"] * len(indices))
        
        elif cluster_type == "pure":
            # High purity → Use cluster's dominant label (0, 1, or 2)
            dominant = cluster_info['dominant_label']
            purity = cluster_info['purity']
            predictions[indices] = dominant
            confidence[indices] = purity
            methods.extend(["pure"] * len(indices))
        
        elif cluster_type == "medium_purity":
            # Borderline mixed cluster → NON-NORMAL
            predictions[indices] = 1
            confidence[indices] = cluster_info['purity']
            methods.extend(["medium_purity"] * len(indices))
        
        else:  # low_purity
            # Use k-NN vote for 3-way classification
            if not use_knn or (knn_index is None and not use_faiss):
                # Fallback: predict NON-NORMAL for ambiguous cases
                predictions[indices] = 1
                confidence[indices] = 0.5
                methods.extend(["fallback"] * len(indices))
                continue
            
            # Get k-NN for each test sample in this cluster
            cluster_test_embeddings = test_embeddings[indices]
            
            # Use FAISS or sklearn depending on availability
            if use_faiss and knn_index is not None:
                # FAISS k-NN search (GPU or CPU)
                # Process in batches to avoid GPU OOM
                n_queries = len(cluster_test_embeddings)
                batch_size = GPU_KNN_BATCH_SIZE if use_gpu else n_queries
                
                all_neighbor_indices = []
                for batch_start in range(0, n_queries, batch_size):
                    batch_end = min(batch_start + batch_size, n_queries)
                    batch_embeddings = cluster_test_embeddings[batch_start:batch_end].astype(np.float32)
                    
                    # FAISS search returns (distances, indices)
                    # For Inner Product (cosine), higher is better
                    _, batch_indices = knn_index.search(batch_embeddings, KNN_NEIGHBORS)
                    all_neighbor_indices.append(batch_indices)
                
                neighbor_indices = np.vstack(all_neighbor_indices)
            else:
                # Sklearn k-NN search (CPU only)
                distances, neighbor_indices = knn_model.kneighbors(cluster_test_embeddings)
            
            # Vote for each test sample (3-way)
            for i, sample_idx in enumerate(indices):
                neighbor_labels = knn_train_labels[neighbor_indices[i]]
                
                # Count votes for each class (0=Normal, 1=NonNormal, 2=Anomaly)
                votes = np.bincount(neighbor_labels, minlength=3)
                vote_normal = votes[0]
                vote_nonnormal = votes[1]
                vote_anomaly = votes[2]
                
                # 3-way classification logic
                if vote_normal >= 8:  # 8-10 normal → NORMAL
                    predictions[sample_idx] = 0
                    confidence[sample_idx] = vote_normal / KNN_NEIGHBORS
                elif vote_anomaly >= 8:  # 8-10 anomaly → ANOMALY
                    predictions[sample_idx] = 2
                    confidence[sample_idx] = vote_anomaly / KNN_NEIGHBORS
                else:  # Borderline/ambiguous → NON-NORMAL
                    predictions[sample_idx] = 1
                    confidence[sample_idx] = KNN_MEDIUM_CONFIDENCE
                
                methods.append("knn")
    
    return predictions, confidence, methods


def calculate_metrics(y_true, y_pred, y_confidence=None, method_labels=None):
    """
    Calculate comprehensive classification metrics (3-way classification)
    """
    print("\n📊 Calculating metrics (3-way classification)...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # 3x3 confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Per-class metrics using classification_report
    report = classification_report(
        y_true, y_pred, 
        labels=[0, 1, 2],
        target_names=['NORMAL', 'NON-NORMAL', 'ANOMALY'],
        output_dict=True,
        zero_division=0
    )
    
    print(f"\n{'='*70}")
    print("OVERALL METRICS (3-WAY CLASSIFICATION)")
    print(f"{'='*70}")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"                  Precision  Recall   F1-Score  Support")
    for label_name in ['NORMAL', 'NON-NORMAL', 'ANOMALY']:
        p = report[label_name]['precision']
        r = report[label_name]['recall']
        f = report[label_name]['f1-score']
        s = int(report[label_name]['support'])
        print(f"  {label_name:12s}    {p:.4f}    {r:.4f}    {f:.4f}   {s:,}")
    
    print(f"\nMacro Avg:          {report['macro avg']['precision']:.4f}    "
          f"{report['macro avg']['recall']:.4f}    {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg:       {report['weighted avg']['precision']:.4f}    "
          f"{report['weighted avg']['recall']:.4f}    {report['weighted avg']['f1-score']:.4f}")
    
    print(f"\n3x3 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 N      NN     A")
    print(f"    True  N  [{cm[0,0]:>6} {cm[0,1]:>6} {cm[0,2]:>6}]")
    print(f"          NN [{cm[1,0]:>6} {cm[1,1]:>6} {cm[1,2]:>6}]")
    print(f"          A  [{cm[2,0]:>6} {cm[2,1]:>6} {cm[2,2]:>6}]")
    
    # Critical error analysis
    if cm[2].sum() > 0:
        print(f"\nCritical Errors:")
        print(f"  A → N (Anomaly missed as Normal):     {cm[2,0]:,} ({cm[2,0]/cm[2].sum()*100:.1f}%)")
        print(f"  A → NN (Anomaly downgrade):           {cm[2,1]:,} ({cm[2,1]/cm[2].sum()*100:.1f}%)")
    
    if cm[1].sum() > 0:
        print(f"\nNon-Normal Errors:")
        print(f"  NN → N (Non-Normal missed as Normal): {cm[1,0]:,} ({cm[1,0]/cm[1].sum()*100:.1f}%)")
        print(f"  NN → A (Non-Normal escalated):        {cm[1,2]:,} ({cm[1,2]/cm[1].sum()*100:.1f}%)")
    
    # Per-method analysis (if available)
    if method_labels is not None:
        print(f"\n{'='*70}")
        print("PER-METHOD METRICS")
        print(f"{'='*70}")
        
        method_counts = Counter(method_labels)
        for method in sorted(method_counts.keys()):
            mask = np.array(method_labels) == method
            method_acc = accuracy_score(y_true[mask], y_pred[mask])
            n_samples = mask.sum()
            
            # Count per class
            y_true_method = y_true[mask]
            n_normal = np.sum(y_true_method == 0)
            n_nonnormal = np.sum(y_true_method == 1)
            n_anomaly = np.sum(y_true_method == 2)
            
            print(f"\n{method.upper():12s}: {n_samples:,} samples ({n_samples/len(y_true)*100:.1f}%)")
            print(f"  Distribution: N={n_normal:,} NN={n_nonnormal:,} A={n_anomaly:,}")
            print(f"  Accuracy: {method_acc:.4f}")
            mask = np.array(method_labels) == method
            acc = accuracy_score(y_true[mask], y_pred[mask])
            count = np.sum(mask)
            pct = count / len(y_pred) * 100
            
            print(f"\n{method.upper():10s}: {count:,} samples ({pct:.1f}%)")
            print(f"  Accuracy: {acc:.4f}")
    
    # Save metrics to file
    with open(OUTPUT_METRICS, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"TESTING RESULTS - {DATASET} {ALGORITHM.upper()} {EMBEDDING_TYPE.upper()}\n")
        f.write("3-WAY CLASSIFICATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, 
                                      labels=[0, 1, 2],
                                      target_names=['NORMAL', 'NON-NORMAL', 'ANOMALY']))
        f.write("\n\n3x3 Confusion Matrix:\n")
        f.write("                 Predicted\n")
        f.write("                 N      NN     A\n")
        f.write(f"    True  N  [{cm[0,0]:>6} {cm[0,1]:>6} {cm[0,2]:>6}]\n")
        f.write(f"          NN [{cm[1,0]:>6} {cm[1,1]:>6} {cm[1,2]:>6}]\n")
        f.write(f"          A  [{cm[2,0]:>6} {cm[2,1]:>6} {cm[2,2]:>6}]\n")
    
    print(f"\n✓ Metrics saved to: {OUTPUT_METRICS}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }


def visualize_results(cluster_df, y_true, y_pred, metrics):
    """
    Create visualizations:
    1. Cluster purity distribution
    2. Confusion matrix heatmap
    3. Cluster type distribution
    """
    print("\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cluster Purity Distribution
    ax1 = axes[0, 0]
    ax1.hist(cluster_df['purity'], bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(PURITY_THRESHOLD_HIGH, color='red', linestyle='--', 
                label=f'High={PURITY_THRESHOLD_HIGH}')
    ax1.axvline(PURITY_THRESHOLD_MEDIUM, color='orange', linestyle='--', 
                label=f'Medium={PURITY_THRESHOLD_MEDIUM}')
    ax1.set_xlabel('Cluster Purity')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Cluster Purity Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Confusion Matrix (3x3)
    ax2 = axes[0, 1]
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Normal', 'Non-Normal', 'Anomaly'],
                yticklabels=['Normal', 'Non-Normal', 'Anomaly'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('3x3 Confusion Matrix')
    
    # 3. Cluster Type Distribution
    ax3 = axes[1, 0]
    type_counts = cluster_df['cluster_type'].value_counts()
    type_counts.plot(kind='bar', ax=ax3, color=['green', 'blue', 'orange', 'red'])
    ax3.set_xlabel('Cluster Type')
    ax3.set_ylabel('Number of Clusters')
    ax3.set_title('Cluster Type Distribution')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Cluster Size Distribution (log scale)
    ax4 = axes[1, 1]
    ax4.hist(cluster_df['n_samples'], bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(VERY_SMALL_CLUSTER_THRESHOLD, color='orange', linestyle='--',
                label=f'Very small={VERY_SMALL_CLUSTER_THRESHOLD}')
    ax4.axvline(SMALL_CLUSTER_THRESHOLD, color='red', linestyle='--',
                label=f'Small={SMALL_CLUSTER_THRESHOLD}')
    ax4.set_xlabel('Cluster Size (samples)')
    ax4.set_ylabel('Number of Clusters')
    ax4.set_title('Cluster Size Distribution')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_overview.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved overview plot")
    
    # Separate detailed confusion matrix (3x3)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=['Normal', 'Non-Normal', 'Anomaly'],
                yticklabels=['Normal', 'Non-Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'3x3 Confusion Matrix - {DATASET} {ALGORITHM.upper()}')
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFUSION_MATRIX, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved 3x3 confusion matrix")
    
    plt.close('all')


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("COMPREHENSIVE CLUSTER TESTING PIPELINE")
    print("3-WAY CLASSIFICATION (Normal/Non-Normal/Anomaly)")
    print("="*70)
    print(f"\nDataset: {DATASET}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"Embedding: {EMBEDDING_TYPE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # ========================================================================
    # STEP 0: Detect GPU capabilities
    # ========================================================================
    gpu_info = detect_gpu_capabilities()
    
    # ========================================================================
    # STEP 1: Load 3-way ground truth labels from templates
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOAD 3-WAY GROUND TRUTH LABELS")
    print("="*70)
    
    training_gt_labels = load_metadata_labels_3way(
        METADATA_TSV_PATH,
        NORMAL_TEMPLATE_PATH,
        NONNORMAL_TEMPLATE_PATH
    )
    
    # ========================================================================
    # STEP 2: Load training cluster results
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: LOAD TRAINING CLUSTER RESULTS")
    print("="*70)
    
    print(f"\nLoading cluster labels: {TRAINING_LABELS_PATH}")
    training_cluster_labels = np.load(TRAINING_LABELS_PATH)
    print(f"   ✓ Loaded {len(training_cluster_labels):,} cluster assignments")
    
    # Verify length match
    if len(training_cluster_labels) != len(training_gt_labels):
        print(f"   ⚠️ Length mismatch!")
        print(f"   Cluster labels: {len(training_cluster_labels):,}")
        print(f"   Ground truth:   {len(training_gt_labels):,}")
        min_len = min(len(training_cluster_labels), len(training_gt_labels))
        print(f"   → Truncating to {min_len:,} samples")
        training_cluster_labels = training_cluster_labels[:min_len]
        training_gt_labels = training_gt_labels[:min_len]
    
    n_clusters = len(set(training_cluster_labels) - {-1})
    n_noise = np.sum(training_cluster_labels == -1)
    print(f"\nClusters found: {n_clusters}")
    print(f"Noise points: {n_noise:,} ({n_noise/len(training_cluster_labels)*100:.2f}%)")
    
    # ========================================================================
    # STEP 3: Analyze cluster characteristics
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: ANALYZE CLUSTER CHARACTERISTICS")
    print("="*70)
    
    cluster_df, cluster_dict = analyze_cluster_characteristics(
        training_cluster_labels, training_gt_labels
    )
    
    # Save cluster analysis
    cluster_df.to_csv(OUTPUT_CLUSTER_ANALYSIS, index=False)
    print(f"\n✓ Cluster analysis saved to: {OUTPUT_CLUSTER_ANALYSIS}")
    
    # ========================================================================
    # STEP 4: Load training embeddings (for k-NN)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: LOAD TRAINING EMBEDDINGS")
    print("="*70)
    
    print(f"\nLoading training embeddings: {TRAINING_EMBEDDINGS_PATH}")
    training_embeddings = np.load(TRAINING_EMBEDDINGS_PATH, mmap_mode='r')
    print(f"   ✓ Shape: {training_embeddings.shape}")
    
    # Truncate if needed (match cluster labels length)
    if len(training_embeddings) != len(training_cluster_labels):
        print(f"   ⚠️ Truncating embeddings to {len(training_cluster_labels):,}")
        training_embeddings = training_embeddings[:len(training_cluster_labels)]
    
    # ========================================================================
    # STEP 5: Load testing data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: LOAD TESTING DATA")
    print("="*70)
    
    # Load multiple testing sets with separate metadata
    test_embeddings, test_gt_labels, test_set_info = load_multiple_testing_sets(
        TESTING_SETS,
        NORMAL_TEMPLATE_PATH,
        NONNORMAL_TEMPLATE_PATH
    )
    
    # ========================================================================
    # STEP 6: Assign test samples to clusters
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: ASSIGN TEST SAMPLES TO CLUSTERS")
    print("="*70)
    
    if ALGORITHM == "kmeans":
        print("\nLoading K-Means model...")
        model = joblib.load(TRAINED_MODEL_PATH)
        print(f"   ✓ Model loaded: {type(model).__name__}")
        
        print("\nPredicting cluster assignments for test data...")
        test_cluster_labels = model.predict(test_embeddings)
        print(f"   ✓ Assigned {len(test_cluster_labels):,} test samples")
        
    else:  # dbscan
        print("\nFor DBSCAN, using k-NN to assign test samples to nearest cluster...")
        
        # Normalize for cosine distance
        if USE_COSINE_DISTANCE:
            print("   Normalizing embeddings...")
            training_embeddings_norm = normalize_embeddings_gpu(
                training_embeddings, 
                use_gpu=gpu_info['use_gpu'], 
                inplace=False
            )
            test_embeddings_norm = normalize_embeddings_gpu(
                test_embeddings, 
                use_gpu=gpu_info['use_gpu'], 
                inplace=False
            )
        else:
            training_embeddings_norm = training_embeddings
            test_embeddings_norm = test_embeddings
        
        # Use FAISS if available for faster cluster assignment
        if FAISS_AVAILABLE and gpu_info['use_gpu']:
            print("   Building FAISS index for cluster assignment (GPU)...")
            index = build_faiss_knn_index(
                training_embeddings_norm, 
                use_gpu=True, 
                use_cosine=USE_COSINE_DISTANCE
            )
            
            print("   Finding nearest training sample for each test sample...")
            # Process in batches
            n_test = len(test_embeddings_norm)
            batch_size = GPU_KNN_BATCH_SIZE
            all_indices = []
            
            for batch_start in tqdm(range(0, n_test, batch_size), desc="   Assigning"):
                batch_end = min(batch_start + batch_size, n_test)
                batch_embeddings = test_embeddings_norm[batch_start:batch_end].astype(np.float32)
                _, batch_indices = index.search(batch_embeddings, 1)
                all_indices.append(batch_indices.flatten())
            
            indices = np.concatenate(all_indices)
        else:
            # Fallback to sklearn
            print("   Building sklearn k-NN for cluster assignment (CPU)...")
            knn = NearestNeighbors(
                n_neighbors=1, 
                metric='cosine' if USE_COSINE_DISTANCE else 'euclidean', 
                n_jobs=-1
            )
            knn.fit(training_embeddings_norm)
            
            print("   Finding nearest training sample for each test sample...")
            distances, indices_matrix = knn.kneighbors(test_embeddings_norm)
            indices = indices_matrix.flatten()
        
        # Assign cluster based on nearest neighbor
        test_cluster_labels = training_cluster_labels[indices]
        print(f"   ✓ Assigned {len(test_cluster_labels):,} test samples")
    
    # ========================================================================
    # STEP 7: Hybrid prediction
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: HYBRID PREDICTION")
    print("="*70)
    
    predictions, confidence, methods = hybrid_predict(
        test_cluster_labels, cluster_dict,
        training_embeddings, training_gt_labels,
        test_embeddings, use_knn=True, gpu_info=gpu_info
    )
    
    # Save predictions
    np.save(OUTPUT_PREDICTIONS, predictions)
    print(f"\n✓ Predictions saved to: {OUTPUT_PREDICTIONS}")
    
    # ========================================================================
    # STEP 8: Calculate metrics
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: CALCULATE METRICS")
    print("="*70)
    
    metrics = calculate_metrics(test_gt_labels, predictions, confidence, methods)
    
    # ========================================================================
    # STEP 9: Create detailed results CSV
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 9: SAVE DETAILED RESULTS")
    print("="*70)
    
    # Assign test set name to each sample
    test_set_names = []
    for i in range(len(test_gt_labels)):
        for set_info in test_set_info:
            if set_info['start_idx'] <= i < set_info['end_idx']:
                test_set_names.append(set_info['name'])
                break
    
    results_df = pd.DataFrame({
        'test_set': test_set_names,
        'cluster_id': test_cluster_labels,
        'true_label': test_gt_labels,
        'predicted_label': predictions,
        'confidence': confidence,
        'method': methods,
        'correct': (test_gt_labels == predictions).astype(int)
    })
    
    results_df.to_csv(OUTPUT_DETAILED_RESULTS, index=False)
    print(f"✓ Detailed results saved to: {OUTPUT_DETAILED_RESULTS}")
    print(f"  Total rows: {len(results_df):,}")
    
    # Print per-set accuracy
    print(f"\n📊 Per-Set Performance:")
    for set_info in test_set_info:
        set_name = set_info['name']
        set_mask = results_df['test_set'] == set_name
        set_acc = results_df[set_mask]['correct'].mean()
        set_samples = set_mask.sum()
        print(f"   {set_name:12s}: {set_acc:.4f} accuracy ({set_samples:,} samples)")
    
    # Calculate detailed per-set metrics
    print(f"\n📈 Calculating per-set detailed metrics...")
    per_set_metrics = []
    for set_info in test_set_info:
        set_name = set_info['name']
        set_mask = results_df['test_set'] == set_name
        
        set_y_true = results_df[set_mask]['true_label'].values
        set_y_pred = results_df[set_mask]['predicted_label'].values
        
        # Per-class metrics for this set
        set_report = classification_report(
            set_y_true, set_y_pred,
            labels=[0, 1, 2],
            target_names=['NORMAL', 'NON-NORMAL', 'ANOMALY'],
            output_dict=True,
            zero_division=0
        )
        
        per_set_metrics.append({
            'test_set': set_name,
            'n_samples': int(set_mask.sum()),
            'accuracy': accuracy_score(set_y_true, set_y_pred),
            'normal_precision': set_report['NORMAL']['precision'],
            'normal_recall': set_report['NORMAL']['recall'],
            'normal_f1': set_report['NORMAL']['f1-score'],
            'nonnormal_precision': set_report['NON-NORMAL']['precision'],
            'nonnormal_recall': set_report['NON-NORMAL']['recall'],
            'nonnormal_f1': set_report['NON-NORMAL']['f1-score'],
            'anomaly_precision': set_report['ANOMALY']['precision'],
            'anomaly_recall': set_report['ANOMALY']['recall'],
            'anomaly_f1': set_report['ANOMALY']['f1-score'],
            'macro_f1': set_report['macro avg']['f1-score'],
            'weighted_f1': set_report['weighted avg']['f1-score']
        })
    
    per_set_df = pd.DataFrame(per_set_metrics)
    per_set_df.to_csv(OUTPUT_PER_SET_METRICS, index=False)
    print(f"✓ Per-set metrics saved to: {OUTPUT_PER_SET_METRICS}")
    
    # ========================================================================
    # STEP 10: Visualize results
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 10: CREATE VISUALIZATIONS")
    print("="*70)
    
    visualize_results(cluster_df, test_gt_labels, predictions, metrics)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("TESTING PIPELINE COMPLETED")
    print("="*70)
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print(f"\nKey files:")
    print(f"  - Predictions:       {OUTPUT_PREDICTIONS.name}")
    print(f"  - Cluster analysis:  {OUTPUT_CLUSTER_ANALYSIS.name}")
    print(f"  - Metrics:           {OUTPUT_METRICS.name}")
    print(f"  - Per-set metrics:   {OUTPUT_PER_SET_METRICS.name}")
    print(f"  - Detailed results:  {OUTPUT_DETAILED_RESULTS.name}")
    print(f"  - Visualizations:    analysis_overview.png, confusion_matrix.png")
    
    print(f"\n🎯 Final Results (3-Way Classification):")
    print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Normal F1:        {metrics['report']['NORMAL']['f1-score']:.4f}")
    print(f"  Non-Normal F1:    {metrics['report']['NON-NORMAL']['f1-score']:.4f}")
    print(f"  Anomaly F1:       {metrics['report']['ANOMALY']['f1-score']:.4f}")
    print(f"  Macro Avg F1:     {metrics['report']['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
