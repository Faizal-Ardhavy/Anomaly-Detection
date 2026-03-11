"""
Comprehensive Cluster Testing Pipeline for Log Anomaly Detection
GROUND TRUTH: 2-Class (Normal / Non-Normal) based on test set name
PREDICTION: 3-Class (Normal / Non-Normal / Anomaly) based on cluster assignment

Strategy for MANY clusters (dozens to hundreds):
1. Load ground truth labels based on test set name:
   - Test set 'normal' → ALL samples = NORMAL (0)
   - Test set 'nonnormal' → ALL samples = NON-NORMAL (1)

2. Analyze TRAINING cluster characteristics using template matching:
   - Pure clusters (>95%) → Assign dominant label from training
   - Mixed clusters → Use hybrid prediction strategy

3. Hybrid prediction strategy for TEST samples:
   - Noise points (DBSCAN) → ANOMALY (label=2)
   - Very small clusters (<50) → ANOMALY (label=2)
   - Small clusters (50-200) → NON-NORMAL (label=1)
   - Pure clusters (>95%) → Trust cluster's dominant label (0/1/2)
   - Medium purity (70-95%) → NON-NORMAL (label=1)
   - Low purity (<70%) → k-NN vote for 3-way classification

4. Calculate 2x3 metrics: 2 ground truth classes, 3 predicted classes

5. Visualize: Cluster purity, confusion matrix, prediction distribution

Supports:
- K-Means and DBSCAN
- BGL and Thunderbird datasets  
- Base/PCA256/PCA128 embeddings
- Ground truth from test set names (NOT template matching!)
- Full testing dataset (no sampling)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import csv
import pickle
import gc
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

# Path to testing data - MULTIPLE SETS (Ground truth based on set name!)
# Each testing set should have embeddings file and a name indicating its class
# Set name 'normal' → ground truth = NORMAL (0)
# Set name 'nonnormal' → ground truth = NON-NORMAL (1)
TESTING_SETS = [
    {
        'name': 'normal',  # Ground truth: ALL = NORMAL (0)
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_testing/after_preprocessed_bgl_normal_embeddings.npy"),
        'metadata': Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_bgl_normal_meta.tsv")  # Optional, only needed if you want to do template-based analysis
    },
    {
        'name': 'nonnormal',  # Ground truth: ALL = NON-NORMAL (1)
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_testing/after_preprocessed_bgl_non_normal_embeddings.npy"),
        'metadata': Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_bgl_non_normal_meta.tsv")  # Optional, only needed if you want to do template-based analysis
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

# Checkpoint paths (for resume capability)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_STEP6 = CHECKPOINT_DIR / "step6_test_cluster_labels.npy"
CHECKPOINT_STEP7_PRED = CHECKPOINT_DIR / "step7_predictions.npy"
CHECKPOINT_STEP7_CONF = CHECKPOINT_DIR / "step7_confidence.npy"
CHECKPOINT_STEP7_METHODS = CHECKPOINT_DIR / "step7_methods.npy"
CHECKPOINT_METADATA = CHECKPOINT_DIR / "checkpoint_metadata.pkl"

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
                                chunksize: int = 1_000_000) -> np.ndarray:
    """
    Load ground truth labels using template-based 3-way classification
    
    Args:
        tsv_path: Path to metadata TSV file
        normal_template_path: Path to normal template
        nonnormal_template_path: Path to non-normal template
        use_chunking: If True, use streaming/chunking for large files (default: True)
        chunksize: Number of rows per chunk (default: 1M rows)
    
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


def load_multiple_testing_sets(testing_sets, normal_template_path=None, nonnormal_template_path=None):
    """
    Load multiple testing sets - Ground truth based on set name
    
    Args:
        testing_sets: List of dicts with keys: 'name', 'embeddings', 'metadata' (optional)
        normal_template_path: Not used (kept for compatibility)
        nonnormal_template_path: Not used (kept for compatibility)
    
    Returns:
        combined_embeddings: numpy array of all embeddings concatenated
        combined_labels: numpy array of all ground truth labels (based on set name)
        test_set_info: list of dicts with per-set statistics
    
    Ground Truth Assignment:
        - Set name 'normal' → ALL samples = NORMAL (0)
        - Set name 'nonnormal' → ALL samples = NON-NORMAL (1)
        - Other names → Check if contains 'normal' or 'nonnormal'
    """
    print(f"\n📦 Loading {len(testing_sets)} testing sets...")
    print(f"   📌 Ground truth assigned based on set name (not template matching)")
    
    all_embeddings = []
    all_labels = []
    test_set_info = []
    
    for test_set in testing_sets:
        name = test_set['name']
        embeddings_path = test_set['embeddings']
        
        print(f"\n  📂 Loading test set: {name}")
        print(f"     Embeddings: {embeddings_path.name}")
        
        # Load embeddings
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        embeddings = np.load(embeddings_path, mmap_mode='r')
        print(f"     ✓ Loaded embeddings: {embeddings.shape}")
        
        # Assign ground truth based on set name (NOT template matching!)
        n_samples = len(embeddings)
        name_lower = name.lower()
        
        if 'normal' in name_lower and 'non' not in name_lower:
            # Set name contains "normal" (but not "nonnormal") → NORMAL
            ground_truth_label = 0  # NORMAL
            label_name = "NORMAL"
        elif 'nonnormal' in name_lower or 'non_normal' in name_lower or 'non-normal' in name_lower:
            # Set name contains "nonnormal" → NON-NORMAL
            ground_truth_label = 1  # NON-NORMAL
            label_name = "NON-NORMAL"
        else:
            # Unknown set name, try to infer or raise error
            print(f"     ⚠️ WARNING: Cannot infer ground truth from set name '{name}'")
            print(f"     → Please rename set to include 'normal' or 'nonnormal'")
            raise ValueError(f"Cannot determine ground truth for test set: {name}")
        
        # Create labels array (all samples have same label based on set name)
        labels = np.full(n_samples, ground_truth_label, dtype=np.int32)
        print(f"     ✓ Ground truth: ALL {n_samples:,} samples = {label_name} ({ground_truth_label})")
        
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
                   test_embeddings, use_knn=True):
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
    
    # Pre-normalize embeddings if needed
    if USE_COSINE_DISTANCE:
        print("   Normalizing embeddings for cosine distance...")
        if NORMALIZE_INPLACE:
            # In-place normalization to save memory (copy=False)
            # WARNING: modifies original arrays, but they are mmap views so it's safe
            print("      Using in-place normalization (memory efficient)")
            training_embeddings = normalize(training_embeddings, norm='l2', copy=False)
            test_embeddings = normalize(test_embeddings, norm='l2', copy=False)
        else:
            training_embeddings = normalize(training_embeddings, norm='l2')
            test_embeddings = normalize(test_embeddings, norm='l2')
    
    # Build k-NN model for mixed clusters (if needed)
    knn_model = None
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
        
        print(f"   Building k-NN model (k={KNN_NEIGHBORS}, samples={len(knn_train_embeddings):,})...")
        knn_model = NearestNeighbors(
            n_neighbors=KNN_NEIGHBORS,
            metric='cosine' if USE_COSINE_DISTANCE else 'euclidean',
            algorithm='auto',
            n_jobs=-1
        )
        knn_model.fit(knn_train_embeddings)
        
        # Store labels for vote counting
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
            if not use_knn or knn_model is None:
                # Fallback: predict NON-NORMAL for ambiguous cases
                predictions[indices] = 1
                confidence[indices] = 0.5
                methods.extend(["fallback"] * len(indices))
                continue
            
            # Get k-NN for each test sample in this cluster
            cluster_test_embeddings = test_embeddings[indices]
            distances, neighbor_indices = knn_model.kneighbors(cluster_test_embeddings)
            
            # Vote for each test sample (3-way)
            # Use the labels from k-NN model (might be subsampled)
            knn_labels = knn_model.train_labels if hasattr(knn_model, 'train_labels') else training_labels
            
            for i, sample_idx in enumerate(indices):
                neighbor_labels = knn_labels[neighbor_indices[i]]
                
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
    Calculate comprehensive classification metrics
    
    Ground Truth: 2-class (NORMAL=0, NON-NORMAL=1)
    Predictions: 3-class (NORMAL=0, NON-NORMAL=1, ANOMALY=2)
    
    Result: 2x3 confusion matrix 
    """
    print("\n📊 Calculating metrics...")
    
    # Detect unique ground truth classes
    unique_true = sorted(set(y_true))
    unique_pred = sorted(set(y_pred))
    
    print(f"   Ground truth classes: {unique_true} → {[['NORMAL', 'NON-NORMAL', 'ANOMALY'][i] for i in unique_true]}")
    print(f"   Prediction classes:   {unique_pred} → {[['NORMAL', 'NON-NORMAL', 'ANOMALY'][i] for i in unique_pred]}")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix: rows=true classes, cols=predicted classes
    # If ground truth has 2 classes (0,1) and predictions have 3 classes (0,1,2) → 2x3 matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_true)
    
    # Per-class metrics - only for ground truth classes
    true_class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    pred_class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    
    # Calculate metrics only for classes that exist in ground truth
    report_labels = [i for i in unique_true if i in [0, 1, 2]]
    report_names = [true_class_names[i] for i in report_labels]
    
    report = classification_report(
        y_true, y_pred, 
        labels=report_labels,
        target_names=report_names,
        output_dict=True,
        zero_division=0
    )
    
    print(f"\n{'='*70}")
    if len(unique_true) == 2:
        print("OVERALL METRICS (2-CLASS GROUND TRUTH vs 3-CLASS PREDICTION)")
    else:
        print("OVERALL METRICS")
    print(f"{'='*70}")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print(f"\nPer-Class Metrics (Ground Truth Classes Only):")
    print(f"                  Precision  Recall   F1-Score  Support")
    for label_name in report_names:
        p = report[label_name]['precision']
        r = report[label_name]['recall']
        f = report[label_name]['f1-score']
        s = int(report[label_name]['support'])
        print(f"  {label_name:12s}    {p:.4f}    {r:.4f}    {f:.4f}   {s:,}")
    
    print(f"\nMacro Avg:          {report['macro avg']['precision']:.4f}    "
          f"{report['macro avg']['recall']:.4f}    {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg:       {report['weighted avg']['precision']:.4f}    "
          f"{report['weighted avg']['recall']:.4f}    {report['weighted avg']['f1-score']:.4f}")
    
    # Dynamic confusion matrix display
    print(f"\nConfusion Matrix ({len(unique_true)}x{len(unique_pred)}):")
    print(f"                 Predicted")
    
    # Header
    header = "                 "
    for pred_class in unique_pred:
        header += f"{pred_class_names[pred_class][:2]:>7}"
    print(header)
    
    # Rows
    for i, true_class in enumerate(unique_true):
        row = f"    True  {true_class_names[true_class][:2]:2s} ["
        for j, pred_class in enumerate(unique_pred):
            if j < cm.shape[1]:
                row += f"{cm[i,j]:>6} "
            else:
                row += "     0 "
        row += "]"
        print(row)
    
    # Error analysis
    if len(unique_true) == 2:
        # 2-class ground truth analysis
        print(f"\n📊 Prediction Distribution:")
        for i, true_class in enumerate(unique_true):
            true_name = true_class_names[true_class]
            total = cm[i].sum()
            print(f"\n  {true_name} ({total:,} samples):")
            for j, pred_class in enumerate(unique_pred):
                if j < cm.shape[1]:
                    count = cm[i, j]
                    pct = count / total * 100 if total > 0 else 0
                    pred_name = pred_class_names[pred_class]
                    status = "✓ Correct" if true_class == pred_class else "✗ Wrong"
                    print(f"    → Predicted as {pred_name:12s}: {count:7,} ({pct:5.2f}%) {status}")
    else:
        # 3-class analysis (legacy)
        if len(unique_true) > 2 and 2 in unique_true:
            anomaly_idx = unique_true.index(2)
            if cm[anomaly_idx].sum() > 0:
                print(f"\nCritical Errors:")
                print(f"  A → N (Anomaly missed as Normal):     {cm[anomaly_idx,0]:,} ({cm[anomaly_idx,0]/cm[anomaly_idx].sum()*100:.1f}%)")
                if cm.shape[1] > 1:
                    print(f"  A → NN (Anomaly downgrade):           {cm[anomaly_idx,1]:,} ({cm[anomaly_idx,1]/cm[anomaly_idx].sum()*100:.1f}%)")
        
        if 1 in unique_true:
            nn_idx = unique_true.index(1)
            if cm[nn_idx].sum() > 0:
                print(f"\nNon-Normal Errors:")
                print(f"  NN → N (Non-Normal missed as Normal): {cm[nn_idx,0]:,} ({cm[nn_idx,0]/cm[nn_idx].sum()*100:.1f}%)")
                if cm.shape[1] > 2:
                    print(f"  NN → A (Non-Normal escalated):        {cm[nn_idx,2]:,} ({cm[nn_idx,2]/cm[nn_idx].sum()*100:.1f}%)")
    
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
            
            # Count per class in ground truth
            y_true_method = y_true[mask]
            class_counts = {}
            for cls in unique_true:
                class_counts[true_class_names[cls]] = np.sum(y_true_method == cls)
            
            print(f"\n{method.upper():12s}: {n_samples:,} samples ({n_samples/len(y_true)*100:.1f}%)")
            dist_str = "  Distribution: " + " ".join([f"{name}={count:,}" for name, count in class_counts.items()])
            print(dist_str)
            print(f"  Accuracy: {method_acc:.4f}")
    
    # Save metrics to file
    with open(OUTPUT_METRICS, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"TESTING RESULTS - {DATASET} {ALGORITHM.upper()} {EMBEDDING_TYPE.upper()}\n")
        if len(unique_true) == 2:
            f.write("2-CLASS GROUND TRUTH vs 3-CLASS PREDICTION\n")
        else:
            f.write("3-WAY CLASSIFICATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, 
                                      labels=report_labels,
                                      target_names=report_names))
        f.write(f"\n\nConfusion Matrix ({len(unique_true)}x{len(unique_pred)}):\n")
        f.write("                 Predicted\n")
        
        # Header
        header = "                 "
        for pred_class in unique_pred:
            header += f"{pred_class_names[pred_class][:2]:>7}"
        f.write(header + "\n")
        
        # Rows
        for i, true_class in enumerate(unique_true):
            row = f"    True  {true_class_names[true_class][:2]:2s} ["
            for j, pred_class in enumerate(unique_pred):
                if j < cm.shape[1]:
                    row += f"{cm[i,j]:>6} "
                else:
                    row += "     0 "
            row += "]\n"
            f.write(row)
    
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
    
    # 2. Confusion Matrix (dynamic size)
    ax2 = axes[0, 1]
    cm = metrics['confusion_matrix']
    
    # Determine labels dynamically
    unique_true = sorted(set(y_true))
    unique_pred = sorted(set(y_pred))
    class_names = ['Normal', 'Non-Normal', 'Anomaly']
    
    true_labels = [class_names[i] for i in unique_true]
    pred_labels = [class_names[i] for i in unique_pred]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=pred_labels,
                yticklabels=true_labels)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Ground Truth')
    ax2.set_title(f'{len(unique_true)}x{len(unique_pred)} Confusion Matrix')
    
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
    
    # Separate detailed confusion matrix (dynamic size)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=pred_labels,
                yticklabels=true_labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.title(f'{len(unique_true)}x{len(unique_pred)} Confusion Matrix - {DATASET} {ALGORITHM.upper()}')
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFUSION_MATRIX, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved confusion matrix ({len(unique_true)}x{len(unique_pred)})")
    
    plt.close('all')


def analyze_prediction_distribution(y_true, y_pred, y_confidence=None, prediction_methods=None, test_set_names=None):
    """
    Detailed analysis of prediction distribution per ground truth class
    
    Shows where mispredictions go (e.g., NORMAL → NON-NORMAL vs NORMAL → ANOMALY)
    
    Args:
        y_true: Ground truth labels (0=NORMAL, 1=NON-NORMAL)
        y_pred: Predicted labels (0=NORMAL, 1=NON-NORMAL, 2=ANOMALY)
        y_confidence: Confidence scores (optional)
        prediction_methods: Prediction methods used (optional)
        test_set_names: Test set names per sample (optional)
    
    Returns:
        dict with distribution statistics
    """
    print("\n" + "="*70)
    print("STEP 11: DETAILED PREDICTION DISTRIBUTION ANALYSIS")
    print("="*70)
    
    class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    unique_true = sorted(set(y_true))
    unique_pred = sorted(set(y_pred))
    
    distribution_stats = {}
    
    # ========================================================================
    # 1. Overall Distribution per Ground Truth Class
    # ========================================================================
    print("\n📊 Prediction Distribution by Ground Truth Class:")
    print("="*70)
    
    for true_label in unique_true:
        mask = y_true == true_label
        n_total = np.sum(mask)
        true_name = class_names[true_label]
        
        print(f"\n{true_name} Ground Truth ({n_total:,} samples):")
        print("-" * 70)
        
        distribution_stats[true_label] = {
            'name': true_name,
            'n_total': n_total,
            'predictions': {}
        }
        
        # Count predictions for this ground truth class
        for pred_label in unique_pred:
            pred_mask = (y_true == true_label) & (y_pred == pred_label)
            n_pred = np.sum(pred_mask)
            pct = (n_pred / n_total) * 100
            pred_name = class_names[pred_label]
            
            # Mark correct/wrong
            is_correct = (true_label == pred_label)
            marker = "✓ CORRECT" if is_correct else "✗ ERROR"
            
            print(f"  → Predicted as {pred_name:12s}: {n_pred:8,} ({pct:6.2f}%) {marker}")
            
            distribution_stats[true_label]['predictions'][pred_label] = {
                'name': pred_name,
                'count': n_pred,
                'percentage': pct,
                'is_correct': is_correct
            }
        
        # Calculate error breakdown
        correct_mask = (y_true == true_label) & (y_pred == true_label)
        n_correct = np.sum(correct_mask)
        n_errors = n_total - n_correct
        error_pct = (n_errors / n_total) * 100
        
        print(f"\n  Summary:")
        print(f"    Correct: {n_correct:,} ({100 - error_pct:.2f}%)")
        print(f"    Errors:  {n_errors:,} ({error_pct:.2f}%)")
    
    # ========================================================================
    # 2. Error Analysis: Where do mispredictions go?
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 Error Breakdown (Mispredictions):")
    print("="*70)
    
    for true_label in unique_true:
        true_name = class_names[true_label]
        mask = y_true == true_label
        error_mask = (y_true == true_label) & (y_pred != true_label)
        n_errors = np.sum(error_mask)
        n_total = np.sum(mask)
        
        if n_errors == 0:
            print(f"\n{true_name}: No errors! 100% accuracy")
            continue
        
        print(f"\n{true_name} Errors ({n_errors:,} / {n_total:,} = {n_errors/n_total*100:.2f}%):")
        print("-" * 70)
        
        # Breakdown by prediction
        for pred_label in unique_pred:
            if pred_label == true_label:
                continue  # Skip correct predictions
            
            pred_mask = (y_true == true_label) & (y_pred == pred_label)
            n_pred = np.sum(pred_mask)
            
            if n_pred == 0:
                continue
            
            pct_of_errors = (n_pred / n_errors) * 100
            pct_of_total = (n_pred / n_total) * 100
            pred_name = class_names[pred_label]
            
            print(f"  → Misclassified as {pred_name:12s}: {n_pred:8,}")
            print(f"     ({pct_of_errors:5.1f}% of errors, {pct_of_total:5.2f}% of total {true_name})")
            
            # Show top methods causing this error (if available)
            if prediction_methods is not None:
                method_counts = Counter(prediction_methods[pred_mask])
                top_methods = method_counts.most_common(3)
                if top_methods:
                    print(f"     Top methods: ", end="")
                    for method, count in top_methods:
                        print(f"{method}={count:,} ", end="")
                    print()
    
    # ========================================================================
    # 3. Prediction Method Analysis (if available)
    # ========================================================================
    if prediction_methods is not None:
        print("\n" + "="*70)
        print("🔧 Error Analysis by Prediction Method:")
        print("="*70)
        
        unique_methods = sorted(set(prediction_methods))
        
        for method in unique_methods:
            method_mask = np.array(prediction_methods) == method
            n_method = np.sum(method_mask)
            
            if n_method == 0:
                continue
            
            print(f"\n{method.upper()} Method ({n_method:,} samples):")
            print("-" * 70)
            
            # Error rate per ground truth class
            for true_label in unique_true:
                true_name = class_names[true_label]
                mask = method_mask & (y_true == true_label)
                n_total = np.sum(mask)
                
                if n_total == 0:
                    continue
                
                n_correct = np.sum(mask & (y_pred == true_label))
                n_errors = n_total - n_correct
                error_rate = (n_errors / n_total) * 100 if n_total > 0 else 0
                
                print(f"  {true_name:12s}: {n_total:7,} samples, "
                      f"{n_correct:7,} correct, {n_errors:7,} errors ({error_rate:5.2f}%)")
                
                # Show error distribution for this method + ground truth
                if n_errors > 0:
                    for pred_label in unique_pred:
                        if pred_label == true_label:
                            continue
                        error_mask = mask & (y_pred == pred_label)
                        n_err = np.sum(error_mask)
                        if n_err > 0:
                            pred_name = class_names[pred_label]
                            print(f"    → {n_err:7,} misclassified as {pred_name}")
    
    # ========================================================================
    # 4. Per-Test-Set Analysis (if available)
    # ========================================================================
    if test_set_names is not None:
        print("\n" + "="*70)
        print("📂 Error Analysis by Test Set:")
        print("="*70)
        
        unique_sets = sorted(set(test_set_names))
        
        for set_name in unique_sets:
            set_mask = np.array(test_set_names) == set_name
            n_set = np.sum(set_mask)
            
            print(f"\n{set_name.upper()} Test Set ({n_set:,} samples):")
            print("-" * 70)
            
            # Ground truth distribution (should be uniform per set)
            for true_label in unique_true:
                mask = set_mask & (y_true == true_label)
                n_total = np.sum(mask)
                
                if n_total == 0:
                    continue
                
                true_name = class_names[true_label]
                print(f"\n  {true_name} ({n_total:,} samples):")
                
                # Prediction distribution
                for pred_label in unique_pred:
                    pred_mask = mask & (y_pred == pred_label)
                    n_pred = np.sum(pred_mask)
                    pct = (n_pred / n_total) * 100 if n_total > 0 else 0
                    pred_name = class_names[pred_label]
                    
                    is_correct = (pred_label == true_label)
                    marker = "✓" if is_correct else "✗"
                    
                    print(f"    {marker} → {pred_name:12s}: {n_pred:7,} ({pct:6.2f}%)")
    
    # ========================================================================
    # 5. Save Distribution Statistics to CSV
    # ========================================================================
    print("\n" + "="*70)
    print("💾 Saving distribution statistics...")
    
    # Create detailed distribution table
    rows = []
    for true_label in unique_true:
        true_name = class_names[true_label]
        mask = y_true == true_label
        n_total = np.sum(mask)
        
        for pred_label in unique_pred:
            pred_name = class_names[pred_label]
            pred_mask = (y_true == true_label) & (y_pred == pred_label)
            n_pred = np.sum(pred_mask)
            pct = (n_pred / n_total) * 100 if n_total > 0 else 0
            
            rows.append({
                'ground_truth': true_name,
                'ground_truth_label': true_label,
                'prediction': pred_name,
                'prediction_label': pred_label,
                'count': n_pred,
                'percentage': pct,
                'total_in_class': n_total,
                'is_correct': (true_label == pred_label)
            })
    
    df_dist = pd.DataFrame(rows)
    output_dist = OUTPUT_DIR / "prediction_distribution.csv"
    df_dist.to_csv(output_dist, index=False)
    print(f"✓ Distribution table saved: {output_dist}")
    
    # ========================================================================
    # 6. Create Distribution Visualization
    # ========================================================================
    print("\n📊 Creating distribution visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Stacked bar chart for prediction distribution
    ax1 = axes[0]
    
    # Prepare data for stacked bar
    true_labels_plot = []
    pred_distributions = {pred: [] for pred in unique_pred}
    
    for true_label in unique_true:
        true_name = class_names[true_label]
        true_labels_plot.append(true_name)
        
        mask = y_true == true_label
        n_total = np.sum(mask)
        
        for pred_label in unique_pred:
            pred_mask = (y_true == true_label) & (y_pred == pred_label)
            n_pred = np.sum(pred_mask)
            pct = (n_pred / n_total) * 100 if n_total > 0 else 0
            pred_distributions[pred_label].append(pct)
    
    # Create stacked bar chart
    x_pos = np.arange(len(true_labels_plot))
    colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange
    bottom = np.zeros(len(true_labels_plot))
    
    for pred_label in unique_pred:
        pred_name = class_names[pred_label]
        values = pred_distributions[pred_label]
        ax1.bar(x_pos, values, bottom=bottom, label=f'Pred: {pred_name}', 
                color=colors[pred_label], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 2:  # Only show label if bar is big enough
                ax1.text(i, b + v/2, f'{v:.1f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=10)
        
        bottom += values
    
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_xlabel('Ground Truth Class', fontsize=12)
    ax1.set_title('Prediction Distribution by Ground Truth Class', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(true_labels_plot, fontsize=11)
    ax1.legend(title='Predictions', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Right plot: Error breakdown (absolute counts)
    ax2 = axes[1]
    
    # Prepare error data
    error_data = []
    error_labels = []
    error_colors_list = []
    
    for true_label in unique_true:
        true_name = class_names[true_label]
        mask = y_true == true_label
        n_total = np.sum(mask)
        
        # Count correct
        correct_mask = (y_true == true_label) & (y_pred == true_label)
        n_correct = np.sum(correct_mask)
        
        error_data.append(n_correct)
        error_labels.append(f'{true_name}\nCorrect')
        error_colors_list.append('#2ecc71')  # Green for correct
        
        # Count errors by type
        for pred_label in unique_pred:
            if pred_label == true_label:
                continue
            
            pred_mask = (y_true == true_label) & (y_pred == pred_label)
            n_pred = np.sum(pred_mask)
            
            if n_pred > 0:
                pred_name = class_names[pred_label]
                error_data.append(n_pred)
                error_labels.append(f'{true_name}→{pred_name}\nError')
                error_colors_list.append('#e74c3c' if pred_label == 2 else '#f39c12')
    
    # Create bar chart
    x_pos_err = np.arange(len(error_data))
    bars = ax2.bar(x_pos_err, error_data, color=error_colors_list, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Count (samples)', fontsize=12)
    ax2.set_xlabel('Prediction Type', fontsize=12)
    ax2.set_title('Correct vs Mispredictions (Absolute Counts)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos_err)
    ax2.set_xticklabels(error_labels, fontsize=9, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visibility
    
    plt.tight_layout()
    output_dist_plot = OUTPUT_DIR / "prediction_distribution.png"
    plt.savefig(output_dist_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plot saved: {output_dist_plot}")
    
    plt.close('all')
    
    print("\n" + "="*70)
    print("✓ Prediction distribution analysis complete!")
    print("="*70)
    
    return distribution_stats


def fast_cluster_assignment_faiss(training_embeddings, training_cluster_labels, 
                                   test_embeddings, use_cosine=True, 
                                   nlist=1024, nprobe=64, batch_size=50000):
    """
    FAST cluster assignment using FAISS IVF (Approximate k-NN)
    
    10-100x faster than exact k-NN with ~95-99% accuracy
    
    Args:
        training_embeddings: Training embeddings (normalized if cosine)
        training_cluster_labels: Cluster assignment for training data
        test_embeddings: Test embeddings to assign
        use_cosine: Use cosine distance (L2 on normalized vectors)
        nlist: Number of Voronoi cells (more = slower but more accurate)
        nprobe: Number of cells to visit during search (more = slower but more accurate)
        batch_size: Process test data in batches (memory efficient)
    
    Returns:
        test_cluster_labels: Cluster assignment for test data
    
    Note:
        - nlist=1024, nprobe=64: Good balance (10-50x speedup, ~98% accuracy)
        - nlist=2048, nprobe=128: More accurate (5-20x speedup, ~99% accuracy)
        - nlist=512, nprobe=32: Fastest (50-100x speedup, ~95% accuracy)
    """
    try:
        import faiss
        import gc
    except ImportError:
        print("   ⚠️ FAISS not available, falling back to sklearn (SLOW!)")
        return None
    
    print(f"\n   🚀 Using FAISS IVF for FAST approximate k-NN...")
    print(f"      nlist={nlist} Voronoi cells, nprobe={nprobe} cells searched")
    print(f"      Expected: 10-50x faster, ~98% accuracy vs exact k-NN")
    
    # Normalize embeddings if using cosine distance
    if use_cosine:
        print(f"      Normalizing embeddings for cosine similarity...")
        training_norm = normalize(np.array(training_embeddings), norm='l2', copy=True)
        test_norm = normalize(test_embeddings, norm='l2', copy=True)
    else:
        training_norm = np.array(training_embeddings, dtype=np.float32)
        test_norm = test_embeddings.astype(np.float32)
    
    # Ensure contiguous arrays
    training_norm = np.ascontiguousarray(training_norm, dtype=np.float32)
    test_norm = np.ascontiguousarray(test_norm, dtype=np.float32)
    
    d = training_norm.shape[1]  # Embedding dimension
    
    print(f"      Training data: {len(training_norm):,} samples, dim={d}")
    print(f"      Test data: {len(test_norm):,} samples")
    
    # Build IVF index
    print(f"      Building FAISS IVF index...")
    quantizer = faiss.IndexFlatL2(d)  # Quantizer for Voronoi cells
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # Train index (clustering training data into Voronoi cells)
    print(f"      Training index (clustering into {nlist} cells)...")
    
    # For very large datasets, train on subset
    if len(training_norm) > 1_000_000:
        print(f"         Sampling 1M points for training (large dataset optimization)...")
        train_sample_idx = np.random.choice(len(training_norm), 1_000_000, replace=False)
        index.train(training_norm[train_sample_idx])
    else:
        index.train(training_norm)
    
    # Add all training data to index
    print(f"      Adding {len(training_norm):,} training vectors to index...")
    index.add(training_norm)
    
    # Set search parameters
    index.nprobe = nprobe  # How many cells to visit during search
    
    print(f"      ✓ Index built! Starting k-NN search...")
    print(f"      Processing {len(test_norm):,} test samples in batches of {batch_size:,}...")
    
    # Search in batches (memory efficient)
    all_indices = []
    n_batches = (len(test_norm) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(test_norm), batch_size), desc="      Searching", total=n_batches):
        batch_end = min(i + batch_size, len(test_norm))
        batch = test_norm[i:batch_end]
        
        # Search k=1 nearest neighbors
        distances, indices = index.search(batch, 1)
        all_indices.append(indices.flatten())
    
    # Combine results
    nearest_indices = np.concatenate(all_indices)
    
    # Map indices to cluster labels
    test_cluster_labels = training_cluster_labels[nearest_indices]
    
    print(f"      ✓ Assigned {len(test_cluster_labels):,} test samples using FAISS IVF!")
    
    # Memory cleanup
    del index, quantizer, training_norm, test_norm, all_indices, nearest_indices
    gc.collect()
    
    return test_cluster_labels


def fast_cluster_assignment_sklearn_batched(training_embeddings, training_cluster_labels,
                                            test_embeddings, use_cosine=True, batch_size=10000):
    """
    Fallback: Batched sklearn k-NN (slower but exact)
    
    Process test data in small batches to show progress
    """
    print(f"\n   Using batched sklearn k-NN (exact, slower)...")
    
    # Normalize if cosine
    if use_cosine:
        training_norm = normalize(np.array(training_embeddings), norm='l2', copy=True)
        test_norm = normalize(test_embeddings, norm='l2', copy=True)
        metric = 'cosine'
    else:
        training_norm = np.array(training_embeddings, dtype=np.float32)
        test_norm = test_embeddings.astype(np.float32)
        metric = 'euclidean'
    
    # Build k-NN model
    print(f"      Building k-NN index...")
    knn = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1, algorithm='auto')
    knn.fit(training_norm)
    
    print(f"      Searching {len(test_norm):,} test samples in batches of {batch_size:,}...")
    
    # Search in batches with progress bar
    all_indices = []
    n_batches = (len(test_norm) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(test_norm), batch_size), desc="      Searching", total=n_batches):
        batch_end = min(i + batch_size, len(test_norm))
        batch = test_norm[i:batch_end]
        
        distances, indices = knn.kneighbors(batch)
        all_indices.append(indices.flatten())
    
    # Combine results
    nearest_indices = np.concatenate(all_indices)
    test_cluster_labels = training_cluster_labels[nearest_indices]
    
    print(f"      ✓ Assigned {len(test_cluster_labels):,} test samples!")
    
    return test_cluster_labels


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Comprehensive Cluster Testing Pipeline
    
    Ground Truth: 2-CLASS (based on test file name)
        - normal_embeddings.npy → ALL = NORMAL (0)
        - nonnormal_embeddings.npy → ALL = NON-NORMAL (1)
    
    Predictions: 3-CLASS (based on cluster assignment)
        - NORMAL (0), NON-NORMAL (1), ANOMALY (2)
    
    Result: 2x3 confusion matrix showing prediction distribution
    """
    print("="*70)
    print("COMPREHENSIVE CLUSTER TESTING PIPELINE")
    print("2-CLASS GROUND TRUTH + 3-CLASS PREDICTION")
    print("="*70)
    print(f"\nDataset: {DATASET}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"Embedding: {EMBEDDING_TYPE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    
    # ========================================================================
    # CHECK FOR EXISTING CHECKPOINTS (RESUME CAPABILITY)
    # ========================================================================
    checkpoint_exists = all([
        CHECKPOINT_STEP6.exists(),
        CHECKPOINT_STEP7_PRED.exists(),
        CHECKPOINT_STEP7_CONF.exists(),
        CHECKPOINT_STEP7_METHODS.exists(),
        CHECKPOINT_METADATA.exists()
    ])
    
    if checkpoint_exists:
        print("\n" + "🔄 " + "="*68)
        print("CHECKPOINT DETECTED - RESUME FROM STEP 8")
        print("="*70)
        print("\n✅ Found existing checkpoint files:")
        print(f"   - {CHECKPOINT_STEP6.name}")
        print(f"   - {CHECKPOINT_STEP7_PRED.name}")
        print(f"   - {CHECKPOINT_STEP7_CONF.name}")
        print(f"   - {CHECKPOINT_STEP7_METHODS.name}")
        print(f"   - {CHECKPOINT_METADATA.name}")
        
        user_choice = input("\n⚠️  Resume from checkpoint? (y/n) [default: y]: ").strip().lower()
        if user_choice in ['', 'y', 'yes']:
            print("\n📂 Loading checkpoint data...")
            
            # Load checkpoint
            test_cluster_labels = np.load(CHECKPOINT_STEP6, allow_pickle=False)
            predictions = np.load(CHECKPOINT_STEP7_PRED, allow_pickle=False)
            confidence = np.load(CHECKPOINT_STEP7_CONF, allow_pickle=False)
            methods = np.load(CHECKPOINT_STEP7_METHODS, allow_pickle=True)
            
            with open(CHECKPOINT_METADATA, 'rb') as f:
                checkpoint_meta = pickle.load(f)
            
            test_gt_labels = checkpoint_meta['test_gt_labels']
            test_set_info = checkpoint_meta['test_set_info']
            cluster_df = checkpoint_meta['cluster_df']
            
            print(f"   ✓ Loaded test_cluster_labels: {len(test_cluster_labels):,}")
            print(f"   ✓ Loaded predictions: {len(predictions):,}")
            print(f"   ✓ Loaded confidence: {len(confidence):,}")
            print(f"   ✓ Loaded methods: {len(methods):,}")
            print(f"   ✓ Loaded test_gt_labels: {len(test_gt_labels):,}")
            print(f"   ✓ Loaded test_set_info: {len(test_set_info)} sets")
            print(f"   ✓ Loaded cluster_df: {len(cluster_df)} clusters")
            
            print("\n🚀 Skipping STEP 1-7, jumping to STEP 8...")
            
            # Jump directly to STEP 8
            goto_step_8 = True
        else:
            print("\n🔄 User chose to restart from beginning...")
            goto_step_8 = False
    else:
        print("\n📌 No checkpoint found - Starting from STEP 1")
        goto_step_8 = False
    # STEP 1-7: Run if no checkpoint or user chose restart
    # ========================================================================
    if not goto_step_8:
        # ====================================================================
        # STEP 1: Load training data with 3-way labels (for cluster analysis)
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 1: LOAD TRAINING DATA (3-way for cluster characterization)")
        print("="*70)
        
        training_gt_labels = load_metadata_labels_3way(
            METADATA_TSV_PATH,
            NORMAL_TEMPLATE_PATH,
            NONNORMAL_TEMPLATE_PATH
        )
        
        # ====================================================================
        # STEP 2: Load training cluster results
        # ====================================================================
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
        
        # ====================================================================
        # STEP 3: Analyze cluster characteristics
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 3: ANALYZE CLUSTER CHARACTERISTICS")
        print("="*70)
        
        cluster_df, cluster_dict = analyze_cluster_characteristics(
            training_cluster_labels, training_gt_labels
        )
        
        # Save cluster analysis
        cluster_df.to_csv(OUTPUT_CLUSTER_ANALYSIS, index=False)
        print(f"\n✓ Cluster analysis saved to: {OUTPUT_CLUSTER_ANALYSIS}")
        
        # ====================================================================
        # STEP 4: Load training embeddings (for k-NN)
        # ====================================================================
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
        
        # ====================================================================
        # STEP 5: Load testing data (2-class ground truth from file names)
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 5: LOAD TESTING DATA (2-class ground truth)")
        print("="*70)
        
        # Load multiple testing sets - ground truth based on file name
        test_embeddings, test_gt_labels, test_set_info = load_multiple_testing_sets(
            TESTING_SETS,
            NORMAL_TEMPLATE_PATH,
            NONNORMAL_TEMPLATE_PATH
        )
        
        # ====================================================================
        # STEP 6: Assign test samples to clusters
        # ====================================================================
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
            
            # Try FAISS IVF first (10-100x faster, ~98% accuracy)
            test_cluster_labels = fast_cluster_assignment_faiss(
                training_embeddings, 
                training_cluster_labels,
                test_embeddings,
                use_cosine=USE_COSINE_DISTANCE,
                nlist=1024,      # 1024 Voronoi cells (good for 1M+ samples)
                nprobe=64,       # Search 64 cells (good balance: speed vs accuracy)
                batch_size=50000 # Process 50K test samples at a time
            )
            
            # Fallback to batched sklearn if FAISS failed
            if test_cluster_labels is None:
                print("\n   ⚠️ FAISS unavailable or failed, using batched sklearn (slower but exact)...")
                test_cluster_labels = fast_cluster_assignment_sklearn_batched(
                    training_embeddings,
                    training_cluster_labels,
                    test_embeddings,
                    use_cosine=USE_COSINE_DISTANCE,
                    batch_size=10000  # Smaller batches for sklearn
                )
            
            print(f"   ✓ Cluster assignment complete!")
            print(f"   ✓ Assigned {len(test_cluster_labels):,} test samples to clusters")
        
        # SAVE CHECKPOINT AFTER STEP 6 (most expensive step)
        print(f"\n💾 Saving STEP 6 checkpoint...")
        np.save(CHECKPOINT_STEP6, test_cluster_labels)
        print(f"   ✓ Saved: {CHECKPOINT_STEP6.name}")
        
        # ====================================================================
        # STEP 7: Hybrid prediction (3-class output)
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 7: HYBRID PREDICTION (3-class output)")
        print("="*70)
        
        predictions, confidence, methods = hybrid_predict(
            test_cluster_labels, cluster_dict,
            training_embeddings, training_gt_labels,
            test_embeddings, use_knn=True
        )
        
        # Save predictions
        np.save(OUTPUT_PREDICTIONS, predictions)
        print(f"\n✓ Predictions saved to: {OUTPUT_PREDICTIONS}")
        
        # SAVE CHECKPOINT AFTER STEP 7 (prediction complete)
        print(f"\n💾 Saving STEP 7 checkpoint...")
        np.save(CHECKPOINT_STEP7_PRED, predictions)
        np.save(CHECKPOINT_STEP7_CONF, confidence)
        np.save(CHECKPOINT_STEP7_METHODS, methods)
        
        # Save metadata (needed for resume)
        checkpoint_meta = {
            'test_gt_labels': test_gt_labels,
            'test_set_info': test_set_info,
            'cluster_df': cluster_df
        }
        with open(CHECKPOINT_METADATA, 'wb') as f:
            pickle.dump(checkpoint_meta, f)
        
        print(f"   ✓ Saved: {CHECKPOINT_STEP7_PRED.name}")
        print(f"   ✓ Saved: {CHECKPOINT_STEP7_CONF.name}")
        print(f"   ✓ Saved: {CHECKPOINT_STEP7_METHODS.name}")
        print(f"   ✓ Saved: {CHECKPOINT_METADATA.name}")
        print(f"\n✅ All checkpoints saved! You can resume from STEP 8 if needed.")
    
    # ========================================================================
    # STEP 8: Calculate metrics (2-class ground truth vs 3-class predictions)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: CALCULATE METRICS (2x3 confusion matrix)")
    print("="*70)
    
    metrics = calculate_metrics(test_gt_labels, predictions, confidence, methods)
    
    # ========================================================================
    # STEP 9: Create detailed results CSV
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 9: SAVE DETAILED RESULTS")
    print("="*70)
    
    # Assign test set name to each sample
    # DEBUG: Print test_set_info to diagnose coverage issues
    print(f"\n🔍 Debugging test_set_info:")
    total_expected = len(test_gt_labels)
    for set_info in test_set_info:
        print(f"   {set_info['name']:12s}: [{set_info['start_idx']:7,} → {set_info['end_idx']:7,}] "
              f"= {set_info['n_samples']:,} samples")
    
    # Calculate total coverage
    total_covered = sum(s['n_samples'] for s in test_set_info)
    print(f"\n   Total expected:  {total_expected:,}")
    print(f"   Total coverage:  {total_covered:,}")
    
    # Validate indices are sequential and non-overlapping
    needs_rebuild = False
    if total_covered != total_expected:
        needs_rebuild = True
        print(f"   ⚠️ Total mismatch detected!")
    else:
        # Check for gaps/overlaps in indices
        sorted_sets = sorted(test_set_info, key=lambda x: x['start_idx'])
        for i, set_info in enumerate(sorted_sets):
            expected_start = 0 if i == 0 else sorted_sets[i-1]['end_idx']
            actual_start = set_info['start_idx']
            
            if actual_start != expected_start:
                needs_rebuild = True
                print(f"   ⚠️ Index mismatch for '{set_info['name']}': "
                      f"expected start={expected_start:,}, got {actual_start:,}")
                break
    
    if needs_rebuild:
        print(f"   ⚠️ MISMATCH: Rebuilding test_set_info from actual data...")
        
        # Rebuild from TESTING_SETS (ground truth configuration)
        test_set_info = []
        current_idx = 0
        for test_set in TESTING_SETS:
            name = test_set['name']
            
            # Count samples from metadata file (if available) or embeddings file
            if 'metadata' in test_set and test_set['metadata'].exists():
                metadata_path = test_set['metadata']
                # Quick count using pandas (memory efficient)
                df_temp = pd.read_csv(metadata_path, sep='\t', usecols=['label'])
                n_samples = len(df_temp)
                del df_temp
                print(f"      Rebuilt {name:12s}: [{current_idx:7,} → "
                      f"{current_idx + n_samples:7,}] = {n_samples:,} samples (from metadata)")
            elif 'embeddings' in test_set and test_set['embeddings'].exists():
                # Count from embeddings file
                embeddings_path = test_set['embeddings']
                emb = np.load(embeddings_path)
                n_samples = len(emb)
                del emb
                print(f"      Rebuilt {name:12s}: [{current_idx:7,} → "
                      f"{current_idx + n_samples:7,}] = {n_samples:,} samples (from embeddings)")
            else:
                print(f"      ⚠️ Cannot count samples for '{name}': no metadata or embeddings found")
                continue
            
            test_set_info.append({
                'name': name,
                'start_idx': current_idx,
                'end_idx': current_idx + n_samples,
                'n_samples': n_samples
            })
            current_idx += n_samples
        
        # Verify rebuild
        total_covered = sum(s['n_samples'] for s in test_set_info)
        if total_covered != total_expected:
            print(f"   ❌ ERROR: Still mismatch after rebuild! ({total_covered:,} vs {total_expected:,})")
            print(f"   → Will use 'unknown' for unassigned samples")
    
    # Create test_set_names array using vectorized approach (faster)
    print(f"\n📝 Assigning test set names...")
    test_set_names = np.full(len(test_gt_labels), 'unknown', dtype=object)
    
    for set_info in test_set_info:
        start = set_info['start_idx']
        end = set_info['end_idx']
        name = set_info['name']
        
        if end <= len(test_set_names):
            test_set_names[start:end] = name
            print(f"   ✓ {name:12s}: assigned {end-start:,} samples [{start:,} → {end:,})")
        else:
            print(f"   ⚠️ {name:12s}: end_idx ({end:,}) exceeds array length ({len(test_set_names):,})")
            # Assign what we can
            test_set_names[start:] = name
            print(f"      → Assigned {len(test_set_names)-start:,} samples instead")
    
    # Count unknowns
    n_unknown = np.sum(test_set_names == 'unknown')
    if n_unknown > 0:
        print(f"\n   ⚠️ WARNING: {n_unknown:,} samples assigned to 'unknown'")
    else:
        print(f"\n   ✓ All samples successfully assigned!")
    
    # Validate all arrays have same length
    print(f"\n📏 Validating array lengths:")
    print(f"   test_gt_labels:     {len(test_gt_labels):,}")
    print(f"   test_cluster_labels: {len(test_cluster_labels):,}")
    print(f"   predictions:        {len(predictions):,}")
    print(f"   confidence:         {len(confidence):,}")
    print(f"   methods:            {len(methods):,}")
    print(f"   test_set_names:     {len(test_set_names):,}")
    
    # Check for length mismatches
    expected_len = len(test_gt_labels)
    if not all(len(arr) == expected_len for arr in [test_cluster_labels, predictions, confidence, methods, test_set_names]):
        print(f"\n❌ ERROR: Length mismatch detected!")
        print(f"   Expected length: {expected_len:,}")
        print(f"   Truncating/padding arrays to match...")
        
        # Truncate to minimum length
        min_len = min(len(test_gt_labels), len(test_cluster_labels), len(predictions), 
                      len(confidence), len(methods), len(test_set_names))
        print(f"   Using minimum length: {min_len:,}")
        
        test_gt_labels = test_gt_labels[:min_len]
        test_cluster_labels = test_cluster_labels[:min_len]
        predictions = predictions[:min_len]
        confidence = confidence[:min_len]
        methods = methods[:min_len]
        test_set_names = test_set_names[:min_len]
    
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
    
    # Detect unique classes in ground truth
    unique_true = sorted(set(test_gt_labels))
    class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    
    for set_info in test_set_info:
        set_name = set_info['name']
        set_mask = results_df['test_set'] == set_name
        
        set_y_true = results_df[set_mask]['true_label'].values
        set_y_pred = results_df[set_mask]['predicted_label'].values
        
        # Per-class metrics for this set (dynamic based on ground truth classes)
        report_labels = [i for i in unique_true if i in [0, 1, 2]]
        report_names = [class_names[i] for i in report_labels]
        
        set_report = classification_report(
            set_y_true, set_y_pred,
            labels=report_labels,
            target_names=report_names,
            output_dict=True,
            zero_division=0
        )
        
        # Build metrics dict dynamically
        metrics_dict = {
            'test_set': set_name,
            'n_samples': int(set_mask.sum()),
            'accuracy': accuracy_score(set_y_true, set_y_pred),
        }
        
        # Add per-class metrics only for ground truth classes
        for label_idx, label_name in zip(report_labels, report_names):
            label_key = label_name.lower().replace('-', '')
            metrics_dict[f'{label_key}_precision'] = set_report[label_name]['precision']
            metrics_dict[f'{label_key}_recall'] = set_report[label_name]['recall']
            metrics_dict[f'{label_key}_f1'] = set_report[label_name]['f1-score']
        
        metrics_dict['macro_f1'] = set_report['macro avg']['f1-score']
        metrics_dict['weighted_f1'] = set_report['weighted avg']['f1-score']
        
        per_set_metrics.append(metrics_dict)
    
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
    # STEP 11: Detailed Prediction Distribution Analysis
    # ========================================================================
    distribution_stats = analyze_prediction_distribution(
        test_gt_labels, 
        predictions, 
        confidence, 
        methods,
        test_set_names
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("TESTING PIPELINE COMPLETED")
    print("="*70)
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print(f"\nKey files:")
    print(f"  - Predictions:           {OUTPUT_PREDICTIONS.name}")
    print(f"  - Cluster analysis:      {OUTPUT_CLUSTER_ANALYSIS.name}")
    print(f"  - Metrics:               {OUTPUT_METRICS.name}")
    print(f"  - Per-set metrics:       {OUTPUT_PER_SET_METRICS.name}")
    print(f"  - Detailed results:      {OUTPUT_DETAILED_RESULTS.name}")
    print(f"  - Prediction distribution: prediction_distribution.csv")
    print(f"  - Visualizations:        analysis_overview.png, confusion_matrix.png, prediction_distribution.png")
    
    # Detect unique ground truth and prediction classes for summary
    unique_true = sorted(set(test_gt_labels))
    unique_pred = sorted(set(predictions))
    class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    
    print(f"\n🎯 Final Results:")
    print(f"  Ground Truth: {len(unique_true)}-class → {[class_names[i] for i in unique_true]}")
    print(f"  Predictions:  {len(unique_pred)}-class → {[class_names[i] for i in unique_pred]}")
    print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
    
    # Print F1 scores only for classes present in ground truth
    for i in unique_true:
        label_name = class_names[i]
        if label_name in metrics['report']:
            print(f"  {label_name:12s} F1:  {metrics['report'][label_name]['f1-score']:.4f}")
    
    print(f"  Macro Avg F1:     {metrics['report']['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
