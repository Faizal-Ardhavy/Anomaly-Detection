"""
Comprehensive Cluster Testing Pipeline for Log Anomaly Detection
GROUND TRUTH: 2-Class (Normal / Non-Normal) based on test set name
PREDICTION: 3-Class (Normal / Non-Normal / Anomaly) based on cluster assignment

UNSUPERVISED APPROACH - NO TEMPLATE-BASED LABELING!

Strategy:
1. Load ground truth labels based on test set name:
   - Test set 'normal' → ALL samples = NORMAL (0)
   - Test set 'nonnormal' → ALL samples = NON-NORMAL (1)

2. Analyze training cluster characteristics (UNSUPERVISED):
   - Size-based classification (noise/very_small/small/regular)
   - Silhouette Score: measures cluster cohesion and separation (-1 to +1)
     * +1: excellent cluster quality
     * 0: overlapping clusters
     * -1: misassigned samples

3. Hybrid prediction strategy for TEST samples:
   - Noise points (DBSCAN) → ANOMALY (label=2)
   - Very small clusters (<50) → ANOMALY (label=2)
   - Small clusters (50-200) → NON-NORMAL (label=1)
   - Regular clusters (≥200) → k-NN vote using cluster IDs as pseudo-labels

4. k-NN voting uses cluster IDs (not ground truth labels):
   - Neighbor's cluster type determines vote
   - noise/very_small → ANOMALY
   - small → NON-NORMAL
   - regular → NORMAL

5. Calculate 2x3 metrics: 2 ground truth classes, 3 predicted classes

6. Visualize: Cluster sizes, silhouette scores, confusion matrix, prediction distribution

Supports:
- K-Means and DBSCAN
- BGL and Thunderbird datasets  
- Base/PCA256/PCA128 embeddings
- Ground truth from test set names ONLY (templates removed!)
- Silhouette Score for cluster quality assessment
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
    confusion_matrix, classification_report, silhouette_score, silhouette_samples
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
    TRAINED_MODEL_PATH = Path("kmeans/bgl_base_k_params/model_kmeans_log.pkl")
    TRAINING_LABELS_PATH = Path("kmeans/bgl_base_k_params/cluster_labels.npy")
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
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector/after_preprocessed_bgl_normal_embeddings.npy"),
        'metadata': Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_bgl_normal_meta.tsv")  # Optional, only needed if you want to do template-based analysis
    },
    {
        'name': 'nonnormal',  # Ground truth: ALL = NON-NORMAL (1)
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector/after_preprocessed_bgl_non_normal_embeddings.npy"),
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

# Semi-supervised cluster labeling strategy
USE_METADATA_LABELING = True        # Use training metadata to label clusters (RECOMMENDED)
METADATA_SAMPLE_SIZE = 1000         # Samples per cluster for metadata check (or full if smaller)
MAJORITY_THRESHOLD = 0.70           # ≥70% majority → assign that class label

# Robust metadata-labeling safeguards (helps avoid all-normal collapse)
FORCE_DBSCAN_NOISE_AS_ANOMALY = True
MIN_METADATA_EVIDENCE = 30          # Minimum known-label samples before trusting metadata vote
NORMAL_STRONG_THRESHOLD = 0.85      # Strong normal evidence required for NORMAL assignment
NONNORMAL_PRESENCE_THRESHOLD = 0.15 # If >=15% non-normal evidence, mark as NON-NORMAL
AMBIGUOUS_CLUSTER_AS_ANOMALY = True

# Legacy threshold (used only when metadata labeling is disabled)
MIN_CLUSTER_SIZE_FOR_LABELING = 50

# K-Means adaptive size rules (recommended for large datasets)
# Example: 0.2% of 3.9M ≈ 7,800 samples
KMEANS_ANOMALY_CLUSTER_RATIO = 0.002   # Cluster < 0.2% of training set → ANOMALY
KMEANS_SMALL_CLUSTER_RATIO = 0.005     # Cluster < 0.5% of training set → NON-NORMAL (legacy mode)
# Additional guardrail using average cluster size (helps when K changes a lot)
KMEANS_ANOMALY_AVG_CLUSTER_RATIO = 0.04  # < 4% of avg cluster size → ANOMALY candidate
KMEANS_SMALL_AVG_CLUSTER_RATIO = 0.12    # < 12% of avg cluster size → NON-NORMAL candidate

# Legacy: Size-based classification (if USE_METADATA_LABELING = False)
VERY_SMALL_CLUSTER_THRESHOLD = 50   # < 50 samples → ANOMALY
SMALL_CLUSTER_THRESHOLD = 200       # 50-200 samples → NON-NORMAL
PURITY_THRESHOLD_HIGH = 0.95        # Legacy pure-cluster threshold
PURITY_THRESHOLD_MEDIUM = 0.70      # Legacy medium-purity threshold

# Legacy: k-NN parameters (not needed with metadata labeling)
KNN_NEIGHBORS = 10                  # For k-NN vote in ambiguous cases
KNN_HIGH_CONFIDENCE = 0.80          # 8/10 vote = high confidence
KNN_MEDIUM_CONFIDENCE = 0.60        # 6/10 vote = medium confidence

USE_COSINE_DISTANCE = True          # Normalize embeddings (recommended for BERT)

# Distance-based reject option to avoid forced assignments into known clusters
ENABLE_DISTANCE_REJECTION = True
REJECTION_REFERENCE_SAMPLE_SIZE = 200_000
REJECTION_QUANTILE = 0.995
REJECTION_DISTANCE_MULTIPLIER = 1.10

# Large dataset optimization parameters
SUBSAMPLE_KNN_TRAINING = True       # Subsample training data for k-NN (for huge datasets)
KNN_SUBSAMPLE_SIZE = 1_000_000      # Max training samples for k-NN (1M samples)
NORMALIZE_INPLACE = True            # Use copy=False to save memory during normalization

# GPU acceleration for Silhouette Score (optional - uses external rapids-pip kernel)
USE_GPU_SILHOUETTE = True           # Try GPU via rapids-pip kernel, fallback to CPU
GPU_KERNEL_NAME = "rapids-pip"      # Jupyter kernel name with cuML installed

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
# GPU ACCELERATION UTILITIES
# ============================================================================

def get_kernel_python_path(kernel_name="rapids-pip"):
    """
    Get Python executable path from Jupyter kernel specification
    Returns None if kernel not found or error occurs
    """
    import subprocess
    import json
    
    try:
        # Get kernel spec directory
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            return None
        
        specs = json.loads(result.stdout)
        if kernel_name not in specs.get("kernelspecs", {}):
            return None
        
        kernel_dir = Path(specs["kernelspecs"][kernel_name]["resource_dir"])
        kernel_json = kernel_dir / "kernel.json"
        
        if not kernel_json.exists():
            return None
        
        with open(kernel_json, 'r') as f:
            kernel_config = json.load(f)
        
        # argv[0] is usually the Python executable
        python_path = kernel_config.get("argv", [])[0] if kernel_config.get("argv") else None
        
        if python_path and Path(python_path).exists():
            return python_path
        
        return None
        
    except Exception as e:
        print(f"⚠ Could not detect {kernel_name} kernel: {e}")
        return None


def compute_silhouette_gpu(
    embeddings, labels, 
    kernel_name="rapids-pip",
    sample_size=None
):
    """
    Compute silhouette score using GPU via external rapids-pip Jupyter kernel
    Falls back to CPU sklearn if GPU computation fails
    
    Args:
        embeddings: np.ndarray of shape (n_samples, n_features)
        labels: np.ndarray of cluster labels
        kernel_name: Name of Jupyter kernel with cuML installed
        sample_size: Optional, subsample for large datasets
        
    Returns:
        overall_score: float, overall silhouette score
        sample_scores: np.ndarray, per-sample silhouette scores
    """
    import subprocess
    import tempfile
    import json
    
    # Try GPU computation
    try:
        print(f"🚀 Attempting GPU silhouette computation via {kernel_name} kernel...")
        
        # Get Python path from kernel
        python_path = get_kernel_python_path(kernel_name)
        if not python_path:
            raise RuntimeError(f"Kernel {kernel_name} not found or invalid")
        
        print(f"   Found Python: {python_path}")
        
        # Create temporary directory for data exchange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Handle sampling if needed
            if sample_size and len(embeddings) > sample_size:
                print(f"   Sampling {sample_size:,} from {len(embeddings):,} samples...")
                indices = np.random.choice(len(embeddings), sample_size, replace=False)
                emb_subset = embeddings[indices]
                lbl_subset = labels[indices]
            else:
                emb_subset = embeddings
                lbl_subset = labels
                indices = None
            
            # Save data to temporary files
            emb_path = tmpdir / "embeddings.npy"
            lbl_path = tmpdir / "labels.npy"
            result_path = tmpdir / "result.npz"
            
            np.save(emb_path, emb_subset)
            np.save(lbl_path, lbl_subset)
            
            # Create GPU computation script
            script_path = tmpdir / "compute_gpu_silhouette.py"
            with open(script_path, 'w') as f:
                f.write(f'''import numpy as np
import sys

try:
    from cuml.metrics import silhouette_score, silhouette_samples
    print("✓ cuML imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"✗ cuML import failed: {{e}}", file=sys.stderr)
    sys.exit(1)

# Load data
emb = np.load("{emb_path}")
lbl = np.load("{lbl_path}")
print(f"✓ Loaded {{len(emb):,}} embeddings, {{len(np.unique(lbl))}} clusters", file=sys.stderr)

# Compute silhouette on GPU
print("Computing silhouette score on GPU...", file=sys.stderr)
overall = float(silhouette_score(emb, lbl))
samples = silhouette_samples(emb, lbl)
print(f"✓ GPU computation complete: {{overall:.4f}}", file=sys.stderr)

# Save results
np.savez("{result_path}", overall=overall, samples=samples)
print("✓ Results saved", file=sys.stderr)
''')
            
            # Execute with rapids-pip Python
            print(f"   Launching GPU computation...")
            result = subprocess.run(
                [python_path, str(script_path)],
                capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )
            
            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                raise RuntimeError(f"GPU script failed: {error_msg}")
            
            # Print GPU process output
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    print(f"   {line}")
            
            # Load results
            if not result_path.exists():
                raise RuntimeError("GPU computation produced no output file")
            
            data = np.load(result_path)
            overall_score = float(data['overall'])
            sample_scores_subset = data['samples']
            
            # Expand to full size if we sampled
            if indices is not None:
                sample_scores = np.zeros(len(embeddings))
                sample_scores[indices] = sample_scores_subset
            else:
                sample_scores = sample_scores_subset
            
            print(f"✅ GPU silhouette successful: {overall_score:.4f}")
            return overall_score, sample_scores
            
    except Exception as e:
        print(f"⚠ GPU computation failed: {e}")
        print(f"⚠ Falling back to CPU sklearn computation...")
        
        # Fallback to CPU
        if sample_size and len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            overall_score = silhouette_score(embeddings[indices], labels[indices])
            sample_scores_subset = silhouette_samples(embeddings[indices], labels[indices])
            sample_scores = np.zeros(len(embeddings))
            sample_scores[indices] = sample_scores_subset
        else:
            overall_score = silhouette_score(embeddings, labels)
            sample_scores = silhouette_samples(embeddings, labels)
        
        print(f"✓ CPU silhouette complete: {overall_score:.4f}")
        return overall_score, sample_scores


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_label_token(value) -> str:
    """Normalize metadata/template label token for robust matching."""
    if pd.isna(value):
        return ''
    return str(value).strip().upper()

def sample_non_noise_indices(cluster_labels, non_noise_mask, sample_size, seed=42):
    """
    Sample non-noise indices efficiently for very large arrays.

    For high non-noise ratios, avoid building full `np.flatnonzero(non_noise_mask)`
    first (which can be slow and memory-heavy on 100M+ rows). Falls back to the
    flatnonzero path when needed.
    """
    rng = np.random.default_rng(seed)
    total_n = len(cluster_labels)
    n_non_noise = int(non_noise_mask.sum())

    if n_non_noise <= sample_size:
        return np.flatnonzero(non_noise_mask)

    if n_non_noise == total_n:
        return np.sort(rng.choice(total_n, sample_size, replace=False))

    ratio = n_non_noise / max(total_n, 1)

    # Fast path for huge datasets with very small noise proportion.
    if total_n >= 10_000_000 and ratio >= 0.95:
        target_pool = int(sample_size * 1.5)
        collected = []
        collected_n = 0

        for _ in range(30):
            need = target_pool - collected_n
            if need <= 0:
                break

            draw = max(sample_size, int((need / max(ratio, 1e-9)) * 1.2))
            draw = min(draw, total_n)

            candidate = rng.integers(0, total_n, size=draw, dtype=np.int64)
            valid = candidate[cluster_labels[candidate] != -1]
            if len(valid) > 0:
                collected.append(valid)
                collected_n += len(valid)

        if collected_n > 0:
            pool = np.unique(np.concatenate(collected))
            if len(pool) >= sample_size:
                return np.sort(rng.choice(pool, sample_size, replace=False))

        print("   ⚠️ Fast sampler pool not enough, fallback to flatnonzero...")

    non_noise_indices = np.flatnonzero(non_noise_mask)
    return np.sort(rng.choice(non_noise_indices, sample_size, replace=False))

def load_kmeans_model_compat(model_path: Path):
    """
    Load K-Means model with compatibility fallback for NumPy RNG pickle format
    differences across environments/versions.
    """
    try:
        return joblib.load(model_path)
    except ValueError as e:
        msg = str(e)
        # Common cross-version issue:
        # ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module.
        if 'BitGenerator module' in msg or 'MT19937' in msg:
            print("   ⚠️ Detected NumPy RNG pickle compatibility issue. Retrying with compatibility patch...")
            try:
                import numpy.random._pickle as np_random_pickle

                original_ctor = np_random_pickle.__bit_generator_ctor

                def _compat_bit_generator_ctor(bit_generator_name):
                    # Older pickles may store class objects instead of string names.
                    if isinstance(bit_generator_name, type):
                        bit_generator_name = bit_generator_name.__name__
                    elif hasattr(bit_generator_name, '__name__'):
                        bit_generator_name = bit_generator_name.__name__
                    return original_ctor(bit_generator_name)

                np_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
                model = joblib.load(model_path)
                print("   ✓ Model loaded with NumPy compatibility patch")
                return model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model due to NumPy compatibility issue. "
                    f"Try using the same NumPy version used during training. Details: {e2}"
                ) from e2
        raise


def find_kmeans_training_embeddings_for_dim(original_path: Path, dataset: str,
                                            embedding_type: str, target_dim: int):
    """
    Try common path patterns to locate training embeddings with the required feature dimension.
    Returns tuple(path, memmap_array) or None.
    """
    dataset_name = dataset.lower()
    emb_type = embedding_type.lower()

    candidates = [original_path]
    original_str = str(original_path)

    if emb_type != "base":
        candidates.extend([
            Path(original_str.replace("_dataset_vector/", f"_dataset_vector_{emb_type}/")),
            Path(original_str.replace("_dataset_vector/", f"_dataset_vector_{emb_type}_testing/")),
            Path(original_str.replace(
                f"after_preprocessed_{dataset_name}_embeddings.npy",
                f"after_preprocessed_{dataset_name}_{emb_type}_embeddings.npy"
            )),
            Path(f"/media/bioinfo04/Expansion/2427051003_dataset_vector_{emb_type}/after_preprocessed_{dataset_name}_{emb_type}_embeddings.npy"),
            Path(f"/media/bioinfo04/Expansion/2427051003_dataset_vector_{emb_type}/after_preprocessed_{dataset_name}_embeddings.npy"),
        ])

    seen = set()
    for path in candidates:
        norm = str(path)
        if norm in seen:
            continue
        seen.add(norm)

        if not path.exists():
            continue

        try:
            emb = np.load(path, mmap_mode='r')
            if emb.ndim == 2 and emb.shape[1] == target_dim:
                return path, emb
        except Exception:
            continue

    return None


def build_kmeans_centroids_from_labels(training_embeddings, training_cluster_labels,
                                       chunk_size=300000):
    """
    Build cluster centroids from embeddings and precomputed cluster labels using chunked streaming.
    """
    labels = np.asarray(training_cluster_labels)
    unique_clusters = sorted(int(c) for c in np.unique(labels) if int(c) != -1)
    if not unique_clusters:
        raise RuntimeError("No valid cluster IDs found in training labels for centroid fallback")

    cluster_ids = np.array(unique_clusters, dtype=np.int64)
    cluster_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    n_features = int(training_embeddings.shape[1])
    sums = np.zeros((len(cluster_ids), n_features), dtype=np.float64)
    counts = np.zeros(len(cluster_ids), dtype=np.int64)

    total = len(labels)
    for start in tqdm(range(0, total, chunk_size), desc="   Building centroids"):
        end = min(start + chunk_size, total)
        emb_chunk = np.asarray(training_embeddings[start:end], dtype=np.float32)
        lbl_chunk = np.asarray(labels[start:end], dtype=np.int64)

        for cid in np.unique(lbl_chunk):
            if cid == -1:
                continue
            idx = cluster_to_idx.get(int(cid))
            if idx is None:
                continue

            mask = (lbl_chunk == cid)
            if np.any(mask):
                sums[idx] += emb_chunk[mask].sum(axis=0, dtype=np.float64)
                counts[idx] += int(mask.sum())

    valid = counts > 0
    if not np.any(valid):
        raise RuntimeError("All centroid counts are zero. Check training labels/embeddings alignment.")

    centroids = (sums[valid] / counts[valid, None]).astype(np.float32)
    cluster_ids = cluster_ids[valid].astype(np.int32)
    return cluster_ids, centroids


def predict_kmeans_from_centroids(test_embeddings, cluster_ids, centroids,
                                  use_cosine=True, batch_size=20000):
    """
    Predict cluster IDs by nearest centroid in batches to keep RAM usage stable.
    """
    n_test = len(test_embeddings)
    predictions = np.empty(n_test, dtype=np.int32)

    if use_cosine:
        centroids_ref = normalize(centroids.astype(np.float32), norm='l2', copy=True)
    else:
        centroids_ref = centroids.astype(np.float32)
        centroids_sq = np.sum(centroids_ref * centroids_ref, axis=1)[None, :]

    for start in tqdm(range(0, n_test, batch_size), desc="   Assigning by centroids"):
        end = min(start + batch_size, n_test)
        batch = np.asarray(test_embeddings[start:end], dtype=np.float32)

        if use_cosine:
            batch = normalize(batch, norm='l2', copy=False)
            scores = batch @ centroids_ref.T
            best = np.argmax(scores, axis=1)
        else:
            dot = batch @ centroids_ref.T
            batch_sq = np.sum(batch * batch, axis=1, keepdims=True)
            dist2 = batch_sq + centroids_sq - 2.0 * dot
            best = np.argmin(dist2, axis=1)

        predictions[start:end] = cluster_ids[best]

    return predictions

def load_template_events(template_path: Path) -> set:
    """
    Load Label set from template TSV file
    
    Returns: set of Labels (e.g., {'-', 'APPREAD', 'KERNDTLB', ...})
    """
    print(f"   Loading template: {template_path.name}")
    
    if not template_path.exists():
        print(f"   ⚠️ Template file not found: {template_path}")
        return set()

    # Try to read with a TSV header first
    try:
        df = pd.read_csv(template_path, sep='\t', dtype=str, engine='python', encoding='utf-8')
    except Exception:
        # Fallback: we'll inspect the file manually
        df = None

    # If dataframe loaded and has a Label-like column (case-insensitive), use it
    if df is not None:
        label_col = None
        for col in df.columns:
            if str(col).strip().lower() == 'label':
                label_col = col
                break
        if label_col is not None:
            label_set = set(df[label_col].map(normalize_label_token))
            label_set.discard('')
            print(f"   ✓ Found {len(label_set)} unique Labels (column: {label_col})")
            return label_set

    # If we reach here, either pandas didn't parse header correctly (e.g., file has leading metadata lines)
    # or there is no header. Scan file to find a header line containing 'Label' or fallback to first column values.
    header_idx = None
    with open(template_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Look for a tab-separated header that includes 'Label'
            if '\t' in line_stripped:
                cols = [c.strip().lower() for c in line_stripped.split('\t')]
                if 'label' in cols:
                    header_idx = i
                    break

    if header_idx is not None:
        try:
            df2 = pd.read_csv(template_path, sep='\t', header=0, skiprows=header_idx, dtype=str, engine='python', encoding='utf-8')
            # Find label column name (original case)
            label_col = None
            for col in df2.columns:
                if str(col).strip().lower() == 'label':
                    label_col = col
                    break
            if label_col is not None:
                label_set = set(df2[label_col].map(normalize_label_token))
                label_set.discard('')
                print(f"   ✓ Found {len(label_set)} unique Labels (header at line {header_idx+1})")
                return label_set
        except Exception as e:
            print(f"   ⚠️ Failed to parse header at line {header_idx+1}: {e}")

    # Last resort: treat the file as plain lines and extract first tab-separated column (skip obvious metadata lines)
    labels = []
    with open(template_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Skip obvious metadata summary lines (e.g., "Total ... = <num>")
            if s.lower().startswith('total') and '=' in s:
                continue
            # If tab-separated, take first column, else whole line
            if '\t' in s:
                labels.append(s.split('\t', 1)[0].strip())
            else:
                labels.append(s)

    label_set = set(labels)
    label_set = {normalize_label_token(v) for v in label_set}
    label_set.discard('')
    print(f"   ✓ Inferred {len(label_set)} unique Labels (plain-line fallback)")
    return label_set
    
    # df = pd.read_csv(template_path, sep='\t')
    
    # if 'Label' not in df.columns:
    #     raise ValueError(f"Label column not found in {template_path}")
    
    # label_set = set(df['Label'].unique())
    # print(f"   ✓ Found {len(label_set)} unique Labels")
    
    # return label_set


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
        normalized_label = normalize_label_token(label_val)
        if normalized_label == '':
            labels.append(2)  # Missing/empty label = ANOMALY
            stats['unknown'] += 1
        elif normalized_label in normal_events:
            labels.append(0)  # NORMAL
            stats['normal'] += 1
        elif normalized_label in nonnormal_events:
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
            normalized_label = normalize_label_token(label_val)
            if normalized_label == '':
                labels.append(2)
                stats['unknown'] += 1
            elif normalized_label in normal_events:
                labels.append(0)
                stats['normal'] += 1
            elif normalized_label in nonnormal_events:
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


def build_metadata_label_memmap(tsv_path: Path, normal_events: set, nonnormal_events: set,
                                memmap_path: Path = None, chunksize: int = 1_000_000):
    """
    Create (or reuse) a disk-backed uint8 memmap with one label per metadata row:
      0 = NORMAL, 1 = NON-NORMAL, 2 = ANOMALY/unknown

    This function is streaming-friendly and avoids loading the whole TSV into RAM.
    It does two streaming passes (count lines, then parse by chunks) but uses minimal memory.
    Returns a numpy memmap-like array for fast index access.
    """
    import os, json
    from numpy.lib.format import open_memmap

    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {tsv_path}")

    if memmap_path is None:
        memmap_path = CHECKPOINT_DIR / (tsv_path.name + '.labels.npy')
    meta_path = memmap_path.with_suffix('.meta.json')

    tsv_mtime = os.path.getmtime(tsv_path)

    # Reuse existing memmap if metadata unchanged
    if memmap_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get('tsv_mtime') == tsv_mtime:
                print(f"   ✓ Reusing existing metadata labels memmap: {memmap_path.name}")
                return open_memmap(memmap_path, mode='r')
        except Exception:
            pass

    # Fast line count (binary read)
    print("   ⚙️ Counting metadata rows (fast scan)...")
    with open(tsv_path, 'rb') as f:
        n_bytes = 1024 * 1024
        total_lines = 0
        for chunk in iter(lambda: f.read(n_bytes), b''):
            total_lines += chunk.count(b'\n')

    # Detect header presence and label column name (cheap sample)
    label_col = None
    header_rows = 0
    try:
        df_head = pd.read_csv(tsv_path, sep='\t', nrows=0)
        for col in df_head.columns:
            if str(col).strip().lower() == 'label':
                label_col = col
                break
    except Exception:
        # fallback: inspect first 10 text lines
        with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
            for i in range(10):
                line = f.readline()
                if not line:
                    break
                if '\t' in line and 'label' in line.lower():
                    label_col = 'label'
                    break

    # Compute effective data rows (subtract header if detected)
    if label_col is not None:
        data_rows = total_lines - 1
    else:
        data_rows = total_lines

    print(f"   ✓ Detected ~{data_rows:,} metadata rows")

    # Create memmap file
    print(f"   ⚙️ Creating memmap ({memmap_path.name}) with dtype=uint8")
    mm = open_memmap(memmap_path, mode='w+', dtype='uint8', shape=(data_rows,))

    # Stream-parse label column in chunks and populate memmap
    read_cols = [label_col] if label_col is not None else None
    pos = 0
    chunk_iter = pd.read_csv(tsv_path, sep='\t', usecols=read_cols, dtype=str, chunksize=chunksize)
    for chunk_num, chunk in enumerate(chunk_iter, 1):
        if label_col is None:
            # If no header/label column, assume first column
            series = chunk.iloc[:, 0].map(normalize_label_token)
        else:
            series = chunk[label_col].map(normalize_label_token)

        # Vectorized mapping: NORMAL=0, NON-NORMAL=1, else 2
        series = series.fillna('')
        isin_normal = series.isin(normal_events)
        isin_nonnormal = series.isin(nonnormal_events)
        arr = np.where(isin_normal, 0, np.where(isin_nonnormal, 1, 2)).astype('uint8')
        L = len(arr)
        mm[pos:pos+L] = arr
        pos += L

        if chunk_num % 10 == 0:
            print(f"      Processed {pos:,}/{data_rows:,} rows ({chunk_num} chunks)")

    mm.flush()

    # Save metadata for reuse
    try:
        meta = {'tsv_mtime': tsv_mtime, 'n_rows': int(data_rows)}
        meta_path.write_text(json.dumps(meta))
    except Exception:
        pass

    print(f"   ✓ Built metadata labels memmap: {memmap_path.name}")
    return mm


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
    current_idx = 0
    
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
            'start_idx': current_idx,
            'end_idx': current_idx + len(labels),
            'n_samples': len(labels)
        })
        
        all_embeddings.append(embeddings)
        all_labels.append(labels)
        current_idx += len(labels)
    
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


def analyze_cluster_characteristics(cluster_labels, embeddings=None, 
                                    compute_silhouette=True, 
                                    silhouette_sample_size=1000000,
                                    metadata_tsv_path=None,
                                    normal_template_path=None,
                                    nonnormal_template_path=None):
    """
    Analyze each cluster with SEMI-SUPERVISED metadata-based labeling
    
    Args:
        cluster_labels: Cluster assignments for each sample
        embeddings: Sample embeddings (needed for silhouette score)
        compute_silhouette: Whether to compute silhouette scores (default: True)
        silhouette_sample_size: Max samples for silhouette (default: 100K for speed)
        metadata_tsv_path: Path to training metadata TSV (for metadata labeling)
        normal_template_path: Path to normal event template
        nonnormal_template_path: Path to non-normal event template
    
    Strategy:
        IF USE_METADATA_LABELING = True (RECOMMENDED):
            1. Load training metadata TSV
            2. For each cluster: sample → check metadata → majority vote
            3. Assign cluster_label: NORMAL (0), NON-NORMAL (1), or ANOMALY (2)
        
        ELSE (Legacy size-based):
            1. Size < 50 → ANOMALY
            2. Size 50-200 → NON-NORMAL  
            3. Size ≥ 200 → Needs k-NN (unreliable)
    
    Silhouette Score:
    - Range: -1 to +1
    - +1: Sample very well matched to cluster, far from others (excellent)
    - 0: Sample on boundary between clusters (ambiguous)
    - -1: Sample possibly assigned to wrong cluster (poor)
    
    Returns:
    - DataFrame with cluster statistics including cluster_label
    - Dict with cluster_id → cluster_info
    """
    if USE_METADATA_LABELING:
        print("\n🔍 Analyzing cluster characteristics (SEMI-SUPERVISED: Metadata-based labeling)...")
    else:
        print("\n🔍 Analyzing cluster characteristics (UNSUPERVISED: Size-based)...")
    
    unique_clusters = sorted(set(cluster_labels))
    cluster_info = []
    total_training_samples = len(cluster_labels)

    # Size thresholds are only used in legacy (non-metadata) mode.
    if not USE_METADATA_LABELING:
        # Adaptive thresholds for K-Means (avoid fixed tiny thresholds on large datasets)
        if ALGORITHM == "kmeans":
            n_kmeans_clusters = len(set(cluster_labels) - {-1})
            avg_cluster_size = total_training_samples / max(n_kmeans_clusters, 1)

            # Base thresholds by dataset scale
            base_anomaly_threshold = max(
                MIN_CLUSTER_SIZE_FOR_LABELING,
                int(total_training_samples * KMEANS_ANOMALY_CLUSTER_RATIO)
            )
            base_small_threshold = max(
                SMALL_CLUSTER_THRESHOLD,
                int(total_training_samples * KMEANS_SMALL_CLUSTER_RATIO)
            )

            # Guardrail thresholds by average cluster size (sensitive to K)
            avg_anomaly_threshold = max(
                MIN_CLUSTER_SIZE_FOR_LABELING,
                int(avg_cluster_size * KMEANS_ANOMALY_AVG_CLUSTER_RATIO)
            )
            avg_small_threshold = max(
                SMALL_CLUSTER_THRESHOLD,
                int(avg_cluster_size * KMEANS_SMALL_AVG_CLUSTER_RATIO)
            )

            # Final thresholds: choose the stricter (smaller) boundary to avoid over-labeling on huge datasets
            adaptive_anomaly_threshold = min(base_anomaly_threshold, avg_anomaly_threshold)
            adaptive_small_threshold = min(base_small_threshold, avg_small_threshold)
            adaptive_small_threshold = max(adaptive_small_threshold, adaptive_anomaly_threshold + 1)

            print(f"   K-Means adaptive thresholds:")
            print(f"      Train samples: {total_training_samples:,}, clusters: {n_kmeans_clusters:,}, avg cluster size: {avg_cluster_size:,.0f}")
            print(f"      Base anomaly threshold: {base_anomaly_threshold:,} ({KMEANS_ANOMALY_CLUSTER_RATIO*100:.2f}% of train)")
            print(f"      Avg-based anomaly threshold: {avg_anomaly_threshold:,} ({KMEANS_ANOMALY_AVG_CLUSTER_RATIO*100:.2f}% of avg cluster)")
            print(f"      FINAL ANOMALY if size < {adaptive_anomaly_threshold:,}")
            print(f"      Base non-normal threshold: {base_small_threshold:,} ({KMEANS_SMALL_CLUSTER_RATIO*100:.2f}% of train)")
            print(f"      Avg-based non-normal threshold: {avg_small_threshold:,} ({KMEANS_SMALL_AVG_CLUSTER_RATIO*100:.2f}% of avg cluster)")
            print(f"      FINAL NON-NORMAL if size < {adaptive_small_threshold:,}")
        else:
            adaptive_anomaly_threshold = MIN_CLUSTER_SIZE_FOR_LABELING
            adaptive_small_threshold = SMALL_CLUSTER_THRESHOLD
    else:
        adaptive_anomaly_threshold = None
        adaptive_small_threshold = None
        print(f"   Metadata labeling mode: robust majority voting")
        print(f"      MAJORITY_THRESHOLD={MAJORITY_THRESHOLD:.0%}")
        print(f"      NORMAL_STRONG_THRESHOLD={NORMAL_STRONG_THRESHOLD:.0%}")
        print(f"      NONNORMAL_PRESENCE_THRESHOLD={NONNORMAL_PRESENCE_THRESHOLD:.0%}")
        print(f"      MIN_METADATA_EVIDENCE={MIN_METADATA_EVIDENCE}")
    
    # Load metadata for label assignment (if metadata labeling enabled)
    metadata_df = None
    normal_events = None
    nonnormal_events = None
    metadata_labels = None
    
    if USE_METADATA_LABELING and metadata_tsv_path and normal_template_path and nonnormal_template_path:
        print(f"\n📖 Loading training metadata for cluster labeling...")
        
        # Load templates
        print(f"   Loading templates...")
        normal_events = load_template_events(normal_template_path)
        nonnormal_events = load_template_events(nonnormal_template_path)
        print(f"   Template coverage: NORMAL={len(normal_events):,}, NON-NORMAL={len(nonnormal_events):,}")
        if len(normal_events) <= 1 or len(nonnormal_events) <= 1:
            print("   ⚠️ Very low template cardinality detected. Check template format and label column parsing.")
        
        # Load metadata TSV
        print(f"   Loading metadata TSV: {metadata_tsv_path.name}")
        if not metadata_tsv_path.exists():
            print(f"   ⚠️ Metadata file not found, falling back to size-based classification")
        else:
            # Use streaming memmap ONLY for Thunderbird (huge dataset); keep BGL on legacy path
            if DATASET.lower() == 'thunderbird':
                try:
                    # Build or reuse a compact uint8 memmap with per-row labels (0=Normal,1=NonNormal,2=Anomaly)
                    metadata_labels = build_metadata_label_memmap(metadata_tsv_path, normal_events, nonnormal_events)
                    print(f"   ✓ Metadata labels memmap shape: {metadata_labels.shape}")
                    metadata_df = None
                except Exception as e:
                    print(f"   ⚠️ Streaming memmap build failed: {e}. Falling back to safe chunked load.")
                    try:
                        metadata_df = pd.read_csv(metadata_tsv_path, sep='\t', usecols=['label'], dtype=str)
                        print(f"   ✓ Loaded {len(metadata_df):,} metadata rows (labels only)")
                        metadata_labels = None
                    except Exception as e2:
                        print(f"   ❌ Failed to load metadata TSV: {e2}. Falling back to size-based classification")
                        metadata_df = None
                        metadata_labels = None
            else:
                # BGL or other datasets: keep legacy behavior (try to load label column into memory first)
                try:
                    metadata_df = pd.read_csv(metadata_tsv_path, sep='\t', usecols=['label'], dtype=str)
                    print(f"   ✓ Loaded {len(metadata_df):,} metadata rows (labels only, legacy mode)")
                    metadata_labels = None
                except Exception as e:
                    print(f"   ⚠️ Loading label-only failed ({e}), trying chunked streaming as fallback")
                    try:
                        labels_array = _load_metadata_chunked(metadata_tsv_path, normal_events, nonnormal_events, chunksize=1000000)
                        metadata_labels = labels_array
                        metadata_df = None
                        print(f"   ✓ Loaded labels via chunked streaming: {len(labels_array):,} rows")
                    except Exception as e2:
                        print(f"   ❌ Failed to load metadata via chunked streaming: {e2}. Falling back to size-based classification")
                        metadata_df = None
                        metadata_labels = None
    
    # Calculate silhouette scores if embeddings provided
    overall_silhouette = None
    cluster_silhouette_lookup = {}
    
    if compute_silhouette and embeddings is not None:
        print(f"\n   Computing Silhouette scores...")
        
        # Filter out noise points (-1) as silhouette doesn't apply to noise
        non_noise_mask = (cluster_labels != -1)
        n_non_noise = int(non_noise_mask.sum())
        
        if n_non_noise > 0:
            # Adaptive cap: Thunderbird is extremely large, so force smaller sample for safe RAM/compute
            effective_sample_size = int(silhouette_sample_size)
            if DATASET.lower() == 'thunderbird':
                safe_cap = 200_000
                if effective_sample_size > safe_cap:
                    print(f"   ⚠️ Thunderbird detected: reducing silhouette sample from {effective_sample_size:,} to {safe_cap:,} for memory safety")
                    effective_sample_size = safe_cap

            if n_non_noise > effective_sample_size:
                print(f"   Sampling {effective_sample_size:,} non-noise points from {n_non_noise:,} for silhouette...")
                sampled_indices = sample_non_noise_indices(
                    cluster_labels,
                    non_noise_mask,
                    effective_sample_size,
                    seed=42
                )
            else:
                print(f"   Using all {n_non_noise:,} non-noise points for silhouette")
                if n_non_noise == len(cluster_labels):
                    sampled_indices = np.arange(len(cluster_labels))
                else:
                    sampled_indices = np.flatnonzero(non_noise_mask)

            print(f"   Preparing sampled labels ({len(sampled_indices):,})...")
            sample_labels = cluster_labels[sampled_indices]
            print(f"   Loading sampled embeddings ({len(sampled_indices):,}) from source array...")
            sample_embeddings = embeddings[sampled_indices]
            print("   ✓ Sampled embeddings loaded")

            # Silhouette requires >=2 clusters in sampled data
            n_unique_sample_clusters = len(np.unique(sample_labels))
            if n_unique_sample_clusters < 2:
                print("   ⚠️ Silhouette skipped: sampled data has < 2 clusters")
            else:
                # We already sampled; pass sample_size=None to avoid second sampling inside helper
                if USE_GPU_SILHOUETTE:
                    overall_silhouette, sample_silhouette_scores = compute_silhouette_gpu(
                        sample_embeddings,
                        sample_labels,
                        kernel_name=GPU_KERNEL_NAME,
                        sample_size=None
                    )
                else:
                    overall_silhouette = silhouette_score(sample_embeddings, sample_labels)
                    sample_silhouette_scores = silhouette_samples(sample_embeddings, sample_labels)
                    print(f"   ✓ Silhouette Score: {overall_silhouette:.4f}")

                # Build per-cluster silhouette mean from sampled points only
                sample_df = pd.DataFrame({
                    'cluster_id': sample_labels,
                    'sil': sample_silhouette_scores
                })
                cluster_silhouette_lookup = sample_df.groupby('cluster_id')['sil'].mean().to_dict()
                print(f"   ✓ Computed sampled silhouette for {len(cluster_silhouette_lookup):,} clusters")
        else:
            print(f"   ⚠️ All samples are noise, cannot compute silhouette")
    
    # Analyze each cluster
    for cluster_id in tqdm(unique_clusters, desc="Analyzing clusters"):
        mask = cluster_labels == cluster_id
        n_samples = np.sum(mask)
        cluster_indices = np.where(mask)[0]
        
        # Initialize cluster info
        cluster_label = None  # 0=NORMAL, 1=NON-NORMAL, 2=ANOMALY
        label_name = None
        pct_normal = 0.0
        pct_nonnormal = 0.0
        count_normal = 0
        count_nonnormal = 0
        labeling_reason = None
        
        # METADATA-BASED LABELING (if enabled and data available)
        if (metadata_df is not None or metadata_labels is not None) and normal_events and nonnormal_events:
            if cluster_id == -1 and FORCE_DBSCAN_NOISE_AS_ANOMALY:
                cluster_label = 2
                label_name = 'ANOMALY'
                labeling_reason = 'dbscan_noise'
                total = 0
            else:
                # Majority-only labeling with safeguards.
                if n_samples > METADATA_SAMPLE_SIZE:
                    sample_indices = np.random.choice(cluster_indices, METADATA_SAMPLE_SIZE, replace=False)
                else:
                    sample_indices = cluster_indices

                # Count normal vs non-normal from metadata (use memmap labels when available)
                if metadata_labels is not None:
                    for idx in sample_indices:
                        if idx < len(metadata_labels):
                            code = int(metadata_labels[idx])
                            if code == 0:
                                count_normal += 1
                            elif code == 1:
                                count_nonnormal += 1
                elif metadata_df is not None:
                    for idx in sample_indices:
                        if idx < len(metadata_df):  # Safety check
                            event_label = normalize_label_token(metadata_df.iloc[idx]['label'])
                            if event_label in normal_events:
                                count_normal += 1
                            elif event_label in nonnormal_events:
                                count_nonnormal += 1

                total = count_normal + count_nonnormal

                if total < MIN_METADATA_EVIDENCE:
                    cluster_label = 2
                    label_name = 'ANOMALY'
                    labeling_reason = 'low_metadata_evidence'
                else:
                    pct_normal = count_normal / total
                    pct_nonnormal = count_nonnormal / total

                    if pct_nonnormal >= MAJORITY_THRESHOLD:
                        cluster_label = 1
                        label_name = 'NON-NORMAL'
                        labeling_reason = 'nonnormal_majority'
                    elif pct_normal >= NORMAL_STRONG_THRESHOLD and pct_nonnormal < NONNORMAL_PRESENCE_THRESHOLD:
                        cluster_label = 0
                        label_name = 'NORMAL'
                        labeling_reason = 'normal_strong_majority'
                    elif pct_nonnormal >= NONNORMAL_PRESENCE_THRESHOLD:
                        cluster_label = 1
                        label_name = 'NON-NORMAL'
                        labeling_reason = 'nonnormal_presence'
                    elif pct_normal >= MAJORITY_THRESHOLD:
                        cluster_label = 0
                        label_name = 'NORMAL'
                        labeling_reason = 'normal_majority'
                    elif AMBIGUOUS_CLUSTER_AS_ANOMALY:
                        cluster_label = 2
                        label_name = 'ANOMALY'
                        labeling_reason = 'mixed_ambiguous'
                    else:
                        cluster_label = 1
                        label_name = 'NON-NORMAL'
                        labeling_reason = 'mixed_fallback_nonnormal'
        
        # FALLBACK: SIZE-BASED CLASSIFICATION (legacy or when metadata unavailable)
        else:
            if cluster_id == -1:
                cluster_label = 2
                label_name = 'ANOMALY'
                labeling_reason = 'noise'
            elif n_samples < adaptive_anomaly_threshold:
                cluster_label = 2
                label_name = 'ANOMALY'
                labeling_reason = 'very_small_size'
            elif n_samples < adaptive_small_threshold:
                cluster_label = 1
                label_name = 'NON-NORMAL'
                labeling_reason = 'small_size'
            else:
                # Regular size → needs k-NN (unreliable, will be handled in prediction)
                cluster_label = None  # Will use k-NN in prediction
                label_name = 'REGULAR'
                labeling_reason = 'regular_size'
        
        # Silhouette mean for this cluster (from sampled points only)
        cluster_silhouette = None
        if cluster_id != -1:
            cluster_silhouette = cluster_silhouette_lookup.get(cluster_id)
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'n_samples': n_samples,
            'cluster_label': cluster_label,
            'label_name': label_name,
            'pct_normal': pct_normal,
            'pct_nonnormal': pct_nonnormal,
            'count_normal': count_normal,
            'count_nonnormal': count_nonnormal,
            'labeling_reason': labeling_reason,
            'silhouette_score': cluster_silhouette
        })
    
    df = pd.DataFrame(cluster_info)
    
    # Summary statistics
    print(f"\n{'='*70}")
    if metadata_df is not None or metadata_labels is not None:
        print("CLUSTER CHARACTERISTICS SUMMARY (SEMI-SUPERVISED: Metadata-based)")
    else:
        print("CLUSTER CHARACTERISTICS SUMMARY (Size-based)")
    print(f"{'='*70}")
    
    print(f"\nTotal clusters: {len(df)}")
    
    # Count by label (if metadata labeling was used)
    if 'label_name' in df.columns and (metadata_df is not None or metadata_labels is not None):
        print(f"\nCluster Labels (Metadata-based):")
        label_counts = df['label_name'].value_counts()
        for label, count in label_counts.items():
            samples = df[df['label_name'] == label]['n_samples'].sum()
            print(f"  {label:12s}: {count:4d} clusters, {samples:,} samples")
        
        # Show labeling reasons
        print(f"\nLabeling Reasons:")
        reason_counts = df['labeling_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason:20s}: {count:4d} clusters")
    
    # Size statistics
    print(f"\nCluster Size Statistics:")
    print(f"  Mean:   {df['n_samples'].mean():.0f}")
    print(f"  Median: {df['n_samples'].median():.0f}")
    print(f"  Min:    {df['n_samples'].min()}")
    print(f"  Max:    {df['n_samples'].max():,}")
    
    # Silhouette statistics (exclude noise and NaN)
    if 'silhouette_score' in df.columns:
        valid_sil = df[df['cluster_id'] != -1]['silhouette_score'].dropna()
        if len(valid_sil) > 0:
            print(f"\nSilhouette Score Statistics (excluding noise):")
            print(f"  Overall Mean: {overall_silhouette:.4f}" if overall_silhouette else "  Overall Mean: N/A")
            print(f"  Per-Cluster Mean: {valid_sil.mean():.4f}")
            print(f"  Per-Cluster Median: {valid_sil.median():.4f}")
            print(f"  Per-Cluster Std: {valid_sil.std():.4f}")
            print(f"  Per-Cluster Min: {valid_sil.min():.4f}")
            print(f"  Per-Cluster Max: {valid_sil.max():.4f}")
            
            # Interpretation guide
            print(f"\n  Interpretation:")
            print(f"    > 0.7  : Strong structure, well-separated clusters")
            print(f"    0.5-0.7: Reasonable structure")
            print(f"    0.25-0.5: Weak structure, overlapping clusters")
            print(f"    < 0.25 : No substantial structure")
    
    # Create lookup dict
    cluster_dict = df.set_index('cluster_id').to_dict('index')
    
    return df, cluster_dict


def hybrid_predict(test_cluster_labels, cluster_dict, 
                   training_embeddings=None, training_cluster_labels=None, 
                   test_embeddings=None, use_knn=False):
    """
    Predict test samples based on their cluster assignment
    
    Strategy:
        IF USE_METADATA_LABELING = True (SEMI-SUPERVISED):
            → Simply lookup cluster label from cluster_dict
            → cluster_label: NORMAL (0), NON-NORMAL (1), ANOMALY (2)
            → Fast & accurate!
        
        ELSE (Legacy size-based with k-NN fallback):
            → Small clusters: direct assignment
            → Regular clusters: k-NN vote (slow & unreliable)
    
    Args:
        test_cluster_labels: Cluster assignments for test samples
        cluster_dict: Dict with cluster characteristics
        training_embeddings: Not needed for metadata-based (legacy only)
        training_cluster_labels: Not needed for metadata-based (legacy only)
        test_embeddings: Not needed for metadata-based (legacy only)
        use_knn: Whether to use k-NN for unlabeled clusters (legacy only)
    
    Returns:
        - predictions (numpy array: 0/1/2)
        - confidence scores (numpy array)
        - prediction_method (list of strings)
    """
    if USE_METADATA_LABELING:
        print("\n🎯 Predicting test samples (SEMI-SUPERVISED: Direct cluster lookup)...")
    else:
        print("\n🎯 Predicting test samples (UNSUPERVISED: Size-based + k-NN)...")
    
    n_test = len(test_cluster_labels)
    predictions = np.zeros(n_test, dtype=np.int32)  # Will store 0, 1, or 2
    confidence = np.zeros(n_test, dtype=np.float32)
    methods = []
    
    # METADATA-BASED PREDICTION (Simple & Fast!)
    if USE_METADATA_LABELING:
        print(f"   Using metadata-labeled clusters...")
        
        for i, cluster_id in enumerate(tqdm(test_cluster_labels, desc="Predicting")):
            if cluster_id in cluster_dict:
                cluster_info = cluster_dict[cluster_id]
                cluster_label = cluster_info.get('cluster_label')
                label_reason = cluster_info.get('labeling_reason', 'unknown')
                
                if cluster_label is not None:
                    predictions[i] = cluster_label
                    
                    # Confidence based on metadata purity
                    if label_reason in {'normal_majority', 'normal_strong_majority'}:
                        confidence[i] = cluster_info.get('pct_normal', 0.7)
                        methods.append('metadata_normal')
                    elif label_reason in {'nonnormal_majority', 'nonnormal_presence', 'mixed_fallback_nonnormal'}:
                        confidence[i] = cluster_info.get('pct_nonnormal', 0.7)
                        methods.append('metadata_nonnormal')
                    elif label_reason in {'dbscan_noise', 'low_metadata_evidence', 'mixed_ambiguous'}:
                        predictions[i] = 2  # ANOMALY
                        confidence[i] = 0.80 if label_reason == 'dbscan_noise' else 0.50
                        methods.append(label_reason)
                    else:
                        confidence[i] = 0.60
                        methods.append(label_reason)
                else:
                    # No label (shouldn't happen with metadata)
                    predictions[i] = 2  # ANOMALY (safe default)
                    confidence[i] = 0.30
                    methods.append('unlabeled')
            else:
                # Unknown cluster (shouldn't happen but handle it)
                predictions[i] = 2  # ANOMALY
                confidence[i] = 0.20
                methods.append('unknown_cluster')
        
        # Summary
        print(f"\n   ✓ Predicted {n_test:,} test samples")
        method_counts = Counter(methods)
        print(f"\n   Prediction Methods Used:")
        for method, count in method_counts.most_common():
            print(f"      {method:20s}: {count:,} samples ({count/n_test*100:.1f}%)")
        
        return predictions, confidence, np.array(methods)
    
    # LEGACY: SIZE-BASED + k-NN PREDICTION (Complex & Slow)
    else:
        print(f"   ⚠️ Using legacy size-based prediction (metadata labeling disabled)")
        print(f"   ⚠️ This approach is UNRELIABLE - consider enabling USE_METADATA_LABELING")
        knn_train_embeddings = training_embeddings
        knn_train_cluster_labels = training_cluster_labels
        
        print(f"   Building k-NN model (k={KNN_NEIGHBORS}, samples={len(knn_train_embeddings):,})...")
        print(f"   NOTE: Using cluster IDs as pseudo-labels (unsupervised approach)")
        knn_model = NearestNeighbors(
            n_neighbors=KNN_NEIGHBORS,
            metric='cosine' if USE_COSINE_DISTANCE else 'euclidean',
            algorithm='auto',
            n_jobs=-1
        )
        knn_model.fit(knn_train_embeddings)
        
        # Store cluster labels for vote counting
        knn_model.train_cluster_labels = knn_train_cluster_labels
    
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
        
        # Decision based on cluster type (UNSUPERVISED)
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
        
        else:  # regular clusters
            # Use k-NN vote with cluster IDs as pseudo-labels
            if not use_knn or knn_model is None:
                # Fallback: predict NORMAL for regular clusters
                predictions[indices] = 0
                confidence[indices] = 0.6
                methods.extend(["no_knn_fallback"] * len(indices))
                continue
            
            # Get k-NN for each test sample in this cluster
            cluster_test_embeddings = test_embeddings[indices]
            distances, neighbor_indices = knn_model.kneighbors(cluster_test_embeddings)
            
            # Get cluster labels from k-NN model
            knn_cluster_labels = knn_model.train_cluster_labels if hasattr(knn_model, 'train_cluster_labels') else training_cluster_labels
            
            for i, sample_idx in enumerate(indices):
                neighbor_cluster_ids = knn_cluster_labels[neighbor_indices[i]]
                
                # Convert neighbor cluster IDs to predictions:
                # -1 (noise) → 2 (ANOMALY)
                # Small clusters → 1 (NON-NORMAL)
                # Regular clusters → 0 (NORMAL)
                neighbor_preds = []
                for nc in neighbor_cluster_ids:
                    if nc == -1:
                        neighbor_preds.append(2)  # Noise → ANOMALY
                    elif nc in cluster_dict:
                        nc_type = cluster_dict[nc]['cluster_type']
                        if nc_type in ['noise', 'very_small']:
                            neighbor_preds.append(2)  # ANOMALY
                        elif nc_type == 'small':
                            neighbor_preds.append(1)  # NON-NORMAL
                        else:
                            neighbor_preds.append(0)  # NORMAL
                    else:
                        neighbor_preds.append(0)  # Unknown → NORMAL
                
                # Majority vote
                vote_counts = np.bincount(neighbor_preds, minlength=3)
                pred_label = np.argmax(vote_counts)
                vote_confidence = vote_counts[pred_label] / KNN_NEIGHBORS
                
                predictions[sample_idx] = pred_label
                confidence[sample_idx] = vote_confidence
                methods.append(f"knn_vote_{pred_label}")
    
    return predictions, confidence, methods


def calculate_metrics(y_true, y_pred, y_confidence=None, method_labels=None):
    """
    Calculate comprehensive classification metrics
    
    Ground Truth: 2-class (NORMAL=0, NON-NORMAL=1)
    Predictions: 3-class (NORMAL=0, NON-NORMAL=1, ANOMALY=2)
    
    Result: 2x3 confusion matrix 
    """
    print("\n📊 Calculating metrics...")
    
    # Ground-truth rows are dynamic, predicted columns are fixed to 3 classes (0/1/2)
    unique_true = sorted(set(y_true))
    present_pred = sorted(set(y_pred))
    true_labels = [cls for cls in [0, 1, 2] if cls in unique_true]
    pred_labels = [0, 1, 2]

    print(f"   Ground truth classes: {true_labels} → {[['NORMAL', 'NON-NORMAL', 'ANOMALY'][i] for i in true_labels]}")
    print(f"   Prediction classes:   {present_pred} → {[['NORMAL', 'NON-NORMAL', 'ANOMALY'][i] for i in present_pred]}")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Build full 3x3 matrix first, then keep only rows present in ground truth.
    # This keeps columns fixed at [NORMAL, NON-NORMAL, ANOMALY].
    cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm = cm_full[true_labels, :]
    
    # Per-class metrics - only for ground truth classes
    true_class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    pred_class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    
    # Calculate metrics only for classes that exist in ground truth
    report_labels = true_labels
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
    print(f"\nConfusion Matrix ({len(true_labels)}x{len(pred_labels)}):")
    print(f"                 Predicted")
    
    # Header
    header = "                 "
    for pred_class in pred_labels:
        header += f"{pred_class_names[pred_class][:2]:>7}"
    print(header)
    
    # Rows
    for i, true_class in enumerate(true_labels):
        row = f"    True  {true_class_names[true_class][:2]:2s} ["
        for j, _ in enumerate(pred_labels):
            row += f"{cm[i,j]:>6} "
        row += "]"
        print(row)
    
    # Error analysis
    if len(unique_true) == 2:
        # 2-class ground truth analysis
        print(f"\n📊 Prediction Distribution:")
        for i, true_class in enumerate(true_labels):
            true_name = true_class_names[true_class]
            total = cm[i].sum()
            print(f"\n  {true_name} ({total:,} samples):")
            for j, pred_class in enumerate(pred_labels):
                count = cm[i, j]
                pct = count / total * 100 if total > 0 else 0
                pred_name = pred_class_names[pred_class]
                status = "✓ Correct" if true_class == pred_class else "✗ Wrong"
                print(f"    → Predicted as {pred_name:12s}: {count:7,} ({pct:5.2f}%) {status}")
    else:
        # 3-class analysis (legacy)
        if len(true_labels) > 2 and 2 in true_labels:
            anomaly_idx = true_labels.index(2)
            if cm[anomaly_idx].sum() > 0:
                print(f"\nCritical Errors:")
                print(f"  A → N (Anomaly missed as Normal):     {cm[anomaly_idx,0]:,} ({cm[anomaly_idx,0]/cm[anomaly_idx].sum()*100:.1f}%)")
                print(f"  A → NN (Anomaly downgrade):           {cm[anomaly_idx,1]:,} ({cm[anomaly_idx,1]/cm[anomaly_idx].sum()*100:.1f}%)")
        
        if 1 in true_labels:
            nn_idx = true_labels.index(1)
            if cm[nn_idx].sum() > 0:
                print(f"\nNon-Normal Errors:")
                print(f"  NN → N (Non-Normal missed as Normal): {cm[nn_idx,0]:,} ({cm[nn_idx,0]/cm[nn_idx].sum()*100:.1f}%)")
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
            for cls in true_labels:
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
        f.write(f"\n\nConfusion Matrix ({len(true_labels)}x{len(pred_labels)}):\n")
        f.write("                 Predicted\n")
        
        # Header
        header = "                 "
        for pred_class in pred_labels:
            header += f"{pred_class_names[pred_class][:2]:>7}"
        f.write(header + "\n")
        
        # Rows
        for i, true_class in enumerate(true_labels):
            row = f"    True  {true_class_names[true_class][:2]:2s} ["
            for j, _ in enumerate(pred_labels):
                row += f"{cm[i,j]:>6} "
            row += "]\n"
            f.write(row)
    
    print(f"\n✓ Metrics saved to: {OUTPUT_METRICS}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'true_labels': true_labels,
        'pred_labels': pred_labels
    }


def visualize_results(cluster_df, y_true, y_pred, metrics):
    """
    Create visualizations:
    1. Silhouette score distribution (cluster quality metric)
    2. Confusion matrix heatmap (prediction accuracy)
    3. Cluster label distribution (metadata-based or size-based)
    4. Cluster size distribution (with labeling threshold)
    """
    print("\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Silhouette Score Distribution (UNSUPERVISED cluster quality)
    ax1 = axes[0, 0]
    if 'silhouette_score' in cluster_df.columns:
        # Filter out NaN (noise clusters don't have silhouette)
        valid_silhouette = cluster_df['silhouette_score'].dropna()
        if len(valid_silhouette) > 0:
            ax1.hist(valid_silhouette, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.axvline(0, color='red', linestyle='--', label='Zero (boundary)')
            ax1.axvline(valid_silhouette.mean(), color='green', linestyle='--', 
                       label=f'Mean={valid_silhouette.mean():.3f}')
            ax1.set_xlabel('Silhouette Score')
            ax1.set_ylabel('Number of Clusters')
            ax1.set_title('Cluster Quality: Silhouette Score Distribution')
            ax1.legend()
            ax1.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No silhouette scores available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Silhouette Score Distribution (N/A)')
    else:
        ax1.text(0.5, 0.5, 'Silhouette computation disabled', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Silhouette Score Distribution (N/A)')
    ax1.grid(alpha=0.3)
    
    # 2. Confusion Matrix (dynamic size)
    ax2 = axes[0, 1]
    cm = metrics['confusion_matrix']
    
    # Determine labels dynamically
    unique_true = metrics.get('true_labels', sorted(set(y_true)))
    unique_pred = metrics.get('pred_labels', [0, 1, 2])
    class_names = ['Normal', 'Non-Normal', 'Anomaly']
    
    true_labels = [class_names[i] for i in unique_true]
    pred_labels = [class_names[i] for i in unique_pred]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=pred_labels,
                yticklabels=true_labels)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Ground Truth')
    ax2.set_title(f'{len(unique_true)}x{len(unique_pred)} Confusion Matrix')
    
    # 3. Cluster Label Distribution (Metadata-based)
    ax3 = axes[1, 0]
    if 'label_name' in cluster_df.columns:
        # Metadata-based labeling
        label_counts = cluster_df['label_name'].value_counts()
        colors = {'NORMAL': 'green', 'NON-NORMAL': 'orange', 'ANOMALY': 'red'}
        label_colors = [colors.get(label, 'gray') for label in label_counts.index]
        label_counts.plot(kind='bar', ax=ax3, color=label_colors)
        ax3.set_xlabel('Cluster Label (Metadata-based)')
        ax3.set_ylabel('Number of Clusters')
        ax3.set_title('Cluster Label Distribution (Semi-supervised)')
    else:
        # Legacy: cluster_type if no metadata labeling
        if 'cluster_type' in cluster_df.columns:
            type_counts = cluster_df['cluster_type'].value_counts()
            type_counts.plot(kind='bar', ax=ax3, color=['green', 'blue', 'orange', 'red'])
            ax3.set_xlabel('Cluster Type')
            ax3.set_ylabel('Number of Clusters')
            ax3.set_title('Cluster Type Distribution (Size-based)')
        else:
            ax3.text(0.5, 0.5, 'No cluster labeling available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cluster Distribution (N/A)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Cluster Size Distribution (log scale)
    ax4 = axes[1, 1]
    ax4.hist(cluster_df['n_samples'], bins=50, edgecolor='black', alpha=0.7)
    
    if not USE_METADATA_LABELING:
        # Show legacy size-based thresholds
        ax4.axvline(VERY_SMALL_CLUSTER_THRESHOLD, color='orange', linestyle='--',
                    label=f'Very small={VERY_SMALL_CLUSTER_THRESHOLD}')
        ax4.axvline(SMALL_CLUSTER_THRESHOLD, color='red', linestyle='--',
                    label=f'Small={SMALL_CLUSTER_THRESHOLD}')
    
    ax4.set_xlabel('Cluster Size (samples)')
    ax4.set_ylabel('Number of Clusters')
    if USE_METADATA_LABELING:
        ax4.set_title('Cluster Size Distribution (Metadata-based, no size cutoff)')
    else:
        ax4.set_title('Cluster Size Distribution (Size-based)')
    ax4.set_yscale('log')
    handles, labels = ax4.get_legend_handles_labels()
    if handles:
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
    print("STEP 8: DETAILED PREDICTION DISTRIBUTION ANALYSIS")
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
        dict with keys: labels, distances, rejection_threshold
    
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

    n_train_total = len(training_embeddings)
    n_test = len(test_embeddings)
    train_labels = np.asarray(training_cluster_labels)

    # Optional downsampling for very large training sets to keep memory bounded.
    if SUBSAMPLE_KNN_TRAINING and n_train_total > KNN_SUBSAMPLE_SIZE:
        print(f"      Subsampling training set: {n_train_total:,} -> {KNN_SUBSAMPLE_SIZE:,}")
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(n_train_total, size=KNN_SUBSAMPLE_SIZE, replace=False))
        training_arr = np.asarray(training_embeddings[sample_idx], dtype=np.float32)
        train_labels = train_labels[sample_idx]
    else:
        training_arr = np.asarray(training_embeddings, dtype=np.float32)

    if use_cosine:
        print(f"      Normalizing training vectors for cosine similarity...")
        if not training_arr.flags.writeable:
            training_arr = training_arr.copy()
        normalize(training_arr, norm='l2', copy=False)

    training_arr = np.ascontiguousarray(training_arr, dtype=np.float32)
    d = training_arr.shape[1]

    print(f"      Training data: {len(training_arr):,} samples, dim={d}")
    print(f"      Test data: {n_test:,} samples")
    
    # Build IVF index
    print(f"      Building FAISS IVF index...")
    effective_nlist = max(1, min(nlist, len(training_arr), max(64, len(training_arr) // 100)))
    quantizer = faiss.IndexFlatL2(d)  # Quantizer for Voronoi cells
    index = faiss.IndexIVFFlat(quantizer, d, effective_nlist, faiss.METRIC_L2)
    
    # Train index (clustering training data into Voronoi cells)
    print(f"      Training index (clustering into {effective_nlist} cells)...")
    
    # For very large datasets, train on subset
    if len(training_arr) > 1_000_000:
        print(f"         Sampling 1M points for training (large dataset optimization)...")
        train_sample_idx = np.random.choice(len(training_arr), 1_000_000, replace=False)
        index.train(training_arr[train_sample_idx])
    else:
        index.train(training_arr)
    
    # Add all training data to index
    print(f"      Adding {len(training_arr):,} training vectors to index...")
    add_batch_size = 200000
    for start in tqdm(range(0, len(training_arr), add_batch_size), desc="      Indexing"):
        end = min(start + add_batch_size, len(training_arr))
        index.add(training_arr[start:end])
    
    # Set search parameters
    index.nprobe = nprobe  # How many cells to visit during search

    # Build reference threshold for reject option from in-distribution neighbors.
    rejection_threshold = None
    if ENABLE_DISTANCE_REJECTION and len(training_arr) > 2:
        ref_size = min(REJECTION_REFERENCE_SAMPLE_SIZE, len(training_arr))
        ref_idx = np.random.choice(len(training_arr), size=ref_size, replace=False)
        d_ref, _ = index.search(np.ascontiguousarray(training_arr[ref_idx]), 2)
        # 2nd neighbor approximates non-self nearest distance.
        d_ref = d_ref[:, 1]
        rejection_threshold = float(np.quantile(d_ref, REJECTION_QUANTILE) * REJECTION_DISTANCE_MULTIPLIER)
        print(f"      Distance rejection enabled: threshold={rejection_threshold:.6f}")
    
    print(f"      ✓ Index built! Starting k-NN search...")
    print(f"      Processing {n_test:,} test samples in batches of {batch_size:,}...")
    
    # Search in batches (memory efficient)
    nearest_indices = np.empty(n_test, dtype=np.int64)
    nearest_distances = np.empty(n_test, dtype=np.float32)
    n_batches = (n_test + batch_size - 1) // batch_size

    for i in tqdm(range(0, n_test, batch_size), desc="      Searching", total=n_batches):
        batch_end = min(i + batch_size, n_test)
        batch = np.array(test_embeddings[i:batch_end], dtype=np.float32, copy=True)
        if use_cosine:
            normalize(batch, norm='l2', copy=False)

        # Search k=1 nearest neighbors
        dist, indices = index.search(np.ascontiguousarray(batch), 1)
        nearest_indices[i:batch_end] = indices.ravel()
        nearest_distances[i:batch_end] = dist.ravel().astype(np.float32)
    
    # Map indices to cluster labels
    test_cluster_labels = train_labels[nearest_indices]
    
    print(f"      ✓ Assigned {len(test_cluster_labels):,} test samples using FAISS IVF!")
    
    # Memory cleanup
    del index, quantizer, training_arr, nearest_indices
    gc.collect()
    
    return {
        'labels': test_cluster_labels,
        'distances': nearest_distances,
        'rejection_threshold': rejection_threshold,
    }


def fast_cluster_assignment_sklearn_batched(training_embeddings, training_cluster_labels,
                                            test_embeddings, use_cosine=True, batch_size=10000):
    """
    Fallback: Batched sklearn k-NN (slower but exact)
    
    Process test data in small batches to show progress
    """
    print(f"\n   Using batched sklearn k-NN (exact, slower)...")
    
    n_train_total = len(training_embeddings)
    n_test = len(test_embeddings)
    train_labels = np.asarray(training_cluster_labels)

    if SUBSAMPLE_KNN_TRAINING and n_train_total > KNN_SUBSAMPLE_SIZE:
        print(f"      Subsampling training set: {n_train_total:,} -> {KNN_SUBSAMPLE_SIZE:,}")
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(n_train_total, size=KNN_SUBSAMPLE_SIZE, replace=False))
        training_norm = np.asarray(training_embeddings[sample_idx], dtype=np.float32)
        train_labels = train_labels[sample_idx]
    else:
        training_norm = np.asarray(training_embeddings, dtype=np.float32)

    if use_cosine:
        if not training_norm.flags.writeable:
            training_norm = training_norm.copy()
        normalize(training_norm, norm='l2', copy=False)
        metric = 'cosine'
    else:
        metric = 'euclidean'
    
    # Build k-NN model
    print(f"      Building k-NN index...")
    knn = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1, algorithm='auto')
    knn.fit(training_norm)

    rejection_threshold = None
    if ENABLE_DISTANCE_REJECTION and len(training_norm) > 2:
        ref_size = min(REJECTION_REFERENCE_SAMPLE_SIZE, len(training_norm))
        ref_idx = np.random.choice(len(training_norm), size=ref_size, replace=False)
        d_ref, _ = knn.kneighbors(training_norm[ref_idx], n_neighbors=2)
        d_ref = d_ref[:, 1]
        rejection_threshold = float(np.quantile(d_ref, REJECTION_QUANTILE) * REJECTION_DISTANCE_MULTIPLIER)
        print(f"      Distance rejection enabled: threshold={rejection_threshold:.6f}")
    
    print(f"      Searching {n_test:,} test samples in batches of {batch_size:,}...")
    
    # Search in batches with progress bar
    nearest_indices = np.empty(n_test, dtype=np.int64)
    nearest_distances = np.empty(n_test, dtype=np.float32)
    n_batches = (n_test + batch_size - 1) // batch_size

    for i in tqdm(range(0, n_test, batch_size), desc="      Searching", total=n_batches):
        batch_end = min(i + batch_size, n_test)
        batch = np.array(test_embeddings[i:batch_end], dtype=np.float32, copy=True)
        if use_cosine:
            normalize(batch, norm='l2', copy=False)

        dist, indices = knn.kneighbors(batch)
        nearest_indices[i:batch_end] = indices.ravel()
        nearest_distances[i:batch_end] = dist.ravel().astype(np.float32)

    test_cluster_labels = train_labels[nearest_indices]
    
    print(f"      ✓ Assigned {len(test_cluster_labels):,} test samples!")
    
    return {
        'labels': test_cluster_labels,
        'distances': nearest_distances,
        'rejection_threshold': rejection_threshold,
    }


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
            test_cluster_distances = None
            
            # Load checkpoint
            test_cluster_labels = np.load(CHECKPOINT_STEP6, allow_pickle=True)
            predictions = np.load(CHECKPOINT_STEP7_PRED, allow_pickle=True)
            confidence = np.load(CHECKPOINT_STEP7_CONF, allow_pickle=True)
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
        test_cluster_distances = None
        # ====================================================================
        # STEP 1: Load training cluster results (UNSUPERVISED)
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 1: LOAD TRAINING CLUSTER RESULTS (UNSUPERVISED)")
        print("="*70)
        print("NOTE: No longer loading template-based labels - using pure clustering")
        
        print(f"\nLoading cluster labels: {TRAINING_LABELS_PATH}")
        training_cluster_labels = np.load(TRAINING_LABELS_PATH)
        print(f"   ✓ Loaded {len(training_cluster_labels):,} cluster assignments")
        
        n_clusters = len(set(training_cluster_labels) - {-1})
        n_noise = np.sum(training_cluster_labels == -1)
        print(f"\nClusters found: {n_clusters}")
        print(f"Noise points: {n_noise:,} ({n_noise/len(training_cluster_labels)*100:.2f}%)")
        
        # ====================================================================
        # STEP 2: Load training embeddings (for silhouette + k-NN)
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 2: LOAD TRAINING EMBEDDINGS (for silhouette + k-NN)")
        print("="*70)
        
        print(f"\nLoading training embeddings: {TRAINING_EMBEDDINGS_PATH}")
        # Try memmap first for minimal RAM usage; robust fallbacks for several formats
        try:
            training_embeddings = np.load(TRAINING_EMBEDDINGS_PATH, mmap_mode='r')
            print("   ✓ Loaded via memmap")
        except ValueError as e:
            msg = str(e).lower()
            # If numpy complains about pickled content, attempt safe fallbacks
            if 'pickled' in msg or 'pickle' in msg:
                print("   ⚠️ File may contain pickled/unknown format — attempting fallbacks:")
                # 1) Try loading with allow_pickle=True (may OOM)
                try:
                    training_embeddings = np.load(TRAINING_EMBEDDINGS_PATH, allow_pickle=True)
                    print("   ✓ Loaded with allow_pickle=True")
                    if getattr(training_embeddings, 'dtype', None) == object:
                        try:
                            print("   🔁 Converting object array into numeric ndarray (may use significant RAM)...")
                            training_embeddings = np.vstack(training_embeddings).astype(np.float32)
                            print(f"   ✓ Converted to numeric ndarray with shape: {training_embeddings.shape}")
                        except Exception as conv_e:
                            print(f"   ❌ Failed to convert object array to numeric: {conv_e}")
                            raise
                except Exception as e2:
                    # allow_pickle load failed (e.g., UnpicklingError) → try raw binary memmap heuristics
                    print(f"   ⚠️ allow_pickle load failed: {e2}")
                    print("   🔎 Trying to interpret file as raw float32 binary (no .npy header)")
                    try:
                        import os
                        fsize = os.path.getsize(TRAINING_EMBEDDINGS_PATH)
                        # Common embedding dims to try (BERT/PCA variants)
                        candidate_dims = [768, 512, 384, 300, 256, 128]
                        integer_candidates = []
                        for d in candidate_dims:
                            if fsize % (4 * d) == 0:
                                n = fsize // (4 * d)
                                integer_candidates.append((int(n), d))
                                print(f"      possible shape: ({int(n)},{d}) based on file size")

                        # Prefer candidate matching training labels length if available
                        chosen = None
                        if 'training_cluster_labels' in locals():
                            target_n = len(training_cluster_labels)
                            for n, d in integer_candidates:
                                if n == target_n:
                                    chosen = (n, d)
                                    break

                        # If no exact match, pick unique candidate if only one
                        if chosen is None and len(integer_candidates) == 1:
                            chosen = integer_candidates[0]

                        if chosen is not None:
                            n, d = chosen
                            print(f"   ✓ Loading as raw memmap with shape=({n},{d})")
                            training_embeddings = np.memmap(str(TRAINING_EMBEDDINGS_PATH), dtype='float32', mode='r', shape=(n, d))
                        else:
                            raise RuntimeError("Could not determine raw memmap shape automatically.\n" \
                                               "Check how the file was written or run header inspection script.")
                    except Exception as e3:
                        print(f"   ❌ Raw memmap fallback failed: {e3}")
                        raise
            else:
                raise
        print(f"   ✓ Shape: {training_embeddings.shape}")
        
        # Truncate if needed (match cluster labels length)
        if len(training_embeddings) != len(training_cluster_labels):
            print(f"   ⚠️ Truncating embeddings to {len(training_cluster_labels):,}")
            training_embeddings = training_embeddings[:len(training_cluster_labels)]
        
        # ====================================================================
        # STEP 3: Analyze cluster characteristics (UNSUPERVISED + Silhouette)
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 3: ANALYZE CLUSTER CHARACTERISTICS (+ Silhouette Score)")
        print("="*70)
        
        cluster_df, cluster_dict = analyze_cluster_characteristics(
            training_cluster_labels,
            embeddings=training_embeddings,
            compute_silhouette=True,
            silhouette_sample_size=100000,  # Sample 100K for speed
            metadata_tsv_path=METADATA_TSV_PATH if USE_METADATA_LABELING else None,
            normal_template_path=NORMAL_TEMPLATE_PATH if USE_METADATA_LABELING else None,
            nonnormal_template_path=NONNORMAL_TEMPLATE_PATH if USE_METADATA_LABELING else None
        )
        
        # Save cluster analysis
        cluster_df.to_csv(OUTPUT_CLUSTER_ANALYSIS, index=False)
        print(f"\n✓ Cluster analysis saved to: {OUTPUT_CLUSTER_ANALYSIS}")
        
        # ====================================================================
        # STEP 4: Load testing data (2-class ground truth from file names)
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 4: LOAD TESTING DATA (2-class ground truth)")
        print("="*70)
        
        # Load multiple testing sets - ground truth based on file name (NOT templates!)
        test_embeddings, test_gt_labels, test_set_info = load_multiple_testing_sets(
            TESTING_SETS
        )
        
        # ====================================================================
        # STEP 5: Assign test samples to clusters
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 5: ASSIGN TEST SAMPLES TO CLUSTERS (k-NN)")
        print("="*70)
        
        if ALGORITHM == "kmeans":
            print("\nLoading K-Means model...")
            model = None
            try:
                model = load_kmeans_model_compat(TRAINED_MODEL_PATH)
                print(f"   ✓ Model loaded: {type(model).__name__}")
            except Exception as model_err:
                print(f"   ⚠️ K-Means model load failed: {model_err}")
                print("   ⚠️ Falling back to centroid-based assignment from cluster labels")

            if model is not None:
                print("\nPredicting cluster assignments for test data...")
                test_cluster_labels = model.predict(test_embeddings)
                print(f"   ✓ Assigned {len(test_cluster_labels):,} test samples")
            else:
                training_embeddings_for_kmeans = training_embeddings
                labels_for_centroids = training_cluster_labels

                # If feature dimensions don't match, try common path patterns for the selected embedding type.
                if training_embeddings_for_kmeans.shape[1] != test_embeddings.shape[1]:
                    print(
                        f"   ⚠️ Feature mismatch: train_dim={training_embeddings_for_kmeans.shape[1]} "
                        f"vs test_dim={test_embeddings.shape[1]}"
                    )
                    print("   🔎 Searching for matching training embeddings path...")
                    resolved = find_kmeans_training_embeddings_for_dim(
                        TRAINING_EMBEDDINGS_PATH,
                        DATASET,
                        EMBEDDING_TYPE,
                        int(test_embeddings.shape[1])
                    )
                    if resolved is None:
                        raise RuntimeError(
                            "Could not find training embeddings with matching feature dimension for "
                            f"EMBEDDING_TYPE={EMBEDDING_TYPE}. "
                            "Update TRAINING_EMBEDDINGS_PATH to the correct file."
                        )

                    resolved_path, training_embeddings_for_kmeans = resolved
                    print(f"   ✓ Using fallback training embeddings: {resolved_path}")

                if len(training_embeddings_for_kmeans) != len(labels_for_centroids):
                    min_len = min(len(training_embeddings_for_kmeans), len(labels_for_centroids))
                    print(
                        f"   ⚠️ Alignment mismatch for centroid build. "
                        f"Truncating to {min_len:,} rows"
                    )
                    training_embeddings_for_kmeans = training_embeddings_for_kmeans[:min_len]
                    labels_for_centroids = labels_for_centroids[:min_len]

                centroid_cache_path = CHECKPOINT_DIR / (
                    f"kmeans_centroids_{DATASET.lower()}_{EMBEDDING_TYPE.lower()}_"
                    f"{training_embeddings_for_kmeans.shape[1]}d.npz"
                )

                if centroid_cache_path.exists():
                    print(f"\nLoading centroid cache: {centroid_cache_path.name}")
                    cache = np.load(centroid_cache_path)
                    cluster_ids = cache['cluster_ids'].astype(np.int32)
                    centroids = cache['centroids'].astype(np.float32)
                    print(f"   ✓ Loaded {len(cluster_ids)} centroids")
                else:
                    print("\nBuilding centroid cache from training labels...")
                    cluster_ids, centroids = build_kmeans_centroids_from_labels(
                        training_embeddings_for_kmeans,
                        labels_for_centroids,
                        chunk_size=300000
                    )
                    np.savez(centroid_cache_path, cluster_ids=cluster_ids, centroids=centroids)
                    print(f"   ✓ Saved centroid cache: {centroid_cache_path.name}")

                print("\nPredicting cluster assignments for test data (centroid fallback)...")
                test_cluster_labels = predict_kmeans_from_centroids(
                    test_embeddings,
                    cluster_ids,
                    centroids,
                    use_cosine=USE_COSINE_DISTANCE,
                    batch_size=20000
                )
                print(f"   ✓ Assigned {len(test_cluster_labels):,} test samples")
            
        else:  # dbscan
            print("\nFor DBSCAN, using k-NN to assign test samples to nearest cluster...")
            test_cluster_distances = None
            rejection_threshold = None
            
            # Try FAISS IVF first (10-100x faster, ~98% accuracy)
            faiss_result = fast_cluster_assignment_faiss(
                training_embeddings, 
                training_cluster_labels,
                test_embeddings,
                use_cosine=USE_COSINE_DISTANCE,
                nlist=1024,      # 1024 Voronoi cells (good for 1M+ samples)
                nprobe=64,       # Search 64 cells (good balance: speed vs accuracy)
                batch_size=50000 # Process 50K test samples at a time
            )
            
            # Fallback to batched sklearn if FAISS failed
            if faiss_result is None:
                print("\n   ⚠️ FAISS unavailable or failed, using batched sklearn (slower but exact)...")
                sklearn_result = fast_cluster_assignment_sklearn_batched(
                    training_embeddings,
                    training_cluster_labels,
                    test_embeddings,
                    use_cosine=USE_COSINE_DISTANCE,
                    batch_size=10000  # Smaller batches for sklearn
                )
                test_cluster_labels = sklearn_result['labels']
                test_cluster_distances = sklearn_result['distances']
                rejection_threshold = sklearn_result['rejection_threshold']
            else:
                test_cluster_labels = faiss_result['labels']
                test_cluster_distances = faiss_result['distances']
                rejection_threshold = faiss_result['rejection_threshold']

            if ENABLE_DISTANCE_REJECTION and rejection_threshold is not None and test_cluster_distances is not None:
                reject_mask = test_cluster_distances > rejection_threshold
                n_rejected = int(np.sum(reject_mask))
                if n_rejected > 0:
                    test_cluster_labels = np.array(test_cluster_labels, copy=True)
                    test_cluster_labels[reject_mask] = -1
                    print(
                        f"   ⚠️ Distance rejection applied: {n_rejected:,}/{len(test_cluster_labels):,} "
                        f"samples mapped to cluster -1 (noise/anomaly)"
                    )
                else:
                    print("   ✓ Distance rejection active, no sample exceeded threshold")
            
            print(f"   ✓ Cluster assignment complete!")
            print(f"   ✓ Assigned {len(test_cluster_labels):,} test samples to clusters")
        
        # SAVE CHECKPOINT AFTER STEP 6 (most expensive step)
        print(f"\n💾 Saving STEP 6 checkpoint...")
        np.save(CHECKPOINT_STEP6, test_cluster_labels)
        print(f"   ✓ Saved: {CHECKPOINT_STEP6.name}")
        
        # ====================================================================
        # STEP 6: Hybrid prediction (3-class output)
        # ====================================================================
        print("\n" + "="*70)
        if USE_METADATA_LABELING:
            print("STEP 6: PREDICTION (SEMI-SUPERVISED: Metadata-based)")
        else:
            print("STEP 6: PREDICTION (UNSUPERVISED: Size-based)")
        print("="*70)
        
        predictions, confidence, methods = hybrid_predict(
            test_cluster_labels, cluster_dict,
            training_embeddings=training_embeddings if not USE_METADATA_LABELING else None,
            training_cluster_labels=training_cluster_labels if not USE_METADATA_LABELING else None,
            test_embeddings=test_embeddings if not USE_METADATA_LABELING else None,
            use_knn=False if USE_METADATA_LABELING else True
        )
        
        # Save predictions
        np.save(OUTPUT_PREDICTIONS, predictions)
        print(f"\n✓ Predictions saved to: {OUTPUT_PREDICTIONS}")
        
        # SAVE CHECKPOINT AFTER STEP 6 (prediction complete)
        print(f"\n💾 Saving STEP 6 checkpoint...")
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
    # STEP 7: Calculate metrics (2-class ground truth vs 3-class predictions)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: CALCULATE METRICS (2x3 confusion matrix)")
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
        'nn_distance': test_cluster_distances if test_cluster_distances is not None else np.full(len(test_gt_labels), np.nan),
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
    # STEP 10: Detailed Prediction Distribution Analysis
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
