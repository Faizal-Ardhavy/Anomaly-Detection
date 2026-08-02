"""
Cluster Testing Pipeline - C1: Pseudo-Labeling (Tutorial-Style)
================================================================

VARIANT: C1 - Per-Sample Pseudo-Labeling Semi-Supervised Learning
- KMeans/DBSCAN clusters from training kept (NOT rebuilt)
- NO cluster-level majority vote for final labeling
- Per-sample 2-class label from metadata (matches actual dataset structure):
    * Label = '-' (dash)  -> NORMAL  (0)
    * Label = anything    -> ANOMALY (1)
- Cluster characteristics (size, noise) become FEATURES for the classifier,
  not label determinants. The 3rd class (NON-NORMAL) is removed.
- 20% of training = "labeled" pool, 80% = "unlabeled" pool (matches tutorial)
- Classifier (Logistic Regression) learns from labeled pool
- High-confidence predictions on unlabeled pool become pseudo-labels
- Combined set retrains the classifier
- Final classifier predicts TEST set (2-class output: NORMAL / ANOMALY)

Tutorial mapping:
- "Small labeled set"    -> 20% of training with known metadata labels
- "Large unlabeled set"  -> 80% of training (labels hidden)
- "Hold-out test set"    -> Test set (2-class ground truth)

Pipeline:
1. Load training clusters + embeddings + metadata
2. Assign per-sample 2-class label from metadata (no majority vote, no 3rd class)
3. Sample 20% as labeled, hide 80%
4. Train LogReg on labeled pool
5. Pseudo-label 80% (high-confidence predictions)
6. Retrain LogReg on (labeled + pseudo-labeled)
7. Predict test set with final classifier
8. Metrics (2x2 confusion matrix), iteration history

Ground Truth: 2-class (Normal / Anomaly) from test set name
Prediction:   2-class (Normal / Anomaly) from classifier
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
from sklearn.linear_model import LogisticRegression
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset & Algorithm Selection
DATASET = "Thunderbird"  # "BGL" or "Thunderbird"
ALGORITHM = "dbscan"     # "kmeans" or "dbscan"
EMBEDDING_TYPE = "base"  # "base", "pca256", or "pca128"

# Template paths for per-sample metadata labeling
if DATASET == "BGL":
    NORMAL_TEMPLATE_PATH = Path("log_processing/bgl/bgl_normal_template.txt")
    NONNORMAL_TEMPLATE_PATH = Path("log_processing/bgl/bgl_nonNormal_template.txt")
    METADATA_TSV_PATH = Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_bgl_meta.tsv")
else:  # Thunderbird
    NORMAL_TEMPLATE_PATH = Path("log_processing/thunderbird/thunderbird_normal_template.txt")
    NONNORMAL_TEMPLATE_PATH = Path("log_processing/thunderbird/thunderbird_nonNormal_template.txt")
    METADATA_TSV_PATH = Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_thunderbird_meta.tsv")

# Path to training results
if ALGORITHM == "kmeans":
    TRAINED_MODEL_PATH = Path("kmeans/thunderbird_k_params/model_kmeans_log.pkl")
    TRAINING_LABELS_PATH = Path("kmeans/thunderbird_k_params/cluster_labels.npy")
    TRAINING_EMBEDDINGS_PATH = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector/after_preprocessed_thunderbird_embeddings.npy")
else:  # dbscan
    TRAINING_LABELS_PATH = Path("dbscan/thunderbird_base_model/dbscan_labels.npy")
    TRAINING_CONFIG_PATH = Path("dbscan/thunderbird_base_model/dbscan_config.npy")
    TRAINING_EMBEDDINGS_PATH = Path("/media/bioinfo04/Expansion/2427051003_dataset_vector/after_preprocessed_thunderbird_embeddings.npy")

# Path to testing data
TESTING_SETS = [
    {
        'name': 'normal',
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_testing/after_preprocessed_thunderbird_normal_embeddings.npy"),
        'metadata': Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_thunderbird_normal_meta.tsv")
    },
    {
        'name': 'anomaly',
        'embeddings': Path("/media/bioinfo04/Expansion/2427051003_dataset_vector_testing/after_preprocessed_thunderbird_non_normal_embeddings.npy"),
        'metadata': Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data/after_preprocessed_thunderbird_non_normal_meta.tsv")
    }
]

# ============================================================================
# C1 PSEUDO-LABELING PARAMETERS (Tutorial-Style)
# ============================================================================

# Train/test split: fraction of training data used as "labeled" pool
LABELED_FRACTION = 0.20  # 20% labeled, 80% unlabeled

# 2-class label mapping from metadata (matches actual dataset)
#   Label = '-'  -> NORMAL  (0)
#   Label = <anything else> -> ANOMALY (1)
LABEL_NORMAL_CODE = 0
LABEL_ANOMALY_CODE = 1
METADATA_NORMAL_TOKEN = '-'  # In templates, normal is denoted by a single dash

# Pseudo-labeling parameters (matches tutorial)
PSEUDO_CONFIDENCE_THRESHOLD = 0.90
MAX_ITERATIONS = 3
MIN_PSEUDO_PER_ITERATION = 100
CONVERGENCE_TOL = 0.001

# Classifier
USE_NOISY_STUDENT = True
CLASSIFIER_C = 1.0
CLASSIFIER_MAX_ITER = 1000
CLASSIFIER_SAMPLE_CAP = 1_000_000  # Max samples for training (memory safety)
LABELED_SUBSAMPLE_CAP = 500_000   # Cap labeled pool to this for first iteration

# Cluster ID as feature
USE_CLUSTER_ID_AS_FEATURE = True
MIN_CLUSTER_ID_FREQ = 5

# Distance
USE_COSINE_DISTANCE = True

# Output paths
OUTPUT_DIR = Path("testing_results_ssl") / f"{DATASET.lower()}_{ALGORITHM}_{EMBEDDING_TYPE}_c1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREDICTIONS = OUTPUT_DIR / "predictions.npy"
OUTPUT_CONFIDENCE = OUTPUT_DIR / "confidence.npy"
OUTPUT_CLUSTER_LABELS = OUTPUT_DIR / "test_cluster_labels.npy"
OUTPUT_PSEUDO_HISTORY = OUTPUT_DIR / "pseudo_label_history.csv"
OUTPUT_TRAIN_LABELS = OUTPUT_DIR / "training_sample_labels.npy"
OUTPUT_METRICS = OUTPUT_DIR / "metrics.txt"
OUTPUT_CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"
OUTPUT_ITERATION_PLOT = OUTPUT_DIR / "iteration_progression.png"
OUTPUT_DETAILED_RESULTS = OUTPUT_DIR / "detailed_results.csv"


# ============================================================================
# HELPER FUNCTIONS (reused)
# ============================================================================

def normalize_label_token(value) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip().upper()


def load_template_events(template_path: Path) -> set:
    """DEPRECATED: kept only for backward-compat. Returns {'-'} as NORMAL marker."""
    return {'-'}


def load_metadata_codes_for_dataset(metadata_tsv_path, normal_template_path=None,
                                    nonnormal_template_path=None):
    """Read metadata TSV and assign per-sample 2-class label.

    Simple rule (matches actual dataset: only 2 labels exist):
      - label column == '-' (or empty/NaN) -> 0 (NORMAL)
      - label column == anything else      -> 1 (ANOMALY)

    Templates are NOT used. Kept in signature for backward-compat.
    """
    print(f"   Reading metadata TSV: {Path(metadata_tsv_path).name}")
    print(f"   Rule: label == '-' or empty -> NORMAL (0), else -> ANOMALY (1)")

    chunksize = 5_000_000
    codes_list = []
    sample_labels = []
    total_rows = 0
    matched_normal = 0
    matched_anomaly = 0

    for chunk in pd.read_csv(metadata_tsv_path, sep='\t', usecols=['label'],
                              dtype=str, chunksize=chunksize):
        chunk_codes = np.ones(len(chunk), dtype=np.uint8)  # default: ANOMALY
        vals = chunk['label'].values
        for i, val in enumerate(vals):
            if len(sample_labels) < 10:
                sample_labels.append(repr(val)[:50])
            # Default ANOMALY, override to NORMAL only if '-' or empty
            if pd.isna(val):
                chunk_codes[i] = 0
                matched_normal += 1
            else:
                tok = str(val).strip()
                if tok == '' or tok == '-':
                    chunk_codes[i] = 0
                    matched_normal += 1
                else:
                    matched_anomaly += 1
            total_rows += 1
        codes_list.append(chunk_codes)

    print(f"   Sample label values (first 10): {sample_labels}")
    print(f"   Total rows processed: {total_rows:,}")
    print(f"   NORMAL  ('-'):  {matched_normal:,} ({matched_normal/total_rows*100:.2f}%)")
    print(f"   ANOMALY (else): {matched_anomaly:,} ({matched_anomaly/total_rows*100:.2f}%)")

    if not codes_list:
        return np.array([], dtype=np.uint8)
    return np.concatenate(codes_list)


def build_metadata_label_memmap(tsv_path, normal_events, nonnormal_events,
                                memmap_path=None, chunksize=1_000_000):
    """Build/reuse a compact uint8 memmap with per-row label codes (0=N, 1=NN, 2=other).

    NOTE: Kept for backward-compat but NOT used in main flow anymore.
    """
    if memmap_path is None:
        memmap_path = OUTPUT_DIR / "metadata_labels.memmap"
    memmap_path = Path(memmap_path)
    if memmap_path.exists():
        size_bytes = memmap_path.stat().st_size
        n = size_bytes
        return np.memmap(memmap_path, dtype=np.uint8, mode='r', shape=(n,))
    n_rows = sum(1 for _ in open(tsv_path, 'rb')) - 1
    mm = np.memmap(memmap_path, dtype=np.uint8, mode='w+', shape=(n_rows,))
    idx = 0
    for chunk in pd.read_csv(tsv_path, sep='\t', usecols=['label'],
                              dtype=str, chunksize=chunksize):
        for val in chunk['label'].values:
            tok = normalize_label_token(val)
            if tok in normal_events:
                mm[idx] = 0
            elif tok in nonnormal_events:
                mm[idx] = 1
            else:
                mm[idx] = 2
            idx += 1
    mm.flush()
    return np.memmap(memmap_path, dtype=np.uint8, mode='r', shape=(n_rows,))


def _load_metadata_chunked(tsv_path, normal_events, nonnormal_events, chunksize=1_000_000):
    codes = []
    for chunk in pd.read_csv(tsv_path, sep='\t', usecols=['label'],
                              dtype=str, chunksize=chunksize):
        for val in chunk['label'].values:
            tok = normalize_label_token(val)
            if tok in normal_events:
                codes.append(0)
            elif tok in nonnormal_events:
                codes.append(1)
            else:
                codes.append(2)
    return np.array(codes, dtype=np.uint8)


def load_multiple_testing_sets(testing_sets):
    all_embeddings = []
    all_gt_labels = []
    test_set_info = []
    current_idx = 0
    for test_set in testing_sets:
        name = test_set['name'].lower().strip()
        if name in ('normal',):
            gt = 0
        elif name in ('anomaly', 'nonnormal', 'non_normal', 'non-normal'):
            gt = 1
        else:
            raise ValueError(f"Unknown test set name: {name}")
        emb = np.load(test_set['embeddings'], mmap_mode='r')
        n_samples = len(emb)
        all_embeddings.append(emb)
        all_gt_labels.append(np.full(n_samples, gt, dtype=np.int32))
        test_set_info.append({
            'name': name, 'start_idx': current_idx,
            'end_idx': current_idx + n_samples, 'n_samples': n_samples
        })
        current_idx += n_samples
        print(f"   {name}: {n_samples:,} samples, GT={['NORMAL','ANOMALY'][gt]}")
    return (np.concatenate(all_embeddings, axis=0),
            np.concatenate(all_gt_labels, axis=0),
            test_set_info)


def predict_kmeans_from_centroids(test_embeddings, cluster_ids, centroids,
                                  use_cosine=True, batch_size=20000):
    n_test = len(test_embeddings)
    predictions = np.empty(n_test, dtype=np.int32)
    if use_cosine:
        centroids_ref = normalize(centroids.astype(np.float32), norm='l2', copy=True)
    else:
        centroids_ref = centroids.astype(np.float32)
        centroids_sq = np.sum(centroids_ref * centroids_ref, axis=1)[None, :]
    for start in tqdm(range(0, n_test, batch_size), desc="   Centroid assign"):
        end = min(start + batch_size, n_test)
        batch = np.asarray(test_embeddings[start:end], dtype=np.float32)
        if use_cosine:
            batch = normalize(batch, norm='l2', copy=False)
            scores = batch @ centroids_ref.T
            predictions[start:end] = cluster_ids[np.argmax(scores, axis=1)]
        else:
            dot = batch @ centroids_ref.T
            batch_sq = np.sum(batch * batch, axis=1, keepdims=True)
            dist2 = batch_sq + centroids_sq - 2.0 * dot
            predictions[start:end] = cluster_ids[np.argmin(dist2, axis=1)]
    return predictions


def build_kmeans_centroids_from_labels(training_embeddings, training_cluster_labels,
                                       chunk_size=300000):
    labels = np.asarray(training_cluster_labels)
    unique_clusters = sorted(int(c) for c in np.unique(labels) if int(c) != -1)
    cluster_ids = np.array(unique_clusters, dtype=np.int64)
    cluster_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    n_features = int(training_embeddings.shape[1])
    sums = np.zeros((len(cluster_ids), n_features), dtype=np.float64)
    counts = np.zeros(len(cluster_ids), dtype=np.int64)
    for start in tqdm(range(0, len(labels), chunk_size), desc="   Building centroids"):
        end = min(start + chunk_size, len(labels))
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
    centroids = (sums[valid] / counts[valid, None]).astype(np.float32)
    cluster_ids = cluster_ids[valid].astype(np.int32)
    return cluster_ids, centroids


def load_kmeans_model_compat(model_path):
    try:
        return joblib.load(model_path)
    except ValueError as e:
        msg = str(e)
        if 'BitGenerator module' in msg or 'MT19937' in msg:
            import numpy.random._pickle as np_random_pickle
            original_ctor = np_random_pickle.__bit_generator_ctor
            def _compat(bit_gen_name):
                if isinstance(bit_gen_name, type):
                    bit_gen_name = bit_gen_name.__name__
                return original_ctor(bit_gen_name)
            np_random_pickle.__bit_generator_ctor = _compat
            return joblib.load(model_path)
        raise


def find_kmeans_training_embeddings_for_dim(original_path, dataset, embedding_type, target_dim):
    candidates = [Path(original_path)]
    emb_type = embedding_type.lower()
    if emb_type != "base":
        candidates.extend([
            Path(f"/media/bioinfo04/Expansion/2427051003_dataset_vector_{emb_type}/after_preprocessed_{dataset.lower()}_{emb_type}_embeddings.npy"),
            Path(f"/media/bioinfo04/Expansion/2427051003_dataset_vector_{emb_type}/after_preprocessed_{dataset.lower()}_embeddings.npy"),
        ])
    for path in candidates:
        if not path.exists():
            continue
        try:
            emb = np.load(path, mmap_mode='r')
            if emb.ndim == 2 and emb.shape[1] == target_dim:
                return path, emb
        except Exception:
            continue
    return None


def fast_cluster_assignment_sklearn_batched(training_embeddings, training_cluster_labels,
                                            test_embeddings, use_cosine=True, batch_size=10000):
    print("   Using sklearn batched k-NN for cluster assignment...")
    train_emb = np.asarray(training_embeddings, dtype=np.float32)
    if use_cosine:
        train_emb = normalize(train_emb, norm='l2', copy=False)
    valid_mask = training_cluster_labels != -1
    train_emb_valid = train_emb[valid_mask]
    train_labels_valid = training_cluster_labels[valid_mask]
    knn = NearestNeighbors(n_neighbors=1, metric='cosine' if use_cosine else 'euclidean', n_jobs=-1)
    knn.fit(train_emb_valid)
    n_test = len(test_embeddings)
    labels = np.empty(n_test, dtype=np.int32)
    distances = np.empty(n_test, dtype=np.float32)
    for start in tqdm(range(0, n_test, batch_size), desc="   k-NN batched"):
        end = min(start + batch_size, n_test)
        batch = np.asarray(test_embeddings[start:end], dtype=np.float32)
        if use_cosine:
            batch = normalize(batch, norm='l2', copy=False)
        dist, idx = knn.kneighbors(batch)
        labels[start:end] = train_labels_valid[idx[:, 0]]
        distances[start:end] = dist[:, 0]
    return {'labels': labels, 'distances': distances, 'rejection_threshold': None}


def fast_cluster_assignment_faiss(training_embeddings, training_cluster_labels,
                                  test_embeddings, use_cosine=True,
                                  nlist=1024, nprobe=64, batch_size=50000):
    try:
        import faiss
    except ImportError:
        return None
    train_emb = np.asarray(training_embeddings, dtype=np.float32)
    if use_cosine:
        train_emb = normalize(train_emb, norm='l2', copy=False)
    valid_mask = training_cluster_labels != -1
    train_emb_valid = train_emb[valid_mask]
    train_labels_valid = training_cluster_labels[valid_mask].astype(np.int32)
    d = train_emb_valid.shape[1]
    n_train = len(train_emb_valid)
    if n_train < 256:
        return None
    quantizer = faiss.IndexFlatIP(d) if use_cosine else faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, min(nlist, n_train // 10),
                                faiss.METRIC_INNER_PRODUCT if use_cosine else faiss.METRIC_L2)
    print("   Training FAISS index...")
    index.train(train_emb_valid)
    index.add(train_emb_valid)
    index.nprobe = nprobe
    n_test = len(test_embeddings)
    labels = np.empty(n_test, dtype=np.int32)
    distances = np.empty(n_test, dtype=np.float32)
    print("   Querying FAISS in batches...")
    for start in tqdm(range(0, n_test, batch_size), desc="   FAISS batched"):
        end = min(start + batch_size, n_test)
        batch = np.asarray(test_embeddings[start:end], dtype=np.float32)
        if use_cosine:
            batch = normalize(batch, norm='l2', copy=False)
        D, I = index.search(batch, 1)
        labels[start:end] = train_labels_valid[I[:, 0]]
        if use_cosine:
            distances[start:end] = 1.0 - D[:, 0]
        else:
            distances[start:end] = np.sqrt(np.maximum(D[:, 0], 0))
    return {'labels': labels, 'distances': distances, 'rejection_threshold': None}


# ============================================================================
# C1 CORE: PER-SAMPLE LABELING FROM CLUSTERS + METADATA
# ============================================================================

def compute_cluster_sizes(cluster_labels):
    """Return dict[cluster_id] -> n_samples. Excludes noise (-1) from size-based logic."""
    sizes = Counter(cluster_labels.tolist())
    return dict(sizes)


def assign_per_sample_labels(cluster_labels, metadata_codes):
    """
    Per-sample 2-class label from metadata WITHOUT majority vote.

    Rules per sample i:
    - metadata code == 0 (label = '-' in template = NORMAL) -> 0 (NORMAL)
    - anything else (non-dash label, unknown, etc.)          -> 1 (ANOMALY)

    Cluster characteristics (size, noise) are NOT used as label determinants.
    They become features for the classifier instead.
    """
    print("\n🏷️  Assigning per-sample 2-class labels (no majority vote)...")
    print(f"   METADATA_NORMAL_TOKEN = '{METADATA_NORMAL_TOKEN}'")
    print(f"   Code 0 (label = '{METADATA_NORMAL_TOKEN}') -> NORMAL (0)")
    print(f"   Anything else                          -> ANOMALY (1)")

    n = len(cluster_labels)
    sample_labels = np.ones(n, dtype=np.int32)  # default: ANOMALY

    normal_count = 0
    anomaly_count = 0
    chunk = 500_000
    for start in tqdm(range(0, n, chunk), desc="   Labeling samples"):
        end = min(start + chunk, n)
        codes = metadata_codes[start:end]
        labels = np.where(codes == 0, 0, 1).astype(np.int32)
        sample_labels[start:end] = labels

    normal_count = int((sample_labels == 0).sum())
    anomaly_count = int((sample_labels == 1).sum())

    print(f"   Sample label distribution:")
    print(f"      NORMAL  : {normal_count:>10,} ({normal_count/n*100:.2f}%)")
    print(f"      ANOMALY : {anomaly_count:>10,} ({anomaly_count/n*100:.2f}%)")
    return sample_labels


def split_labeled_unlabeled(sample_labels, fraction=LABELED_FRACTION, seed=42):
    """
    Tutorial-style split: fraction of training samples -> labeled, rest -> unlabeled.

    Stratified across 2 classes (NORMAL, ANOMALY) so each class is represented
    in the labeled pool.
    """
    print(f"\n✂️  Splitting training into labeled ({fraction:.0%}) / unlabeled ({1-fraction:.0%})...")
    rng = np.random.default_rng(seed)
    n = len(sample_labels)
    labeled_idx = []
    class_names = {0: 'NORMAL', 1: 'ANOMALY'}
    for cls in [0, 1]:
        cls_idx = np.where(sample_labels == cls)[0]
        n_take = max(1, int(len(cls_idx) * fraction))
        if len(cls_idx) > 0:
            chosen = rng.choice(cls_idx, n_take, replace=False)
            labeled_idx.extend(chosen.tolist())
            print(f"      Class {cls} ({class_names[cls]}): {n_take:,} / {len(cls_idx):,} labeled")
    labeled_idx = np.array(sorted(labeled_idx), dtype=np.int64)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    unlabeled_idx = np.where(~labeled_mask)[0]
    print(f"   Total labeled:   {len(labeled_idx):,}")
    print(f"   Total unlabeled: {len(unlabeled_idx):,}")
    return labeled_idx, unlabeled_idx


# ============================================================================
# C1 CORE: FEATURE BUILDER + CLASSIFIER
# ============================================================================

def build_onehot_cluster_mapping(cluster_ids, min_freq=MIN_CLUSTER_ID_FREQ):
    unique, counts = np.unique(cluster_ids, return_counts=True)
    valid = unique[counts >= min_freq]
    if -1 not in valid and -1 in unique:
        valid = np.append(valid, -1)
    return {int(c): i for i, c in enumerate(valid)}, len(valid)


def append_cluster_onehot(embeddings, cluster_ids, cluster_to_idx, n_clusters):
    n_samples = len(embeddings)
    onehot = np.zeros((n_samples, n_clusters), dtype=np.float32)
    for i, cid in enumerate(cluster_ids):
        idx = cluster_to_idx.get(int(cid))
        if idx is not None:
            onehot[i, idx] = 1.0
    return np.concatenate([np.asarray(embeddings, dtype=np.float32), onehot], axis=1)


def load_embeddings_chunked(embeddings_source, indices, chunk_size=50_000, desc="   Loading"):
    """Load specific rows from a memmap into a contiguous array, in chunks."""
    n = len(indices)
    out = None
    feat_dim = None
    for start in tqdm(range(0, n, chunk_size), desc=desc):
        end = min(start + chunk_size, n)
        idx_chunk = indices[start:end]
        emb_chunk = np.asarray(embeddings_source[idx_chunk], dtype=np.float32)
        if out is None:
            feat_dim = emb_chunk.shape[1]
            out = np.empty((n, feat_dim), dtype=np.float32)
        out[start:end] = emb_chunk
    return out


def train_ssl_classifier(X_train, y_train, C=CLASSIFIER_C, max_iter=CLASSIFIER_MAX_ITER):
    print(f"   Training LogReg (binary, C={C}, max_iter={max_iter}, samples={len(X_train):,})...")
    clf = LogisticRegression(
        C=C, max_iter=max_iter, n_jobs=-1,
        class_weight='balanced', solver='lbfgs'
    )
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"   ✓ Train accuracy: {train_acc:.4f}")
    return clf


def pseudo_label_unlabeled(clf, X_unlabeled, threshold=PSEUDO_CONFIDENCE_THRESHOLD):
    """Predict on unlabeled set, return (pseudo_mask, pseudo_labels)."""
    proba = clf.predict_proba(X_unlabeled)
    max_probs = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    pseudo_mask = max_probs >= threshold
    n_pseudo = int(pseudo_mask.sum())
    print(f"   Pseudo-labels @ {threshold:.0%}: {n_pseudo:,} / {len(X_unlabeled):,}")
    return pseudo_mask, predictions, max_probs


# ============================================================================
# METRICS + VISUALIZATION
# ============================================================================

def calculate_metrics(y_true, y_pred):
    print("\n📊 Calculating metrics...")
    unique_true = sorted(set(y_true))
    true_labels = [cls for cls in [0, 1] if cls in unique_true]
    pred_labels = [0, 1]
    class_names = ['NORMAL', 'ANOMALY']

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=['NORMAL', 'ANOMALY'],
        output_dict=True, zero_division=0
    )

    print(f"\n{'='*70}")
    print("OVERALL METRICS (C1 Pseudo-Labeling, 2-class)")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    for name in ['NORMAL', 'ANOMALY']:
        r = report[name]
        print(f"  {name:8s}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
              f"F1={r['f1-score']:.4f}  Support={int(r['support']):,}")
    print(f"\nConfusion Matrix (2x2):")
    print(f"                 Predicted")
    header = "                 " + "".join([f"{class_names[c]:>7}" for c in pred_labels])
    print(header)
    for i, tc in enumerate(true_labels):
        row = f"    True  {class_names[tc]:6s} [" + \
              "".join([f"{cm[i,j]:>6} " for j in range(2)]) + "]"
        print(row)

    with open(OUTPUT_METRICS, 'w') as f:
        f.write(f"C1 Pseudo-Labeling (2-class) - {DATASET} {ALGORITHM.upper()} {EMBEDDING_TYPE.upper()}\n")
        f.write(f"LABELED_FRACTION={LABELED_FRACTION}\n")
        f.write(f"PSEUDO_CONFIDENCE_THRESHOLD={PSEUDO_CONFIDENCE_THRESHOLD}\n")
        f.write(f"MAX_ITERATIONS={MAX_ITERATIONS}\n")
        f.write(f"USE_NOISY_STUDENT={USE_NOISY_STUDENT}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, labels=[0, 1],
                                      target_names=['NORMAL', 'ANOMALY']))
        f.write(f"\nConfusion Matrix (2x2):\n{header}\n")
        for i, tc in enumerate(true_labels):
            row = f"    True  {class_names[tc]:6s} [" + \
                  "".join([f"{cm[i,j]:>6} " for j in range(2)]) + "]\n"
            f.write(row)
    print(f"\n✓ Metrics saved to: {OUTPUT_METRICS}")
    return {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm,
            'true_labels': true_labels, 'pred_labels': pred_labels}


def visualize_results(metrics, history, y_true):
    print("\n📈 Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cm = metrics['confusion_matrix']
    class_names = ['Normal', 'Anomaly']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=[class_names[i] for i in metrics['true_labels']])
    axes[0].set_title('Confusion Matrix (C1 Pseudo-Labeling, 2-class)')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')

    if history:
        iters = [h['iteration'] for h in history]
        n_pseudo = [h['n_pseudo'] for h in history]
        n_train = [h['n_train_total'] for h in history]
        ax2 = axes[1]
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('# Pseudo-labels', color='tab:blue')
        ax2.bar(iters, n_pseudo, color='tab:blue', alpha=0.6, label='# New pseudo')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax3 = ax2.twinx()
        ax3.set_ylabel('# Total training samples', color='tab:green')
        ax3.plot(iters, n_train, color='tab:green', marker='s', label='Total train')
        ax3.tick_params(axis='y', labelcolor='tab:green')
        plt.title('Pseudo-Labeling Iteration Progress')
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFUSION_MATRIX, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_CONFUSION_MATRIX.name}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print(f"C1 PSEUDO-LABELING - {DATASET} {ALGORITHM.upper()} {EMBEDDING_TYPE.upper()}")
    print("="*70)
    print(f"  LABELED_FRACTION             = {LABELED_FRACTION}")
    print(f"  PSEUDO_CONFIDENCE_THRESHOLD  = {PSEUDO_CONFIDENCE_THRESHOLD}")
    print(f"  MAX_ITERATIONS               = {MAX_ITERATIONS}")
    print(f"  USE_NOISY_STUDENT            = {USE_NOISY_STUDENT}")
    print(f"  USE_CLUSTER_ID_AS_FEATURE    = {USE_CLUSTER_ID_AS_FEATURE}")

    # ------------------------------------------------------------------
    # STEP 1: Load training data
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: LOAD TRAINING DATA")
    print("="*70)
    training_cluster_labels = np.load(TRAINING_LABELS_PATH)
    print(f"   Training cluster labels: {len(training_cluster_labels):,} samples, "
          f"{len(set(training_cluster_labels))} clusters")

    # Load embeddings: try multiple strategies since file may be pickled or raw
    print(f"   Loading training embeddings: {TRAINING_EMBEDDINGS_PATH}")
    training_embeddings = None
    fsize = TRAINING_EMBEDDINGS_PATH.stat().st_size

    # Strategy 1: standard np.load (handles .npy header)
    try:
        training_embeddings = np.load(TRAINING_EMBEDDINGS_PATH, mmap_mode='r')
        print(f"   ✓ Loaded with mmap (has .npy header)")
    except Exception as e1:
        print(f"   ⚠️ Standard load failed: {type(e1).__name__}: {str(e1)[:80]}")

        # Strategy 2: file might be pickled object
        try:
            training_embeddings = np.load(TRAINING_EMBEDDINGS_PATH, allow_pickle=True)
            print(f"   ✓ Loaded with allow_pickle=True (in-memory)")
        except Exception as e2:
            print(f"   ⚠️ Pickle load failed: {type(e2).__name__}: {str(e2)[:80]}")

            # Strategy 3: raw float32 memmap (most likely for big embedding files)
            print(f"   🔍 Attempting raw float32 memmap...")
            print(f"   File size: {fsize:,} bytes, target rows: {len(training_cluster_labels):,}")

            with open(TRAINING_EMBEDDINGS_PATH, 'rb') as fh:
                first_bytes = fh.read(6)

            # Check for .npy header magic
            has_npy_header = (first_bytes == b'\x93NUMPY')
            target_n = len(training_cluster_labels)

            if has_npy_header:
                # Has header but failed earlier - try reading header manually
                try:
                    arr = np.load(TRAINING_EMBEDDINGS_PATH, allow_pickle=False)
                    if isinstance(arr, np.ndarray) and arr.ndim == 2:
                        training_embeddings = arr
                        print(f"   ✓ Loaded raw array (shape={arr.shape})")
                except Exception:
                    pass

            if training_embeddings is None:
                # Infer shape from file size: total_bytes = n * d * 4 (float32)
                target_bytes = target_n * 4
                if fsize % target_bytes != 0:
                    print(f"   ⚠️ File size {fsize:,} not divisible by rows×4. Trying generic dims...")
                found = False
                for d in [128, 256, 384, 512, 768, 1024, 1536, 2048]:
                    if fsize == target_n * d * 4:
                        training_embeddings = np.memmap(
                            str(TRAINING_EMBEDDINGS_PATH),
                            dtype='float32', mode='r', shape=(target_n, d)
                        )
                        print(f"   ✓ Loaded as raw memmap with shape=({target_n},{d})")
                        found = True
                        break
                if not found:
                    # Last resort: assume the most likely dim and report mismatch
                    guessed_d = fsize // (target_n * 4) if target_n else 256
                    raise RuntimeError(
                        f"Cannot auto-detect embedding shape.\n"
                        f"  File size: {fsize:,} bytes\n"
                        f"  Target rows: {target_n:,}\n"
                        f"  Bytes per row if float32: {fsize // max(target_n, 1):,}\n"
                        f"  Guessed dim: {guessed_d}\n"
                        f"  Tried dims: [128, 256, 384, 512, 768, 1024, 1536, 2048]"
                    )
    print(f"   ✓ Shape: {training_embeddings.shape}")
    if len(training_embeddings) != len(training_cluster_labels):
        n = min(len(training_embeddings), len(training_cluster_labels))
        print(f"   ⚠️ Truncating to {n:,}")
        training_embeddings = training_embeddings[:n]
        training_cluster_labels = training_cluster_labels[:n]

    # ------------------------------------------------------------------
    # STEP 2: Per-sample 2-class label (REPLACES majority vote)
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 2: ASSIGN PER-SAMPLE LABELS (no majority vote)")
    print("="*70)
    metadata_codes = load_metadata_codes_for_dataset(
        METADATA_TSV_PATH, NORMAL_TEMPLATE_PATH, NONNORMAL_TEMPLATE_PATH
    )
    print(f"   Metadata codes: {len(metadata_codes):,}")
    sample_labels = assign_per_sample_labels(training_cluster_labels, metadata_codes)
    np.save(OUTPUT_TRAIN_LABELS, sample_labels)

    # ------------------------------------------------------------------
    # STEP 3: Split labeled / unlabeled
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 3: SPLIT LABELED / UNLABELED")
    print("="*70)
    labeled_idx, unlabeled_idx = split_labeled_unlabeled(sample_labels, LABELED_FRACTION)

    # ------------------------------------------------------------------
    # STEP 4: Build feature matrix + train initial classifier
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4: TRAIN INITIAL CLASSIFIER (on labeled pool)")
    print("="*70)

    if USE_CLUSTER_ID_AS_FEATURE:
        cluster_to_idx, n_clusters_feat = build_onehot_cluster_mapping(training_cluster_labels)
        print(f"   Cluster one-hot features: {n_clusters_feat}")
    else:
        cluster_to_idx, n_clusters_feat = {}, 0

    # Subsample labeled if too large
    labeled_idx_use = labeled_idx
    if len(labeled_idx) > LABELED_SUBSAMPLE_CAP:
        rng = np.random.default_rng(42)
        labeled_idx_use = rng.choice(labeled_idx, LABELED_SUBSAMPLE_CAP, replace=False)
        print(f"   Subsampled labeled: {len(labeled_idx):,} -> {len(labeled_idx_use):,}")

    print(f"   Loading labeled embeddings...")
    X_labeled_emb = load_embeddings_chunked(
        training_embeddings, labeled_idx_use, desc="   Loading labeled"
    )
    if USE_CLUSTER_ID_AS_FEATURE:
        X_labeled = append_cluster_onehot(
            X_labeled_emb, training_cluster_labels[labeled_idx_use],
            cluster_to_idx, n_clusters_feat
        )
        del X_labeled_emb; gc.collect()
    else:
        X_labeled = X_labeled_emb
    y_labeled = sample_labels[labeled_idx_use]
    print(f"   ✓ X_labeled: {X_labeled.shape}, y distribution: {Counter(y_labeled.tolist())}")

    # Load test data (for final prediction)
    print("\n   Loading test data for final prediction...")
    test_embeddings, test_gt_labels, test_set_info = load_multiple_testing_sets(TESTING_SETS)

    # Assign test to clusters
    print("\n" + "="*70)
    print("STEP 5: ASSIGN TEST SAMPLES TO TRAINING CLUSTERS")
    print("="*70)
    if ALGORITHM == "kmeans":
        model = None
        try:
            model = load_kmeans_model_compat(TRAINED_MODEL_PATH)
            test_cluster_labels = model.predict(test_embeddings)
        except Exception as e:
            print(f"   ⚠️ KMeans model load failed: {e}, using centroid fallback")
            train_emb = training_embeddings
            if train_emb.shape[1] != test_embeddings.shape[1]:
                resolved = find_kmeans_training_embeddings_for_dim(
                    TRAINING_EMBEDDINGS_PATH, DATASET, EMBEDDING_TYPE,
                    int(test_embeddings.shape[1])
                )
                train_emb = resolved[1] if resolved else train_emb
            cluster_ids, centroids = build_kmeans_centroids_from_labels(
                train_emb, training_cluster_labels, chunk_size=300000
            )
            test_cluster_labels = predict_kmeans_from_centroids(
                test_embeddings, cluster_ids, centroids,
                use_cosine=USE_COSINE_DISTANCE, batch_size=20000
            )
    else:
        faiss_result = fast_cluster_assignment_faiss(
            training_embeddings, training_cluster_labels,
            test_embeddings, use_cosine=USE_COSINE_DISTANCE
        )
        if faiss_result is None:
            sklearn_result = fast_cluster_assignment_sklearn_batched(
                training_embeddings, training_cluster_labels,
                test_embeddings, use_cosine=USE_COSINE_DISTANCE
            )
            test_cluster_labels = sklearn_result['labels']
        else:
            test_cluster_labels = faiss_result['labels']
    print(f"   ✓ Assigned {len(test_cluster_labels):,} test samples")
    np.save(OUTPUT_CLUSTER_LABELS, test_cluster_labels)

    # ------------------------------------------------------------------
    # STEP 6: Iterative pseudo-labeling
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 6: ITERATIVE PSEUDO-LABELING")
    print("="*70)

    # Maintain a list of training indices (labeled + pseudo-labeled)
    train_indices = list(labeled_idx_use)  # will grow
    train_labels_combined = list(y_labeled.tolist())

    history = []
    final_clf = None
    prev_test_predictions = None

    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {it}/{MAX_ITERATIONS} ---")
        print(f"   Training pool: {len(train_indices):,} samples "
              f"(labeled={len(labeled_idx_use):,} + pseudo={len(train_indices)-len(labeled_idx_use):,})")

        # Train classifier
        X_train = np.array(train_indices)  # placeholder, will be replaced
        # Build X_train from current train_indices
        # (to keep memory low, we re-load and concatenate)
        # For simplicity, just reload labeled and pseudo indices
        all_idx = np.array(train_indices, dtype=np.int64)
        if len(all_idx) > CLASSIFIER_SAMPLE_CAP:
            rng = np.random.default_rng(123 + it)
            all_idx = rng.choice(all_idx, CLASSIFIER_SAMPLE_CAP, replace=False)
        X_train_emb = load_embeddings_chunked(
            training_embeddings, all_idx, desc=f"   It{it} load train"
        )
        if USE_CLUSTER_ID_AS_FEATURE:
            X_train = append_cluster_onehot(
                X_train_emb, training_cluster_labels[all_idx],
                cluster_to_idx, n_clusters_feat
            )
            del X_train_emb; gc.collect()
        else:
            X_train = X_train_emb
        # Map train_indices -> labels using a dict for O(1) lookup
        idx_to_label = {idx: lbl for idx, lbl in zip(train_indices, train_labels_combined)}
        y_train_combined = np.array([idx_to_label[int(i)] for i in all_idx], dtype=np.int32)

        final_clf = train_ssl_classifier(X_train, y_train_combined)

        # Predict on UNLABELED training samples (the ones we still haven't pseudo-labeled)
        unlabeled_remaining = np.setdiff1d(unlabeled_idx, np.array(train_indices))
        n_remaining = len(unlabeled_remaining)
        print(f"   Unlabeled remaining: {n_remaining:,}")

        if n_remaining == 0:
            print(f"   ✓ All training samples have been pseudo-labeled")
            break

        if n_remaining > CLASSIFIER_SAMPLE_CAP:
            rng = np.random.default_rng(999 + it)
            unlabeled_sample = rng.choice(unlabeled_remaining, CLASSIFIER_SAMPLE_CAP, replace=False)
        else:
            unlabeled_sample = unlabeled_remaining

        X_unlabeled_emb = load_embeddings_chunked(
            training_embeddings, unlabeled_sample, desc=f"   It{it} load unlabeled"
        )
        if USE_CLUSTER_ID_AS_FEATURE:
            X_unlabeled = append_cluster_onehot(
                X_unlabeled_emb, training_cluster_labels[unlabeled_sample],
                cluster_to_idx, n_clusters_feat
            )
            del X_unlabeled_emb; gc.collect()
        else:
            X_unlabeled = X_unlabeled_emb

        pseudo_mask, pseudo_preds, pseudo_confs = pseudo_label_unlabeled(final_clf, X_unlabeled)
        n_pseudo = int(pseudo_mask.sum())

        # Add pseudo-labels to training set
        new_pseudo_idx = unlabeled_sample[pseudo_mask]
        new_pseudo_lbl = pseudo_preds[pseudo_mask]
        train_indices.extend(new_pseudo_idx.tolist())
        train_labels_combined.extend(new_pseudo_lbl.tolist())

        # Evaluate on test set (early monitoring)
        X_test_emb = np.asarray(test_embeddings, dtype=np.float32)
        if USE_CLUSTER_ID_AS_FEATURE:
            X_test = append_cluster_onehot(
                X_test_emb, test_cluster_labels, cluster_to_idx, n_clusters_feat
            )
        else:
            X_test = X_test_emb
        test_preds = final_clf.predict(X_test)
        test_proba = final_clf.predict_proba(X_test).max(axis=1)
        test_acc = accuracy_score(test_gt_labels, test_preds)
        print(f"   Test accuracy (early): {test_acc:.4f}")

        history.append({
            'iteration': it,
            'n_pseudo': n_pseudo,
            'n_train_total': len(train_indices),
            'accuracy': test_acc
        })

        # Convergence
        if prev_test_predictions is not None:
            change = float(np.mean(test_preds != prev_test_predictions))
            print(f"   Test prediction change: {change:.4f}")
            if change < CONVERGENCE_TOL:
                print(f"   ✓ Converged")
                break
        prev_test_predictions = test_preds

        if n_pseudo < MIN_PSEUDO_PER_ITERATION and it > 1:
            print(f"   ✓ Stopped: too few new pseudo-labels")
            break

    # ------------------------------------------------------------------
    # STEP 7: Final test prediction
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 7: FINAL TEST PREDICTION")
    print("="*70)
    if final_clf is None:
        raise RuntimeError("Classifier was never trained")
    X_test_emb = np.asarray(test_embeddings, dtype=np.float32)
    if USE_CLUSTER_ID_AS_FEATURE:
        X_test = append_cluster_onehot(X_test_emb, test_cluster_labels, cluster_to_idx, n_clusters_feat)
    else:
        X_test = X_test_emb
    final_predictions = final_clf.predict(X_test)
    final_proba = final_clf.predict_proba(X_test)
    final_confidence = final_proba.max(axis=1)
    print(f"   ✓ Predicted {len(final_predictions):,} test samples")
    print(f"   Distribution: {Counter(final_predictions.tolist())}")

    # ------------------------------------------------------------------
    # STEP 8: Save outputs + metrics
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 8: SAVE OUTPUTS + METRICS")
    print("="*70)
    np.save(OUTPUT_PREDICTIONS, final_predictions)
    np.save(OUTPUT_CONFIDENCE, final_confidence)
    pd.DataFrame(history).to_csv(OUTPUT_PSEUDO_HISTORY, index=False)
    test_set_names = np.empty(len(test_gt_labels), dtype=object)
    for info in test_set_info:
        test_set_names[info['start_idx']:info['end_idx']] = info['name']
    detailed_df = pd.DataFrame({
        'sample_idx': np.arange(len(test_gt_labels)),
        'test_set': test_set_names,
        'true_label': test_gt_labels,
        'cluster_id': test_cluster_labels,
        'predicted_label': final_predictions,
        'confidence': final_confidence,
    })
    detailed_df.to_csv(OUTPUT_DETAILED_RESULTS, index=False)
    print(f"   ✓ Predictions:  {OUTPUT_PREDICTIONS}")
    print(f"   ✓ Confidence:   {OUTPUT_CONFIDENCE}")
    print(f"   ✓ History:      {OUTPUT_PSEUDO_HISTORY}")
    print(f"   ✓ Detailed CSV: {OUTPUT_DETAILED_RESULTS}")

    metrics = calculate_metrics(test_gt_labels, final_predictions)
    visualize_results(metrics, history, test_gt_labels)

    print("\n" + "="*70)
    print("✅ C1 PSEUDO-LABELING COMPLETE")
    print("="*70)
    print(f"   Final accuracy: {metrics['accuracy']:.4f}")
    print(f"   Total iterations: {len(history)}")
    print(f"   Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
