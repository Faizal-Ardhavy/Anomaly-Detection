# K-Means Testing Script
# Testing data baru menggunakan trained model dari training

import numpy as np
from pathlib import Path
import gc
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from collections import Counter
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# ============================================================================
# CONFIGURATION - EDIT PATHS DI SINI!
# ============================================================================

# Path ke trained model dari training
TRAINED_MODEL_PATH = Path("model_kmeans_log.pkl")
TRAINING_CENTROIDS_PATH = Path("cluster_centroids.npy")
TRAINING_LABELS_PATH = Path("cluster_labels.npy")  # Optional, untuk comparison

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
OUTPUT_TEST_LABELS = Path("kmeans_labels_test.npy")
OUTPUT_TEST_DISTANCES = Path("kmeans_distances_test.npy")  # Distance ke centroid terdekat
OUTPUT_ANALYSIS_CSV = Path("kmeans_test_analysis.csv")

RANDOM_STATE = 42
SAMPLE_FOR_METRICS = 50000  # Sample untuk compute metrics jika dataset test besar

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_embedding_dim(file_path: Path) -> int:
    """Auto-detect embedding dimension from filename"""
    filename = file_path.name.lower()
    if 'pca256' in filename:
        return 256
    elif 'pca128' in filename:
        return 128
    else:
        return 768

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
# LOAD TRAINED MODEL
# ============================================================================

print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

if not TRAINED_MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {TRAINED_MODEL_PATH}")

model = joblib.load(TRAINED_MODEL_PATH)
print(f"✓ Loaded model: {TRAINED_MODEL_PATH}")

# Get model info
n_clusters = model.n_clusters
model_type = type(model).__name__
print(f"\nModel Information:")
print(f"  Type: {model_type}")
print(f"  Number of clusters (K): {n_clusters}")
print(f"  Random state: {model.random_state}")

# Load centroids
if TRAINING_CENTROIDS_PATH.exists():
    centroids = np.load(TRAINING_CENTROIDS_PATH)
    print(f"  Centroids shape: {centroids.shape}")
else:
    centroids = model.cluster_centers_
    print(f"  Centroids from model: {centroids.shape}")

# Load training labels (optional, for distribution comparison)
training_dist = None
if TRAINING_LABELS_PATH.exists():
    labels_train = np.load(TRAINING_LABELS_PATH)
    training_dist = Counter(labels_train)
    print(f"\nTraining Distribution:")
    for k in range(n_clusters):
        count = training_dist.get(k, 0)
        pct = (count / len(labels_train)) * 100
        print(f"  Cluster {k}: {count:,} samples ({pct:.2f}%)")

# ============================================================================
# LOAD TESTING DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING TESTING DATA")
print("="*70)

emb_test = load_embeddings_from_files(TESTING_EMBEDDINGS_PATHS)
print(f"\nTesting data shape: {emb_test.shape}")

# Verify dimension matches
if emb_test.shape[1] != centroids.shape[1]:
    raise ValueError(
        f"Dimension mismatch! Testing data: {emb_test.shape[1]}, "
        f"Model expects: {centroids.shape[1]}"
    )

n_test = emb_test.shape[0]
print(f"Total testing samples: {n_test:,}")

# ============================================================================
# PREDICTION ON TESTING DATA
# ============================================================================

print("\n" + "="*70)
print("PREDICTING CLUSTERS FOR TESTING DATA")
print("="*70)

# For large datasets, predict in chunks
CHUNK_SIZE = 100_000

if n_test > CHUNK_SIZE:
    print(f"Large dataset detected, processing in chunks of {CHUNK_SIZE:,}...")
    
    labels_test = np.zeros(n_test, dtype=np.int32)
    distances_test = np.zeros(n_test, dtype=np.float32)
    
    n_chunks = (n_test + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for start in tqdm(range(0, n_test, CHUNK_SIZE), desc='Predicting', total=n_chunks, unit='chunk'):
        end = min(start + CHUNK_SIZE, n_test)
        chunk = emb_test[start:end]
        
        # Convert memmap to array if needed
        if isinstance(chunk, np.memmap):
            chunk = np.array(chunk)
        
        # Predict
        labels_test[start:end] = model.predict(chunk)
        
        # Compute distance to nearest centroid
        chunk_centroids = centroids[labels_test[start:end]]
        distances_test[start:end] = np.linalg.norm(chunk - chunk_centroids, axis=1)
    
    print(f"✓ Prediction complete")
else:
    print("Processing entire dataset at once...")
    
    if isinstance(emb_test, np.memmap):
        emb_test = np.array(emb_test)
    
    labels_test = model.predict(emb_test)
    
    # Compute distance to nearest centroid
    test_centroids = centroids[labels_test]
    distances_test = np.linalg.norm(emb_test - test_centroids, axis=1)
    
    print(f"✓ Prediction complete")

# Save results
np.save(OUTPUT_TEST_LABELS, labels_test)
np.save(OUTPUT_TEST_DISTANCES, distances_test)

print(f"\n✅ Saved: {OUTPUT_TEST_LABELS} ({len(labels_test):,} labels)")
print(f"✅ Saved: {OUTPUT_TEST_DISTANCES}")

# ============================================================================
# CLUSTER DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("CLUSTER DISTRIBUTION ANALYSIS")
print("="*70)

test_dist = Counter(labels_test)

print(f"\nTesting Distribution:")
for k in range(n_clusters):
    count = test_dist.get(k, 0)
    pct = (count / n_test) * 100
    print(f"  Cluster {k}: {count:,} samples ({pct:.2f}%)")

# Compare with training distribution
if training_dist is not None:
    print(f"\n📊 Training vs Testing Comparison:")
    print(f"\n  {'Cluster':<10s} {'Training':>15s} {'Testing':>15s} {'Difference':>15s}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    
    for k in range(n_clusters):
        train_count = training_dist.get(k, 0)
        test_count = test_dist.get(k, 0)
        train_pct = (train_count / len(labels_train)) * 100
        test_pct = (test_count / n_test) * 100
        diff_pct = test_pct - train_pct
        
        print(f"  {k:<10d} {train_pct:>14.2f}% {test_pct:>14.2f}% {diff_pct:>+14.2f}%")
    
    # Distribution stability assessment
    max_diff = max(abs((test_dist.get(k, 0) / n_test) - (training_dist.get(k, 0) / len(labels_train))) 
                   for k in range(n_clusters)) * 100
    
    if max_diff < 5.0:
        stability = "✅ HIGH (distributions very similar)"
    elif max_diff < 10.0:
        stability = "⚠️ MODERATE (some differences)"
    else:
        stability = "❌ LOW (significant differences)"
    
    print(f"\n  Distribution Stability: {stability}")
    print(f"  Max difference: {max_diff:.2f}%")

# ============================================================================
# DISTANCE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("DISTANCE TO CENTROID ANALYSIS")
print("="*70)

print(f"\nDistance Statistics (testing data):")
print(f"  Min distance:    {distances_test.min():.4f}")
print(f"  Max distance:    {distances_test.max():.4f}")
print(f"  Mean distance:   {distances_test.mean():.4f}")
print(f"  Median distance: {np.median(distances_test):.4f}")
print(f"  Std distance:    {distances_test.std():.4f}")

# Distance percentiles
percentiles = [50, 75, 90, 95, 99]
print(f"\n  Distance Percentiles:")
for p in percentiles:
    val = np.percentile(distances_test, p)
    print(f"    {p}th: {val:.4f}")

# Per-cluster distance analysis
print(f"\n📊 Distance by Cluster:")
print(f"\n  {'Cluster':<10s} {'Count':>12s} {'Mean Dist':>12s} {'Std Dist':>12s}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

cluster_stats = []
for k in range(n_clusters):
    mask = labels_test == k
    if mask.sum() > 0:
        cluster_distances = distances_test[mask]
        mean_dist = cluster_distances.mean()
        std_dist = cluster_distances.std()
        print(f"  {k:<10d} {mask.sum():>12,} {mean_dist:>12.4f} {std_dist:>12.4f}")
        
        cluster_stats.append({
            'cluster': k,
            'count': int(mask.sum()),
            'pct': float(mask.sum() / n_test * 100),
            'mean_distance': float(mean_dist),
            'std_distance': float(std_dist),
            'min_distance': float(cluster_distances.min()),
            'max_distance': float(cluster_distances.max())
        })
    else:
        print(f"  {k:<10d} {0:>12,} {0.0:>12.4f} {0.0:>12.4f}")

# ============================================================================
# QUALITY METRICS
# ============================================================================

print("\n" + "="*70)
print("CLUSTERING QUALITY METRICS")
print("="*70)

# For large datasets, compute metrics on sample
if n_test > SAMPLE_FOR_METRICS:
    print(f"\nComputing metrics on sample ({SAMPLE_FOR_METRICS:,} samples)...")
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(n_test, SAMPLE_FOR_METRICS, replace=False)
    
    if isinstance(emb_test, np.memmap):
        emb_sample = np.array(emb_test[sample_idx])
    else:
        emb_sample = emb_test[sample_idx]
    
    labels_sample = labels_test[sample_idx]
else:
    print(f"\nComputing metrics on full testing set...")
    if isinstance(emb_test, np.memmap):
        emb_sample = np.array(emb_test)
    else:
        emb_sample = emb_test
    labels_sample = labels_test

# Compute metrics
try:
    silhouette = silhouette_score(emb_sample, labels_sample, sample_size=min(50000, len(emb_sample)))
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"    Interpretation: {'Good' if silhouette > 0.5 else 'Moderate' if silhouette > 0.3 else 'Poor'}")
except Exception as e:
    print(f"  Silhouette Score: N/A ({e})")
    silhouette = None

try:
    dbi = davies_bouldin_score(emb_sample, labels_sample)
    print(f"  Davies-Bouldin Index: {dbi:.4f} (lower is better)")
    print(f"    Interpretation: {'Good' if dbi < 1.0 else 'Moderate' if dbi < 2.0 else 'Poor'}")
except Exception as e:
    print(f"  Davies-Bouldin Index: N/A ({e})")
    dbi = None

try:
    ch = calinski_harabasz_score(emb_sample, labels_sample)
    print(f"  Calinski-Harabasz Score: {ch:.2f} (higher is better)")
    print(f"    Interpretation: {'Good' if ch > 1000 else 'Moderate' if ch > 500 else 'Poor'}")
except Exception as e:
    print(f"  Calinski-Harabasz Index: N/A ({e})")
    ch = None

# Model inertia on testing data
if hasattr(model, 'inertia_'):
    # Compute inertia on testing data
    test_centroids = centroids[labels_test]
    if isinstance(emb_test, np.memmap):
        # Compute in chunks for memmap
        test_inertia = 0.0
        for start in range(0, n_test, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_test)
            chunk = np.array(emb_test[start:end])
            chunk_centroids = test_centroids[start:end]
            test_inertia += np.sum((chunk - chunk_centroids) ** 2)
    else:
        test_inertia = np.sum((emb_test - test_centroids) ** 2)
    
    print(f"  Inertia (testing): {test_inertia:,.2f}")
    print(f"    (Sum of squared distances to nearest centroid)")

# ============================================================================
# OUTLIER DETECTION (based on distance)
# ============================================================================

print("\n" + "="*70)
print("OUTLIER DETECTION (based on distance to centroid)")
print("="*70)

# Define outlier threshold (e.g., 95th percentile)
outlier_threshold = np.percentile(distances_test, 95)
outliers_mask = distances_test > outlier_threshold
n_outliers = outliers_mask.sum()

print(f"\nOutlier threshold (95th percentile): {outlier_threshold:.4f}")
print(f"Number of outliers: {n_outliers:,} ({n_outliers/n_test*100:.2f}%)")

if n_outliers > 0:
    print(f"\nOutliers by cluster:")
    for k in range(n_clusters):
        cluster_outliers = (labels_test == k) & outliers_mask
        n_cluster_outliers = cluster_outliers.sum()
        cluster_size = (labels_test == k).sum()
        if cluster_size > 0:
            pct = (n_cluster_outliers / cluster_size) * 100
            print(f"  Cluster {k}: {n_cluster_outliers:,} / {cluster_size:,} ({pct:.2f}%)")

# ============================================================================
# SAVE ANALYSIS REPORT
# ============================================================================

print("\n" + "="*70)
print("SAVING ANALYSIS REPORT")
print("="*70)

# Create detailed report
report_data = []
for stat in cluster_stats:
    k = stat['cluster']
    
    # Count outliers in this cluster
    cluster_mask = labels_test == k
    cluster_outliers = (cluster_mask & outliers_mask).sum()
    
    report_data.append({
        'cluster_id': k,
        'test_count': stat['count'],
        'test_percentage': stat['pct'],
        'mean_distance': stat['mean_distance'],
        'std_distance': stat['std_distance'],
        'min_distance': stat['min_distance'],
        'max_distance': stat['max_distance'],
        'outliers': cluster_outliers,
        'outlier_percentage': (cluster_outliers / stat['count'] * 100) if stat['count'] > 0 else 0
    })

df_report = pd.DataFrame(report_data)

# Add summary row
summary = {
    'cluster_id': 'TOTAL',
    'test_count': n_test,
    'test_percentage': 100.0,
    'mean_distance': distances_test.mean(),
    'std_distance': distances_test.std(),
    'min_distance': distances_test.min(),
    'max_distance': distances_test.max(),
    'outliers': n_outliers,
    'outlier_percentage': (n_outliers / n_test * 100)
}
df_report = pd.concat([df_report, pd.DataFrame([summary])], ignore_index=True)

df_report.to_csv(OUTPUT_ANALYSIS_CSV, index=False)
print(f"✅ Saved: {OUTPUT_ANALYSIS_CSV}")

print("\n" + df_report.to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Cluster size distribution
ax1 = plt.subplot(2, 3, 1)
clusters = list(range(n_clusters))
counts = [test_dist.get(k, 0) for k in clusters]
ax1.bar(clusters, counts, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Cluster ID')
ax1.set_ylabel('Number of Samples')
ax1.set_title('Testing: Cluster Size Distribution')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Distance distribution (overall)
ax2 = plt.subplot(2, 3, 2)
ax2.hist(distances_test, bins=50, alpha=0.7, edgecolor='black')
ax2.axvline(outlier_threshold, color='r', linestyle='--', label=f'95th percentile: {outlier_threshold:.2f}')
ax2.set_xlabel('Distance to Nearest Centroid')
ax2.set_ylabel('Frequency')
ax2.set_title('Distance Distribution')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Distance by cluster (boxplot)
ax3 = plt.subplot(2, 3, 3)
cluster_distances_list = [distances_test[labels_test == k] for k in range(n_clusters)]
ax3.boxplot(cluster_distances_list, labels=clusters)
ax3.set_xlabel('Cluster ID')
ax3.set_ylabel('Distance to Centroid')
ax3.set_title('Distance Distribution by Cluster')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Cluster percentage distribution
ax4 = plt.subplot(2, 3, 4)
percentages = [(test_dist.get(k, 0) / n_test * 100) for k in clusters]
ax4.bar(clusters, percentages, alpha=0.7, edgecolor='black', color='orange')
ax4.set_xlabel('Cluster ID')
ax4.set_ylabel('Percentage (%)')
ax4.set_title('Testing: Cluster Percentage Distribution')
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Training vs Testing comparison (if available)
if training_dist is not None:
    ax5 = plt.subplot(2, 3, 5)
    train_pcts = [(training_dist.get(k, 0) / len(labels_train) * 100) for k in clusters]
    test_pcts = percentages
    
    x = np.arange(len(clusters))
    width = 0.35
    ax5.bar(x - width/2, train_pcts, width, label='Training', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, test_pcts, width, label='Testing', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Cluster ID')
    ax5.set_ylabel('Percentage (%)')
    ax5.set_title('Training vs Testing Distribution')
    ax5.set_xticks(x)
    ax5.set_xticklabels(clusters)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

# Plot 6: Outlier percentage by cluster
ax6 = plt.subplot(2, 3, 6)
outlier_pcts = [(report_data[k]['outlier_percentage']) for k in range(n_clusters)]
ax6.bar(clusters, outlier_pcts, alpha=0.7, edgecolor='black', color='red')
ax6.set_xlabel('Cluster ID')
ax6.set_ylabel('Outlier Percentage (%)')
ax6.set_title('Outliers by Cluster (>95th percentile)')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_test_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: kmeans_test_analysis.png")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ TESTING COMPLETE - SUMMARY")
print("="*70)

print(f"\n📊 Dataset:")
print(f"  Testing samples: {n_test:,}")
print(f"  Dimensions: {emb_test.shape[1]}")
print(f"  Clusters (K): {n_clusters}")

print(f"\n📈 Quality Metrics:")
if silhouette is not None:
    print(f"  Silhouette Score: {silhouette:.4f}")
if dbi is not None:
    print(f"  Davies-Bouldin Index: {dbi:.4f}")
if ch is not None:
    print(f"  Calinski-Harabasz: {ch:.2f}")

print(f"\n📍 Distance Analysis:")
print(f"  Mean distance: {distances_test.mean():.4f}")
print(f"  Std distance: {distances_test.std():.4f}")
print(f"  Outliers (>95th): {n_outliers:,} ({n_outliers/n_test*100:.2f}%)")

if training_dist is not None:
    print(f"\n🔄 Distribution Stability: {stability}")
    print(f"  Max difference: {max_diff:.2f}%")

print(f"\n💾 Output Files:")
print(f"  - {OUTPUT_TEST_LABELS}")
print(f"  - {OUTPUT_TEST_DISTANCES}")
print(f"  - {OUTPUT_ANALYSIS_CSV}")
print(f"  - kmeans_test_analysis.png")

print("\n" + "="*70)
print("Done! Check output files for detailed results.")
print("="*70)
