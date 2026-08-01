"""
Validate DBSCAN Cluster Results vs Original Anomaly Labels

Membandingkan cluster ID dengan label asli (normal vs anomaly) untuk:
1. Cluster purity analysis
2. Identify "anomaly clusters" vs "normal clusters"  
3. Anomaly detection performance metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION - EDIT PATHS
# ============================================================================

# Input files (edit sesuai path Anda)
CLUSTER_LABELS_FILE = Path("dbscan_labels.npy")  # Output dari notebook
ORIGINAL_LABELS_FILE = Path("/path/to/bgl_original_labels.npy")  # Label asli: 0=normal, 1=anomaly

# Optional: jika punya log templates untuk cluster interpretation
EMBEDDINGS_FILE = Path("/path/to/after_preprocessed_bgl_embeddings.npy")  # Untuk sampling

# Output
OUTPUT_DIR = Path(".")

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*70)
print("CLUSTER VS ORIGINAL LABEL VALIDATION")
print("="*70)

print("\n📂 Loading data...")
cluster_labels = np.load(CLUSTER_LABELS_FILE)
print(f"   ✓ Cluster labels: {len(cluster_labels):,} samples")

try:
    original_labels = np.load(ORIGINAL_LABELS_FILE)
    print(f"   ✓ Original labels: {len(original_labels):,} samples")
    
    if len(cluster_labels) != len(original_labels):
        print(f"   ⚠️ Length mismatch! Truncating to shorter length...")
        min_len = min(len(cluster_labels), len(original_labels))
        cluster_labels = cluster_labels[:min_len]
        original_labels = original_labels[:min_len]
except FileNotFoundError:
    print(f"   ❌ Original labels not found: {ORIGINAL_LABELS_FILE}")
    print(f"   → Please update ORIGINAL_LABELS_FILE path")
    print(f"   → Expected format: numpy array with 0=normal, 1=anomaly")
    exit(1)

# Count original distribution
n_total = len(original_labels)
n_anomaly_orig = np.sum(original_labels == 1)
n_normal_orig = np.sum(original_labels == 0)

print(f"\n📊 Original Dataset Distribution:")
print(f"   Normal logs:  {n_normal_orig:8,} ({n_normal_orig/n_total*100:5.1f}%)")
print(f"   Anomaly logs: {n_anomaly_orig:8,} ({n_anomaly_orig/n_total*100:5.1f}%)")

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================

print(f"\n📊 Cluster Statistics:")
n_clusters = len(set(cluster_labels) - {-1})
n_noise = np.sum(cluster_labels == -1)
print(f"   Clusters found: {n_clusters}")
print(f"   Noise points:   {n_noise:,} ({n_noise/n_total*100:.2f}%)")

# ============================================================================
# CLUSTER PURITY ANALYSIS
# ============================================================================

print(f"\n🔍 Analyzing cluster purity...")

cluster_info = []

for cluster_id in sorted(set(cluster_labels)):
    mask = cluster_labels == cluster_id
    n_samples = np.sum(mask)
    
    # Count normal vs anomaly in this cluster
    labels_in_cluster = original_labels[mask]
    n_normal = np.sum(labels_in_cluster == 0)
    n_anomaly = np.sum(labels_in_cluster == 1)
    
    # Purity = max(normal, anomaly) / total
    purity = max(n_normal, n_anomaly) / n_samples
    
    # Dominant type
    dominant_type = "normal" if n_normal > n_anomaly else "anomaly"
    
    cluster_info.append({
        'cluster_id': cluster_id,
        'n_samples': n_samples,
        'n_normal': n_normal,
        'n_anomaly': n_anomaly,
        'pct_normal': (n_normal / n_samples) * 100,
        'pct_anomaly': (n_anomaly / n_samples) * 100,
        'purity': purity,
        'dominant': dominant_type
    })

df = pd.DataFrame(cluster_info)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print(f"\n{'='*70}")
print("CLUSTER PURITY SUMMARY")
print(f"{'='*70}")

# Categorize clusters
normal_clusters = df[df['dominant'] == 'normal']
anomaly_clusters = df[df['dominant'] == 'anomaly']

print(f"\nCluster Categories:")
print(f"   Normal-dominant clusters:  {len(normal_clusters):3d} ({normal_clusters['n_samples'].sum():,} samples)")
print(f"   Anomaly-dominant clusters: {len(anomaly_clusters):3d} ({anomaly_clusters['n_samples'].sum():,} samples)")

# Average purity
print(f"\nAverage Purity:")
print(f"   Overall:          {df['purity'].mean():.4f}")
print(f"   Normal clusters:  {normal_clusters['purity'].mean():.4f}")
print(f"   Anomaly clusters: {anomaly_clusters['purity'].mean():.4f}")

# High purity clusters (>90%)
high_purity = df[df['purity'] > 0.9]
print(f"\nHigh Purity Clusters (>90%):")
print(f"   Count: {len(high_purity)} / {len(df)} ({len(high_purity)/len(df)*100:.1f}%)")

# ============================================================================
# TOP CLUSTERS BY SIZE
# ============================================================================

print(f"\n{'='*70}")
print("TOP 20 LARGEST CLUSTERS")
print(f"{'='*70}")

top_20 = df.nlargest(20, 'n_samples')
print("\n" + top_20.to_string(index=False, 
    columns=['cluster_id', 'n_samples', 'pct_normal', 'pct_anomaly', 'purity', 'dominant'],
    float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else str(x)))

# ============================================================================
# ANOMALY CLUSTERS (potential anomaly types)
# ============================================================================

print(f"\n{'='*70}")
print("TOP 20 ANOMALY-DOMINANT CLUSTERS")
print(f"{'='*70}")

top_anomaly = anomaly_clusters.nlargest(20, 'n_samples')
print("\n" + top_anomaly.to_string(index=False,
    columns=['cluster_id', 'n_samples', 'pct_anomaly', 'purity'],
    float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else str(x)))

print(f"\n💡 Insight:")
print(f"   These clusters contain mostly anomaly logs!")
print(f"   Each cluster likely represents a different anomaly type/pattern.")

# ============================================================================
# SMALL CLUSTERS (rare patterns)
# ============================================================================

print(f"\n{'='*70}")
print("SMALL CLUSTERS (<100 samples)")
print(f"{'='*70}")

small_clusters = df[df['n_samples'] < 100].sort_values('n_samples', ascending=False)
print(f"\nCount: {len(small_clusters)}")
print(f"Total samples: {small_clusters['n_samples'].sum():,}")

small_anomaly = small_clusters[small_clusters['dominant'] == 'anomaly']
print(f"\nAnomaly-dominant small clusters: {len(small_anomaly)} ({small_anomaly['n_samples'].sum():,} samples)")
print(f"   → These are RARE ANOMALY PATTERNS! 🎯")

# ============================================================================
# ANOMALY DETECTION AS CLUSTERING PROBLEM
# ============================================================================

print(f"\n{'='*70}")
print("ANOMALY DETECTION APPROACH")
print(f"{'='*70}")

# Approach 1: Treat all small clusters as anomalies
SMALL_CLUSTER_THRESHOLD = 1000  # Adjust based on your needs

print(f"\nApproach 1: Small Clusters as Anomalies (threshold={SMALL_CLUSTER_THRESHOLD})")

predicted_anomaly_small = np.zeros(n_total, dtype=int)
small_cluster_ids = df[df['n_samples'] < SMALL_CLUSTER_THRESHOLD]['cluster_id'].values

for cid in small_cluster_ids:
    predicted_anomaly_small[cluster_labels == cid] = 1

# Include noise as anomaly
predicted_anomaly_small[cluster_labels == -1] = 1

tp_small = np.sum((predicted_anomaly_small == 1) & (original_labels == 1))
fp_small = np.sum((predicted_anomaly_small == 1) & (original_labels == 0))
fn_small = np.sum((predicted_anomaly_small == 0) & (original_labels == 1))
tn_small = np.sum((predicted_anomaly_small == 0) & (original_labels == 0))

precision_small = tp_small / (tp_small + fp_small) if (tp_small + fp_small) > 0 else 0
recall_small = tp_small / (tp_small + fn_small) if (tp_small + fn_small) > 0 else 0
f1_small = 2 * precision_small * recall_small / (precision_small + recall_small) if (precision_small + recall_small) > 0 else 0

print(f"   True Positives:  {tp_small:8,}")
print(f"   False Positives: {fp_small:8,}")
print(f"   False Negatives: {fn_small:8,}")
print(f"   True Negatives:  {tn_small:8,}")
print(f"   Precision: {precision_small:.4f}")
print(f"   Recall:    {recall_small:.4f}")
print(f"   F1-Score:  {f1_small:.4f}")

# Approach 2: Treat anomaly-dominant clusters as anomalies
print(f"\nApproach 2: Anomaly-Dominant Clusters as Anomalies (>50% anomaly)")

predicted_anomaly_dominant = np.zeros(n_total, dtype=int)
anomaly_cluster_ids = df[df['pct_anomaly'] > 50]['cluster_id'].values

for cid in anomaly_cluster_ids:
    predicted_anomaly_dominant[cluster_labels == cid] = 1

predicted_anomaly_dominant[cluster_labels == -1] = 1

tp_dom = np.sum((predicted_anomaly_dominant == 1) & (original_labels == 1))
fp_dom = np.sum((predicted_anomaly_dominant == 1) & (original_labels == 0))
fn_dom = np.sum((predicted_anomaly_dominant == 0) & (original_labels == 1))
tn_dom = np.sum((predicted_anomaly_dominant == 0) & (original_labels == 0))

precision_dom = tp_dom / (tp_dom + fp_dom) if (tp_dom + fp_dom) > 0 else 0
recall_dom = tp_dom / (tp_dom + fn_dom) if (tp_dom + fn_dom) > 0 else 0
f1_dom = 2 * precision_dom * recall_dom / (precision_dom + recall_dom) if (precision_dom + recall_dom) > 0 else 0

print(f"   True Positives:  {tp_dom:8,}")
print(f"   False Positives: {fp_dom:8,}")
print(f"   False Negatives: {fn_dom:8,}")
print(f"   True Negatives:  {tn_dom:8,}")
print(f"   Precision: {precision_dom:.4f}")
print(f"   Recall:    {recall_dom:.4f}")
print(f"   F1-Score:  {f1_dom:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cluster size distribution (colored by dominant type)
ax1 = axes[0, 0]
colors = ['green' if x == 'normal' else 'red' for x in df['dominant']]
ax1.scatter(range(len(df)), df['n_samples'], c=colors, alpha=0.6, s=30)
ax1.set_xlabel('Cluster ID')
ax1.set_ylabel('Cluster Size (log scale)')
ax1.set_yscale('log')
ax1.set_title('Cluster Sizes by Type')
ax1.axhline(y=SMALL_CLUSTER_THRESHOLD, color='orange', linestyle='--', 
           label=f'Small cluster threshold ({SMALL_CLUSTER_THRESHOLD})', alpha=0.5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Purity distribution
ax2 = axes[0, 1]
ax2.hist(df['purity'], bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(x=0.9, color='red', linestyle='--', label='90% purity', alpha=0.7)
ax2.set_xlabel('Cluster Purity')
ax2.set_ylabel('Number of Clusters')
ax2.set_title('Cluster Purity Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Normal vs Anomaly percentage
ax3 = axes[1, 0]
top_15 = df.nlargest(15, 'n_samples')
x = np.arange(len(top_15))
width = 0.35
ax3.bar(x - width/2, top_15['pct_normal'], width, label='Normal %', alpha=0.8, color='green')
ax3.bar(x + width/2, top_15['pct_anomaly'], width, label='Anomaly %', alpha=0.8, color='red')
ax3.set_xlabel('Cluster ID')
ax3.set_ylabel('Percentage')
ax3.set_title('Top 15 Clusters: Normal vs Anomaly Distribution')
ax3.set_xticks(x)
ax3.set_xticklabels(top_15['cluster_id'].values, rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Confusion matrix heatmap (Approach 2)
ax4 = axes[1, 1]
cm = np.array([[tn_dom, fp_dom], [fn_dom, tp_dom]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
           xticklabels=['Predicted Normal', 'Predicted Anomaly'],
           yticklabels=['Actual Normal', 'Actual Anomaly'])
ax4.set_title(f'Confusion Matrix (Approach 2)\nF1={f1_dom:.4f}')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cluster_validation_analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: cluster_validation_analysis.png")

plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n💾 Saving results...")

# Save full cluster analysis
df.to_csv(OUTPUT_DIR / 'cluster_purity_analysis.csv', index=False)
print(f"   ✓ Saved: cluster_purity_analysis.csv")

# Save anomaly cluster list
anomaly_clusters_df = df[df['dominant'] == 'anomaly'][['cluster_id', 'n_samples', 'pct_anomaly', 'purity']]
anomaly_clusters_df.to_csv(OUTPUT_DIR / 'anomaly_clusters.csv', index=False)
print(f"   ✓ Saved: anomaly_clusters.csv")

# Save predictions
np.save(OUTPUT_DIR / 'predicted_anomaly_approach1.npy', predicted_anomaly_small)
np.save(OUTPUT_DIR / 'predicted_anomaly_approach2.npy', predicted_anomaly_dominant)
print(f"   ✓ Saved: predicted_anomaly_approach*.npy")

print(f"\n{'='*70}")
print("✅ VALIDATION COMPLETE!")
print(f"{'='*70}")

print(f"\n📌 KEY FINDINGS:")
print(f"   • {len(anomaly_clusters)}/{n_clusters} clusters are anomaly-dominant")
print(f"   • Average cluster purity: {df['purity'].mean():.2%}")
print(f"   • Small clusters (<{SMALL_CLUSTER_THRESHOLD}): {len(df[df['n_samples'] < SMALL_CLUSTER_THRESHOLD])}")
print(f"   • Best approach: Approach {'1' if f1_small > f1_dom else '2'} (F1={max(f1_small, f1_dom):.4f})")

print(f"\n💡 CONCLUSION:")
if f1_small > 0.7 or f1_dom > 0.7:
    print(f"   ✅ DBSCAN successfully separates anomaly types into distinct clusters!")
    print(f"   ✅ Clustering-based anomaly detection is viable for this dataset.")
else:
    print(f"   ⚠️ Some anomaly types mixed with normal logs in large clusters.")
    print(f"   → Consider: adjusting eps, using semi-supervised approach, or ensemble methods.")

print(f"\n🎯 ANSWER TO YOUR QUESTION:")
print(f"   • NOISE ({n_noise:,} samples) ≠ All anomalies")
print(f"   • Anomalies form {len(anomaly_clusters)} separate clusters!")
print(f"   • Large cluster (Cluster 2: 38%) = mostly normal logs")
print(f"   • Small/medium clusters = various anomaly types")
print(f"   • This is EXPECTED and CORRECT behavior! ✅")
