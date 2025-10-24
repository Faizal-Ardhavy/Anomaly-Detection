"""
Eksperimen: Membandingkan Hasil Klasterisasi DENGAN dan TANPA Normalisasi
Untuk dataset log yang sudah diubah menjadi BERT embeddings
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd

# ==============================================================================
#           1. LOAD DATA
# ==============================================================================
print("="*80)
print("EKSPERIMEN: NORMALISASI vs TANPA NORMALISASI")
print("="*80)

try:
    # Load embeddings yang sudah dibuat dengan BERT
    apache_vectors = np.load('apache_embeddings.npy')
    proxifier_vectors = np.load('proxifier_embeddings.npy')
    print("✓ Berhasil memuat embeddings dari file .npy")
    semantic_vectors = np.vstack([apache_vectors, proxifier_vectors])
except FileNotFoundError:
    print("✗ File embeddings tidak ditemukan, menggunakan data dummy")
    # Data dummy dengan 3 cluster yang jelas
    cluster1 = np.random.randn(200, 768) * 0.3 + np.array([0] * 768)
    cluster2 = np.random.randn(200, 768) * 0.3 + np.array([2] * 768)
    cluster3 = np.random.randn(200, 768) * 0.3 + np.array([4] * 768)
    semantic_vectors = np.vstack([cluster1, cluster2, cluster3])

print(f"Shape data: {semantic_vectors.shape}")
print(f"Jumlah log: {semantic_vectors.shape[0]}")
print(f"Dimensi vektor: {semantic_vectors.shape[1]}")

# ==============================================================================
#           2. PERSIAPAN SKENARIO NORMALISASI
# ==============================================================================
normalization_scenarios = {
    "Tanpa Normalisasi": semantic_vectors,
    "StandardScaler (Z-score)": StandardScaler().fit_transform(semantic_vectors),
    "MinMaxScaler (0-1)": MinMaxScaler().fit_transform(semantic_vectors),
    "L2 Normalizer": Normalizer(norm='l2').fit_transform(semantic_vectors)
}

print("\n" + "="*80)
print("SKENARIO YANG AKAN DIUJI:")
print("="*80)
for i, name in enumerate(normalization_scenarios.keys(), 1):
    print(f"{i}. {name}")

# ==============================================================================
#           3. FUNGSI EVALUASI
# ==============================================================================
def evaluate_clustering(vectors, labels, algorithm_name):
    """Evaluasi kualitas clustering dengan berbagai metrik"""
    
    # Filter out noise points (label = -1) untuk DBSCAN
    mask = labels != -1
    filtered_vectors = vectors[mask]
    filtered_labels = labels[mask]
    
    n_clusters = len(set(filtered_labels)) - (1 if -1 in filtered_labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_clustered': len(filtered_labels)
    }
    
    # Hitung metrik hanya jika ada minimal 2 cluster dan data yang cukup
    if n_clusters >= 2 and len(filtered_labels) > n_clusters:
        try:
            metrics['silhouette'] = silhouette_score(filtered_vectors, filtered_labels)
            metrics['davies_bouldin'] = davies_bouldin_score(filtered_vectors, filtered_labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(filtered_vectors, filtered_labels)
        except:
            metrics['silhouette'] = np.nan
            metrics['davies_bouldin'] = np.nan
            metrics['calinski_harabasz'] = np.nan
    else:
        metrics['silhouette'] = np.nan
        metrics['davies_bouldin'] = np.nan
        metrics['calinski_harabasz'] = np.nan
    
    return metrics

# ==============================================================================
#           4. EKSPERIMEN DBSCAN
# ==============================================================================
print("\n" + "="*80)
print("EKSPERIMEN 1: DBSCAN")
print("="*80)

dbscan_results = []

for scenario_name, vectors in normalization_scenarios.items():
    print(f"\n📊 Testing: {scenario_name}")
    
    # Parameter DBSCAN - disesuaikan untuk high-dimensional BERT embeddings
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    labels = dbscan.fit_predict(vectors)
    
    metrics = evaluate_clustering(vectors, labels, "DBSCAN")
    
    result = {
        'Skenario': scenario_name,
        'Jumlah Cluster': metrics['n_clusters'],
        'Noise Points': metrics['n_noise'],
        'Silhouette Score': metrics['silhouette'],
        'Davies-Bouldin Score': metrics['davies_bouldin'],
        'Calinski-Harabasz Score': metrics['calinski_harabasz']
    }
    
    dbscan_results.append(result)
    
    print(f"  - Cluster: {metrics['n_clusters']}")
    print(f"  - Noise: {metrics['n_noise']}")
    if not np.isnan(metrics['silhouette']):
        print(f"  - Silhouette: {metrics['silhouette']:.4f}")
        print(f"  - Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
        print(f"  - Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")

# ==============================================================================
#           5. EKSPERIMEN K-MEANS
# ==============================================================================
print("\n" + "="*80)
print("EKSPERIMEN 2: K-MEANS")
print("="*80)

kmeans_results = []
k = 5  # Jumlah cluster yang diinginkan

for scenario_name, vectors in normalization_scenarios.items():
    print(f"\n📊 Testing: {scenario_name}")
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    
    metrics = evaluate_clustering(vectors, labels, "K-Means")
    
    result = {
        'Skenario': scenario_name,
        'Jumlah Cluster': k,
        'Silhouette Score': metrics['silhouette'],
        'Davies-Bouldin Score': metrics['davies_bouldin'],
        'Calinski-Harabasz Score': metrics['calinski_harabasz']
    }
    
    kmeans_results.append(result)
    
    print(f"  - K: {k}")
    if not np.isnan(metrics['silhouette']):
        print(f"  - Silhouette: {metrics['silhouette']:.4f}")
        print(f"  - Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
        print(f"  - Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")

# ==============================================================================
#           6. VISUALISASI HASIL
# ==============================================================================
print("\n" + "="*80)
print("MEMBUAT VISUALISASI PERBANDINGAN")
print("="*80)

# Convert to DataFrame for easier comparison
df_dbscan = pd.DataFrame(dbscan_results)
df_kmeans = pd.DataFrame(kmeans_results)

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Perbandingan Metrik Clustering: Dengan vs Tanpa Normalisasi', 
             fontsize=16, fontweight='bold')

# DBSCAN Plots
scenarios_short = ['Tanpa Norm', 'Z-score', 'MinMax', 'L2']

# Plot 1: DBSCAN - Silhouette Score
ax1 = axes[0, 0]
ax1.bar(scenarios_short, df_dbscan['Silhouette Score'], color='steelblue')
ax1.set_title('DBSCAN: Silhouette Score\n(Semakin tinggi semakin baik)')
ax1.set_ylabel('Score')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# Plot 2: DBSCAN - Davies-Bouldin Score
ax2 = axes[0, 1]
ax2.bar(scenarios_short, df_dbscan['Davies-Bouldin Score'], color='coral')
ax2.set_title('DBSCAN: Davies-Bouldin Score\n(Semakin rendah semakin baik)')
ax2.set_ylabel('Score')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: DBSCAN - Number of Clusters
ax3 = axes[0, 2]
ax3.bar(scenarios_short, df_dbscan['Jumlah Cluster'], color='lightgreen')
ax3.set_title('DBSCAN: Jumlah Cluster')
ax3.set_ylabel('Jumlah')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: K-Means - Silhouette Score
ax4 = axes[1, 0]
ax4.bar(scenarios_short, df_kmeans['Silhouette Score'], color='steelblue')
ax4.set_title('K-Means: Silhouette Score\n(Semakin tinggi semakin baik)')
ax4.set_ylabel('Score')
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

# Plot 5: K-Means - Davies-Bouldin Score
ax5 = axes[1, 1]
ax5.bar(scenarios_short, df_kmeans['Davies-Bouldin Score'], color='coral')
ax5.set_title('K-Means: Davies-Bouldin Score\n(Semakin rendah semakin baik)')
ax5.set_ylabel('Score')
ax5.grid(axis='y', alpha=0.3)

# Plot 6: K-Means - Calinski-Harabasz Score
ax6 = axes[1, 2]
ax6.bar(scenarios_short, df_kmeans['Calinski-Harabasz Score'], color='plum')
ax6.set_title('K-Means: Calinski-Harabasz Score\n(Semakin tinggi semakin baik)')
ax6.set_ylabel('Score')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('normalization_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi disimpan: normalization_comparison.png")

# ==============================================================================
#           7. KESIMPULAN DAN REKOMENDASI
# ==============================================================================
print("\n" + "="*80)
print("KESIMPULAN DAN REKOMENDASI")
print("="*80)

print("\n📊 HASIL DBSCAN:")
print(df_dbscan.to_string(index=False))

print("\n📊 HASIL K-MEANS:")
print(df_kmeans.to_string(index=False))

print("\n" + "="*80)
print("💡 INTERPRETASI METRIK:")
print("="*80)
print("""
1. SILHOUETTE SCORE (Range: -1 to 1)
   - Nilai mendekati 1: Cluster sangat baik terpisah
   - Nilai mendekati 0: Cluster overlap
   - Nilai negatif: Data mungkin di cluster yang salah

2. DAVIES-BOULDIN SCORE (Range: 0 to ∞)
   - Semakin rendah semakin baik
   - Mengukur rasio within-cluster vs between-cluster distance

3. CALINSKI-HARABASZ SCORE (Range: 0 to ∞)
   - Semakin tinggi semakin baik
   - Mengukur rasio antara between-cluster dispersion dan within-cluster dispersion
""")

print("\n" + "="*80)
print("🎯 REKOMENDASI UNTUK BERT EMBEDDINGS:")
print("="*80)
print("""
Berdasarkan teori dan praktik:

✅ YANG HARUS DILAKUKAN:
   1. Normalisasi TEKS LOG sebelum BERT (preprocessing):
      - Hapus timestamp, IP address, path spesifik
      - Lowercase
      - Hapus karakter khusus yang tidak penting
   
   2. Gunakan COSINE SIMILARITY untuk DBSCAN:
      - DBSCAN(eps=0.5, min_samples=5, metric='cosine')
      - Lebih cocok untuk high-dimensional semantic vectors

❌ YANG TIDAK PERLU:
   1. StandardScaler/MinMaxScaler pada BERT embeddings
      - BERT embeddings sudah ter-normalized dengan baik
      - Bisa menghilangkan informasi semantik
   
   2. L2 Normalizer mungkin OK, tapi biasanya tidak signifikan

📌 CATATAN PENTING:
   - Jika Silhouette score rendah di semua skenario → masalah di data/parameter
   - Tuning parameter (eps, min_samples, k) lebih penting daripada normalisasi
   - Untuk log anomaly detection, preprocessing teks > normalisasi vektor
""")

plt.show()
