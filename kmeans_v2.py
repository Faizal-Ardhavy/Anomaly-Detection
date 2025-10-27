import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
matplotlib.use('Agg')  # supaya gak buka window GUI
import random


# ==============================================================================
#           1. PEMUATAN DAN PERSIAPAN DATA
# ==============================================================================
try:
    semantic_vectors = np.load('combined_embeddings.npy')
    print("Berhasil memuat 'combined_embeddings.npy'.")
except FileNotFoundError:
    print("File .npy tidak ditemukan. Membuat data dummy untuk demonstrasi.")
    cluster1 = np.random.rand(500, 768) + 0.5
    cluster2 = np.random.rand(100, 768) - 0.5
    semantic_vectors = np.vstack([cluster1, cluster2])

print(f"\nDataset berhasil dimuat.")
print(f"Bentuk (shape) data: {semantic_vectors.shape}")

# Membuat log dummy untuk analisis
log_asli = [f"Pesan log gabungan nomor {i+1}" for i in range(semantic_vectors.shape[0])]

# ==============================================================================
#           2. ANALISIS AWAL DENGAN k=2 (NORMAL vs ANOMALI)
# ==============================================================================
print("\n--- Analisis Awal: Menjalankan K-Means dengan k=2 ---")
kmeans_k2 = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
kmeans_k2.fit(semantic_vectors)
unique_clusters, counts = np.unique(kmeans_k2.labels_, return_counts=True)
print("Ukuran cluster untuk k=2:")
for cluster_id, size in zip(unique_clusters, counts):
    print(f"   - Cluster {cluster_id}: {size} anggota")


# Data vektor semantik kamu
# pastikan sudah ada: semantic_vectors = np.load("combined_embeddings.npy")

# ===============================
# PARAMETER EKSPERIMEN
# ===============================
possible_k = range(2, 21)
n_jobs = 2  # gunakan semua core CPU

# ===================================================================
# 1️⃣  Eksperimen 1 - Euclidean (standar)
# ===================================================================
def compute_wcss_euclidean(k, X):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    return k, kmeans.inertia_

print("\n🚀 Eksperimen 1: Euclidean Distance (Parallel)...")
results_euclidean = Parallel(n_jobs=n_jobs)(
    delayed(compute_wcss_euclidean)(k, semantic_vectors) for k in possible_k
)

# urutkan hasil berdasarkan k
results_euclidean.sort(key=lambda x: x[0])
wcss_euclidean = [r[1] for r in results_euclidean]

# Simpan plot
plt.figure(figsize=(12, 6))
plt.plot(possible_k, wcss_euclidean, 'bo-', markerfacecolor='r')
plt.title('Elbow Method Menggunakan Jarak Euclidean')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.xticks(possible_k)
plt.savefig("elbow_euclidean_parallel.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Disimpan: elbow_euclidean_parallel.png")

# ===================================================================
# 2️⃣  Eksperimen 2 - Cosine (dengan normalisasi L2)
# ===================================================================
print("\n🚀 Eksperimen 2: Cosine Distance (Parallel)...")
semantic_vectors_normalized = normalize(semantic_vectors, norm='l2', axis=1)

def compute_wcss_cosine(k, X):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    return k, kmeans.inertia_

results_cosine = Parallel(n_jobs=n_jobs)(
    delayed(compute_wcss_cosine)(k, semantic_vectors_normalized) for k in possible_k
)

results_cosine.sort(key=lambda x: x[0])
wcss_cosine = [r[1] for r in results_cosine]

plt.figure(figsize=(12, 6))
plt.plot(possible_k, wcss_cosine, 'go-', markerfacecolor='b')
plt.title('Elbow Method Menggunakan Jarak Cosine (via Normalisasi)')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS (Inertia pada Data Normalisasi)')
plt.grid(True)
plt.xticks(possible_k)
plt.savefig("elbow_cosine_parallel.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Disimpan: elbow_cosine_parallel.png")

# ===================================================================
# Simpan nilai WCSS ke file npy (biar bisa analisis ulang)
# ===================================================================
np.savez("wcss_results_parallel.npz",
         k_values=list(possible_k),
         wcss_euclidean=wcss_euclidean,
         wcss_cosine=wcss_cosine)

print("📁 Semua hasil disimpan:")
print("- elbow_euclidean_parallel.png")
print("- elbow_cosine_parallel.png")
print("- wcss_results_parallel.npz")


# # ==============================================================================
# #           5. ANALISIS DAN PERBANDINGAN HASIL
# # ==============================================================================
# # Catatan: Tentukan nilai k_optimal_... dari grafik "siku" yang muncul.
# # Nilai di bawah ini hanyalah contoh.
# k_optimal_euclidean = 4 # GANTI SESUAI GRAFIK EUCLIDEAN ANDA
# k_optimal_cosine = 5    # GANTI SESUAI GRAFIK COSINE ANDA

# print(f"\n--- Melatih Model Final Euclidean dengan k={k_optimal_euclidean} ---")
# final_model_euclidean = KMeans(n_clusters=k_optimal_euclidean, init='k-means++', random_state=42, n_init=10)
# final_model_euclidean.fit(semantic_vectors)

# print(f"\n--- Melatih Model Final Cosine dengan k={k_optimal_cosine} ---")
# final_model_cosine = KMeans(n_clusters=k_optimal_cosine, init='k-means++', random_state=42, n_init=10)
# final_model_cosine.fit(semantic_vectors_normalized)

# # Fungsi bantuan untuk menampilkan hasil
# def analyze_clusters(model, data, optimal_k):
#     labels = model.labels_
#     grouped_logs = {i: [] for i in range(optimal_k)}
#     for i, label in enumerate(labels):
#         grouped_logs[label].append(data[i])

#     print("\nUkuran Cluster:")
#     unique, counts = np.unique(labels, return_counts=True)
#     for cluster_id, size in zip(unique, counts):
#         print(f"   - Cluster {cluster_id}: {size} anggota")

#     print("\nContoh Log per Cluster:")
#     for cid, logs in grouped_logs.items():
#         print(f"   --- Cluster {cid} (Contoh 5 log) ---")
#         sample_logs = random.sample(logs, min(len(logs), 5))
#         for log in sample_logs:
#             print(f"      - {log}")

# # Tampilkan analisis untuk kedua model
# print("\n\n=================================================")
# print("          HASIL MODEL JARAK EUCLIDEAN")
# print("=================================================")
# analyze_clusters(final_model_euclidean, log_asli, k_optimal_euclidean)

# print("\n\n=================================================")
# print("           HASIL MODEL JARAK COSINE")
# print("=================================================")
# analyze_clusters(final_model_cosine, log_asli, k_optimal_cosine)