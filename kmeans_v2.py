import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import random

# ==============================================================================
#           1. PEMUATAN DAN PERSIAPAN DATA
# ==============================================================================
try:
    apache_vectors = np.load('apache_embeddings.npy')
    proxifier_vectors = np.load('proxifier_embeddings.npy')
    print("Berhasil memuat 'apache_embeddings.npy' dan 'proxifier_embeddings.npy'.")
    semantic_vectors = np.vstack([apache_vectors, proxifier_vectors])
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

# ==============================================================================
#           3. EKSPERIMEN 1: MENENTUKAN K DENGAN JARAK EUCLIDEAN
# ==============================================================================
print("\n--- Eksperimen 1: Menentukan K Optimal dengan Jarak Euclidean (Standar) ---")
wcss_euclidean = []
possible_k = range(2, 21) # Rentang k dari 2 hingga 20

for k in possible_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(semantic_vectors)
    wcss_euclidean.append(kmeans.inertia_) # .inertia_ adalah WCSS untuk Euclidean

# Plot grafik Elbow untuk Jarak Euclidean
plt.figure(figsize=(12, 6))
plt.plot(possible_k, wcss_euclidean, 'bo-', markerfacecolor='r')
plt.title('Elbow Method Menggunakan Jarak Euclidean')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.xticks(possible_k)
plt.show()


# ==============================================================================
#           4. EKSPERIMEN 2: MENENTUKAN K DENGAN JARAK COSINE
# ==============================================================================
print("\n--- Eksperimen 2: Menentukan K Optimal dengan Jarak Cosine ---")
# Trik: Normalisasi L2 pada vektor. Jarak Euclidean pada data yang dinormalisasi
# setara dengan memaksimalkan Cosine Similarity.
semantic_vectors_normalized = normalize(semantic_vectors, norm='l2', axis=1)
print("Vektor telah dinormalisasi untuk mensimulasikan Jarak Cosine.")

wcss_cosine = []
for k in possible_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(semantic_vectors_normalized)
    wcss_cosine.append(kmeans.inertia_)

# Plot grafik Elbow untuk Jarak Cosine
plt.figure(figsize=(12, 6))
plt.plot(possible_k, wcss_cosine, 'go-', markerfacecolor='b')
plt.title('Elbow Method Menggunakan Jarak Cosine (via Normalisasi)')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS (Inertia pada Data Normalisasi)')
plt.grid(True)
plt.xticks(possible_k)
plt.show()


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