import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import random
import joblib

# ==============================================================================
#           1. PEMUATAN DAN PERSIAPAN DATA
# ==============================================================================
try:
    semantic_vectors = np.load("combined_embeddings.npy")
    print("Berhasil memuat 'combined_embeddings.npy'.")
except FileNotFoundError:
    print("File .npy tidak ditemukan. Membuat data dummy untuk demonstrasi.")
    # Data dummy dengan kepadatan berbeda untuk DBSCAN
    cluster1 = np.random.rand(500, 768) * 0.5
    cluster2 = np.random.rand(300, 768) * 0.5 + 2
    noise = np.random.rand(50, 768) * 5 - 1
    semantic_vectors = np.vstack([cluster1, cluster2, noise])

print(f"\nDataset berhasil dimuat.")
print(f"Bentuk (shape) data: {semantic_vectors.shape}")

# Membuat log dummy untuk analisis
log_asli = [f"Pesan log gabungan nomor {i+1}" for i in range(semantic_vectors.shape[0])]

# ==============================================================================
#           2. MENENTUKAN `eps` MENGGUNAKAN K-DISTANCE GRAPH
# ==============================================================================
# Tentukan nilai min_samples (k) untuk heuristic. Nilai ini bisa dieksperimenkan.
# Nilai umum adalah 2 * dimensi_data, tapi kita mulai dari 10 sesuai skenario.
min_samples_heuristic = 50
print(f"\n--- Menentukan 'eps' menggunakan K-Distance Graph (dengan k={min_samples_heuristic}) ---")

# Hitung jarak ke k tetangga terdekat untuk setiap titik
neighbors = NearestNeighbors(n_neighbors=min_samples_heuristic)
neighbors_fit = neighbors.fit(semantic_vectors)
distances, indices = neighbors_fit.kneighbors(semantic_vectors)

# Ambil jarak ke tetangga ke-k (kolom terakhir) dan urutkan
k_distances = np.sort(distances[:, -1])

# Plot grafik K-Distance
plt.figure(figsize=(12, 6))
plt.plot(k_distances)
plt.title(f'K-Distance Graph (k = {min_samples_heuristic})')
plt.xlabel('Titik Data (diurutkan berdasarkan jarak)')
plt.ylabel(f'Jarak ke Tetangga ke-{min_samples_heuristic}')
plt.grid(True)
plt.show()

print("\nPETUNJUK: Perhatikan grafik di atas. Titik 'lutut' (knee), di mana kurva mulai menanjak tajam,")
print("adalah nilai 'eps' yang baik. Catat nilai pada sumbu Y di titik tersebut.")


# ==============================================================================
#           3. MENJALANKAN DBSCAN DENGAN PARAMETER YANG DIPILIH
# ==============================================================================

# GANTI NILAI INI berdasarkan pengamatan Anda pada grafik K-Distance di atas.
# Nilai di bawah ini hanyalah contoh.
eps_optimal = 1.5 

# Gunakan nilai min_samples yang sama atau sesuaikan untuk eksperimen.
min_samples_optimal = 50

print(f"\n--- Menjalankan DBSCAN dengan eps={eps_optimal} dan min_samples={min_samples_optimal} ---")

# Inisialisasi dan jalankan model DBSCAN
dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples_optimal)
cluster_labels = dbscan.fit_predict(semantic_vectors)

# ==============================================================================
#           4. MENGANALISIS HASIL KLASTERISASI DBSCAN
# ==============================================================================

# Label -1 di DBSCAN berarti titik tersebut adalah noise (anomali)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print("\n--- Ringkasan Hasil DBSCAN ---")
print(f"Jumlah cluster yang ditemukan: {n_clusters}")
print(f"Jumlah anomali (noise) yang terdeteksi: {n_noise}")

# Kelompokkan log asli berdasarkan hasil cluster
grouped_logs = {i: [] for i in range(n_clusters)}
noise_logs = []

for i, label in enumerate(cluster_labels):
    if label == -1:
        noise_logs.append(log_asli[i])
    else:
        grouped_logs[label].append(log_asli[i])

# Tampilkan sampel anomali yang terdeteksi
print("\n--- 🕵️ Sampel Anomali (Noise) yang Terdeteksi ---")
if noise_logs:
    sample_noise = random.sample(noise_logs, min(len(noise_logs), 10))
    for log in sample_noise:
        print(f"   - {log}")
else:
    print("   Tidak ada anomali yang terdeteksi.")

# Tampilkan sampel dari setiap cluster yang terbentuk
print("\n--- 📖 Sampel dari Setiap Cluster yang Ditemukan ---")
if n_clusters > 0:
    for cluster_id, logs_in_cluster in grouped_logs.items():
        print(f"\n   --- Cluster {cluster_id} (Total: {len(logs_in_cluster)} log) ---")
        sample_logs = random.sample(logs_in_cluster, min(len(logs_in_cluster), 5))
        for log in sample_logs:
            print(f"      - {log}")
else:
    print("   Tidak ada cluster yang terbentuk (semua data dianggap noise).")
joblib.dump(dbscan, 'model_dbscan.pkl') 
print("Model DBSCAN telah disimpan!")