import numpy as np
import joblib


# Asumsikan vektor Anda disimpan dalam file 'log_vectors.npy'
# Jika Anda punya file CSV, gunakan pd.read_csv('log_vectors.csv').values
try:
    semantic_vectors = np.load('combined_embeddings.npy')
    print("Berhasil memuat 'combined_embeddings.npy'.")

    # 2. Gabungkan kedua dataset menjadi satu
    # np.vstack (vertical stack) menumpuk array secara vertikal

except FileNotFoundError as e:
    print(f"File tidak ditemukan: {e.filename}. Pastikan kedua file .npy ada di direktori yang sama.")
    print("Membuat data dummy untuk demonstrasi.")
    # Membuat data dummy jika salah satu file tidak ada
    cluster1 = np.random.rand(50, 128) + np.array([0, 5, 0] * 42 + [0, 2])
    cluster2 = np.random.rand(50, 128) + np.array([5, 0, 5] * 42 + [5, 0])
    cluster3 = np.random.rand(50, 128) + np.array([2, 2, 8] * 42 + [2, 8])
    semantic_vectors = np.vstack([cluster1, cluster2, cluster3])


# Periksa bentuk data: (jumlah_log, dimensi_vektor)
print(f"Dataset berhasil dimuat.")
print(f"Bentuk (shape) data: {semantic_vectors.shape}")
print(f"Jumlah log: {semantic_vectors.shape[0]}")
print(f"Dimensi vektor: {semantic_vectors.shape[1]}")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# List untuk menyimpan nilai inertia
inertia_values = []
possible_k = range(2, 16) # Mencoba K dari 2 hingga 15

# print("Mencari nilai K optimal dengan metode Elbow...")
# for k in possible_k:
#     # Inisialisasi dan latih model K-Means
#     kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
#     kmeans.fit(semantic_vectors)
    
#     # Simpan nilai inertia
#     inertia_values.append(kmeans.inertia_)

# # Plot grafik Elbow
# plt.figure(figsize=(10, 6))
# plt.plot(possible_k, inertia_values, 'bo-')
# plt.xlabel('Jumlah Cluster (K)')
# plt.ylabel('Inertia')
# plt.title('Metode Elbow untuk Menentukan K Optimal')
# plt.grid(True)
# plt.xticks(possible_k)
# plt.show()

# Misalkan dari grafik, K optimal adalah 4
optimal_k = 4

# Inisialisasi model K-Means final
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)

# Latih model dengan data vektor
print(f"Melatih model K-Means final dengan K={optimal_k}...")
kmeans_final.fit(semantic_vectors)
print("Model berhasil dilatih!")

# Dapatkan label cluster untuk setiap log
cluster_labels = kmeans_final.labels_

# Dapatkan pusat dari setiap cluster (centroid)
centroids = kmeans_final.cluster_centers_

print(f"Setiap log telah diberi label cluster (0 sampai {optimal_k-1}).")
print(f"Contoh 5 label pertama: {cluster_labels[:5]}")

# Asumsikan Anda memiliki log asli dalam sebuah list atau file
# Misalnya, log_asli = ["log baris 1", "log baris 2", ...]
# Untuk demo, kita buat log dummy
log_asli = [f"Pesan log dummy nomor {i+1}" for i in range(semantic_vectors.shape[0])]

# Mengelompokkan log asli berdasarkan label cluster
grouped_logs = {}
for i in range(optimal_k):
    grouped_logs[i] = []

for i, label in enumerate(cluster_labels):
    grouped_logs[label].append(log_asli[i])

# Tampilkan beberapa contoh log dari setiap cluster
for i in range(optimal_k):
    print(f"\n--- Cluster {i} (Total: {len(grouped_logs[i])} log) ---")
    # Tampilkan 5 log pertama dari cluster ini sebagai sampel
    sample_logs = grouped_logs[i][:5]
    for log in sample_logs:
        print(log)



# ... (setelah Anda menjalankan kmeans_final.fit(semantic_vectors))

# Simpan model ke file
joblib.dump(kmeans_final, 'model_kmeans_log.pkl')
print("Model telah dilatih dan disimpan!")