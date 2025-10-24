import joblib
import numpy as np
from pathlib import Path
from sentence_to_vector import log_to_vector
from sklearn.metrics.pairwise import euclidean_distances
from log_preprocessing import LogPreprocessor

# ==============================================================================
#           FUNGSI HELPER UNTUK DETEKSI ANOMALI
# ==============================================================================

def predict_cluster_or_anomaly(log_message: str, dbscan_model, training_vectors, preprocessor):
    """
    Memprediksi apakah log masuk ke cluster tertentu atau anomali (-1).
    Log akan di-preprocess terlebih dahulu sebelum dijadikan vektor.

    Args:
        log_message: Pesan log baru yang akan diuji.
        dbscan_model: Objek model DBSCAN yang sudah dilatih (dimuat dari file).
        training_vectors: Vektor data asli yang digunakan untuk melatih model.
        preprocessor: LogPreprocessor object untuk preprocessing log.

    Returns:
        Tuple (int, str, str): (cluster_id, status, preprocessed_log)
            cluster_id: -1 untuk anomali, >=0 untuk cluster normal
            status: "Anomali" atau "Normal (Cluster X)"
            preprocessed_log: Log yang sudah dipreprocessing
    """
    # 1. PREPROCESSING LOG TERLEBIH DAHULU (STEP PALING PENTING!)
    preprocessed_log = preprocessor.preprocess(log_message, keep_log_level=True)
    
    # 2. Dapatkan semua core points dari model yang sudah dilatih
    core_point_indices = dbscan_model.core_sample_indices_
    core_points = training_vectors[core_point_indices]
    core_labels = dbscan_model.labels_[core_point_indices]

    # 3. Ubah log yang SUDAH DIPREPROCESSING menjadi vektor
    new_vector = log_to_vector(preprocessed_log, model_name="bert-base-uncased")
    
    # Pastikan vektor berbentuk 2D untuk kalkulasi jarak
    if new_vector.ndim == 1:
        new_vector = new_vector.reshape(1, -1)

    # 4. Hitung jarak dari log baru ke SEMUA core points
    distances = euclidean_distances(new_vector, core_points)

    # 5. Cari jarak terpendek ke salah satu core point dan label-nya
    min_distance_idx = np.argmin(distances)
    min_distance = distances[0, min_distance_idx]
    nearest_cluster = core_labels[min_distance_idx]

    # 6. Terapkan aturan DBSCAN
    # Jika jarak terpendeknya masih dalam radius 'eps', maka ia masuk cluster
    if min_distance <= dbscan_model.eps:
        status = f"Normal (Cluster {nearest_cluster})"
        return int(nearest_cluster), status, preprocessed_log
    else:
        status = "Anomali"
        return -1, status, preprocessed_log


# ==============================================================================
#           LANGKAH 1: MUAT MODEL DBSCAN DARI FILE
# ==============================================================================

model_path = 'model_dbscan.pkl'
try:
    print(f"Memuat model DBSCAN dari: '{model_path}'...")
    dbscan_model = joblib.load(model_path)
    print("✓ Model DBSCAN berhasil dimuat!")
    print(f"  - eps: {dbscan_model.eps}")
    print(f"  - min_samples: {dbscan_model.min_samples}")
except FileNotFoundError:
    print(f"!!! FATAL ERROR: File model '{model_path}' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan skrip training DBSCAN dan menyimpan modelnya.")
    exit()


# ==============================================================================
#           LANGKAH 2: MUAT DATA TRAINING (UNTUK CORE POINTS REFERENCE)
# ==============================================================================

print("\nMemuat data training asli untuk referensi core points...")
try:
    semantic_vectors = np.load("combined_embeddings.npy")
    print(f"✓ Data training dimuat! Shape: {semantic_vectors.shape}")
    print(f"  - Jumlah core points: {len(dbscan_model.core_sample_indices_)}")
    print(f"  - Jumlah cluster ditemukan: {len(set(dbscan_model.labels_)) - (1 if -1 in dbscan_model.labels_ else 0)}")
except FileNotFoundError:
    print("!!! FATAL ERROR: File embeddings tidak ditemukan.")
    print("Pastikan 'combined_embeddings.npy' ada.")
    exit()


# ==============================================================================
#           LANGKAH 3: INISIALISASI PREPROCESSOR
# ==============================================================================

print("\n--- Menginisialisasi Log Preprocessor ---")
preprocessor = LogPreprocessor()
print("✓ Preprocessor siap digunakan!")


# ==============================================================================
#           LANGKAH 4: PROSES FILE LOG UNTUK PREDIKSI
# ==============================================================================

# Gunakan Path dari pathlib untuk menangani path dengan lebih baik
current_dir = Path(__file__).parent
log_file_path = current_dir.parent / "testing_log_generator" / "warning.log"
output_file_path = current_dir / "testing1" / "hasil_dbscan_warning.txt"

print(f"\n--- Memulai Prediksi untuk File Log: '{log_file_path}' ---")
print(f"File exists: {log_file_path.exists()}")
print(f"Output akan disimpan ke: '{output_file_path}' ---")

try:
    with open(str(log_file_path), 'r', encoding='utf-8') as log_file:
        with open(str(output_file_path), "w", encoding="utf-8") as txt_file:  # "w" untuk overwrite
            # Tulis header untuk output file
            txt_file.write("="*100 + "\n")
            txt_file.write("HASIL PREDIKSI DBSCAN DENGAN PREPROCESSING\n")
            txt_file.write("(Cluster >= 0: Normal, Cluster = -1: Anomali)\n")
            txt_file.write("="*100 + "\n\n")
            
            processed_count = 0
            skipped_count = 0
            anomaly_count = 0
            cluster_counts = {}
            
            for i, log_message in enumerate(log_file):
                log_message = log_message.strip()
                if not log_message:
                    skipped_count += 1
                    continue

                try:
                    # ============================================================
                    # STEP 1: PREPROCESSING LOG (YANG PALING PENTING!)
                    # ============================================================
                    preprocessed_log = preprocessor.preprocess(log_message, keep_log_level=True)
                    
                    # Skip jika hasil preprocessing terlalu pendek (mungkin hanya noise)
                    if len(preprocessed_log) < 5:
                        skipped_count += 1
                        continue
                    
                    # ============================================================
                    # STEP 2 & 3: UBAH JADI VEKTOR DAN PREDIKSI CLUSTER/ANOMALI
                    # ============================================================
                    cluster_id, status, preprocessed = predict_cluster_or_anomaly(
                        log_message, dbscan_model, semantic_vectors, preprocessor
                    )
                    
                    # Update statistik
                    if cluster_id == -1:
                        anomaly_count += 1
                    else:
                        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

                    # ============================================================
                    # STEP 4: TULIS HASIL (ORIGINAL + PREPROCESSED + CLUSTER)
                    # ============================================================
                    txt_file.write(f"--- Log #{i+1} ---\n")
                    txt_file.write(f"ORIGINAL    : {log_message}\n")
                    txt_file.write(f"PREPROCESSED: {preprocessed}\n")
                    txt_file.write(f"CLUSTER     : {cluster_id}\n")
                    txt_file.write(f"STATUS      : {status}\n")
                    txt_file.write("\n")
                    
                    processed_count += 1
                    
                    # Print progress setiap 100 log
                    if processed_count % 100 == 0:
                        print(f"✓ Processed {processed_count} logs...")

                except Exception as e:
                    print(f"Error di baris {i+1} saat memproses log '{log_message}': {e}")
                    skipped_count += 1
            
            # Tulis summary di akhir file
            txt_file.write("\n" + "="*100 + "\n")
            txt_file.write("SUMMARY\n")
            txt_file.write("="*100 + "\n")
            txt_file.write(f"Total logs processed : {processed_count}\n")
            txt_file.write(f"Total logs skipped   : {skipped_count}\n")
            txt_file.write(f"Total anomalies      : {anomaly_count}\n")
            txt_file.write(f"Total normal logs    : {processed_count - anomaly_count}\n")
            txt_file.write(f"\nDistribusi Cluster:\n")
            txt_file.write(f"  - Anomali (Cluster -1): {anomaly_count} logs\n")
            for cluster_id in sorted(cluster_counts.keys()):
                txt_file.write(f"  - Cluster {cluster_id}: {cluster_counts[cluster_id]} logs\n")
            
            print(f"\n✓ Selesai! Total {processed_count} log berhasil diproses")
            print(f"  - Normal logs: {processed_count - anomaly_count}")
            print(f"  - Anomali: {anomaly_count}")
            print(f"  - Skipped: {skipped_count} log")
            print(f"  - Hasil disimpan di: {output_file_path}")

except FileNotFoundError:
    print(f"!!! ERROR: File log tidak ditemukan di path: '{log_file_path}'")
except Exception as e:
    print(f"Terjadi error yang tidak terduga: {e}")