import joblib
import numpy as np
from pathlib import Path
from sentence_to_vector import log_to_vector
from log_preprocessing import LogPreprocessor

# ==============================================================================
#           LANGKAH 1: MUAT MODEL DAN CENTROIDS DARI FILE
# ==============================================================================

model_path = 'model_kmeans_log.pkl'
try:
    print(f"Memuat model yang sudah dilatih dari: '{model_path}'...")
    # Memuat objek model K-Means dari file
    kmeans_final = joblib.load(model_path)
    
    # Mengambil centroids dari atribut model yang sudah dimuat
    centroids = kmeans_final.cluster_centers_
    
    print("Model berhasil dimuat!")

except FileNotFoundError:
    print(f"!!! FATAL ERROR: File model '{model_path}' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan skrip training dan menyimpan modelnya terlebih dahulu.")
    exit() # Keluar dari skrip jika model tidak ada


# ==============================================================================
#           LANGKAH 2: INISIALISASI PREPROCESSOR
# ==============================================================================

print("\n--- Menginisialisasi Log Preprocessor ---")
preprocessor = LogPreprocessor()
print("✓ Preprocessor siap digunakan!")

# ==============================================================================
#           LANGKAH 3: PROSES FILE LOG UNTUK PREDIKSI (DENGAN PREPROCESSING)
# ==============================================================================

# Gunakan Path dari pathlib untuk menangani path dengan lebih baik
current_dir = Path(__file__).parent
log_file_path = current_dir.parent / "testing_log_generator" / "info.log"
output_file_path = current_dir / "testing1" / "hasil_kmeans_info.txt"
print(f"\n--- Memulai Prediksi untuk File Log: '{log_file_path}' ---")
print(f"File exists: {log_file_path.exists()}")  # Debug line
print(f"Output akan disimpan ke: '{output_file_path}' ---")

try:
    with open(str(log_file_path), 'r', encoding='utf-8') as log_file:
        with open(str(output_file_path), "a", encoding="utf-8") as txt_file:
            # Tulis header untuk output file
            txt_file.write("="*100 + "\n")
            txt_file.write("HASIL PREDIKSI CLUSTER K-MEANS DENGAN PREPROCESSING\n")
            txt_file.write("="*100 + "\n\n")
            
            processed_count = 0
            skipped_count = 0
            
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
                    # STEP 2: UBAH LOG YANG SUDAH DIPREPROCESSING JADI VEKTOR
                    # ============================================================
                    log_vector_1d = log_to_vector(preprocessed_log, model_name="bert-base-uncased")

                    # RESHAPE vektor menjadi 2D
                    log_vector_2d = log_vector_1d.reshape(1, -1)

                    # Lakukan pengecekan dimensi
                    if log_vector_2d.shape[1] != centroids.shape[1]:
                        print(f"Dimensi vektor tidak cocok! Melewati log: {log_message}")
                        skipped_count += 1
                        continue

                    # ============================================================
                    # STEP 3: PREDIKSI CLUSTER
                    # ============================================================
                    predicted_cluster = kmeans_final.predict(log_vector_2d)

                    # ============================================================
                    # STEP 4: TULIS HASIL (ORIGINAL + PREPROCESSED + CLUSTER)
                    # ============================================================
                    txt_file.write(f"--- Log #{i+1} ---\n")
                    txt_file.write(f"ORIGINAL    : {log_message}\n")
                    txt_file.write(f"PREPROCESSED: {preprocessed_log}\n")
                    txt_file.write(f"CLUSTER     : {predicted_cluster[0]}\n")
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
            txt_file.write(f"Total logs processed: {processed_count}\n")
            txt_file.write(f"Total logs skipped  : {skipped_count}\n")
            txt_file.write(f"Total logs read     : {i+1}\n")
            
            print(f"\n✓ Selesai! Total {processed_count} log berhasil diproses")
            print(f"  Skipped: {skipped_count} log")
            print(f"  Hasil disimpan di: {output_file_path}")

except FileNotFoundError:
    print(f"!!! ERROR: File log tidak ditemukan di path: '{log_file_path}'")
except Exception as e:
    print(f"Terjadi error yang tidak terduga di luar loop: {e}")