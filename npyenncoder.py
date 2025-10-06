import numpy as np

# Ganti dengan path ke file .npy kamu
data = np.load('apache_embeddings.npy')

print(data)
print(type(data))       # Menampilkan tipe data
print(data.shape)       # Menampilkan bentuk array (dimensinya)
