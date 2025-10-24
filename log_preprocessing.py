"""
Preprocessing Log Files - TAHAP PALING PENTING
Normalisasi TEKS LOG sebelum diubah jadi vector semantik dengan BERT
"""

import re
from typing import List
import pandas as pd

class LogPreprocessor:
    """
    Class untuk preprocessing log files sebelum dijadikan BERT embeddings
    Ini adalah normalisasi yang BENAR-BENAR MENINGKATKAN hasil clustering!
    """
    
    def __init__(self):
        # Pattern untuk berbagai jenis informasi yang perlu dihapus/diganti
        self.patterns = {
            # Timestamp patterns
            'timestamp': [
                r'\[.*?\]',  # [Thu Jun 09 06:07:04 2005]
                r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}',  # 2005-06-09 06:07:04
                r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',  # 09/Jun/2005:06:07:04
            ],
            # IP Address patterns
            'ip': [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # 192.168.1.1
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',  # IPv6
            ],
            # File paths (keep struktur, hapus path spesifik)
            'filepath': [
                r'/var/www/[\w/.-]*',  # /var/www/html/...
                r'/usr/[\w/.-]*',  # /usr/local/...
                r'/etc/[\w/.-]*',  # /etc/httpd/...
                r'C:\\[\w\\.-]*',  # C:\Windows\...
                r'[A-Za-z]:\\[\w\\.-]*',  # D:\path\...
            ],
            # URLs
            'url': [
                r'https?://[^\s]+',
            ],
            # Port numbers
            'port': [
                r':\d{2,5}\b',  # :8080, :443
            ],
            # Hex/Memory addresses
            'hex': [
                r'0x[0-9a-fA-F]+',
            ],
            # Process IDs, UIDs
            'ids': [
                r'\b(?:pid|uid|gid)[\s:=]+\d+',
                r'\bchild\s+\d+\b',
            ],
        }
    
    def remove_timestamps(self, text: str) -> str:
        """Hapus timestamps dari log"""
        for pattern in self.patterns['timestamp']:
            text = re.sub(pattern, '', text)
        return text
    
    def remove_ips(self, text: str) -> str:
        """Hapus IP addresses"""
        for pattern in self.patterns['ip']:
            text = re.sub(pattern, '<IP>', text)
        return text
    
    def normalize_paths(self, text: str) -> str:
        """Normalisasi file paths - ganti dengan token generik"""
        for pattern in self.patterns['filepath']:
            text = re.sub(pattern, '<FILEPATH>', text)
        return text
    
    def remove_urls(self, text: str) -> str:
        """Hapus URLs"""
        for pattern in self.patterns['url']:
            text = re.sub(pattern, '<URL>', text)
        return text
    
    def remove_hex_addresses(self, text: str) -> str:
        """Hapus hex/memory addresses"""
        for pattern in self.patterns['hex']:
            text = re.sub(pattern, '<HEX>', text)
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """Normalisasi angka - ganti dengan token atau kategori"""
        # Ganti angka besar (kemungkinan ID) dengan token
        text = re.sub(r'\b\d{5,}\b', '<NUM>', text)
        return text
    
    def remove_special_chars(self, text: str) -> str:
        """Hapus karakter khusus yang tidak perlu"""
        # Hapus karakter encoding
        text = re.sub(r'%[0-9a-fA-F]{2}', '', text)
        # Hapus escape characters
        text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
        text = re.sub(r'\\[rnt]', ' ', text)
        return text
    
    def lowercase(self, text: str) -> str:
        """Convert to lowercase untuk konsistensi"""
        return text.lower()
    
    def remove_extra_spaces(self, text: str) -> str:
        """Hapus spasi berlebih"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_log_level(self, text: str) -> tuple:
        """
        Extract log level (INFO, ERROR, WARNING, NOTICE)
        Returns: (log_level, cleaned_text)
        """
        levels = ['error', 'warning', 'notice', 'info', 'debug', 'critical']
        log_level = 'unknown'
        
        text_lower = text.lower()
        for level in levels:
            if level in text_lower:
                log_level = level
                break
        
        return log_level, text
    
    def preprocess(self, text: str, keep_log_level: bool = True) -> str:
        """
        Preprocessing lengkap untuk satu baris log
        
        Args:
            text: Raw log text
            keep_log_level: Apakah log level (ERROR, INFO, dll) tetap disimpan
        
        Returns:
            Preprocessed log text
        """
        # 1. Extract log level dulu (sebelum lowercase)
        log_level, text = self.extract_log_level(text)
        
        # 2. Hapus timestamp
        text = self.remove_timestamps(text)
        
        # 3. Normalisasi IP addresses
        text = self.remove_ips(text)
        
        # 4. Normalisasi file paths
        text = self.normalize_paths(text)
        
        # 5. Hapus URLs
        text = self.remove_urls(text)
        
        # 6. Hapus hex addresses
        text = self.remove_hex_addresses(text)
        
        # 7. Normalisasi angka
        text = self.normalize_numbers(text)
        
        # 8. Hapus special characters
        text = self.remove_special_chars(text)
        
        # 9. Lowercase
        text = self.lowercase(text)
        
        # 10. Clean extra spaces
        text = self.remove_extra_spaces(text)
        
        # 11. Optional: tambahkan log level di awal
        if keep_log_level and log_level != 'unknown':
            text = f"{log_level}: {text}"
        
        return text
    
    def preprocess_batch(self, log_lines: List[str], keep_log_level: bool = True) -> List[str]:
        """
        Preprocess multiple log lines
        
        Args:
            log_lines: List of raw log texts
            keep_log_level: Whether to keep log level
        
        Returns:
            List of preprocessed log texts
        """
        return [self.preprocess(line, keep_log_level) for line in log_lines]


# ==============================================================================
#           DEMONSTRASI PENGGUNAAN
# ==============================================================================
if __name__ == "__main__":
    print("="*80)
    print("DEMONSTRASI LOG PREPROCESSING")
    print("="*80)
    
    # Contoh log lines dari proxifier
    sample_logs = [
        "[Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP LDAP SDK",
        "[Thu Jun 09 06:07:05 2005] [error] env.createBean2(): Factory error creating channel.jni:jni ( channel.jni, jni)",
        "[Thu Jun 09 07:11:21 2005] [error] [client 204.100.200.22] Directory index forbidden by rule: /var/www/html/",
        "[Fri Jun 10 01:23:50 2005] [error] [client 63.203.254.140] Invalid method in request get /scripts/.%252e/.%252e/winnt/system32/cmd.exe?/c+dir",
        "[Sat Jun 11 03:03:03 2005] [error] [client 202.133.98.6] script not found or unable to stat: /var/www/cgi-bin/awstats",
        "[Thu Jun 09 19:23:32 2005] [notice] jk2_init() Found child 4056 in scoreboard slot 8",
    ]
    
    # Inisialisasi preprocessor
    preprocessor = LogPreprocessor()
    
    print("\n" + "="*80)
    print("PERBANDINGAN: SEBELUM vs SESUDAH PREPROCESSING")
    print("="*80)
    
    for i, log in enumerate(sample_logs, 1):
        print(f"\n--- Log #{i} ---")
        print(f"SEBELUM: {log}")
        
        preprocessed = preprocessor.preprocess(log, keep_log_level=True)
        print(f"SESUDAH: {preprocessed}")
    
    # Batch processing
    print("\n" + "="*80)
    print("BATCH PROCESSING")
    print("="*80)
    
    preprocessed_logs = preprocessor.preprocess_batch(sample_logs)
    
    print(f"\nTotal log diproses: {len(preprocessed_logs)}")
    print("\nHasil preprocessing:")
    for i, log in enumerate(preprocessed_logs, 1):
        print(f"{i}. {log}")
    
    # Save hasil preprocessing
    print("\n" + "="*80)
    print("SIMPAN HASIL PREPROCESSING")
    print("="*80)
    
    # Contoh: load dari file proxifier.log
    try:
        log_file = "../dataset/proxifier.log"
        print(f"Membaca file: {log_file}")
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_logs = f.readlines()
        
        print(f"Total log lines: {len(raw_logs)}")
        
        # Preprocess
        print("Melakukan preprocessing...")
        preprocessed = preprocessor.preprocess_batch(
            [line.strip() for line in raw_logs if line.strip()],
            keep_log_level=True
        )
        
        # Save to file
        output_file = "proxifier_preprocessed.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(preprocessed))
        
        print(f"✓ Hasil disimpan ke: {output_file}")
        print(f"✓ Total log diproses: {len(preprocessed)}")
        
        # Show statistics
        print("\n📊 STATISTIK PREPROCESSING:")
        log_levels = {}
        for log in preprocessed:
            if log.startswith('error:'):
                log_levels['error'] = log_levels.get('error', 0) + 1
            elif log.startswith('notice:'):
                log_levels['notice'] = log_levels.get('notice', 0) + 1
            elif log.startswith('warning:'):
                log_levels['warning'] = log_levels.get('warning', 0) + 1
            elif log.startswith('info:'):
                log_levels['info'] = log_levels.get('info', 0) + 1
            else:
                log_levels['other'] = log_levels.get('other', 0) + 1
        
        for level, count in sorted(log_levels.items()):
            print(f"  - {level.upper()}: {count} logs")
        
    except FileNotFoundError:
        print(f"⚠ File {log_file} tidak ditemukan")
        print("Silakan sesuaikan path file")
    
    print("\n" + "="*80)
    print("💡 KESIMPULAN:")
    print("="*80)
    print("""
    Preprocessing teks log SANGAT PENTING karena:
    
    ✅ Menghilangkan noise (timestamp, IP, path spesifik)
    ✅ BERT bisa fokus pada pola semantik yang benar
    ✅ Log yang mirip akan punya embedding yang mirip
    ✅ Clustering akan lebih akurat
    
    Gunakan log yang sudah dipreprocessing untuk:
    1. Generate BERT embeddings (bert.py)
    2. Clustering dengan DBSCAN/K-Means
    
    JANGAN lakukan normalisasi tambahan (StandardScaler, dll) 
    pada BERT embeddings - tidak perlu dan bisa harmful!
    """)
