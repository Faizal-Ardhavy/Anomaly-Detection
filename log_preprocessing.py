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
            # Timestamp patterns (berbagai format dari berbagai jenis log)
            'timestamp': [
                r'\[.*?\]',  # [Thu Jun 09 06:07:04 2005]
                r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[\.\d+]*',  # 2005-06-09 06:07:04
                r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',  # 09/Jun/2005:06:07:04
                r'\d{8}-\d{1,2}:\d{1,2}:\d{1,2}:\d{1,3}\|',  # 20171224-20:11:16:931| (HealthApp)
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b',  # Feb 24 11:16:38 (Linux/SSH)
                r'\[\d{1,2}\.\d{1,2}\s+\d{1,2}:\d{2}:\d{2}\]',  # [10.30 17:37:51] (Proxifier)
            ],
            # IP Address patterns
            'ip': [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # 192.168.1.1
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',  # IPv6
            ],
            # Hostname/Domain patterns
            'hostname': [
                r'\b[\w-]+\.[\w.-]+\.[\w.-]+\.[\w]+\b',  # proxy.cse.cuhk.edu.hk, adsl-69-230-112-80.dsl.scrm01.pacbell.net
                r'\b[\w-]+\.[\w-]+\.[\w]+\b',  # example.com, mx.dyn.com.ar
            ],
            # File paths (keep struktur, hapus path spesifik)
            'filepath': [
                r'/var/www/[\w/.-]*',  # /var/www/html/...
                r'/usr/[\w/.-]*',  # /usr/local/...
                r'/etc/[\w/.-]*',  # /etc/httpd/...
                r'/scripts/[\w/.-]*',  # /scripts/...
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
            # Process IDs, UIDs, GIDs, TTY
            'ids': [
                r'\b(?:pid|uid|gid)[\s:=]+\d+',
                r'\beuid=\d+',
                r'\bchild\s+\d+\b',
                r'\[\d{4,}\]',  # [6248], [30002312] (process IDs in brackets)
                r'tty=\w+',  # tty=NODEVssh
            ],
            # Program/Service names with PID
            'program': [
                r'\b(?:sshd|ftpd|telnetd|httpd|chrome\.exe)\([\w_]+\)\[\d+\]',  # sshd(pam_unix)[6248]
            ],
            # Bytes/Data size patterns
            'datasize': [
                r'\b\d+\s+bytes(?:\s+\([\d.]+\s+[KMG]B\))?',  # 0 bytes, 1438 bytes (1.40 KB)
                r'\b[\d.]+\s+[KMG]B\b',  # 1.40 KB, 41.2 MB
            ],
            # Duration/Lifetime patterns
            'duration': [
                r'lifetime\s+\d{2}:\d{2}',  # lifetime 00:01
                r'lifetime\s+<\d+\s+sec',  # lifetime <1 sec
            ],
            # HealthApp specific patterns (angka statistik)
            'healthapp_stats': [
                r'=\d+##\d+##\d+##\d+##\d+##\d+',  # =1514117400000##11414##649730##...
                r'totalCalories=\d+',
                r'totalAltitude=\d+',
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
    
    def normalize_hostnames(self, text: str) -> str:
        """Normalisasi hostnames dan domains"""
        for pattern in self.patterns['hostname']:
            text = re.sub(pattern, '<HOSTNAME>', text)
        return text
    
    def remove_program_pids(self, text: str) -> str:
        """Hapus program names dengan PID dalam kurung"""
        for pattern in self.patterns['program']:
            # Extract program name only, remove PID
            text = re.sub(pattern, lambda m: m.group(0).split('(')[0], text)
        return text
    
    def normalize_data_sizes(self, text: str) -> str:
        """Normalisasi ukuran data (bytes, KB, MB)"""
        for pattern in self.patterns['datasize']:
            text = re.sub(pattern, '<SIZE>', text)
        return text
    
    def normalize_durations(self, text: str) -> str:
        """Normalisasi duration/lifetime"""
        for pattern in self.patterns['duration']:
            text = re.sub(pattern, '<DURATION>', text)
        return text
    
    def normalize_healthapp_stats(self, text: str) -> str:
        """Normalisasi statistik HealthApp"""
        for pattern in self.patterns['healthapp_stats']:
            text = re.sub(pattern, '<STATS>', text)
        return text
    
    def remove_process_ids(self, text: str) -> str:
        """Hapus process IDs, UIDs, GIDs, TTY"""
        for pattern in self.patterns['ids']:
            text = re.sub(pattern, '', text)
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
        
        # 2. Hapus timestamp (berbagai format)
        text = self.remove_timestamps(text)
        
        # 3. Hapus program names dengan PID
        text = self.remove_program_pids(text)
        
        # 4. Hapus process IDs, UIDs, GIDs, TTY
        text = self.remove_process_ids(text)
        
        # 5. Normalisasi IP addresses
        text = self.remove_ips(text)
        
        # 6. Normalisasi hostnames/domains
        text = self.normalize_hostnames(text)
        
        # 7. Normalisasi file paths
        text = self.normalize_paths(text)
        
        # 8. Hapus URLs
        text = self.remove_urls(text)
        
        # 9. Hapus hex addresses
        text = self.remove_hex_addresses(text)
        
        # 10. Normalisasi data sizes
        text = self.normalize_data_sizes(text)
        
        # 11. Normalisasi durations
        text = self.normalize_durations(text)
        
        # 12. Normalisasi HealthApp statistics
        text = self.normalize_healthapp_stats(text)
        
        # 13. Normalisasi angka besar
        text = self.normalize_numbers(text)
        
        # 14. Hapus special characters
        text = self.remove_special_chars(text)
        
        # 15. Lowercase
        text = self.lowercase(text)
        
        # 16. Clean extra spaces
        text = self.remove_extra_spaces(text)
        
        # 17. Optional: tambahkan log level di awal
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
    print("DEMONSTRASI LOG PREPROCESSING - MULTI-TYPE LOGS")
    print("="*80)
    
    # Contoh log lines dari BERBAGAI JENIS LOG
    sample_logs = [
        # Apache logs
        "[Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP LDAP SDK",
        "[Thu Jun 09 06:07:05 2005] [error] env.createBean2(): Factory error creating channel.jni:jni ( channel.jni, jni)",
        "[Thu Jun 09 07:11:21 2005] [error] [client 204.100.200.22] Directory index forbidden by rule: /var/www/html/",
        
        # Proxifier logs
        "[10.30 17:37:51] chrome.exe - proxy.cse.cuhk.edu.hk:5070 open through proxy proxy.cse.cuhk.edu.hk:5070 HTTPS",
        "[10.30 17:37:53] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 3755 bytes (3.66 KB) sent, 1776 bytes (1.73 KB) received, lifetime 00:02",
        "[10.30 17:37:54] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 529 bytes sent, 43281385 bytes (41.2 MB) received, lifetime 00:42",
        
        # HealthApp logs
        "20171224-20:11:16:931|Step_SPUtils|30002312|setTodayTotalDetailSteps=1514117400000##11414##649730##8661##25953##16727998",
        "20171224-20:11:16:938|Step_ExtSDM|30002312|calculateCaloriesWithCache totalCalories=178786",
        "20171224-20:11:16:944|Step_StandReportReceiver|30002312|REPORT : 11414 8149 244487 210",
        
        # Linux SSH/FTP logs
        "Feb 24 11:16:38 combo sshd(pam_unix)[6248]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=83.228.113.166  user=root",
        "Feb 24 11:57:02 combo ftpd[6265]: connection from 89.52.108.44 (P6c2c.p.pppool.de) at Fri Feb 24 11:57:02 2006",
        "Feb 25 04:02:59 combo telnetd[6464]: ttloop:  peer died: Invalid or incomplete multibyte or wide character",
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
    
    # Contoh: load dari file gabungan
    try:
        log_file = "../dataset/apache+proxifier+healthapp+mac+ssh+linux.log"
        print(f"Membaca file: {log_file}")
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_logs = f.readlines()
        
        print(f"Total log lines: {len(raw_logs)}")
        
        print("Melakukan preprocessing...")
        size = len(raw_logs)
        preprocessed = preprocessor.preprocess_batch(
            [line.strip() for line in raw_logs[:size] if line.strip()],
            keep_log_level=True
        )
        
        # Save to file
        output_file = "combined_logs_preprocessed.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(preprocessed))
        
        print(f"[OK] Hasil disimpan ke: {output_file}")
        print(f"[OK] Total log diproses: {len(preprocessed)}")
        
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
    
    ✅ Menghilangkan noise (timestamp, IP, path spesifik, PIDs, sizes)
    ✅ Menormalisasi berbagai format log (Apache, Proxifier, HealthApp, Linux/SSH)
    ✅ BERT bisa fokus pada pola semantik yang benar
    ✅ Log yang mirip akan punya embedding yang mirip
    ✅ Clustering akan lebih akurat dan robust
    
    Preprocessing yang dilakukan:
    - Hapus timestamp (berbagai format)
    - Normalisasi IP, hostname, domain
    - Normalisasi file paths
    - Hapus process IDs, UIDs, PIDs
    - Normalisasi data sizes (bytes, KB, MB)
    - Normalisasi durations/lifetimes
    - Normalisasi statistik aplikasi
    - Lowercase untuk konsistensi
    
    Gunakan log yang sudah dipreprocessing untuk:
    1. Generate BERT embeddings (bert.py)
    2. Clustering dengan DBSCAN/K-Means
    
    JANGAN lakukan normalisasi tambahan (StandardScaler, dll) 
    pada BERT embeddings - tidak perlu dan bisa harmful!
    """)
