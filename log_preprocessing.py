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
                r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[\.\d+]*',  # 2005-06-09 06:07:04 (Windows, OpenStack, Hadoop)
                r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',  # 09/Jun/2005:06:07:04 (Apache)
                r'\d{8}-\d{1,2}:\d{1,2}:\d{1,2}:\d{1,3}\|',  # 20171224-20:11:16:931| (HealthApp)
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b',  # Feb 24 11:16:38 (Linux/SSH/Mac)
                r'\[\d{1,2}\.\d{1,2}\s+\d{1,2}:\d{2}:\d{2}\]',  # [10.30 17:37:51] (Proxifier)
                r'\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+',  # 12-17 19:31:36.263 (Android)
                r'\d{6}\s+\d{6}',  # 081109 203518 (HDFS)
                r'\d{10}\s+\d{4}\.\d{2}\.\d{2}',  # 1117838570 2005.06.03 (BGL)
                r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+',  # 2005-06-03-15.42.50.363779 (BGL)
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
            # Process IDs, UIDs, GIDs, TTY, Thread IDs
            'ids': [
                r'\b(?:pid|uid|gid)[\s:=]+\d+',
                r'\beuid=\d+',
                r'\bchild\s+\d+\b',
                r'\[\d{4,}\]',  # [6248], [30002312] (process IDs in brackets)
                r'tty=\w+',  # tty=NODEVssh
                r'\b\d{1,5}\s+\d{1,5}\s+[IWE]\b',  # Android: 1795  1825 I (PID TID Level)
                r'\bthread_\d+\b',  # Thread IDs
                r'\[\w+\]',  # [main], [Thread-1] - thread names in Java logs
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
            # Request IDs, UUIDs, GUIDs (OpenStack, distributed systems)
            'request_ids': [
                r'req-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',  # OpenStack request IDs
                r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',  # UUIDs/GUIDs
                r'\b[0-9a-f]{32}\b',  # 32-char hex strings (tenant IDs, tokens)
            ],
            # Application/Container IDs (Hadoop, YARN)
            'app_container_ids': [
                r'application_\d{13}_\d{4}',  # application_1445062781478_0011
                r'container_\d{13}_\d{4}_\d{2}_\d{6}',  # container_1445182159119_0019_02_000015
                r'appattempt_\d{13}_\d{4}_\d{6}',  # appattempt_1445062781478_0011_000001
                r'attempt_\d{13}_\d{4}_[mr]_\d{6}_\d+',  # attempt IDs
            ],
            # Block IDs (HDFS)
            'block_ids': [
                r'blk_-?\d+',  # blk_-1608999687919862906
            ],
            # Component/Package names (Java, Python packages)
            'components': [
                r'\b(?:org|com|net|edu)\.[\w.]+\.[\w.]+',  # org.apache.hadoop.mapreduce.v2.app.MRAppMaster
            ],
            # Node/Location identifiers (BGL supercomputer)
            'node_locations': [
                r'R\d{2}-M\d+-N\d+-C:J\d{2}-U\d{2}',  # R02-M1-N0-C:J12-U11
            ],
            # Windows-specific patterns
            'windows': [
                r'@0x[0-9a-f]+',  # @0x7fed806eb5d (stack addresses)
                r'C:\\Windows\\[\w\\.-]+',  # C:\Windows\winsxs\...
                r'v\d+\.\d+\.\d+\.\d+',  # v6.1.7601.23505 (version numbers)
            ],
            # Android-specific patterns
            'android': [
                r'\b[A-Z][a-zA-Z]+\$[A-Z][a-zA-Z]+',  # DataNode$DataXceiver, DisplayPowerController
                r'action:[\w.]+',  # action:android.com.huawei.bone.NOTIFY_SPORT_DATA
            ],
            # HTTP status and metrics
            'http_metrics': [
                r'status:\s+\d{3}',  # status: 200
                r'len:\s+\d+',  # len: 1893
                r'time:\s+[\d.]+',  # time: 0.2477829
                r'HTTP/\d\.\d',  # HTTP/1.1
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
    
    def remove_request_ids(self, text: str) -> str:
        """Hapus request IDs, UUIDs, GUIDs (OpenStack, distributed systems)"""
        for pattern in self.patterns['request_ids']:
            text = re.sub(pattern, '<REQUEST_ID>', text)
        return text
    
    def remove_app_container_ids(self, text: str) -> str:
        """Hapus application dan container IDs (Hadoop, YARN)"""
        for pattern in self.patterns['app_container_ids']:
            text = re.sub(pattern, '<APP_ID>', text)
        return text
    
    def remove_block_ids(self, text: str) -> str:
        """Hapus block IDs (HDFS)"""
        for pattern in self.patterns['block_ids']:
            text = re.sub(pattern, '<BLOCK_ID>', text)
        return text
    
    def normalize_components(self, text: str) -> str:
        """Normalisasi component/package names (keep base package only)"""
        # Keep only the first 2-3 parts of package names for context
        # e.g., org.apache.hadoop.mapreduce.v2.app.MRAppMaster -> org.apache.hadoop
        text = re.sub(r'\b((?:org|com|net|edu)\.[\w]+\.[\w]+)\.[\w.]+', r'\1', text)
        return text
    
    def remove_node_locations(self, text: str) -> str:
        """Hapus node/location identifiers (BGL supercomputer)"""
        for pattern in self.patterns['node_locations']:
            text = re.sub(pattern, '<NODE>', text)
        return text
    
    def normalize_windows_specific(self, text: str) -> str:
        """Normalisasi Windows-specific patterns"""
        for pattern in self.patterns['windows']:
            text = re.sub(pattern, '<WIN_SPECIFIC>', text)
        return text
    
    def normalize_android_specific(self, text: str) -> str:
        """Normalisasi Android-specific patterns"""
        for pattern in self.patterns['android']:
            text = re.sub(pattern, '<ANDROID_SPECIFIC>', text)
        return text
    
    def normalize_http_metrics(self, text: str) -> str:
        """Normalisasi HTTP status codes dan metrics"""
        for pattern in self.patterns['http_metrics']:
            text = re.sub(pattern, '<METRIC>', text)
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
        
        # 13. Hapus request IDs, UUIDs (OpenStack, distributed systems)
        text = self.remove_request_ids(text)
        
        # 14. Hapus application/container IDs (Hadoop, YARN)
        text = self.remove_app_container_ids(text)
        
        # 15. Hapus block IDs (HDFS)
        text = self.remove_block_ids(text)
        
        # 16. Normalisasi component/package names
        text = self.normalize_components(text)
        
        # 17. Hapus node/location identifiers (BGL)
        text = self.remove_node_locations(text)
        
        # 18. Normalisasi Windows-specific patterns
        text = self.normalize_windows_specific(text)
        
        # 19. Normalisasi Android-specific patterns
        text = self.normalize_android_specific(text)
        
        # 20. Normalisasi HTTP metrics
        text = self.normalize_http_metrics(text)
        
        # 21. Normalisasi angka besar
        text = self.normalize_numbers(text)
        
        # 22. Hapus special characters
        text = self.remove_special_chars(text)
        
        # 23. Lowercase
        text = self.lowercase(text)
        
        # 24. Clean extra spaces
        text = self.remove_extra_spaces(text)
        
        # 25. Optional: tambahkan log level di awal
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
#           BATCH PROCESSING SEMUA FILE LOG
# ==============================================================================
def process_all_log_files(dataset_dir: str = "../dataset", output_dir: str = "../after_PreProcessed_Dataset"):
    """
    Process semua file .log di dataset directory dan simpan hasilnya
    
    Args:
        dataset_dir: Path ke folder dataset
        output_dir: Path ke folder output untuk hasil preprocessing
    """
    import os
    from pathlib import Path
    
    # Create output directory if not exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = LogPreprocessor()
    
    # Find all .log files in dataset directory (including subdirectories)
    dataset_path = Path(dataset_dir)
    log_files = list(dataset_path.rglob("*.log"))
    
    print("="*100)
    print(f"BATCH PROCESSING - SEMUA FILE LOG DI {dataset_dir}")
    print("="*100)
    print(f"\n✓ Ditemukan {len(log_files)} file .log")
    print(f"✓ Output akan disimpan di: {output_dir}\n")
    
    # Statistics
    total_files_processed = 0
    total_logs_processed = 0
    failed_files = []
    
    # Process each log file
    for i, log_file in enumerate(log_files, 1):
        try:
            # Get relative path for better naming
            rel_path = log_file.relative_to(dataset_path)
            
            # Create output filename
            # Replace path separators with underscores for flat structure
            output_filename = f"AfterPreProcessed_{str(rel_path).replace(os.sep, '_').replace('.log', '.txt')}"
            output_file_path = output_path / output_filename
            
            print(f"\n[{i}/{len(log_files)}] Processing: {rel_path}")
            print(f"    Output: {output_filename}")
            
            # Read log file
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_logs = f.readlines()
            
            print(f"    ✓ Loaded {len(raw_logs)} lines")
            
            # Preprocess
            preprocessed = preprocessor.preprocess_batch(
                [line.strip() for line in raw_logs if line.strip()],
                keep_log_level=True
            )
            
            print(f"    ✓ Preprocessed {len(preprocessed)} logs")
            
            # Save to output file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(preprocessed))
            
            print(f"    ✓ Saved to: {output_file_path}")
            
            total_files_processed += 1
            total_logs_processed += len(preprocessed)
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            failed_files.append((str(rel_path), str(e)))
            continue
    
    # Summary
    print("\n" + "="*100)
    print("📊 SUMMARY")
    print("="*100)
    print(f"\n✓ Total files processed: {total_files_processed}/{len(log_files)}")
    print(f"✓ Total log lines processed: {total_logs_processed:,}")
    print(f"✓ Output directory: {output_dir}")
    
    if failed_files:
        print(f"\n⚠ Failed files: {len(failed_files)}")
        for file, error in failed_files:
            print(f"  - {file}: {error}")
    
    print("\n" + "="*100)
    print("✅ BATCH PROCESSING SELESAI!")
    print("="*100)


# ==============================================================================
#           DEMONSTRASI PENGGUNAAN
# ==============================================================================
if __name__ == "__main__":
    import sys
    
    # Check if user wants to process all files or just demo
    if len(sys.argv) > 1 and sys.argv[1] == "--process-all":
        # Process all log files in dataset
        process_all_log_files(
            dataset_dir="../dataset",
            output_dir="../after_PreProcessed_Dataset"
        )
    else:
        # Demo mode - show examples
        print("="*80)
        print("DEMONSTRASI LOG PREPROCESSING - MULTI-TYPE LOGS (17 JENIS LOG)")
        print("="*80)
        print("\nTip: Jalankan dengan '--process-all' untuk memproses semua file log")
        print("     python log_preprocessing.py --process-all\n")
        print("="*80)
    
    # Contoh log lines dari BERBAGAI JENIS LOG
    sample_logs = [
        # 1. Apache logs
        "[Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP LDAP SDK",
        "[Thu Jun 09 07:11:21 2005] [error] [client 204.100.200.22] Directory index forbidden by rule: /var/www/html/",
        
        # 2. Proxifier logs
        "[10.30 17:37:51] chrome.exe - proxy.cse.cuhk.edu.hk:5070 open through proxy proxy.cse.cuhk.edu.hk:5070 HTTPS",
        "[10.30 17:37:53] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 3755 bytes (3.66 KB) sent, 1776 bytes (1.73 KB) received, lifetime 00:02",
        
        # 3. HealthApp logs
        "20171224-20:11:16:931|Step_SPUtils|30002312|setTodayTotalDetailSteps=1514117400000##11414##649730##8661##25953##16727998",
        "20171224-20:11:16:938|Step_ExtSDM|30002312|calculateCaloriesWithCache totalCalories=178786",
        
        # 4. Linux SSH/FTP logs
        "Feb 24 11:16:38 combo sshd(pam_unix)[6248]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=83.228.113.166  user=root",
        "Feb 24 11:57:02 combo ftpd[6265]: connection from 89.52.108.44 (P6c2c.p.pppool.de) at Fri Feb 24 11:57:02 2006",
        
        # 5. Android logs
        "12-17 19:31:36.263  1795  1825 I PowerManager_screenOn: DisplayPowerStatesetColorFadeLevel: level=1.0",
        "12-17 19:31:36.264  1795  1825 D DisplayPowerController: Animating brightness: target=21, rate=40",
        
        # 6. Windows logs
        "2016-09-28 04:30:30, Info                  CBS    Starting TrustedInstaller initialization.",
        "2016-09-28 04:30:31, Info                  CSI    00000001@2016/9/27:20:30:31.455 WcpInitialize (wcp.dll version 0.0.0.6) called (stack @0x7fed806eb5d @0x7fef9fb9b6d)",
        
        # 7. Mac logs
        "Jul  1 09:00:55 calvisitor-10-105-160-95 kernel[0]: AppleThunderboltNHIType2::prePCIWake - power up complete - took 2 us",
        "Jul  1 09:00:55 calvisitor-10-105-160-95 kernel[0]: AirPort: Link Down on awdl0. Reason 1 (Unspecified).",
        
        # 8. BGL (Blue Gene/L supercomputer) logs
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
        
        # 9. HDFS logs
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating",
        
        # 10. OpenStack logs
        "nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server [req-38101a0b-2096-447d-96ea-a692162415ae 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 \"GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1\" status: 200 len: 1893 time: 0.2477829",
        
        # 11. Hadoop/YARN logs
        "2015-10-17 15:37:56,547 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster for application appattempt_1445062781478_0011_000001",
        "2015-10-17 15:37:56,900 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Kind: YARN_AM_RM_TOKEN, Service: , Ident: (appAttemptId { application_id { id: 11 cluster_timestamp: 1445062781478 } attemptId: 1 } keyId: 471522253)",
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
    print("\nHasil preprocessing (5 pertama):")
    for i, log in enumerate(preprocessed_logs[:5], 1):
        print(f"{i}. {log}")
    
    print("\n" + "="*80)
    print("💡 KESIMPULAN:")
    print("="*80)
    print("""
    Preprocessing teks log SANGAT PENTING karena:
    
    ✅ Menghilangkan noise (timestamp, IP, path spesifik, PIDs, sizes)
    ✅ Menormalisasi 17 JENIS LOG BERBEDA:
       1. Apache Web Server          10. Thunderbird Email Server
       2. Proxifier Proxy Logs        11. Zookeeper Coordination
       3. HealthApp Mobile Logs       12-14. OpenStack Cloud (3 variants)
       4. Linux/SSH/FTP System Logs   15. SSH Authentication
       5. Android Mobile Logs         16. Windows System Logs
       6. Mac/macOS Kernel Logs       17. Hadoop/YARN Container Logs
       7. BGL Supercomputer Logs
       8. HDFS Distributed Storage
       9. HPC Logs
    
    ✅ BERT bisa fokus pada pola semantik yang benar
    ✅ Log yang mirip akan punya embedding yang mirip
    ✅ Clustering akan lebih akurat dan robust
    
    Preprocessing yang dilakukan (25 tahap):
    - Hapus timestamp (10 format berbeda)
    - Normalisasi IP, hostname, domain
    - Normalisasi file paths
    - Hapus process IDs, UIDs, PIDs, Thread IDs
    - Normalisasi data sizes (bytes, KB, MB)
    - Normalisasi durations/lifetimes
    - Normalisasi statistik aplikasi
    - Hapus request IDs, UUIDs, GUIDs (OpenStack)
    - Hapus application/container IDs (Hadoop/YARN)
    - Hapus block IDs (HDFS)
    - Normalisasi package names (Java/Python)
    - Hapus node locations (BGL supercomputer)
    - Normalisasi Windows-specific patterns
    - Normalisasi Android-specific patterns
    - Normalisasi HTTP metrics
    - Lowercase untuk konsistensi
    
    🚀 CARA PENGGUNAAN:
    
    1. Demo mode (tampilkan contoh):
       python log_preprocessing.py
    
    2. Process SEMUA file log di dataset:
       python log_preprocessing.py --process-all
       
       Akan memproses semua .log files di ../dataset dan menyimpan ke:
       ../after_PreProcessed_Dataset/AfterPreProcessed_namafile.txt
    
    Gunakan log yang sudah dipreprocessing untuk:
    1. Generate BERT embeddings (bert.py)
    2. Clustering dengan DBSCAN/K-Means
    
    JANGAN lakukan normalisasi tambahan (StandardScaler, dll) 
    pada BERT embeddings - tidak perlu dan bisa harmful!
    """)
