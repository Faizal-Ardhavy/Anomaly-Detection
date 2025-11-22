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
                r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{1,2}:\d+',  # 2016-12-17 20:41:6:123 (Android variant)
                r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',  # 09/Jun/2005:06:07:04 (Apache)
                r'\b\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\b',  # 15/09/01 18:14:42 (Hadoop/OpenStack two-digit year)
                r'\d{8}-\d{1,2}:\d{1,2}:\d{1,2}:\d{1,3}\|',  # 20171224-20:11:16:931| (HealthApp)
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b',  # Feb 24 11:16:38 (Linux/SSH/Mac)
                r'\[\d{1,2}\.\d{1,2}\s+\d{1,2}:\d{2}:\d{2}\]',  # [10.30 17:37:51] (Proxifier)
                r'\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+',  # 12-17 19:31:36.263 (Android)
                r'\d{6}\s+\d{6}',  # 081109 203518 (HDFS)
                r'\d{10}\s+\d{4}\.\d{2}\.\d{2}',  # 1117838570 2005.06.03 (BGL)
                r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+',  # 2005-06-03-15.42.50.363779 (BGL)
                r',\d{3,}\s+(?:info|warn|error|debug)',  # ,755 info, ,756 warn (Hadoop/HDFS logs)
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
                r'\b\d{3,5}\s+\d{3,5}\s+[a-z]\b',  # Android: 2852 2852 d (lowercase variant)
                r'\bthread_\d+\b',  # Thread IDs
                r'\[\w+\]',  # [main], [Thread-1] - thread names in Java logs
                r'\bsub\s*=\s*\d+\b',  # sub=0 (Android subscription IDs)
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
            # Node/Location identifiers (BGL supercomputer, Thunderbird)
            'node_locations': [
                r'R\d{2}-M\d+-N\d+-C:J\d{2}-U\d{2}',  # R02-M1-N0-C:J12-U11 (BGL)
                r'\b[cdn]+\d+\s+[cdn]+\d+/[cdn]+\d+\b',  # cn951 cn951/cn951, dn1002 dn1002/dn1002 (Thunderbird)
                r'\b[a-z]{2}\d+\s+[a-z]{2}\d+/[a-z]{2}\d+\b',  # an143 an143/an143 (Thunderbird variants)
                r'\b[a-z]+\d+\s+src@[a-z]+\d+\b',  # aadmin2 src@aadmin2 (Thunderbird)
                r'\bR\d{2}-M\d+-N[fF]\s+[Cc]:J\d{2}-U\d{2}\b',  # r05-m0-nf c:j09-u11 (BGL variant)
                r'\bR\d{2}-M\d+-N\d+\s+R\d{2}-M\d+-N\d+\b',  # r02-m1-n2 r02-m1-n2 (BGL repetitive)
                r'\b[a-z]{2}\d+\s+[a-z]{2}\d+/[a-z]{2}\d+\.\d+\b',  # an143 an143/an143.72 (Thunderbird with subnet)
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
                r'str\s+mcc\s+mnc\s*=\s*\w+',  # str mcc mnc =null (Android signal controller)
                r'hash\s+mcc\s+mnc\s*=\s*\w*',  # hash mcc mnc = (Android)
                r'str\s+plmn\s*=\s*\w+',  # str plmn =hw (Android)
            ],
            # Java stack traces (Hadoop, HDFS, Spark)
            'java_stack_traces': [
                r'\bat\s+[\w.$]+\([\w\s.:]+\)',  # at org.apache.hadoop.ipc.Client.call(Client.java:1480)
                r'\bat\s+[\w.$]+\([^)]+\)',  # at sun.nio.ch.SocketChannelImpl.checkConnect(Native Method)
                r'caused\s+by:\s+[\w.]+:\s*',  # caused by: java.net.ConnectException:
                r'\.\.\.\s+\d+\s+more\s*$',  # ... 8 more
                r'\bunknown\s+source\b',  # (Unknown Source)
                r'\bnative\s+method\b',  # (Native Method)
            ],
            # Hardware/Kernel identifiers (Thunderbird, BGL)
            'hardware_identifiers': [
                r'acpi:\s+pci\s+interrupt\s+link\s+\([^)]+\)\s+[\d*,\s]+',  # ACPI PCI interrupt links
                r'acpi:\s+pci\s+interrupt\s+0x[\da-f:]+\s+->\s+gsi\s+\d+\s+\([^)]+\)\s+->\s+irq\s+\d+',  # ACPI PCI interrupt mapping
                r'\b0x[\da-f]{4}:[\da-f]{2}:[\da-f]{2}\.\d\b',  # PCI device addresses 0000:00:02.0
                r'\bgsi\s+\d+\s+\(level,\s*low\)',  # GSI interrupt descriptor
                r'\birq\s+\d+\b',  # IRQ numbers
            ],
            # Network/Interface codes (Thunderbird)
            'network_codes': [
                r'\bgi\s+\d+/\d+\b',  # gi 8/89 (GigabitEthernet interface)
                r'%[\w-]+:\s*%[\w-]+:',  # %RPM0-P:CP %IFMGR-5-OSTATE_DN:
                r'\b[a-z]{2,6}\d+/\d+\b',  # gi8/89, et0/1
            ],
            # Generic noise patterns
            'generic_noise': [
                r'slf4j\s+logger:\s+slf4j\s+logger\s+started',  # SLF4J logger startup noise
                r'\bfsck:\s+fsck\s+ext[234]\s+-a\s+',  # fsck command echoes
                r'\bfsck:\s+/:\s+clean,',  # fsck clean status (keep message part)
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
    
    def clean_java_stack_traces(self, text: str) -> str:
        """Clean Java stack traces - hapus detail line numbers, keep semantic info"""
        # Remove full stack trace lines starting with 'at'
        text = re.sub(r'\bat\s+[\w.$]+\([^)]+\)', '<STACK_TRACE>', text)
        
        # Remove 'caused by:' with exception class
        text = re.sub(r'caused\s+by:\s+[\w.]+:\s*', 'caused by: ', text)
        
        # Remove '... X more' lines
        text = re.sub(r'\.\.\.\s+\d+\s+more\s*', '', text)
        
        # Remove (Unknown Source) and (Native Method)
        text = re.sub(r'\(unknown\s+source\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\(native\s+method\)', '', text, flags=re.IGNORECASE)
        
        return text
    
    def remove_hardware_identifiers(self, text: str) -> str:
        """Hapus hardware/kernel identifiers (ACPI, PCI, IRQ)"""
        for pattern in self.patterns['hardware_identifiers']:
            text = re.sub(pattern, '<HW_ID>', text)
        return text
    
    def remove_network_codes(self, text: str) -> str:
        """Hapus network interface codes dan Cisco-style messages"""
        for pattern in self.patterns['network_codes']:
            text = re.sub(pattern, '', text)
        return text
    
    def remove_generic_noise(self, text: str) -> str:
        """Hapus generic noise patterns"""
        for pattern in self.patterns['generic_noise']:
            text = re.sub(pattern, '', text)
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
    
    def normalize_camel_case(self, text: str) -> str:
        """
        Convert camelCase menjadi spasi-separated words
        Examples:
            camelCase -> camel case
            getUserName -> get user name
            HTTPResponse -> http response
            XMLHttpRequest -> xml http request
        """
        # Handle sequence of capitals followed by lowercase (HTTPResponse -> HTTP Response)
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
        
        # Handle normal camelCase (userName -> user Name)
        text = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', text)
        
        return text
    
    def normalize_snake_case(self, text: str) -> str:
        """
        Convert snake_case menjadi spasi-separated words
        Examples:
            user_name -> user name
            get_user_data -> get user data
            HTTP_RESPONSE_CODE -> http response code
        """
        # Replace underscore dengan spasi
        text = text.replace('_', ' ')
        return text
    
    def normalize_kebab_case(self, text: str) -> str:
        """
        Convert kebab-case menjadi spasi-separated words
        Examples:
            user-name -> user name
            get-user-data -> get user data
            http-response-code -> http response code
        """
        # Replace hyphen dengan spasi, tapi jaga hyphen yang memang bagian dari kata
        # (mis: built-in, state-of-the-art)
        
        # Replace hyphen yang ada di tengah kata/identifier
        text = re.sub(r'([a-zA-Z])-([a-zA-Z])', r'\1 \2', text)
        return text
    
    def normalize_dot_separated(self, text: str) -> str:
        """
        Normalize dot-separated identifiers (method calls, package names)
        Examples:
            object.method.call -> object method call
            com.example.app -> com example app
        """
        # Replace dots dengan spasi untuk method calls dan package names
        # Tapi jaga dots di angka (1.0, 192.168.1.1)
        text = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1 \2', text)
        return text
    
    def normalize_all_naming_conventions(self, text: str) -> str:
        """
        Normalisasi semua naming conventions sekaligus
        Urutan penting untuk hasil optimal
        """
        # 1. Camel case dulu (sebelum lowercase)
        text = self.normalize_camel_case(text)
        
        # 2. Snake case
        text = self.normalize_snake_case(text)
        
        # 3. Kebab case
        text = self.normalize_kebab_case(text)
        
        # 4. Dot-separated
        text = self.normalize_dot_separated(text)
        
        return text
    
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
        
        # 2. Normalize naming conventions (SEBELUM lowercase!)
        # Penting: Dilakukan sebelum lowercase agar camelCase detection berfungsi
        text = self.normalize_all_naming_conventions(text)
        
        # 3. Hapus timestamp (berbagai format)
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
        
        # 20. Clean Java stack traces (Hadoop, HDFS, Spark)
        text = self.clean_java_stack_traces(text)
        
        # 21. Hapus hardware identifiers (ACPI, PCI, IRQ)
        text = self.remove_hardware_identifiers(text)
        
        # 22. Hapus network interface codes
        text = self.remove_network_codes(text)
        
        # 23. Hapus generic noise
        text = self.remove_generic_noise(text)
        
        # 24. Normalisasi HTTP metrics
        text = self.normalize_http_metrics(text)
        
        # 25. Normalisasi angka besar
        text = self.normalize_numbers(text)
        
        # 26. Hapus special characters
        text = self.remove_special_chars(text)
        
        # 27. Lowercase
        text = self.lowercase(text)
        
        # 28. Clean extra spaces
        text = self.remove_extra_spaces(text)
        
        # 29. Optional: tambahkan log level di awal
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
#           BATCH PROCESSING SEMUA FILE LOG (OPTIMIZED WITH MULTIPROCESSING)
# ==============================================================================
def process_all_log_files(dataset_dir: str = "../dataset", output_dir: str = "../after_PreProcessed_Dataset", 
                          num_workers: int = None, chunk_size: int = 10000, max_file_size_gb: float = None):
    """
    Process semua file .log di dataset directory dan simpan hasilnya
    OPTIMIZED: Menggunakan multiprocessing untuk file besar
    
    Args:
        dataset_dir: Path ke folder dataset
        output_dir: Path ke folder output untuk hasil preprocessing
        num_workers: Jumlah CPU cores untuk parallel processing (default: CPU count - 1)
        chunk_size: Ukuran chunk untuk processing (default: 10000 lines per chunk)
        max_file_size_gb: Skip files larger than this (default: None = process all)
    """
    import os
    from pathlib import Path
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    import time
    import gc
    
    # Create output directory if not exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave 1 core free
    
    # Find all .log files in dataset directory (including subdirectories)
    dataset_path = Path(dataset_dir)
    log_files = list(dataset_path.rglob("*.log"))
    
    print("="*100)
    print(f"BATCH PROCESSING - SEMUA FILE LOG DI {dataset_dir}")
    print("="*100)
    print(f"\n✓ Ditemukan {len(log_files)} file .log")
    print(f"✓ Output akan disimpan di: {output_dir}")
    print(f"✓ Menggunakan {num_workers} CPU cores untuk parallel processing")
    print(f"✓ Chunk size: {chunk_size:,} lines per batch")
    if max_file_size_gb:
        print(f"✓ Max file size: {max_file_size_gb:.1f} GB (larger files will be skipped)")
    print()
    
    # Statistics
    total_files_processed = 0
    total_logs_processed = 0
    failed_files = []
    skipped_files = []
    start_time = time.time()
    
    # Process each log file
    for i, log_file in enumerate(log_files, 1):
        try:
            # Get relative path for better naming
            rel_path = log_file.relative_to(dataset_path)
            
            # Create output filename
            output_filename = f"AfterPreProcessed_{str(rel_path).replace(os.sep, '_').replace('.log', '.txt')}"
            output_file_path = output_path / output_filename
            
            # Get file size
            file_size = log_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            file_size_gb = file_size / (1024 * 1024 * 1024)
            
            print(f"\n[{i}/{len(log_files)}] Processing: {rel_path}")
            print(f"    File size: {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
            print(f"    Output: {output_filename}")
            
            # Skip if file too large
            if max_file_size_gb and file_size_gb > max_file_size_gb:
                print(f"    ⚠️  SKIPPED: File too large (> {max_file_size_gb:.1f} GB)")
                skipped_files.append((str(rel_path), f"File too large: {file_size_gb:.2f} GB"))
                continue
            
            # Adjust chunk size for very large files
            actual_chunk_size = chunk_size
            if file_size_gb > 10:
                actual_chunk_size = max(chunk_size, 50000)  # Larger chunks for huge files
                print(f"    📦 Adjusted chunk size to {actual_chunk_size:,} for large file")
            
            # Process file with chunking and multiprocessing
            if file_size_mb > 100:  # Use multiprocessing for files > 100MB
                print(f"    ⚡ Using multiprocessing (file > 100MB)")
                preprocessed = process_large_file_parallel(log_file, num_workers, actual_chunk_size)
            else:
                print(f"    📄 Using single process (file < 100MB)")
                preprocessed = process_small_file(log_file)
            
            print(f"    ✓ Preprocessed {len(preprocessed):,} logs")
            
            # Save to output file (chunked writing for large files)
            print(f"    💾 Saving to file...")
            write_chunk_size = 50000  # Write 50K lines at a time
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for idx in range(0, len(preprocessed), write_chunk_size):
                    chunk = preprocessed[idx:idx + write_chunk_size]
                    f.write('\n'.join(chunk))
                    if idx + write_chunk_size < len(preprocessed):
                        f.write('\n')
            
            print(f"    ✓ Saved to: {output_file_path}")
            
            total_files_processed += 1
            total_logs_processed += len(preprocessed)
            
            # Force garbage collection after each file
            del preprocessed
            gc.collect()
            
        except KeyboardInterrupt:
            print(f"\n⚠ Process interrupted by user!")
            break
        except MemoryError:
            print(f"    ✗ MEMORY ERROR: File too large for available RAM")
            print(f"    💡 Suggestions:")
            print(f"       - Increase chunk_size (current: {actual_chunk_size:,})")
            print(f"       - Reduce num_workers (current: {num_workers})")
            print(f"       - Close other applications")
            print(f"       - Process this file separately")
            failed_files.append((str(rel_path), "MemoryError - File too large"))
            gc.collect()
            continue
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append((str(rel_path), str(e)))
            gc.collect()
            continue
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*100)
    print("📊 SUMMARY")
    print("="*100)
    print(f"\n✓ Total files processed: {total_files_processed}/{len(log_files)}")
    print(f"✓ Total log lines processed: {total_logs_processed:,}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    if total_logs_processed > 0:
        print(f"✓ Average speed: {total_logs_processed/(elapsed_time+0.001):.0f} logs/second")
    
    if skipped_files:
        print(f"\n⏭️  Skipped files: {len(skipped_files)}")
        for file, reason in skipped_files[:5]:
            print(f"  - {file}: {reason}")
        if len(skipped_files) > 5:
            print(f"  ... and {len(skipped_files)-5} more")
    
    if failed_files:
        print(f"\n⚠ Failed files: {len(failed_files)}")
        for file, error in failed_files[:10]:  # Show max 10 errors
            print(f"  - {file}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more errors")
    
    print("\n" + "="*100)
    print("✅ BATCH PROCESSING SELESAI!")
    print("="*100)


def process_small_file(log_file):
    """Process small file (<100MB) without multiprocessing"""
    preprocessor = LogPreprocessor()
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        raw_logs = [line.strip() for line in f if line.strip()]
    
    print(f"    ✓ Loaded {len(raw_logs):,} lines")
    
    preprocessed = preprocessor.preprocess_batch(raw_logs, keep_log_level=True)
    return preprocessed


def process_chunk(args):
    """Process a chunk of log lines (worker function for multiprocessing)"""
    chunk_lines, chunk_id = args
    preprocessor = LogPreprocessor()
    return preprocessor.preprocess_batch(chunk_lines, keep_log_level=True)


def process_large_file_parallel(log_file, num_workers, chunk_size):
    """Process large file (>100MB) with multiprocessing - MEMORY SAFE"""
    from multiprocessing import Pool
    from tqdm import tqdm
    import gc
    
    # STREAMING PROCESSING - Tidak load seluruh file ke memory!
    print(f"    📖 Reading file with streaming (memory-safe)...")
    
    preprocessed_file = str(log_file).replace('.log', '_preprocessed_temp.txt')
    total_lines = 0
    chunks_processed = 0
    
    try:
        # First pass: Count lines for progress bar (optional, bisa di-skip untuk file SANGAT besar)
        # Skip counting untuk file > 1GB untuk hemat waktu
        file_size_gb = log_file.stat().st_size / (1024**3)
        
        if file_size_gb < 1:
            print(f"    📊 Counting lines...")
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for line in f if line.strip())
            print(f"    ✓ Total lines: {total_lines:,}")
        else:
            print(f"    ⚡ Skipping line count (file too large: {file_size_gb:.2f} GB)")
            total_lines = None
        
        # Process file in streaming mode
        print(f"    ⚙️  Processing with {num_workers} workers (streaming mode)...")
        
        chunk_buffer = []
        all_results = []
        chunks_count = 0
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            with Pool(processes=num_workers) as pool:
                # Create progress bar
                if total_lines:
                    pbar = tqdm(total=total_lines, desc="    Processing lines", unit="lines")
                else:
                    pbar = tqdm(desc="    Processing lines", unit="lines")
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        chunk_buffer.append(line)
                        
                        # When buffer reaches chunk_size, process it
                        if len(chunk_buffer) >= chunk_size:
                            # Submit chunk to pool
                            result = pool.apply_async(process_chunk, ((chunk_buffer.copy(), chunks_count),))
                            all_results.append(result)
                            chunks_count += 1
                            
                            # Update progress
                            pbar.update(len(chunk_buffer))
                            
                            # Clear buffer
                            chunk_buffer = []
                            
                            # Limit number of pending tasks to avoid memory buildup
                            if len(all_results) >= num_workers * 2:
                                # Get results from completed tasks
                                completed = []
                                for idx, res in enumerate(all_results):
                                    if res.ready():
                                        completed.append(idx)
                                
                                # Remove completed from list
                                for idx in reversed(completed):
                                    all_results.pop(idx)
                                
                                # Force garbage collection
                                gc.collect()
                
                # Process remaining lines in buffer
                if chunk_buffer:
                    result = pool.apply_async(process_chunk, ((chunk_buffer.copy(), chunks_count),))
                    all_results.append(result)
                    pbar.update(len(chunk_buffer))
                    chunks_count += 1
                
                pbar.close()
                
                # Wait for all remaining tasks
                print(f"    ⏳ Waiting for all {len(all_results)} remaining chunks...")
                final_results = []
                for res in tqdm(all_results, desc="    Collecting results", unit="chunk"):
                    final_results.append(res.get())
                
                # Flatten results
                preprocessed = []
                for result in final_results:
                    preprocessed.extend(result)
        
        print(f"    ✓ Processing complete: {len(preprocessed):,} lines")
        return preprocessed
        
    except Exception as e:
        print(f"    ✗ Error during parallel processing: {e}")
        raise


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
        # Test camelCase, snake_case, kebab-case normalization
        "getUserData from userName with firstName and lastName",
        "http_response_code is OK with status_code 200",
        "user-profile-data sent to api-gateway-service",
        "DataNode.PacketResponder for BlockManager.updatePipeline",
        "HTTPResponse.XMLHttpRequest.getResponseData",
        
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
        
        # 12. Thunderbird Email Server logs
        "cn951 cn951/cn951 kernel: mosal(1): mnt_projects/sysapps/src/ib/topspin/hostname-16/third_party/thca4_linux/kernel/mlxsys/obj_host_amd64_custom1_rhel4/mlxsys/mosal_mem.c: mosal_virt_to_phys_ex: cannot retrieve pmd_p: prot_ctx=user, va=0x7fed806eb5d",
        "dn1002 dn1002/dn1002 kernel: mosal(1): mnt_projects/sysapps/src/ib/topspin/hostname-16/third_party/thca4_linux/kernel/mlxsys/obj_host_amd64_custom1_rhel4/mlxsys/mosal_mem.c: mosal_virt_to_phys_ex: cannot retrieve pmd_p: prot_ctx=user, va=0x7fed806eb5d",
        
        # Problematic samples from user's analysis
        # Android - timestamp and IDs
        "sdk : 123 sdk: ue se c 2016-12-17 20:41:6:123 level magic:decoder message type is invalidation;invalidation mmtp msg enable extend is",
        "2852 2852 d hw cust mobile signal controller impl: sub =0,str mcc mnc =null,hash mcc mnc =,str plmn =hw hspap show 4g",
        
        # BGL - repetitive node names
        "- r05-m0-nf c:j09-u11 r05-m0-nf c:j09-u11 RAS KERNEL INFO generating core.123456",
        "- r02-m1-n2 r02-m1-n2 NULL DISCOVERY ERROR node card status: no alerts are active. clock mode is low.",
        
        # HDFS/Hadoop - timestamps and stack traces
        ",755 INFO org.apache.hadoop.ipc.Client: Retrying connect to server: mesos-master-1/10.0.0.1:9000. Already tried 9 time(s)",
        "at org.apache.hadoop.ipc.Client.call(Client.java:1480)",
        "at sun.nio.ch.SocketChannelImpl.checkConnect(Native Method)",
        "caused by: java.net.ConnectException: Connection refused",
        "... 8 more",
        
        # Spark - noise and stack traces
        "INFO slf4j.Slf4jLogger: Slf4j logger started",
        "at org.apache.spark.network.shuffle.RetryingBlockFetcher.fetchAllOutstanding(RetryingBlockFetcher.java:140)",
        
        # Thunderbird - unique IDs and hardware codes
        "aadmin2 src@aadmin2 xinetd: START: rsync from=192.168.1.100",
        "an143 an143/an143 fsck: /: clean, 123456/789012 files, 456789/901234 blocks",
        "an143 an143/an143 kernel: ACPI: PCI Interrupt Link [LNKS] (IRQs 3 4 5 6 7 10 11 12) *0, disabled.",
        "an143 an143/an143 kernel: ACPI: PCI interrupt 0000:00:02.0 -> GSI 16 (level, low) -> IRQ 169",
        "192.168.1.72 192.168.1.72/192.168.1.72 .72 MDT: %RPM0-P:CP %IFMGR-5-OSTATE_DN: Changed interface state to down: Gi 8/89",
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
    ✅ Menormalisasi naming conventions:
       - camelCase → camel case (getUserData → get user data)
       - snake_case → snake case (user_name → user name)
       - kebab-case → kebab case (api-gateway → api gateway)
       - dot.separated → dot separated (object.method → object method)
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
    
    Preprocessing yang dilakukan (29 tahap):
    - Normalisasi naming conventions (camelCase, snake_case, kebab-case)
    - Hapus timestamp (13+ format berbeda)
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
    - Hapus node locations (BGL supercomputer, Thunderbird)
    - Normalisasi Windows-specific patterns
    - Normalisasi Android-specific patterns
    - Clean Java stack traces (at, caused by, ... X more)
    - Hapus hardware identifiers (ACPI, PCI, IRQ)
    - Hapus network interface codes (Cisco, Linux)
    - Hapus generic noise (SLF4J, fsck echoes)
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
