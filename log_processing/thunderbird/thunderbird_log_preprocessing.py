"""
Thunderbird Log Preprocessing for BERT Anomaly Detection
Optimized for semantic preservation and BERT efficiency
"""

import re
from typing import List
from pathlib import Path


class ThunderbirdLogPreprocessor:
    """
    Preprocessor untuk Thunderbird supercomputer log format
    
    Format input:
    - [UNIX_TS] [DATE] [HOSTNAME] [SYSLOG_TS] [SOURCE] [COMPONENT][PID]: [LEVEL]: [MESSAGE]
    
    Format output:
    [component] [level] [normalized_message]
    """
    
    def __init__(self):
        # Regex patterns for normalization
        self.patterns = {
            # Numbers (integers, floats, scientific notation)
            'number': re.compile(r'\b\d+\.?\d*([eE][+-]?\d+)?\b'),
            
            # Hexadecimal values
            'hex': re.compile(r'\b0[xX][0-9a-fA-F]+\b'),
            
            # IP addresses (IPv4 and IPv6)
            'ipv4': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b'),
            'ipv6': re.compile(r'::ffff:[0-9a-f.:]+'),
            
            # Hostnames (thunderbird format: aadmin1, tbird-admin1, cn822, etc.)
            'hostname': re.compile(r'\b(?:aadmin|badmin|cadmin|dadmin|tbird-admin|tbird-sm|cn|bn)\d+\b', re.IGNORECASE),
            
            # File paths (Unix/Linux style)
            'path': re.compile(r'(?:/[a-zA-Z0-9_\-\.]+){2,}/?'),
            
            # URLs
            'url': re.compile(r'https?://[^\s]+'),
            
            # Email addresses
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Timestamps (various formats)
            'timestamp': re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}[-\s]\d{2}[:.]\d{2}[:.]\d{2}(?:\.\d+)?'),
            
            # Network addresses (AF_INET format)
            'af_inet': re.compile(r'AF_INET\([^)]+\)'),
            
            # User IDs
            'uid': re.compile(r'\(uid=\d+\)'),
            
            # Port numbers (when standalone)
            'port': re.compile(r'\bport\s+\d+\b'),
        }
        
        # Common log levels
        self.log_levels = {'fatal', 'error', 'warning', 'info', 'debug', 'notice'}
        
        # Track statistics
        self.stats = {
            'total_lines': 0,
            'processed_lines': 0,
            'skipped_lines': 0,
            'empty_lines': 0,
            'malformed_lines': 0,
        }
    
    def parse_thunderbird_line(self, line: str) -> dict:
        """
        Parse Thunderbird log line into components
        
        Format (multiple variations):
        - [0] UNIX_TS [1] DATE [2] HOSTNAME [3-5] SYSLOG_TIMESTAMP [6] SOURCE [7] COMPONENT[PID]: [MESSAGE]
        - [0] UNIX_TS [1] DATE [2] HOSTNAME [3-5] SYSLOG_TIMESTAMP [6] SOURCE [7] COMPONENT[PID]: [LEVEL]: [MESSAGE]
        
        Examples:
        - 1131523501 2005.11.09 aadmin1 Nov 10 00:05:01 src@aadmin1 in.tftpd[14620]: tftp: client does not accept options
        - 1131563443 2005.11.09 aadmin1 Nov 9 11:10:43 src@aadmin1 sshd[30057]: Accepted publickey for root from ::ffff:10.100.4.251 port 35558 ssh2
        """
        # Remove leading "- " if present
        line = line.strip()
        if line.startswith('- '):
            line = line[2:]
        
        # Split to get basic structure
        parts = line.split(maxsplit=7)
        
        if len(parts) < 8:
            return None
        
        # Extract component and message from parts[7]
        # Format: component[pid]: message or component[pid]: level: message
        component_msg = parts[7]
        
        # Split by first colon to separate component[pid] from rest
        if ':' not in component_msg:
            return None
        
        component_part, rest = component_msg.split(':', 1)
        rest = rest.strip()
        
        # Extract component name (remove [pid] if present)
        component = re.sub(r'\[\d+\]', '', component_part).strip()
        component = re.sub(r'\(pam_unix\)', '', component).strip()  # Remove (pam_unix)
        
        # If component is a full path, extract just the filename
        if '/' in component:
            component = component.split('/')[-1]  # Get last part after /
        
        # Check if rest starts with a log level
        level = 'info'  # default
        message = rest
        
        # Check if message starts with a known log level
        rest_lower = rest.lower()
        for lvl in self.log_levels:
            if rest_lower.startswith(lvl + ':'):
                level = lvl
                message = rest[len(lvl)+1:].strip()
                break
        
        return {
            'component': component,
            'level': level,
            'message': message
        }
    
    def normalize_message(self, message: str) -> str:
        """
        Normalize message by replacing variables with tokens
        """
        # Apply normalization in specific order (important!)
        
        # 1. URLs first (before other patterns)
        message = self.patterns['url'].sub('<URL>', message)
        
        # 2. Email addresses
        message = self.patterns['email'].sub('<EMAIL>', message)
        
        # 3. AF_INET addresses
        message = self.patterns['af_inet'].sub('<INET>', message)
        
        # 4. IPv6 addresses (before IPv4)
        message = self.patterns['ipv6'].sub('<IP>', message)
        
        # 5. IPv4 addresses
        message = self.patterns['ipv4'].sub('<IP>', message)
        
        # 6. Port numbers
        message = self.patterns['port'].sub('port <NUM>', message)
        
        # 7. File paths
        message = self.patterns['path'].sub('<PATH>', message)
        
        # 8. Timestamps
        message = self.patterns['timestamp'].sub('', message)
        
        # 9. Hostnames
        message = self.patterns['hostname'].sub('<HOST>', message)
        
        # 10. User IDs
        message = self.patterns['uid'].sub('<UID>', message)
        
        # 11. Hexadecimal values (before general numbers)
        message = self.patterns['hex'].sub('<HEX>', message)
        
        # 12. Numbers (integers and floats)
        message = self.patterns['number'].sub('<NUM>', message)
        
        # 13. Lowercase
        message = message.lower()
        
        # 14. Remove special characters (keep only alphanumeric, spaces, <>)
        message = re.sub(r'[^\w\s<>]', ' ', message)
        
        # 15. Collapse multiple spaces
        message = re.sub(r'\s+', ' ', message)
        
        # 16. Strip whitespace
        message = message.strip()
        
        return message
    
    def preprocess_line(self, line: str) -> str:
        """
        Preprocess a single Thunderbird log line
        
        Returns:
            Preprocessed string: "[component] [level] [message]"
            or empty string if line should be skipped
        """
        self.stats['total_lines'] += 1
        
        # Skip empty lines
        if not line or not line.strip():
            self.stats['empty_lines'] += 1
            return ""
        
        # Parse line
        parsed = self.parse_thunderbird_line(line)
        if not parsed:
            self.stats['malformed_lines'] += 1
            return ""
        
        # Normalize message
        normalized_msg = self.normalize_message(parsed['message'])
        
        # Skip if message becomes empty after normalization
        if not normalized_msg:
            self.stats['skipped_lines'] += 1
            return ""
        
        # Combine fields: [component] [level] [message]
        preprocessed = f"{parsed['component'].lower()} {parsed['level'].lower()} {normalized_msg}"
        
        self.stats['processed_lines'] += 1
        return preprocessed
    
    def preprocess_logs(self, lines: List[str]) -> List[str]:
        """
        Preprocess multiple log lines
        
        Returns:
            List of preprocessed log lines
        """
        preprocessed_lines = []
        
        for line in lines:
            preprocessed = self.preprocess_line(line)
            if preprocessed:  # Only add non-empty lines
                preprocessed_lines.append(preprocessed)
        
        return preprocessed_lines
    
    def remove_duplicates(self, logs: List[str]) -> List[str]:
        """
        Remove duplicate log entries while preserving order
        """
        seen = set()
        unique_logs = []
        
        for log in logs:
            if log not in seen:
                seen.add(log)
                unique_logs.append(log)
        
        return unique_logs
    
    def print_stats(self):
        """Print preprocessing statistics"""
        print("\n" + "="*80)
        print("📊 PREPROCESSING STATISTICS")
        print("="*80)
        print(f"\n✓ Total lines read: {self.stats['total_lines']:,}")
        print(f"✓ Successfully processed: {self.stats['processed_lines']:,}")
        print(f"✓ Empty lines: {self.stats['empty_lines']:,}")
        print(f"✓ Malformed lines: {self.stats['malformed_lines']:,}")
        print(f"✓ Skipped (no content): {self.stats['skipped_lines']:,}")
        
        if self.stats['total_lines'] > 0:
            success_rate = (self.stats['processed_lines'] / self.stats['total_lines']) * 100
            print(f"\n✓ Success rate: {success_rate:.2f}%")
    
    def reset_stats(self):
        """Reset statistics counters"""
        for key in self.stats:
            self.stats[key] = 0


def process_thunderbird_file(input_file: str, output_file: str, remove_duplicates: bool = True):
    """
    Process Thunderbird log file and save preprocessed output
    
    Args:
        input_file: Path to input Thunderbird log file
        output_file: Path to output preprocessed file
        remove_duplicates: Whether to remove duplicate entries
    """
    print("="*80)
    print("🔧 THUNDERBIRD LOG PREPROCESSING")
    print("="*80)
    print(f"\n✓ Input file: {input_file}")
    print(f"✓ Output file: {output_file}")
    print(f"✓ Remove duplicates: {remove_duplicates}")
    
    # Initialize preprocessor
    preprocessor = ThunderbirdLogPreprocessor()
    
    # Read input file
    print(f"\n📖 Reading input file...")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"✓ Read {len(lines):,} lines")
    
    # Preprocess
    print(f"\n⚙️  Preprocessing logs...")
    preprocessed_lines = preprocessor.preprocess_logs(lines)
    
    # Remove duplicates if requested
    original_count = len(preprocessed_lines)
    if remove_duplicates:
        print(f"\n🔄 Removing duplicates...")
        preprocessed_lines = preprocessor.remove_duplicates(preprocessed_lines)
        duplicates_removed = original_count - len(preprocessed_lines)
        print(f"✓ Removed {duplicates_removed:,} duplicates ({duplicates_removed/original_count*100:.2f}%)")
    
    # Save output
    print(f"\n💾 Saving preprocessed logs...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in preprocessed_lines:
            f.write(line + '\n')
    
    print(f"✓ Saved {len(preprocessed_lines):,} lines to {output_file}")
    
    # Print statistics
    preprocessor.print_stats()
    
    # Size comparison
    input_size = Path(input_file).stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100
    
    print("\n" + "="*80)
    print("📦 FILE SIZE COMPARISON")
    print("="*80)
    print(f"\n✓ Input size: {input_size:.2f} MB")
    print(f"✓ Output size: {output_size:.2f} MB")
    print(f"✓ Reduction: {reduction:.2f}%")
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Default paths
    default_input = "../../dataset/Thunderbird.log"
    default_output = "../../after_preprocessed_dataset/after_preprocessed_thunderbird.txt"
    
    # Filter out Jupyter/IPython arguments (e.g., '-f')
    filtered_args = [arg for arg in sys.argv if not arg.startswith('-f')]
    
    # Check if custom paths provided
    if len(filtered_args) >= 3:
        input_file = filtered_args[1]
        output_file = filtered_args[2]
    elif len(filtered_args) == 2 and filtered_args[1] in ['--help', '-h']:
        print("Usage: python thunderbird_log_preprocessing.py [input_file] [output_file] [--keep-duplicates]")
        print("\nDefault:")
        print(f"  Input:  {default_input}")
        print(f"  Output: {default_output}")
        print("\nExample:")
        print("  python thunderbird_log_preprocessing.py")
        print("  python thunderbird_log_preprocessing.py Thunderbird.log Thunderbird_preprocessed.txt")
        print("  python thunderbird_log_preprocessing.py Thunderbird.log Thunderbird_preprocessed.txt --keep-duplicates")
        sys.exit(0)
    else:
        # Use default paths
        input_file = default_input
        output_file = default_output
    
    remove_dups = "--keep-duplicates" not in filtered_args
    
    process_thunderbird_file(input_file, output_file, remove_duplicates=remove_dups)
