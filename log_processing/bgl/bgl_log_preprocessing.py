"""
BGL Log Preprocessing for BERT Anomaly Detection
Optimized for semantic preservation and BERT efficiency
"""

import re
import csv
from typing import List, Set
from pathlib import Path


class BGLLogPreprocessor:
    """
    Preprocessor untuk BGL (BlueGene/L) log format
    
    Format input:
    [UNIX_TS] [DATE] [NODE] [TIMESTAMP] [NODE] [COMPONENT] [SUBSYSTEM] [LEVEL] [MESSAGE]
    
    Format output:
    [component] [subsystem] [level] [normalized_message]
    """
    
    def __init__(self):
        # Regex patterns for normalization
        self.patterns = {
            # Numbers (integers, floats, scientific notation)
            'number': re.compile(r'\b\d+\.?\d*([eE][+-]?\d+)?\b'),
            
            # Hexadecimal values
            'hex': re.compile(r'\b0[xX][0-9a-fA-F]+\b'),
            
            # Node IDs (BGL format: RXX-MX-NX-C:JXX-UXX or similar)
            'node': re.compile(r'\b[A-Z]\d{2,3}-[A-Z]\d+-[A-Z]{1,2}(-[A-Z])?:[A-Z]\d{2}-[A-Z]\d{2}\b'),
            
            # IP addresses (IPv4)
            'ip': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b'),
            
            # File paths (Unix/Linux style)
            'path': re.compile(r'(?:/[a-zA-Z0-9_\-\.]+)+/?'),
            
            # URLs
            'url': re.compile(r'https?://[^\s]+'),
            
            # Email addresses
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Timestamps (various formats)
            'timestamp': re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}[-\s]\d{2}[:.]\d{2}[:.]\d{2}(?:\.\d+)?'),
        }
        
        # Track statistics
        self.stats = {
            'total_lines': 0,
            'processed_lines': 0,
            'skipped_lines': 0,
            'empty_lines': 0,
            'malformed_lines': 0,
        }
    
    def parse_bgl_line(self, line: str) -> dict:
        """
        Parse BGL log line into components
        
        Supports TWO formats:
        
        Format 1 (with - prefix):
        - [0] UNIX timestamp [1] Date [2] Node [3] Timestamp [4] Node [5] Component [6] Subsystem [7] Level [8] Message
        
        Format 2 (with label prefix):
        [0] Label [1] UNIX timestamp [2] Date [3] Node [4] Timestamp [5] Node [6] Component [7] Subsystem [8] Level [9] Message
        
        Note: Some logs have "- " prefix, need to skip it
        """
        # Remove leading "- " if present, but preserve label
        raw = line.rstrip('\n')
        line = raw.strip()

        label = ''
        if line.startswith('- '):
            label = '-'
            line = line[2:].lstrip()

        # Check if first field is a label (not a timestamp)
        first_word = line.split(maxsplit=1)[0]

        if first_word.isalpha() or (not first_word.isdigit()):
            # Format 2: has label prefix (e.g., "KERNMNTF")
            # [Label] [UNIX_TS] [Date] [Node] [Timestamp] [Node] [Component] [Subsystem] [Level] [Message...]
            parts = line.split(maxsplit=9)
            if len(parts) < 10:
                return None
            return {
                'label': parts[0],
                'unix_ts': parts[1],
                'date': parts[2],
                'node': parts[3],
                'ts': parts[4],
                'node_repeat': parts[5],
                'component': parts[6],
                'subsystem': parts[7],
                'level': parts[8],
                'message': parts[9],
                'raw': raw
            }
        else:
            # Format 1: no label prefix
            # [UNIX_TS] [Date] [Node] [Timestamp] [Node] [Component] [Subsystem] [Level] [Message...]
            parts = line.split(maxsplit=8)
            if len(parts) < 9:
                return None
            return {
                'label': label,
                'unix_ts': parts[0],
                'date': parts[1],
                'node': parts[2],
                'ts': parts[3],
                'node_repeat': parts[4],
                'component': parts[5],
                'subsystem': parts[6],
                'level': parts[7],
                'message': parts[8],
                'raw': raw
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
        
        # 3. IP addresses (before numbers to avoid partial matches)
        message = self.patterns['ip'].sub('<IP>', message)
        
        # 4. File paths
        message = self.patterns['path'].sub('<PATH>', message)
        
        # 5. Timestamps
        message = self.patterns['timestamp'].sub('', message)
        
        # 6. Node IDs
        message = self.patterns['node'].sub('<NODE>', message)
        
        # 7. Hexadecimal values (before general numbers)
        message = self.patterns['hex'].sub('<HEX>', message)
        
        # 8. Numbers (integers and floats)
        message = self.patterns['number'].sub('<NUM>', message)
        
        # 9. Lowercase
        message = message.lower()
        
        # 10. Remove special characters (keep only alphanumeric, spaces, <>)
        message = re.sub(r'[^\w\s<>]', ' ', message)
        
        # 11. Collapse multiple spaces
        message = re.sub(r'\s+', ' ', message)
        
        # 12. Strip whitespace
        message = message.strip()
        
        return message
    
    def preprocess_line(self, line: str) -> str:
        """
        Preprocess a single BGL log line
        
        Returns:
            Preprocessed string: "[component] [subsystem] [level] [message]"
            or empty string if line should be skipped
        """
        self.stats['total_lines'] += 1
        
        # Skip empty lines
        if not line or not line.strip():
            self.stats['empty_lines'] += 1
            return ""
        
        # Parse line and extract metadata
        parsed = self.parse_bgl_line(line)
        if not parsed:
            self.stats['malformed_lines'] += 1
            return "", None

        # Normalize message
        raw_msg = parsed.get('message', '')
        normalized_msg = self.normalize_message(raw_msg)

        # Skip if message becomes empty after normalization
        if not normalized_msg:
            self.stats['skipped_lines'] += 1
            return "", None

        # Combine fields WITHOUT node ID
        preprocessed = f"{parsed['component'].lower()} {parsed['subsystem'].lower()} {parsed['level'].lower()} {normalized_msg}"

        # Build metadata
        metadata = {
            'label': parsed.get('label',''),
            'unix_ts': parsed.get('unix_ts',''),
            'date': parsed.get('date',''),
            'node': parsed.get('node',''),
            'ts': parsed.get('ts',''),
            'node_repeat': parsed.get('node_repeat',''),
            'component': parsed.get('component',''),
            'subsystem': parsed.get('subsystem',''),
            'level': parsed.get('level',''),
            'ips': self.patterns['ip'].findall(raw_msg) or [],
            'raw_message': raw_msg,
            'raw_line': parsed.get('raw','')
        }

        self.stats['processed_lines'] += 1
        return preprocessed, metadata
    
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


def process_bgl_file(input_file: str, output_file: str, remove_duplicates: bool = True):
    """
    Process BGL log file and save preprocessed output
    
    Args:
        input_file: Path to input BGL log file
        output_file: Path to output preprocessed file
        remove_duplicates: Whether to remove duplicate entries
    """
    print("="*80)
    print("🔧 BGL LOG PREPROCESSING")
    print("="*80)
    print(f"\n✓ Input file: {input_file}")
    print(f"✓ Output file: {output_file}")
    print(f"✓ Remove duplicates: {remove_duplicates}")
    
    # Initialize preprocessor
    preprocessor = BGLLogPreprocessor()
    
    # Stream input and write messages + metadata TSV
    output_path = Path(output_file)
    meta_output_default = Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data") / (output_path.name.replace('.txt','') + '_meta.tsv')
    meta_output = str(meta_output_default)

    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    Path(meta_output_default.parent).mkdir(parents=True, exist_ok=True)

    total_lines = 0
    written_lines = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as inf, \
        open(output_file, 'w', encoding='utf-8') as outf, \
        open(meta_output, 'w', encoding='utf-8', newline='') as metaf:

        writer = csv.writer(metaf, delimiter='\t')
        header = ['label','unix_ts','date','node','ts','node_repeat','component','subsystem','level','ips','raw_message','raw_line']
        writer.writerow(header)

        for line in inf:
            total_lines += 1
            preprocessed, meta = preprocessor.preprocess_line(line)

            # Only write when preprocessing produced a valid entry and metadata
            if not preprocessed or meta is None:
                continue

            outf.write(preprocessed + '\n')

            row = [
                meta.get('label',''), meta.get('unix_ts',''), meta.get('date',''), meta.get('node',''), meta.get('ts',''), meta.get('node_repeat',''),
                meta.get('component',''), meta.get('subsystem',''), meta.get('level',''), '|'.join(meta.get('ips',[])), meta.get('raw_message',''), meta.get('raw_line','')
            ]
            writer.writerow(row)
            written_lines += 1

    print(f"✓ Processed and wrote {written_lines:,} lines to dataset and metadata files")
    
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
    # Example usage with argparse
    import argparse

    default_input = "../../../dataset/BGL/BGL.log"
    default_output = "/media/bioinfo04/Expansion/after_preprocessed_dataset/after_preprocessed_bgl.txt"

    parser = argparse.ArgumentParser(description='BGL log preprocessing')
    parser.add_argument('input_file', nargs='?', default=default_input, help='Path to BGL.log')
    parser.add_argument('output_file', nargs='?', default=default_output, help='Path to output preprocessed messages')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate messages (after preprocessing)')
    parser.add_argument('--sample-normal', type=int, default=None, help='Number of normal (label "-") lines to sample')
    parser.add_argument('--sample-non', type=int, default=None, help='Number of non-normal lines to sample')
    parser.add_argument('--meta-output', type=str, default=None, help='Path to metadata TSV output (overrides default)')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    remove_dups = True if getattr(args, 'remove_duplicates', False) else False
    sample_normal = args.sample_normal
    sample_non = args.sample_non

    # TODO: meta-output override not yet wired into process function
    process_bgl_file(input_file, output_file, remove_duplicates=remove_dups)
