"""
BGL Log Preprocessing for BERT Anomaly Detection
Optimized for semantic preservation and BERT efficiency
"""

import re
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
        # Remove leading "- " if present
        line = line.strip()
        if line.startswith('- '):
            line = line[2:]
        
        # Check if first field is a label (not a timestamp)
        first_word = line.split(maxsplit=1)[0]
        
        if first_word.isalpha() or (not first_word.isdigit()):
            # Format 2: has label prefix (e.g., "KERNMNTF")
            # [Label] [UNIX_TS] [Date] [Node] [Timestamp] [Node] [Component] [Subsystem] [Level] [Message...]
            parts = line.split(maxsplit=9)
            if len(parts) < 10:
                return None
            return {
                'component': parts[6],
                'subsystem': parts[7],
                'level': parts[8],
                'message': parts[9]
            }
        else:
            # Format 1: no label prefix
            # [UNIX_TS] [Date] [Node] [Timestamp] [Node] [Component] [Subsystem] [Level] [Message...]
            parts = line.split(maxsplit=8)
            if len(parts) < 9:
                return None
            return {
                'component': parts[5],
                'subsystem': parts[6],
                'level': parts[7],
                'message': parts[8]
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
        
        # Parse line
        parsed = self.parse_bgl_line(line)
        if not parsed:
            self.stats['malformed_lines'] += 1
            return ""
        
        # Normalize message
        normalized_msg = self.normalize_message(parsed['message'])
        
        # Skip if message becomes empty after normalization
        if not normalized_msg:
            self.stats['skipped_lines'] += 1
            return ""
        
        # Combine fields WITHOUT node ID
        # Format: [component] [subsystem] [level] [message]
        preprocessed = f"{parsed['component'].lower()} {parsed['subsystem'].lower()} {parsed['level'].lower()} {normalized_msg}"
        
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
    
    if len(sys.argv) < 3:
        print("Usage: python bgl_log_preprocessing.py <input_file> <output_file> [--keep-duplicates]")
        print("\nExample:")
        print("  python bgl_log_preprocessing.py BGL.log BGL_preprocessed.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_dups = "--keep-duplicates" not in sys.argv
    
    process_bgl_file(input_file, output_file, remove_duplicates=remove_dups)
