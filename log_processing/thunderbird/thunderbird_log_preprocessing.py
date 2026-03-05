"""
Thunderbird Log Preprocessing for BERT Anomaly Detection
Optimized for semantic preservation and BERT efficiency
"""

import re
import csv
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
        # Preserve raw line and detect optional leading label marker
        raw = line.rstrip('\n')
        line = raw.strip()

        label = ''
        if line.startswith('- '):
            label = '-'
            line = line[2:].lstrip()

        # Attempt robust split: unix_ts, date, host, sys_ts (3 tokens), source, remainder
        parts = line.split(maxsplit=7)
        if len(parts) < 7:
            return None

        unix_ts = parts[0]
        date = parts[1]
        host = parts[2]
        sys_ts = ' '.join(parts[3:6]) if len(parts) >= 6 else ''
        source = parts[6] if len(parts) >= 7 else ''

        remainder = parts[7] if len(parts) >= 8 else ''
        if not remainder:
            return None

        # remainder example: component[pid]: level: message  OR  component[pid]: message
        if ':' not in remainder:
            return None

        component_part, rest = remainder.split(':', 1)
        rest = rest.strip()

        # Extract PID if present
        pid_match = re.search(r'\[(\d+)\]', component_part)
        pid = pid_match.group(1) if pid_match else ''

        # Clean component name
        component = re.sub(r'\[\d+\]', '', component_part).strip()
        component = re.sub(r'\(pam_unix\)', '', component).strip()
        if '/' in component:
            component = component.split('/')[-1]

        # Detect level at start of rest (e.g., "error: message")
        level = 'info'
        level_match = re.match(r'^(?P<lvl>' + '|'.join(self.log_levels) + r')\s*:\s*(?P<msg>.*)$', rest, re.IGNORECASE)
        if level_match:
            level = level_match.group('lvl').lower()
            message = level_match.group('msg').strip()
        else:
            message = rest

        return {
            'label': label,
            'unix_ts': unix_ts,
            'date': date,
            'host': host,
            'sys_ts': sys_ts,
            'source': source,
            'component': component,
            'pid': pid,
            'level': level,
            'message': message,
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
        
        # 3. AF_INET addresses
        message = self.patterns['af_inet'].sub('<INET>', message)
        
        # 4. IPv6 addresses (before IPv4)
        message = self.patterns['ipv6'].sub('<IP>', message)
        
        # 5. IPv4 addresses
        message = self.patterns['ipv4'].sub('<IP>', message)
        
        # 6. Port numbers -> normalize to <PORT>
        message = self.patterns['port'].sub('port <PORT>', message)
        
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
        
        # Parse line and extract metadata
        parsed = self.parse_thunderbird_line(line)
        if not parsed:
            self.stats['malformed_lines'] += 1
            return "", None

        # Build metadata: capture ips, ports, urls, emails, timestamps from raw message
        raw_msg = parsed.get('message', '')
        ips = self.patterns['ipv4'].findall(raw_msg) or []
        ips6 = self.patterns['ipv6'].findall(raw_msg) or []
        af_inet = self.patterns['af_inet'].findall(raw_msg) or []
        urls = self.patterns['url'].findall(raw_msg) or []
        emails = self.patterns['email'].findall(raw_msg) or []
        ports = re.findall(r'port\s+(\d+)', raw_msg, re.IGNORECASE) or []
        timestamps = self.patterns['timestamp'].findall(raw_msg) or []

        # Normalize message: replace PID occurrences with <PID> if pid present
        normalized_input = raw_msg
        if parsed.get('pid'):
            normalized_input = re.sub(r'\[' + re.escape(parsed['pid']) + r'\]', '<PID>', normalized_input)

        normalized_msg = self.normalize_message(normalized_input)

        # Skip if message becomes empty after normalization
        if not normalized_msg:
            self.stats['skipped_lines'] += 1
            return "", None

        # Only return the normalized message (no metadata) for dataset
        preprocessed = normalized_msg

        # Build metadata dictionary
        metadata = {
            'label': parsed.get('label', ''),
            'unix_ts': parsed.get('unix_ts', ''),
            'date': parsed.get('date', ''),
            'host': parsed.get('host', ''),
            'sys_ts': parsed.get('sys_ts', ''),
            'source': parsed.get('source', ''),
            'component': parsed.get('component', ''),
            'pid': parsed.get('pid', ''),
            'level': parsed.get('level', ''),
            'ips': ips + ips6 + af_inet,
            'ports': ports,
            'urls': urls,
            'emails': emails,
            'timestamps': timestamps,
            'raw_message': raw_msg,
            'raw_line': parsed.get('raw', '')
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
        metadata_list = []

        for line in lines:
            preprocessed, meta = self.preprocess_line(line)
            if preprocessed:
                preprocessed_lines.append(preprocessed)
            if meta:
                metadata_list.append(meta)

        return preprocessed_lines, metadata_list
    
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


def process_thunderbird_file(input_file: str, output_file: str, remove_duplicates: bool = True, sample_normal: int = None, sample_non: int = None):
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

    # Prepare output paths
    output_path = Path(output_file)
    meta_output_default = Path("/media/bioinfo04/Expansion/after_preprocessed_meta_data") / (output_path.name.replace('.txt', '') + '_meta.tsv')
    meta_output = str(meta_output_default)

    print(f"\n📖 Reading input file and streaming preprocessing...")
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    Path(meta_output_default.parent).mkdir(parents=True, exist_ok=True)

    total_lines = 0
    written_lines = 0
    # Sampling mode: collect up to sample_normal and sample_non lines only
    if sample_normal is not None or sample_non is not None:
        want_normal = sample_normal or 0
        want_non = sample_non or 0
        got_normal = 0
        got_non = 0

        with open(input_file, 'r', encoding='utf-8', errors='ignore') as inf, \
             open(output_file, 'w', encoding='utf-8') as outf, \
             open(meta_output, 'w', encoding='utf-8', newline='') as metaf:

            writer = csv.writer(metaf, delimiter='\t')
            header = ['label','unix_ts','date','host','sys_ts','source','component','pid','level',
                      'ips','ports','urls','emails','timestamps','raw_message','raw_line']
            writer.writerow(header)

            for line in inf:
                if got_normal >= want_normal and got_non >= want_non:
                    break

                parsed = preprocessor.parse_thunderbird_line(line)
                # Determine label: '-' => normal, else non-normal
                label = parsed.get('label') if parsed else ''
                is_normal = (label == '-')

                # Decide to include
                if is_normal and got_normal < want_normal:
                    preprocessed, meta = preprocessor.preprocess_line(line)
                    outf.write((preprocessed or '') + '\n')
                    if meta is None:
                        meta = {'label': label or '', 'unix_ts':'','date':'','host':'','sys_ts':'','source':'','component':'','pid':'','level':'','ips':[],'ports':[],'urls':[],'emails':[],'timestamps':[],'raw_message':'','raw_line':line.rstrip('\n')}
                    row = [meta.get('label',''), meta.get('unix_ts',''), meta.get('date',''), meta.get('host',''), meta.get('sys_ts',''), meta.get('source',''),
                           meta.get('component',''), meta.get('pid',''), meta.get('level',''), '|'.join(meta.get('ips',[])), '|'.join(meta.get('ports',[])), '|'.join(meta.get('urls',[])),
                           '|'.join(meta.get('emails',[])), '|'.join(meta.get('timestamps',[])), meta.get('raw_message',''), meta.get('raw_line','')]
                    writer.writerow(row)
                    got_normal += 1
                    total_lines += 1

                elif not is_normal and got_non < want_non:
                    preprocessed, meta = preprocessor.preprocess_line(line)
                    outf.write((preprocessed or '') + '\n')
                    if meta is None:
                        meta = {'label': label or '', 'unix_ts':'','date':'','host':'','sys_ts':'','source':'','component':'','pid':'','level':'','ips':[],'ports':[],'urls':[],'emails':[],'timestamps':[],'raw_message':'','raw_line':line.rstrip('\n')}
                    row = [meta.get('label',''), meta.get('unix_ts',''), meta.get('date',''), meta.get('host',''), meta.get('sys_ts',''), meta.get('source',''),
                           meta.get('component',''), meta.get('pid',''), meta.get('level',''), '|'.join(meta.get('ips',[])), '|'.join(meta.get('ports',[])), '|'.join(meta.get('urls',[])),
                           '|'.join(meta.get('emails',[])), '|'.join(meta.get('timestamps',[])), meta.get('raw_message',''), meta.get('raw_line','')]
                    writer.writerow(row)
                    got_non += 1
                    total_lines += 1

        print(f"✓ Sampled {got_normal} normal and {got_non} non-normal lines and wrote {total_lines} records")
    else:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as inf, \
             open(output_file, 'w', encoding='utf-8') as outf, \
             open(meta_output, 'w', encoding='utf-8', newline='') as metaf:

            writer = csv.writer(metaf, delimiter='\t')
            # Write header for metadata TSV
            header = ['label','unix_ts','date','host','sys_ts','source','component','pid','level',
                      'ips','ports','urls','emails','timestamps','raw_message','raw_line']
            writer.writerow(header)

            for line in inf:
                total_lines += 1
                preprocessed, meta = preprocessor.preprocess_line(line)

                # Only write records when preprocessing produced a valid entry
                if not preprocessed or meta is None:
                    continue

                # Write preprocessed message and metadata in lock-step
                outf.write(preprocessed + '\n')

                # Flatten lists into pipes to keep TSV structure
                row = [
                    meta.get('label',''), meta.get('unix_ts',''), meta.get('date',''), meta.get('host',''), meta.get('sys_ts',''), meta.get('source',''),
                    meta.get('component',''), meta.get('pid',''), meta.get('level',''),
                    '|'.join(meta.get('ips',[])), '|'.join(meta.get('ports',[])), '|'.join(meta.get('urls',[])),
                    '|'.join(meta.get('emails',[])), '|'.join(meta.get('timestamps',[])), meta.get('raw_message',''), meta.get('raw_line','')
                ]
                writer.writerow(row)
                written_lines += 1

        print(f"✓ Processed and wrote {written_lines:,} lines to dataset and metadata files")

    # If sampling branch used, written_lines will have been updated there.
    # For the non-sampling branch we printed written_lines above; print final tally here too.
    try:
        final_written = written_lines
    except NameError:
        final_written = total_lines
    print(f"✓ Processed and wrote {final_written:,} lines to dataset and metadata files")
    
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
    import argparse
    # Default paths
    default_input = "/home/bioinfo04/Desktop/Pak Akmal/2427051003/dataset/thunderbird_non_normal_testing.txt"
    default_output = "/media/bioinfo04/Expansion/after_preprocessed_dataset_testing/after_preprocessed_thunderbird_non_normal.txt"

    parser = argparse.ArgumentParser(description='Thunderbird log preprocessing')
    parser.add_argument('input_file', nargs='?', default=default_input, help='Path to Thunderbird.log')
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

    # If user provided meta-output, adjust internal meta path (not currently used)
    meta_override = args.meta_output if args.meta_output else None

    process_thunderbird_file(input_file, output_file, remove_duplicates=remove_dups, sample_normal=sample_normal, sample_non=sample_non)
