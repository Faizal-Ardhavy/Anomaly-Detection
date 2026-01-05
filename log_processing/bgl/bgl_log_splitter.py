"""
BGL log splitter

- Sample N lines total with specified normal/non-normal ratio using reservoir sampling per class
- Use streaming reservoir sampling to avoid loading entire file
- Write two output files (no preprocessing):
  - <input_dir>/bgl_normal_testing.txt
  - <input_dir>/bgl_non_normal_testing.txt

Usage:
    python bgl_log_splitter.py [--input INPUT_FILE] [--total 2000000] [--ratio 0.7] [--seed 42]
"""

from pathlib import Path
import re
import random
import argparse
import sys
from collections import Counter

# Helper: find default_input path from preprocessing file
def find_default_input(preproc_path: Path) -> Path:
    if not preproc_path.exists():
        return None
    text = preproc_path.read_text(encoding='utf-8', errors='ignore')
    m = re.search(r"default_input\s*=\s*([\'\"])(.+?)\1", text)
    if not m:
        return None
    rel = m.group(2)
    resolved = (preproc_path.parent / rel).resolve()
    return resolved


def is_normal_line(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return False
    first = s.split(maxsplit=1)[0]
    return first == '-'


def reservoir_sample_two_classes(input_file: Path, normal_k: int, non_k: int, seed: int = 42, show_progress: bool = False):
    random.seed(seed)
    normal_res = []
    non_res = []
    normal_seen = 0
    non_seen = 0

    total_size = 0
    try:
        total_size = input_file.stat().st_size
    except Exception:
        total_size = 0

    last_percent = -1
    with input_file.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if is_normal_line(line):
                normal_seen += 1
                if len(normal_res) < normal_k:
                    normal_res.append(line.rstrip('\n'))
                else:
                    r = random.randint(0, normal_seen - 1)
                    if r < normal_k:
                        normal_res[r] = line.rstrip('\n')
            else:
                non_seen += 1
                if len(non_res) < non_k:
                    non_res.append(line.rstrip('\n'))
                else:
                    r = random.randint(0, non_seen - 1)
                    if r < non_k:
                        non_res[r] = line.rstrip('\n')

            if show_progress and total_size > 0:
                try:
                    pos = f.tell()
                    percent = int(pos * 100 / total_size)
                except Exception:
                    percent = -1
                if percent != last_percent and percent >= 0:
                    print(f"Progress: {percent}%\r", end='', flush=True)
                    last_percent = percent

    if show_progress and total_size > 0:
        print('\n', end='')

    return normal_res, non_res, normal_seen, non_seen


def write_lines(lines, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as out:
        for ln in lines:
            out.write(ln + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Path to BGL log file (optional)')
    parser.add_argument('--total', type=int, default=800000, help='Total number of samples to extract')
    parser.add_argument('--ratio', type=float, default=0.7, help='Fraction of normal samples (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress', action='store_true', help='Show progress')
    args = parser.parse_args()

    preproc_file = Path(__file__).parent / 'bgl_log_preprocessing.py'
    default_input = find_default_input(preproc_file)

    if args.input:
        input_path = Path(args.input)
    else:
        if default_input and default_input.exists():
            input_path = default_input
        else:
            candidate = Path(__file__).resolve().parents[3] / 'dataset' / 'BGL' / 'BGL.log'
            input_path = candidate

    input_path = input_path.resolve()

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(2)

    total = args.total
    normal_k = int(total * args.ratio)
    non_k = total - normal_k

    print(f"Input file: {input_path}")
    print(f"Total samples target: {total}  (normal: {normal_k}, non-normal: {non_k})")

    normal_res, non_res, normal_seen, non_seen = reservoir_sample_two_classes(input_path, normal_k, non_k, seed=args.seed, show_progress=args.progress)

    print(f"Seen normal lines: {normal_seen:,}, collected: {len(normal_res):,}")
    print(f"Seen non-normal lines: {non_seen:,}, collected: {len(non_res):,}")

    outdir = input_path.parent.resolve()
    normal_out = outdir / 'bgl_normal_testing.txt'
    non_out = outdir / 'bgl_non_normal_testing.txt'

    write_lines(normal_res, normal_out)
    write_lines(non_res, non_out)

    print(f"Wrote normal file: {normal_out} ({len(normal_res):,} lines)")
    print(f"Wrote non-normal file: {non_out} ({len(non_res):,} lines)")

    if len(normal_res) < normal_k or len(non_res) < non_k:
        print("Warning: requested sample size not met for one or both classes. See counts above.")

    # --- Remove sampled lines from original file ---
    print("Removing sampled lines from original file (creating temp then replacing)...")
    to_delete = Counter()
    for ln in normal_res:
        to_delete[ln] += 1
    for ln in non_res:
        to_delete[ln] += 1

    temp_path = input_path.parent / (input_path.name + '.tmp')
    removed = 0
    kept = 0
    try:
        with input_path.open('r', encoding='utf-8', errors='ignore') as infile, temp_path.open('w', encoding='utf-8') as outfile:
            for line in infile:
                line_stripped = line.rstrip('\n')
                if to_delete.get(line_stripped, 0) > 0:
                    to_delete[line_stripped] -= 1
                    removed += 1
                    continue
                outfile.write(line)
                kept += 1

        backup_path = input_path.parent / (input_path.name + '.bak')
        input_path.replace(backup_path)
        temp_path.replace(input_path)

        print(f"Removed {removed:,} lines from original file; backup saved at {backup_path}")
    except Exception as e:
        print(f"Error while removing lines: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


if __name__ == '__main__':
    main()
