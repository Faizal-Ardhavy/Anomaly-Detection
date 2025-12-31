"""
Thunderbird log splitter

- Extract default input path from `thunderbird_log_preprocessing.py` if present.
- Sample 2,000,000 lines total with 70% normal (`-`) and 30% non-normal (labels != `-`).
- Use reservoir sampling per class in a single streaming pass to avoid loading entire file.
- Write two output files (no preprocessing):
  - after_preprocessed_dataset/thunderbird_normal_testing.txt
  - after_preprocessed_dataset/thunderbird_non_normal_testing.txt

Usage:
    python thunderbird_log_spliter.py [--input INPUT_FILE] [--output-dir OUT_DIR] [--total 2000000] [--ratio 0.7] [--seed 42]

"""

from pathlib import Path
import re
import random
import argparse
import sys

# Helper: find default_input path from preprocessing file
def find_default_input(preproc_path: Path) -> Path:
    if not preproc_path.exists():
        return None
    text = preproc_path.read_text(encoding='utf-8', errors='ignore')
    m = re.search(r"default_input\s*=\s*([\'\"])(.+?)\1", text)
    if not m:
        return None
    rel = m.group(2)
    # Resolve relative to preproc_path parent
    resolved = (preproc_path.parent / rel).resolve()
    return resolved


def is_normal_line(line: str) -> bool:
    # Thunderbird dataset: first token '-' indicates non-alert (normal)
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
            # keep original line as-is (do not preprocess)
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
    parser.add_argument('--input', '-i', help='Path to Thunderbird log file (optional)')
    parser.add_argument('--outdir', '-o', help='Output directory', default='../../../after_preprocessed_dataset')
    parser.add_argument('--total', type=int, default=2000000, help='Total number of samples to extract')
    parser.add_argument('--ratio', type=float, default=0.7, help='Fraction of normal samples (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    preproc_file = Path(__file__).parent / 'thunderbird_log_preprocessing.py'
    default_input = find_default_input(preproc_file)

    if args.input:
        input_path = Path(args.input)
    else:
        if default_input and default_input.exists():
            input_path = default_input
        else:
            # try workspace relative path
            candidate = Path(__file__).resolve().parents[3] / 'dataset' / 'Thunderbird.log'
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
    print(f"Output dir: {args.outdir}")

    normal_res, non_res, normal_seen, non_seen = reservoir_sample_two_classes(input_path, normal_k, non_k, seed=args.seed)

    print(f"Seen normal lines: {normal_seen:,}, collected: {len(normal_res):,}")
    print(f"Seen non-normal lines: {non_seen:,}, collected: {len(non_res):,}")

    outdir = (Path(__file__).parent / args.outdir).resolve()
    normal_out = outdir / 'thunderbird_normal_testing.txt'
    non_out = outdir / 'thunderbird_non_normal_testing.txt'

    write_lines(normal_res, normal_out)
    write_lines(non_res, non_out)

    print(f"Wrote normal file: {normal_out} ({len(normal_res):,} lines)")
    print(f"Wrote non-normal file: {non_out} ({len(non_res):,} lines)")

    if len(normal_res) < normal_k or len(non_res) < non_k:
        print("Warning: requested sample size not met for one or both classes. See counts above.")

if __name__ == '__main__':
    main()
