"""Recreate summary artifacts from a saved testing pipeline TXT log.

This script is intentionally limited to what can be reconstructed from the
text log alone:
- overall metrics summary
- confusion matrix visualization
- prediction distribution visualization
- compact JSON / text report

It cannot recreate per-sample artifacts such as detailed_results.csv,
cluster_analysis.csv, or the exact original plots that depend on raw arrays
unless those source files are also available.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None


CLASS_NAMES = ["NORMAL", "NON-NORMAL", "ANOMALY"]


@dataclass
class ParsedLog:
    dataset: str = "unknown"
    algorithm: str = "unknown"
    embedding: str = "unknown"
    overall_accuracy: Optional[float] = None
    ground_truth_classes: List[int] = None
    prediction_classes: List[int] = None
    confusion_matrix: List[List[int]] = None
    per_class_metrics: Dict[str, Dict[str, float]] = None
    prediction_distribution: Dict[str, Dict[str, int]] = None
    cluster_label_summary: Dict[str, int] = None
    cluster_size_stats: Dict[str, float] = None
    per_method_metrics: Dict[str, Dict[str, object]] = None


def _to_int(text: str) -> int:
    return int(text.replace(",", "").strip())


def _parse_int_list(text: str) -> List[int]:
    return [int(item) for item in re.findall(r"-?\d+", text)]


def parse_log_text(text: str) -> ParsedLog:
    parsed = ParsedLog(
        ground_truth_classes=[],
        prediction_classes=[],
        confusion_matrix=[],
        per_class_metrics={},
        prediction_distribution={},
        cluster_label_summary={},
        cluster_size_stats={},
        per_method_metrics={},
    )

    dataset_match = re.search(r"^Dataset:\s*(.+)$", text, re.MULTILINE)
    algorithm_match = re.search(r"^Algorithm:\s*(.+)$", text, re.MULTILINE)
    embedding_match = re.search(r"^Embedding:\s*(.+)$", text, re.MULTILINE)
    accuracy_match = re.search(r"^Overall Accuracy:\s*([0-9.]+)$", text, re.MULTILINE)

    if dataset_match:
        parsed.dataset = dataset_match.group(1).strip()
    if algorithm_match:
        parsed.algorithm = algorithm_match.group(1).strip()
    if embedding_match:
        parsed.embedding = embedding_match.group(1).strip()
    if accuracy_match:
        parsed.overall_accuracy = float(accuracy_match.group(1))

    gt_match = re.search(
        r"Ground truth classes:\s*\[(.*?)\]\s*→\s*\[(.*?)\]",
        text,
    )
    if gt_match:
        parsed.ground_truth_classes = _parse_int_list(gt_match.group(1))

    pred_match = re.search(
        r"Prediction classes:\s*\[(.*?)\]\s*→\s*\[(.*?)\]",
        text,
    )
    if pred_match:
        parsed.prediction_classes = _parse_int_list(pred_match.group(1))

    cluster_summary_block = re.search(
        r"Cluster Labels \(Metadata-based\):\s*(.*?)\n\nLabeling Reasons:",
        text,
        re.DOTALL,
    )
    if cluster_summary_block:
        for line in cluster_summary_block.group(1).splitlines():
            m = re.search(r"^\s*([A-Z\-]+)\s*:\s*([0-9,]+) clusters,\s*([0-9,]+) samples", line)
            if m:
                parsed.cluster_label_summary[m.group(1)] = _to_int(m.group(2))

    size_stats = {
        "mean": r"^\s*Mean:\s*([0-9.]+)$",
        "median": r"^\s*Median:\s*([0-9.]+)$",
        "min": r"^\s*Min:\s*([0-9,]+)$",
        "max": r"^\s*Max:\s*([0-9,]+)$",
    }
    for key, pattern in size_stats.items():
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            parsed.cluster_size_stats[key] = float(m.group(1).replace(",", ""))

    metrics_block = re.search(
        r"Per-Class Metrics \(Ground Truth Classes Only\):\s*\n\s*Precision\s+Recall\s+F1-Score\s+Support\s*\n(.*?)\n\s*Macro Avg:",
        text,
        re.DOTALL,
    )
    if metrics_block:
        for line in metrics_block.group(1).splitlines():
            m = re.search(
                r"^\s*(NORMAL|NON-NORMAL|ANOMALY)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9,]+)",
                line,
            )
            if m:
                parsed.per_class_metrics[m.group(1)] = {
                    "precision": float(m.group(2)),
                    "recall": float(m.group(3)),
                    "f1": float(m.group(4)),
                    "support": _to_int(m.group(5)),
                }

    cm_match = re.search(
        r"Confusion Matrix \(\d+x\d+\):.*?\n(.*?)(?:\n\n📊 Prediction Distribution:|\n\nPER-METHOD METRICS|\Z)",
        text,
        re.DOTALL,
    )
    if cm_match:
        rows = []
        for line in cm_match.group(1).splitlines():
            nums = re.findall(r"\[(.*?)\]", line)
            if nums:
                values = [_to_int(v) for v in re.split(r"\s+", nums[0].strip()) if v.strip()]
                if values:
                    rows.append(values)
        parsed.confusion_matrix = rows

    dist: Dict[str, Dict[str, int]] = {}
    class_section = re.split(r"^\s{0,3}([A-Z\-]+) Ground Truth \(([^)]+)\):\s*$", text, flags=re.MULTILINE)
    if len(class_section) > 1:
        for idx in range(1, len(class_section), 3):
            true_name = class_section[idx]
            block = class_section[idx + 2]
            preds: Dict[str, int] = {}
            for pred_name, count_text in re.findall(r"→ Predicted as\s+([A-Z\-]+)\s*:\s*([0-9,]+)", block):
                preds[pred_name] = _to_int(count_text)
            if preds:
                dist[true_name] = preds
    parsed.prediction_distribution = dist

    method_block = re.search(
        r"PER-METHOD METRICS\s*\n(.*?)\n\n=+\nSTEP 9: SAVE DETAILED RESULTS",
        text,
        re.DOTALL,
    )
    if method_block:
        current_key = None
        for line in method_block.group(1).splitlines():
            m = re.search(
                r"^\s*([A-Z_]+):\s*([0-9,]+) samples \(([0-9.]+)%\)",
                line,
            )
            if m:
                current_key = m.group(1)
                parsed.per_method_metrics[current_key] = {
                    "samples": _to_int(m.group(2)),
                    "percentage": float(m.group(3)),
                }
                continue
            m = re.search(r"^\s*Distribution:\s*NORMAL=([0-9,]+)\s+NON-NORMAL=([0-9,]+)", line)
            if m and current_key:
                parsed.per_method_metrics[current_key]["distribution"] = {
                    "NORMAL": _to_int(m.group(1)),
                    "NON-NORMAL": _to_int(m.group(2)),
                }
                continue
            m = re.search(r"^\s*Accuracy:\s*([0-9.]+)", line)
            if m and current_key:
                parsed.per_method_metrics[current_key]["accuracy"] = float(m.group(1))

    return parsed


def render_confusion_matrix(confusion_matrix: List[List[int]], true_labels: List[int], pred_labels: List[int], out_path: Path, title: str) -> None:
    if not confusion_matrix:
        return

    cm = np.array(confusion_matrix, dtype=int)
    true_names = [CLASS_NAMES[i] for i in true_labels] if true_labels else [CLASS_NAMES[0], CLASS_NAMES[1]]
    pred_names = [CLASS_NAMES[i] for i in pred_labels] if pred_labels else [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', xticklabels=pred_names, yticklabels=true_names, cbar_kws={'label': 'Count'}, ax=ax)
    else:
        image = ax.imshow(cm, cmap='RdYlGn_r')
        plt.colorbar(image, ax=ax, label='Count')
        ax.set_xticks(np.arange(len(pred_names)))
        ax.set_yticks(np.arange(len(true_names)))
        ax.set_xticklabels(pred_names)
        ax.set_yticklabels(true_names)
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                ax.text(col_idx, row_idx, f'{cm[row_idx, col_idx]:d}', ha='center', va='center', color='black')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def render_prediction_distribution(prediction_distribution: Dict[str, Dict[str, int]], out_path: Path, title: str) -> None:
    if not prediction_distribution:
        return

    true_names = list(prediction_distribution.keys())
    pred_names: List[str] = []
    for block in prediction_distribution.values():
        for key in block:
            if key not in pred_names:
                pred_names.append(key)

    colors = {
        'NORMAL': '#2ecc71',
        'NON-NORMAL': '#e74c3c',
        'ANOMALY': '#f39c12',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    x_pos = np.arange(len(true_names))
    bottom = np.zeros(len(true_names))

    for pred_name in pred_names:
        pct_values = []
        for true_name in true_names:
            total = max(sum(prediction_distribution[true_name].values()), 1)
            pct_values.append(100.0 * prediction_distribution[true_name].get(pred_name, 0) / total)
        ax1.bar(x_pos, pct_values, bottom=bottom, label=pred_name, color=colors.get(pred_name, None), edgecolor='black', linewidth=0.5)
        bottom += np.array(pct_values)

    ax1.set_ylabel('Percentage (%)')
    ax1.set_xlabel('Ground Truth Class')
    ax1.set_title('Prediction Distribution by Ground Truth Class')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(true_names)
    ax1.set_ylim(0, 100)
    ax1.legend(title='Predictions')
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    error_labels = []
    error_counts = []
    error_colors = []
    for true_name in true_names:
        correct = prediction_distribution[true_name].get(true_name, 0)
        error_labels.append(f'{true_name} Correct')
        error_counts.append(correct)
        error_colors.append('#27ae60')
        for pred_name, count in prediction_distribution[true_name].items():
            if pred_name != true_name:
                error_labels.append(f'{true_name} → {pred_name}')
                error_counts.append(count)
                error_colors.append('#c0392b' if pred_name == 'ANOMALY' else '#d35400')

    bars = ax2.bar(np.arange(len(error_counts)), error_counts, color=error_colors, edgecolor='black', linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height):,}', ha='center', va='bottom', fontsize=8, rotation=90)
    ax2.set_ylabel('Count (samples)')
    ax2.set_xlabel('Prediction Type')
    ax2.set_title('Correct vs Mispredictions (Absolute Counts)')
    ax2.set_xticks(np.arange(len(error_counts)))
    ax2.set_xticklabels(error_labels, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def render_overview(parsed: ParsedLog, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax = axes[0, 0]
    if parsed.cluster_label_summary:
        names = list(parsed.cluster_label_summary.keys())
        values = [parsed.cluster_label_summary[name] for name in names]
        ax.bar(names, values, color=['#2ecc71', '#e74c3c', '#f39c12'][: len(names)], edgecolor='black')
        ax.set_title('Cluster Label Summary')
        ax.set_ylabel('Clusters')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.axis('off')

    ax = axes[0, 1]
    ax.axis('off')
    summary_lines = [
        f'Dataset: {parsed.dataset}',
        f'Algorithm: {parsed.algorithm}',
        f'Embedding: {parsed.embedding}',
    ]
    if parsed.overall_accuracy is not None:
        summary_lines.append(f'Accuracy: {parsed.overall_accuracy:.4f}')
    if parsed.cluster_size_stats:
        summary_lines.append(f"Cluster mean size: {parsed.cluster_size_stats.get('mean', 0):.0f}")
        summary_lines.append(f"Cluster median size: {parsed.cluster_size_stats.get('median', 0):.0f}")
    ax.text(0.02, 0.98, '\n'.join(summary_lines), va='top', ha='left', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#d0d7de'))

    ax = axes[1, 0]
    if parsed.per_class_metrics:
        labels = list(parsed.per_class_metrics.keys())
        f1s = [parsed.per_class_metrics[label]['f1'] for label in labels]
        ax.bar(labels, f1s, color='#4c78a8', edgecolor='black')
        ax.set_ylim(0, 1)
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.axis('off')

    ax = axes[1, 1]
    if parsed.per_method_metrics:
        labels = list(parsed.per_method_metrics.keys())
        values = [parsed.per_method_metrics[label]['samples'] for label in labels]
        ax.bar(labels, values, color='#f58518', edgecolor='black')
        ax.set_yscale('log')
        ax.set_ylabel('Samples')
        ax.set_title('Prediction Methods')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def write_summary_files(parsed: ParsedLog, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / 'reconstructed_summary.json').open('w', encoding='utf-8') as f:
        json.dump(asdict(parsed), f, indent=2)

    lines = [
        'RECONSTRUCTED TESTING SUMMARY',
        '=' * 72,
        f'Dataset: {parsed.dataset}',
        f'Algorithm: {parsed.algorithm}',
        f'Embedding: {parsed.embedding}',
    ]
    if parsed.overall_accuracy is not None:
        lines.append(f'Overall Accuracy: {parsed.overall_accuracy:.4f}')
    if parsed.per_class_metrics:
        lines.append('')
        lines.append('Per-Class Metrics:')
        for label, metrics in parsed.per_class_metrics.items():
            lines.append(
                f"  {label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}, support={metrics['support']:,}"
            )
    if parsed.prediction_distribution:
        lines.append('')
        lines.append('Prediction Distribution:')
        for true_name, preds in parsed.prediction_distribution.items():
            total = sum(preds.values())
            lines.append(f'  {true_name}: {total:,} samples')
            for pred_name, count in preds.items():
                pct = (100.0 * count / total) if total else 0.0
                lines.append(f'    {pred_name}: {count:,} ({pct:.2f}%)')

    with (out_dir / 'reconstructed_summary.txt').open('w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='Recreate summary artifacts from a testing pipeline TXT log.')
    parser.add_argument('log_file', type=Path, help='Path to the saved TXT log file')
    parser.add_argument('--out-dir', type=Path, default=None, help='Output directory for recreated files')
    args = parser.parse_args()

    if not args.log_file.exists():
        raise FileNotFoundError(f'Log file not found: {args.log_file}')

    text = args.log_file.read_text(encoding='utf-8', errors='replace')
    parsed = parse_log_text(text)

    out_dir = args.out_dir or args.log_file.with_suffix('')
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary_files(parsed, out_dir)

    if parsed.confusion_matrix:
        render_confusion_matrix(
            parsed.confusion_matrix,
            parsed.ground_truth_classes or [0, 1],
            parsed.prediction_classes or [0, 1, 2],
            out_dir / 'confusion_matrix.png',
            f"{parsed.dataset} {parsed.algorithm.upper()} Confusion Matrix",
        )

    if parsed.prediction_distribution:
        render_prediction_distribution(
            parsed.prediction_distribution,
            out_dir / 'prediction_distribution.png',
            f'{parsed.dataset} {parsed.algorithm.upper()} Prediction Distribution',
        )

    render_overview(parsed, out_dir / 'analysis_overview.png')

    print(f'Reconstructed artifacts saved to: {out_dir}')
    print('Generated: reconstructed_summary.json, reconstructed_summary.txt, analysis_overview.png, confusion_matrix.png, prediction_distribution.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())