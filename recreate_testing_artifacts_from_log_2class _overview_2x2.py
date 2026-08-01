"""Recreate summary artifacts from a testing pipeline TXT log, but
merge `NON-NORMAL` and `ANOMALY` into a single `ANOMALY` class (2-class view).

This is a standalone script (doesn't modify the original). Use it like:

    python recreate_testing_artifacts_from_log_2class.py testing_result_log --out-dir results/2class

Outputs: 
  - results/2class/[log_name]/en/...
  - results/2class/[log_name]/id/...
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


CLASS_NAMES_2 = ["NORMAL", "ANOMALY"]


@dataclass
class ParsedLog2:
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


def parse_log_text_to_parsedlog(text: str) -> ParsedLog2:
    """Parse the same log format used by the pipeline into a ParsedLog2 object."""
    parsed = ParsedLog2(
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

    gt_match = re.search(r"Ground truth classes:\s*\[(.*?)\]\s*→\s*\[(.*?)\]", text)
    if gt_match:
        parsed.ground_truth_classes = _parse_int_list(gt_match.group(1))

    pred_match = re.search(r"Prediction classes:\s*\[(.*?)\]\s*→\s*\[(.*?)\]", text)
    if pred_match:
        parsed.prediction_classes = _parse_int_list(pred_match.group(1))

    cluster_summary_block = re.search(
        r"Cluster Labels \(Metadata-based\):\s*(.*?)\n\nLabeling Reasons:", text, re.DOTALL
    )
    if cluster_summary_block:
        for line in cluster_summary_block.group(1).splitlines():
            # Ditambahkan _ untuk menangkap NON_NORMAL
            m = re.search(r"^\s*([A-Z\-_]+)\s*:\s*([0-9,]+) clusters,\s*([0-9,]+) samples", line)
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
            # Diperluas agar menangkap variasi class (NORMAL|NON_NORMAL|NON-NORMAL|ANOMALY)
            m = re.search(r"^\s*([A-Z\-_]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9,]+)", line)
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
    class_section = re.split(r"^\s{0,3}([A-Z\-_]+) Ground Truth \(([^)]+)\):\s*$", text, flags=re.MULTILINE)
    if len(class_section) > 1:
        for idx in range(1, len(class_section), 3):
            true_name = class_section[idx]
            block = class_section[idx + 2]
            preds: Dict[str, int] = {}
            for pred_name, count_text in re.findall(r"→ Predicted as\s+([A-Z\-_]+)\s*:\s*([0-9,]+)", block):
                preds[pred_name] = _to_int(count_text)
            if preds:
                dist[true_name] = preds
    parsed.prediction_distribution = dist

    method_block = re.search(r"PER-METHOD METRICS\s*\n(.*?)\n\n=+\nSTEP 9: SAVE DETAILED RESULTS", text, re.DOTALL)
    if method_block:
        current_key = None
        for line in method_block.group(1).splitlines():
            m = re.search(r"^\s*([A-Z_]+):\s*([0-9,]+) samples \(([0-9.]+)%\)", line)
            if m:
                current_key = m.group(1)
                parsed.per_method_metrics[current_key] = {
                    "samples": _to_int(m.group(2)),
                    "percentage": float(m.group(3)),
                }
                continue
            m = re.search(r"^\s*Distribution:\s*NORMAL=([0-9,]+)\s+([A-Z\-_]+)=([0-9,]+)", line)
            if m and current_key:
                parsed.per_method_metrics[current_key]["distribution"] = {
                    "NORMAL": _to_int(m.group(1)),
                    "ANOMALY": _to_int(m.group(3)), # Menyederhanakan non-normal jadi Anomaly
                }
                continue
            m = re.search(r"^\s*Accuracy:\s*([0-9.]+)", line)
            if m and current_key:
                parsed.per_method_metrics[current_key]["accuracy"] = float(m.group(1))

    return parsed


def merge_to_2class(parsed: ParsedLog2) -> ParsedLog2:
    """Convert parsed structure so that ANY variation of non-normal becomes ANOMALY."""
    pred_order: List[str] = []
    if parsed.prediction_distribution:
        for _, preds in parsed.prediction_distribution.items():
            if preds:
                pred_order = list(preds.keys())
                break

    if not pred_order:
        pred_order = ["NORMAL", "NON-NORMAL", "ANOMALY"]

    cm = parsed.confusion_matrix or []
    if cm:
        max_cols = max(len(r) for r in cm)
        if len(pred_order) < max_cols:
            pred_order = pred_order + [f"COL_{i}" for i in range(len(pred_order), max_cols)]

        merged_cm: List[List[int]] = []
        for row in cm:
            row_vals = list(row) + [0] * (max_cols - len(row))
            normal = 0
            anomaly = 0
            for val, name in zip(row_vals, pred_order):
                if name == "NORMAL":
                    normal += val
                else:
                    anomaly += val
            merged_cm.append([normal, anomaly])
        parsed.confusion_matrix = merged_cm

    new_dist: Dict[str, Dict[str, int]] = {}
    for true_name, preds in (parsed.prediction_distribution or {}).items():
        normal = preds.get("NORMAL", 0)
        anomaly = sum(v for k, v in preds.items() if k != "NORMAL")
        
        # Penangkapan super agresif untuk variasi NON-NORMAL
        gt_key = "ANOMALY" if "NON" in true_name.upper() or "ANOMALY" in true_name.upper() else true_name
        
        if gt_key in new_dist:
            new_dist[gt_key]["NORMAL"] += normal
            new_dist[gt_key]["ANOMALY"] += anomaly
        else:
            new_dist[gt_key] = {"NORMAL": normal, "ANOMALY": anomaly}
    parsed.prediction_distribution = new_dist

    new_pcm: Dict[str, Dict[str, float]] = {}
    for k, v in (parsed.per_class_metrics or {}).items():
        key = "ANOMALY" if "NON" in k.upper() or "ANOMALY" in k.upper() else k
        if key in new_pcm:
            new_pcm[key] = v
        else:
            new_pcm[key] = v
    parsed.per_class_metrics = new_pcm

    cls = parsed.cluster_label_summary or {}
    normal_clusters = cls.get("NORMAL", 0)
    anomaly_clusters = sum(v for k, v in cls.items() if k != "NORMAL")
    
    new_cls: Dict[str, int] = {}
    if normal_clusters:
        new_cls["NORMAL"] = normal_clusters
    if anomaly_clusters:
        new_cls["ANOMALY"] = anomaly_clusters
    parsed.cluster_label_summary = new_cls

    new_pmm = {}
    for method, data in (parsed.per_method_metrics or {}).items():
        # Memaksa ubah nama method jika mengandung NON_NORMAL, NON-NORMAL, dsb.
        new_method = method.upper()
        if "NON" in new_method:
            new_method = new_method.replace("NON_NORMAL", "ANOMALY").replace("NON-NORMAL", "ANOMALY").replace("NONNORMAL", "ANOMALY")
        
        if isinstance(data, dict) and "distribution" in data:
            dist = data["distribution"]
            normal = dist.get("NORMAL", 0)
            anomaly = sum(v for k, v in dist.items() if k != "NORMAL")
            data["distribution"] = {"NORMAL": normal, "ANOMALY": anomaly}
            
        new_pmm[new_method] = data
    parsed.per_method_metrics = new_pmm

    parsed.prediction_classes = [0, 1]
    parsed.ground_truth_classes = parsed.ground_truth_classes or [0, 1]

    return parsed


# --- TRANSLATION HELPER ---
def get_lang_config(lang: str) -> Dict[str, str]:
    if lang == "id":
        return {
            "ANOMALY": "ANOMALI",
            "cm_title": "Matriks Kebingungan",
            "pred_label": "Label Prediksi",
            "gt_label": "Label Sebenarnya",
            "dist_title_1": "Distribusi Prediksi Berdasarkan Kelas Sebenarnya",
            "dist_title_2": "Prediksi Benar vs Salah (Jumlah Absolut)",
            "dist_ylabel_1": "Persentase (%)",
            "dist_xlabel_1": "Kelas Sebenarnya",
            "dist_ylabel_2": "Jumlah (Sampel)",
            "dist_xlabel_2": "Tipe Prediksi",
            "overview_cluster": "Ringkasan Label Klaster",
            "overview_cluster_ylabel": "Jumlah Klaster",
            "overview_f1": "Skor F1 Per Kelas",
            "overview_method": "Metode Prediksi",
            "correct": "Benar"
        }
    else:
        return {
            "ANOMALY": "ANOMALY",
            "cm_title": "Confusion Matrix",
            "pred_label": "Predicted Label",
            "gt_label": "Ground Truth Label",
            "dist_title_1": "Prediction Distribution by Ground Truth Class",
            "dist_title_2": "Correct vs Mispredictions (Absolute Counts)",
            "dist_ylabel_1": "Percentage (%)",
            "dist_xlabel_1": "Ground Truth Class",
            "dist_ylabel_2": "Count (samples)",
            "dist_xlabel_2": "Prediction Type",
            "overview_cluster": "Cluster Label Summary",
            "overview_cluster_ylabel": "Clusters",
            "overview_f1": "Per-Class F1 Score",
            "overview_method": "Prediction Methods",
            "correct": "Correct"
        }

def localize_keys(keys: List[str], lang_cfg: Dict[str, str]) -> List[str]:
    return [lang_cfg.get(k, k) for k in keys]


def render_confusion_matrix_2class(confusion_matrix: List[List[int]], true_names: List[str], pred_names: List[str], out_path: Path, title: str, lang_cfg: Dict[str, str]) -> None:
    if not confusion_matrix:
        return

    cm = np.array(confusion_matrix, dtype=int)
    
    true_names_loc = localize_keys(true_names, lang_cfg)
    pred_names_loc = localize_keys(pred_names, lang_cfg)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r", 
                    xticklabels=pred_names_loc, yticklabels=true_names_loc, 
                    cbar_kws={"label": "Count"}, ax=ax,
                    annot_kws={"size": 16, "color": "black", "weight": "bold"})
    else:
        image = ax.imshow(cm, cmap="RdYlGn_r")
        plt.colorbar(image, ax=ax, label="Count")
        ax.set_xticks(np.arange(len(pred_names_loc)))
        ax.set_yticks(np.arange(len(true_names_loc)))
        
        ax.set_xticklabels(pred_names_loc, fontsize=14, color="black")
        ax.set_yticklabels(true_names_loc, fontsize=14, color="black")
        
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                ax.text(col_idx, row_idx, f"{cm[row_idx, col_idx]:d}", ha="center", va="center", color="black", fontsize=16, fontweight="bold")
    
    plt.xlabel(lang_cfg["pred_label"], fontsize=16, color="black", fontweight="bold")
    plt.ylabel(lang_cfg["gt_label"], fontsize=16, color="black", fontweight="bold")
    plt.title(f"{title} - {lang_cfg['cm_title']}", fontsize=18, color="black", fontweight="bold", pad=15)
    
    plt.xticks(fontsize=14, color="black")
    plt.yticks(fontsize=14, color="black")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_prediction_distribution_2class(prediction_distribution: Dict[str, Dict[str, int]], out_path: Path, title: str, lang_cfg: Dict[str, str]) -> None:
    if not prediction_distribution:
        return

    true_names = list(prediction_distribution.keys())
    true_names_loc = localize_keys(true_names, lang_cfg)

    colors = {"NORMAL": "#2ecc71", "ANOMALY": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    x_pos = np.arange(len(true_names))
    bottom = np.zeros(len(true_names))

    for pred_name in ["NORMAL", "ANOMALY"]:
        pct_values = []
        for true_name in true_names:
            total = max(sum(prediction_distribution[true_name].values()), 1)
            pct_values.append(100.0 * prediction_distribution[true_name].get(pred_name, 0) / total)
        
        pred_loc = lang_cfg.get(pred_name, pred_name)
        ax1.bar(x_pos, pct_values, bottom=bottom, label=pred_loc, color=colors.get(pred_name, None), edgecolor="black", linewidth=0.5)
        bottom += np.array(pct_values)

    ax1.set_ylabel(lang_cfg["dist_ylabel_1"], fontsize=14, color="black", fontweight="bold")
    ax1.set_xlabel(lang_cfg["dist_xlabel_1"], fontsize=14, color="black", fontweight="bold")
    ax1.set_title(lang_cfg["dist_title_1"], fontsize=16, color="black", fontweight="bold", pad=10)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(true_names_loc, fontsize=14, color="black") 
    ax1.tick_params(axis='y', labelsize=12, colors='black') 
    ax1.set_ylim(0, 100)
    
    legend = ax1.legend(title="Predictions", fontsize=12)
    plt.setp(legend.get_title(), fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    error_labels = []
    error_counts = []
    error_colors = []
    for true_name in true_names:
        correct = prediction_distribution[true_name].get("NORMAL" if true_name == "NORMAL" else "ANOMALY", 0)
        t_loc = lang_cfg.get(true_name, true_name)
        
        error_labels.append(f"{t_loc} {lang_cfg['correct']}")
        error_counts.append(correct)
        error_colors.append("#27ae60")
        for pred_name, count in prediction_distribution[true_name].items():
            if (true_name == "NORMAL" and pred_name != "NORMAL") or (true_name != "NORMAL" and pred_name != "ANOMALY"):
                p_loc = lang_cfg.get(pred_name, pred_name)
                error_labels.append(f"{t_loc} → {p_loc}")
                error_counts.append(count)
                error_colors.append("#c0392b" if pred_name == "ANOMALY" else "#d35400")

    bars = ax2.bar(np.arange(len(error_counts)), error_counts, color=error_colors, edgecolor="black", linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height):,}", ha="center", va="bottom", fontsize=12, rotation=0, fontweight="bold", color="black")
            
    ax2.set_ylabel(lang_cfg["dist_ylabel_2"], fontsize=14, color="black", fontweight="bold")
    ax2.set_xlabel(lang_cfg["dist_xlabel_2"], fontsize=14, color="black", fontweight="bold")
    ax2.set_title(lang_cfg["dist_title_2"], fontsize=16, color="black", fontweight="bold", pad=10)
    
    ax2.set_xticks(np.arange(len(error_counts)))
    ax2.set_xticklabels(error_labels, rotation=45, ha="right", fontsize=12, color="black")
    ax2.tick_params(axis='y', labelsize=12, colors='black')
    
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=20, fontweight="bold", y=1.05) 
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_overview_2class(parsed: ParsedLog2, out_path: Path, lang_cfg: Dict[str, str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    if parsed.cluster_label_summary:
        names = list(parsed.cluster_label_summary.keys())
        names_loc = localize_keys(names, lang_cfg)
        values = [parsed.cluster_label_summary[name] for name in names]
        ax.bar(names_loc, values, color=["#2ecc71", "#e74c3c"][: len(names)], edgecolor="black")
        
        ax.set_title(lang_cfg["overview_cluster"], fontsize=20, fontweight="bold", color="black")
        ax.set_ylabel(lang_cfg["overview_cluster_ylabel"], fontsize=18, color="black")
        ax.tick_params(axis='both', labelsize=12, colors='black')
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")

    ax = axes[0, 1]
    ax.axis("off")
    summary_lines = [
        f"Dataset: {parsed.dataset}",
        f"Algorithm: {parsed.algorithm}",
        f"Embedding: {parsed.embedding}",
    ]
    if parsed.overall_accuracy is not None:
        summary_lines.append(f"Accuracy: {parsed.overall_accuracy:.4f}")
    if parsed.cluster_size_stats:
        summary_lines.append(f"Cluster mean size: {parsed.cluster_size_stats.get('mean', 0):.0f}")
        summary_lines.append(f"Cluster median size: {parsed.cluster_size_stats.get('median', 0):.0f}")
        
    ax.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=20, color="black",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#d0d7de"))

    ax = axes[1, 0]
    if parsed.per_class_metrics:
        labels = list(parsed.per_class_metrics.keys())
        labels_loc = localize_keys(labels, lang_cfg)
        f1s = [parsed.per_class_metrics[label]["f1"] for label in labels]
        ax.bar(labels_loc, f1s, color="#4c78a8", edgecolor="black")
        
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 Score", fontsize=18, color="black")
        ax.set_title(lang_cfg["overview_f1"], fontsize=20, fontweight="bold", color="black")
        ax.tick_params(axis='both', labelsize=12, colors='black')
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")

    ax = axes[1, 1]
    if parsed.per_method_metrics:
        labels = list(parsed.per_method_metrics.keys())
        
        # Lokalisasi nama method ke bahasa indonesia jika perlu, atau default ke ANOMALI
        labels_loc = []
        for l in labels:
            if l == "ANOMALY" and lang_cfg.get("ANOMALY") == "ANOMALI":
                labels_loc.append("ANOMALI")
            elif l == "ANOMALY_RULES" and lang_cfg.get("ANOMALY") == "ANOMALI":
                labels_loc.append("ATURAN_ANOMALI")
            else:
                labels_loc.append(l)

        values = [parsed.per_method_metrics[label]["samples"] for label in labels]
        ax.bar(labels_loc, values, color="#f58518", edgecolor="black")
        
        ax.set_yscale("log")
        ax.set_ylabel("Samples", fontsize=18, color="black")
        ax.set_title(lang_cfg["overview_method"], fontsize=20, fontweight="bold", color="black")
        ax.tick_params(axis="x", rotation=45, labelsize=12, colors='black')
        ax.tick_params(axis="y", labelsize=12, colors='black')
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_summary_files_2class(parsed: ParsedLog2, out_dir: Path, lang_cfg: Dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "reconstructed_summary_2class.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(parsed), f, indent=2)

    lines = [
        "RECONSTRUCTED TESTING SUMMARY (2-CLASS)",
        "=" * 72,
        f"Dataset: {parsed.dataset}",
        f"Algorithm: {parsed.algorithm}",
        f"Embedding: {parsed.embedding}",
    ]
    if parsed.overall_accuracy is not None:
        lines.append(f"Overall Accuracy: {parsed.overall_accuracy:.4f}")
    if parsed.per_class_metrics:
        lines.append("")
        lines.append("Per-Class Metrics:")
        for label, metrics in parsed.per_class_metrics.items():
            loc_label = lang_cfg.get(label, label)
            lines.append(
                f"  {loc_label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}, support={metrics['support']:,}"
            )
    if parsed.prediction_distribution:
        lines.append("")
        lines.append("Prediction Distribution:")
        for true_name, preds in parsed.prediction_distribution.items():
            loc_true = lang_cfg.get(true_name, true_name)
            total = sum(preds.values())
            lines.append(f"  {loc_true}: {total:,} samples")
            for pred_name, count in preds.items():
                loc_pred = lang_cfg.get(pred_name, pred_name)
                pct = (100.0 * count / total) if total else 0.0
                lines.append(f"    {loc_pred}: {count:,} ({pct:.2f}%)")

    with (out_dir / "reconstructed_summary_2class.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_for_language(parsed: ParsedLog2, out_dir: Path, lang: str) -> None:
    lang_cfg = get_lang_config(lang)
    lang_dir = out_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    write_summary_files_2class(parsed, lang_dir, lang_cfg)

    true_names = list(parsed.prediction_distribution.keys()) if parsed.prediction_distribution else ["NORMAL", "ANOMALY"]
    pred_names = ["NORMAL", "ANOMALY"]

    if parsed.confusion_matrix:
        render_confusion_matrix_2class(
            parsed.confusion_matrix,
            true_names,
            pred_names,
            lang_dir / "confusion_matrix_2class.png",
            f"{parsed.dataset} {parsed.algorithm.upper()}",
            lang_cfg
        )

    if parsed.prediction_distribution:
        render_prediction_distribution_2class(
            parsed.prediction_distribution,
            lang_dir / "prediction_distribution_2class.png",
            f"{parsed.dataset} {parsed.algorithm.upper()}",
            lang_cfg
        )

    render_overview_2class(parsed, lang_dir / "analysis_overview_2class.png", lang_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recreate 2-class summary artifacts from a testing pipeline TXT log.")
    parser.add_argument("input_path", type=Path, help="Path to the saved TXT log file OR directory containing log files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Base output directory for recreated files")
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Path not found: {args.input_path}")

    if args.input_path.is_dir():
        log_files = list(args.input_path.rglob("*.txt"))
        print(f"Mendeteksi {len(log_files)} file log di dalam folder {args.input_path}...")
    else:
        log_files = [args.input_path]

    for log_file in log_files:
        print(f"\nMemproses: {log_file.name} ...")
        text = log_file.read_text(encoding="utf-8", errors="replace")
        
        try:
            parsed = parse_log_text_to_parsedlog(text)
            parsed = merge_to_2class(parsed)
        except Exception as e:
            print(f"Gagal memproses {log_file.name}: {e}")
            continue

        if args.out_dir:
            base_out_dir = args.out_dir / log_file.stem
        else:
            base_out_dir = log_file.with_suffix(".2class")
        
        base_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Output Bahasa Inggris
        generate_for_language(parsed, base_out_dir, "en")
        
        # 2. Output Bahasa Indonesia
        generate_for_language(parsed, base_out_dir, "id")
        
        print(f"Selesai! Hasil disimpan di: {base_out_dir}")

    print("\nSemua file berhasil diproses!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())