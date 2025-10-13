"""
Utility script to inspect and compare multiple .npy embedding files.

This script helps you:
- View information about multiple .npy files
- Compare dataset sizes and shapes
- Verify data integrity before using in kmeans.py
"""

import numpy as np
import os
import glob
from typing import List, Dict


def inspect_npy_file(filepath: str) -> Dict:
    """
    Inspect a single .npy file and return its information.
    
    Args:
        filepath: Path to the .npy file
        
    Returns:
        Dictionary containing file information
    """
    if not os.path.exists(filepath):
        return {
            'filepath': filepath,
            'exists': False,
            'error': 'File not found'
        }
    
    try:
        data = np.load(filepath)
        
        info = {
            'filepath': filepath,
            'exists': True,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size': data.size,
            'n_samples': data.shape[0] if len(data.shape) > 0 else 0,
            'n_dimensions': data.shape[1] if len(data.shape) > 1 else 0,
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'std_value': float(np.std(data))
        }
        
        return info
        
    except Exception as e:
        return {
            'filepath': filepath,
            'exists': True,
            'error': str(e)
        }


def inspect_multiple_files(pattern: str = "*.npy") -> List[Dict]:
    """
    Inspect multiple .npy files matching a pattern.
    
    Args:
        pattern: Glob pattern for matching files (default: "*.npy")
        
    Returns:
        List of dictionaries containing information for each file
    """
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return []
    
    results = []
    for filepath in sorted(files):
        info = inspect_npy_file(filepath)
        results.append(info)
    
    return results


def print_file_info(info: Dict):
    """
    Pretty print information about a single .npy file.
    
    Args:
        info: Dictionary containing file information
    """
    print(f"\n{'='*70}")
    print(f"File: {info['filepath']}")
    print(f"{'='*70}")
    
    if not info['exists']:
        print(f"❌ Error: {info.get('error', 'Unknown error')}")
        return
    
    if 'error' in info:
        print(f"❌ Error loading file: {info['error']}")
        return
    
    print(f"✅ File loaded successfully")
    print(f"\nBasic Information:")
    print(f"  Shape:           {info['shape']}")
    print(f"  Data Type:       {info['dtype']}")
    print(f"  Total Elements:  {info['size']:,}")
    print(f"  Number of Samples: {info['n_samples']:,}")
    print(f"  Embedding Dimension: {info['n_dimensions']}")
    print(f"  File Size:       {info['file_size_mb']:.2f} MB")
    
    print(f"\nStatistics:")
    print(f"  Min Value:       {info['min_value']:.6f}")
    print(f"  Max Value:       {info['max_value']:.6f}")
    print(f"  Mean Value:      {info['mean_value']:.6f}")
    print(f"  Std Deviation:   {info['std_value']:.6f}")


def compare_datasets(results: List[Dict]):
    """
    Compare multiple datasets and show summary.
    
    Args:
        results: List of dictionaries containing file information
    """
    if not results:
        print("No datasets to compare.")
        return
    
    valid_results = [r for r in results if r['exists'] and 'error' not in r]
    
    if not valid_results:
        print("No valid datasets to compare.")
        return
    
    print(f"\n{'='*70}")
    print("DATASET COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    total_samples = sum(r['n_samples'] for r in valid_results)
    total_size_mb = sum(r['file_size_mb'] for r in valid_results)
    
    print(f"\nTotal Datasets:  {len(valid_results)}")
    print(f"Total Samples:   {total_samples:,}")
    print(f"Total Size:      {total_size_mb:.2f} MB")
    
    # Check dimension consistency
    dimensions = set(r['n_dimensions'] for r in valid_results)
    if len(dimensions) == 1:
        print(f"✅ All datasets have consistent dimension: {dimensions.pop()}")
    else:
        print(f"⚠️  WARNING: Inconsistent dimensions found: {dimensions}")
    
    # Dataset distribution
    print(f"\nDataset Distribution:")
    print(f"{'File':<40} {'Samples':<10} {'Percentage':<10}")
    print(f"{'-'*60}")
    
    for r in valid_results:
        filename = os.path.basename(r['filepath'])
        percentage = (r['n_samples'] / total_samples * 100) if total_samples > 0 else 0
        print(f"{filename:<40} {r['n_samples']:<10,} {percentage:<10.2f}%")


def main():
    """
    Main function to inspect all .npy files in the current directory.
    """
    print("="*70)
    print("NPY File Inspector - Multi-Dataset Analysis")
    print("="*70)
    
    # Find all .npy files
    results = inspect_multiple_files("*.npy")
    
    if not results:
        print("\n❌ No .npy files found in the current directory.")
        print("\nTo generate embeddings:")
        print("  1. Edit bert.py to add your log files")
        print("  2. Run: python bert.py")
        return
    
    # Print detailed information for each file
    for info in results:
        print_file_info(info)
    
    # Print comparison summary
    if len(results) > 1:
        compare_datasets(results)
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    valid_results = [r for r in results if r['exists'] and 'error' not in r]
    
    if valid_results:
        total_samples = sum(r['n_samples'] for r in valid_results)
        
        # Suggest number of clusters
        suggested_clusters = min(max(3, total_samples // 200), 10)
        print(f"\n📊 Suggested number of clusters: {suggested_clusters}")
        print(f"   (Based on {total_samples:,} total samples)")
        
        # Check if ready for kmeans
        dimensions = set(r['n_dimensions'] for r in valid_results)
        if len(dimensions) == 1 and 768 in dimensions:
            print(f"\n✅ All datasets are ready for KMeans clustering")
            print(f"   Run: python kmeans.py")
            print(f"   Or: python example_multi_dataset.py")
        elif len(dimensions) == 1:
            print(f"\n⚠️  Datasets have dimension {dimensions.pop()}")
            print(f"   Expected dimension: 768 (BERT embeddings)")
        else:
            print(f"\n❌ Datasets have inconsistent dimensions: {dimensions}")
            print(f"   Please regenerate embeddings with bert.py")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
