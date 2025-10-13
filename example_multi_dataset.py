"""
Example script demonstrating multi-dataset anomaly detection.

This script shows how to:
1. Load embeddings from multiple .npy files
2. Train a KMeans model on combined datasets
3. Detect anomalies in new log entries
"""

from kmeans import LogAnomalyDetector
import numpy as np


def create_sample_embeddings():
    """
    Create sample embedding files for demonstration purposes.
    This simulates having multiple log datasets.
    """
    print("Creating sample embeddings for demonstration...")
    
    # Create sample embeddings (normally these would come from bert.py)
    # Using random data for demonstration
    np.random.seed(42)
    
    # Simulate Apache logs (500 samples)
    apache_embeddings = np.random.randn(500, 768) * 0.1
    np.save('sample_apache_embeddings.npy', apache_embeddings)
    print("Created sample_apache_embeddings.npy")
    
    # Simulate Nginx logs (300 samples)
    nginx_embeddings = np.random.randn(300, 768) * 0.1 + 0.5
    np.save('sample_nginx_embeddings.npy', nginx_embeddings)
    print("Created sample_nginx_embeddings.npy")
    
    # Simulate System logs (200 samples)
    system_embeddings = np.random.randn(200, 768) * 0.1 - 0.5
    np.save('sample_system_embeddings.npy', system_embeddings)
    print("Created sample_system_embeddings.npy")
    
    print("\nSample embeddings created successfully!")


def example_single_dataset():
    """
    Example 1: Using a single dataset
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Dataset")
    print("="*60)
    
    # Note: This requires actual embeddings from bert.py
    # For demonstration, we'll use sample data
    create_sample_embeddings()
    
    detector = LogAnomalyDetector()
    
    # Load single dataset
    detector.load_embeddings('sample_apache_embeddings.npy')
    
    # Train KMeans
    detector.train_kmeans(n_clusters=3)


def example_multiple_datasets():
    """
    Example 2: Using multiple specific datasets
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Specific Datasets")
    print("="*60)
    
    detector = LogAnomalyDetector()
    
    # Load multiple specific files
    embedding_files = [
        'sample_apache_embeddings.npy',
        'sample_nginx_embeddings.npy',
        'sample_system_embeddings.npy'
    ]
    
    detector.load_embeddings(embedding_files)
    
    # Train KMeans with more clusters for combined data
    detector.train_kmeans(n_clusters=5)


def example_glob_pattern():
    """
    Example 3: Using glob pattern to load all datasets
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Load All Datasets with Glob Pattern")
    print("="*60)
    
    detector = LogAnomalyDetector()
    
    # Load all .npy files starting with 'sample_'
    detector.load_embeddings('sample_*.npy')
    
    # Train KMeans
    detector.train_kmeans(n_clusters=5)


def example_transform_and_detect():
    """
    Example 4: Transform new log strings and detect anomalies
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Transform Logs and Detect Anomalies")
    print("="*60)
    
    detector = LogAnomalyDetector()
    
    # Load all sample embeddings
    detector.load_embeddings('sample_*.npy')
    
    # Train KMeans
    detector.train_kmeans(n_clusters=5)
    
    # Test logs
    test_logs = [
        "INFO: Server started successfully",
        "ERROR: Critical system failure detected",
        "WARNING: High memory usage (95%)",
        "INFO: User logged in",
        "ERROR: Database connection timeout",
    ]
    
    print("\nTesting log entries for anomalies:")
    print("-" * 60)
    
    for log in test_logs:
        is_anomaly, distance, cluster_id, threshold = detector.predict_anomaly(log)
        
        status = "🔴 ANOMALY" if is_anomaly else "✅ NORMAL"
        print(f"\n{status}")
        print(f"  Log: {log}")
        print(f"  Cluster: {cluster_id}")
        print(f"  Distance: {distance:.4f}")
        print(f"  Threshold: {threshold:.4f}")


def example_batch_transform():
    """
    Example 5: Transform multiple logs at once
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Transform Logs to Vectors")
    print("="*60)
    
    detector = LogAnomalyDetector()
    
    # Sample logs to transform
    logs = [
        "User authentication successful",
        "Failed login attempt detected",
        "System backup completed",
        "Network connection established",
        "Error: Out of memory exception"
    ]
    
    print(f"Transforming {len(logs)} log entries...")
    vectors = detector.transform_logs_to_vectors(logs)
    
    print(f"\nResult:")
    print(f"  Shape: {vectors.shape}")
    print(f"  Dimension: {vectors.shape[1]} (BERT embedding size)")
    print(f"  Number of logs: {vectors.shape[0]}")


def cleanup_sample_files():
    """
    Remove sample embedding files created for demonstration.
    """
    import os
    
    sample_files = [
        'sample_apache_embeddings.npy',
        'sample_nginx_embeddings.npy',
        'sample_system_embeddings.npy'
    ]
    
    print("\n" + "="*60)
    print("Cleaning up sample files...")
    print("="*60)
    
    for file in sample_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")


def main():
    """
    Run all examples
    """
    print("="*60)
    print("Multi-Dataset Anomaly Detection Examples")
    print("="*60)
    
    # Run examples
    example_single_dataset()
    example_multiple_datasets()
    example_glob_pattern()
    example_transform_and_detect()
    example_batch_transform()
    
    # Cleanup
    cleanup_sample_files()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nTo use with real data:")
    print("1. Run bert.py to generate embeddings from your log files")
    print("2. Use kmeans.py to load and analyze the embeddings")
    print("3. See README.md for more detailed usage instructions")


if __name__ == "__main__":
    main()
