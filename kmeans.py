import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
import os
from typing import List, Union
import glob


class LogAnomalyDetector:
    """
    A class for detecting anomalies in log data using BERT embeddings and KMeans clustering.
    Supports multiple datasets and can transform raw log strings to semantic vectors.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the LogAnomalyDetector with a BERT model.
        
        Args:
            model_name: Name of the pre-trained BERT model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.kmeans_model = None
        self.embeddings = None
        
    def transform_log_to_vector(self, log_text: str) -> np.ndarray:
        """
        Transform a raw log string to a semantic vector using BERT.
        
        Args:
            log_text: Raw log string to transform
            
        Returns:
            numpy array of shape (768,) representing the log embedding
        """
        # Tokenize the text
        inputs = self.tokenizer(
            log_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            
            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            masked_embeddings = last_hidden_state * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            sentence_embedding = sum_embeddings / sum_mask
            
        return sentence_embedding.squeeze(0).detach().numpy()
    
    def transform_logs_to_vectors(self, log_texts: List[str]) -> np.ndarray:
        """
        Transform multiple raw log strings to semantic vectors.
        
        Args:
            log_texts: List of raw log strings to transform
            
        Returns:
            numpy array of shape (n_logs, 768) containing all embeddings
        """
        embeddings = []
        print(f"Processing {len(log_texts)} log entries...")
        
        for i, text in enumerate(log_texts):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(log_texts)} logs")
            embedding = self.transform_log_to_vector(text)
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    def load_embeddings(self, embedding_files: Union[str, List[str]]) -> np.ndarray:
        """
        Load embeddings from one or multiple .npy files.
        
        Args:
            embedding_files: Single file path or list of file paths to .npy files
                           Can also be a glob pattern like "*.npy"
            
        Returns:
            Combined numpy array containing all embeddings
        """
        if isinstance(embedding_files, str):
            # Check if it's a glob pattern
            if '*' in embedding_files:
                embedding_files = glob.glob(embedding_files)
            else:
                embedding_files = [embedding_files]
        
        all_embeddings = []
        
        for file_path in embedding_files:
            if os.path.exists(file_path):
                print(f"Loading embeddings from: {file_path}")
                embeddings = np.load(file_path)
                
                # Handle different possible shapes
                if len(embeddings.shape) == 3:
                    # If shape is (n, 1, 768), squeeze the middle dimension
                    embeddings = embeddings.squeeze(1)
                
                all_embeddings.append(embeddings)
                print(f"  Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not all_embeddings:
            raise ValueError("No embeddings were loaded. Please check your file paths.")
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        print(f"\nTotal embeddings loaded: {combined_embeddings.shape[0]}")
        print(f"Embedding dimension: {combined_embeddings.shape[1]}")
        
        self.embeddings = combined_embeddings
        return combined_embeddings
    
    def train_kmeans(self, n_clusters: int = 5, random_state: int = 42):
        """
        Train KMeans clustering model on the loaded embeddings.
        
        Args:
            n_clusters: Number of clusters to create
            random_state: Random state for reproducibility
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Please load embeddings first using load_embeddings().")
        
        print(f"\nTraining KMeans with {n_clusters} clusters...")
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.kmeans_model.fit(self.embeddings)
        print("KMeans training completed!")
        
        # Print cluster distribution
        labels = self.kmeans_model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} samples ({count/len(labels)*100:.2f}%)")
        
        return self.kmeans_model
    
    def predict_anomaly(self, log_text: str, threshold_percentile: float = 95) -> tuple:
        """
        Predict if a log entry is anomalous based on distance to cluster center.
        
        Args:
            log_text: Raw log string to analyze
            threshold_percentile: Percentile threshold for anomaly detection
            
        Returns:
            Tuple of (is_anomaly, distance, cluster_id)
        """
        if self.kmeans_model is None:
            raise ValueError("KMeans model not trained. Please train the model first.")
        
        # Transform log to vector
        embedding = self.transform_log_to_vector(log_text).reshape(1, -1)
        
        # Predict cluster
        cluster_id = self.kmeans_model.predict(embedding)[0]
        
        # Calculate distance to cluster center
        distance = np.linalg.norm(embedding - self.kmeans_model.cluster_centers_[cluster_id])
        
        # Calculate threshold based on training data
        if self.embeddings is not None:
            all_distances = []
            labels = self.kmeans_model.labels_
            for i, label in enumerate(labels):
                dist = np.linalg.norm(
                    self.embeddings[i] - self.kmeans_model.cluster_centers_[label]
                )
                all_distances.append(dist)
            
            threshold = np.percentile(all_distances, threshold_percentile)
            is_anomaly = distance > threshold
        else:
            is_anomaly = False
            threshold = 0
        
        return is_anomaly, distance, cluster_id, threshold


def main():
    """
    Example usage of LogAnomalyDetector with multiple datasets.
    """
    # Initialize detector
    detector = LogAnomalyDetector()
    
    # Method 1: Load multiple specific .npy files
    # detector.load_embeddings(['apache_embeddings.npy', 'nginx_embeddings.npy', 'system_embeddings.npy'])
    
    # Method 2: Load all .npy files in current directory using glob pattern
    detector.load_embeddings('*.npy')
    
    # Method 3: Load from a list of files
    # embedding_files = ['apache_embeddings.npy']
    # detector.load_embeddings(embedding_files)
    
    # Train KMeans clustering
    detector.train_kmeans(n_clusters=5)
    
    # Example: Transform a new log string and check for anomaly
    test_log = "User login failed due to wrong password"
    is_anomaly, distance, cluster_id, threshold = detector.predict_anomaly(test_log)
    
    print(f"\nTest Log: {test_log}")
    print(f"Cluster ID: {cluster_id}")
    print(f"Distance to center: {distance:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Is Anomaly: {is_anomaly}")


if __name__ == "__main__":
    main()
