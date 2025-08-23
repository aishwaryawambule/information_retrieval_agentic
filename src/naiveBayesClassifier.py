"""
naive_bayes.py

Core Multinomial Naive Bayes classifier implementation for text classification.
This module contains only the classifier logic without preprocessing dependencies.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Union
from pathlib import Path


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier optimized for text classification.
    Uses log probabilities and vectorized operations for efficiency and numerical stability.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Naive Bayes classifier.
        
        Args:
            alpha (float): Laplace smoothing parameter (default: 1.0)
                         Higher values = more smoothing, less overfitting
                         Lower values = less smoothing, may overfit on small datasets
        """
        self.alpha = alpha
        self.class_log_priors = None
        self.feature_log_probs = None
        self.classes = None
        self.class_names = None
        self.n_features = 0
        self.is_fitted = False
        
        # Store label mappings for string labels
        self.label_to_idx = None
        self.idx_to_label = None
    
    def fit(self, X: np.ndarray, y: Union[np.ndarray, List[str]]):
        """
        Train the Naive Bayes classifier on feature matrix and labels.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
                          Each row represents a document, each column a feature
            y (Union[np.ndarray, List[str]]): Target labels
                                            Can be string labels or integer indices
        """

        print(f"fit called")
        # Handle string labels by converting to indices
        if isinstance(y[0], str):
            unique_labels = sorted(list(set(y)))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            y_encoded = np.array([self.label_to_idx[label] for label in y])
            self.class_names = unique_labels
        else:
            y_encoded = np.array(y)
            self.classes = np.unique(y_encoded)
            self.class_names = [f"Class_{i}" for i in self.classes]
        
        self.classes = np.unique(y_encoded)
        n_classes = len(self.classes)
        n_samples, self.n_features = X.shape
        
        # Calculate class priors using log probabilities for numerical stability
        class_counts = np.bincount(y_encoded)
        self.class_log_priors = np.log(class_counts / n_samples)
        
        # Initialize feature probabilities matrix (n_classes x n_features)
        self.feature_log_probs = np.zeros((n_classes, self.n_features))
        
        # Calculate feature likelihoods for each class using vectorized operations
        for i, class_idx in enumerate(self.classes):
            # Get all samples belonging to this class
            class_mask = (y_encoded == class_idx)
            class_features = X[class_mask]
            
            # Sum feature counts for this class across all documents
            feature_counts = np.sum(class_features, axis=0)
            
            # Apply Laplace smoothing to avoid zero probabilities
            smoothed_counts = feature_counts + self.alpha
            total_count = np.sum(smoothed_counts)
            
            # Calculate log probabilities for numerical stability
            self.feature_log_probs[i] = np.log(smoothed_counts / total_count)
        
        self.is_fitted = True
        print(f"Model trained: {n_samples} samples, {n_classes} classes, {self.n_features} features")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples using Bayes' theorem.
        
        P(class|features) ∝ P(class) * P(features|class)
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes)
        """
        self._check_fitted()
        
        # Calculate log probabilities: log P(class) + sum(log P(feature|class))
        # Using matrix multiplication for vectorized computation
        log_probs = X @ self.feature_log_probs.T + self.class_log_priors
        # Convert log probabilities to probabilities using log-sum-exp trick
        # This prevents numerical underflow
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log_probs)
        probabilities = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels
        """
        self._check_fitted()
        
        # Calculate log probabilities and get class with highest probability
        log_probs = X @ self.feature_log_probs.T + self.class_log_priors
        print("log_probs:", log_probs)
        predicted_indices = np.argmax(log_probs, axis=1)
        print("predicted_indices:", predicted_indices)
        # Convert back to original labels if string labels were used
        if self.idx_to_label:
            predictions = [self.idx_to_label[idx] for idx in predicted_indices]
            return np.array(predictions)
        else:
            return self.classes[predicted_indices]
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels with confidence scores.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and confidence scores (max probability)
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores
    
    def score(self, X: np.ndarray, y: Union[np.ndarray, List[str]]) -> float:
        """
        Calculate accuracy score on test data.
        
        Args:
            X (np.ndarray): Feature matrix
            y (Union[np.ndarray, List[str]]): True labels
            
        Returns:
            float: Accuracy score (fraction of correct predictions)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_feature_importance(self, class_name: str, feature_names: List[str], 
                             top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important features for a specific class based on log probabilities.
        
        Args:
            class_name (str): Name of the class to analyze
            feature_names (List[str]): List of feature names corresponding to feature matrix
            top_n (int): Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature_name, log_probability) tuples
                                   sorted by importance (highest first)
        """
        self._check_fitted()
        
        if self.label_to_idx and class_name in self.label_to_idx:
            class_idx = self.label_to_idx[class_name]
        else:
            raise ValueError(f"Unknown class: {class_name}. Available classes: {self.class_names}")
        
        # Get feature log probabilities for the specified class
        feature_probs = self.feature_log_probs[class_idx]
        
        # Get indices of top features (highest log probabilities)
        top_indices = np.argsort(feature_probs)[-top_n:][::-1]
        
        # Return feature names and their log probabilities
        return [(feature_names[idx], feature_probs[idx]) for idx in top_indices]
    
    def get_class_distribution(self) -> Dict[str, float]:
        """
        Get the class distribution from training data.
        
        Returns:
            Dict[str, float]: Dictionary mapping class names to their prior probabilities
        """
        self._check_fitted()
        
        if self.class_names:
            return {name: np.exp(prior) for name, prior in 
                   zip(self.class_names, self.class_log_priors)}
        else:
            return {f"Class_{i}": np.exp(prior) for i, prior in 
                   enumerate(self.class_log_priors)}
    
    def _check_fitted(self):
        """Check if the classifier has been fitted."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions. Call fit() first.")
    
    def save_model(self, filepath: str):
        """
        Save the trained model to a JSON file.
        
        Args:
            filepath (str): Path where to save the model
        """
        self._check_fitted()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'alpha': self.alpha,
            'class_log_priors': self.class_log_priors.tolist(),
            'feature_log_probs': self.feature_log_probs.tolist(),
            'classes': self.classes.tolist() if self.classes is not None else None,
            'class_names': self.class_names,
            'n_features': self.n_features,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from a JSON file.
        
        Args:
            filepath (str): Path to the saved model file
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.alpha = model_data['alpha']
        self.class_log_priors = np.array(model_data['class_log_priors'])
        self.feature_log_probs = np.array(model_data['feature_log_probs'])
        self.classes = np.array(model_data['classes']) if model_data['classes'] else None
        self.class_names = model_data['class_names']
        self.n_features = model_data['n_features']
        self.label_to_idx = model_data['label_to_idx']
        self.idx_to_label = model_data['idx_to_label']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.
        
        Returns:
            Dict: Model information including parameters and dimensions
        """
        if not self.is_fitted:
            return {"status": "not_fitted", "message": "Model has not been trained yet"}
        
        return {
            "status": "fitted",
            "alpha": self.alpha,
            "n_features": self.n_features,
            "n_classes": len(self.classes),
            "class_names": self.class_names,
            "class_distribution": self.get_class_distribution()
        }