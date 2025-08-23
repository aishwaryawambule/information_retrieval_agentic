"""
document_classifier.py

Streamlined document classification system with essential functionality only.
"""

import json
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.metrics import accuracy_score

# Import modules
from .dataPreprocessing import DatasetManager, TextPreprocessor, FeatureExtractor
from .naiveBayesClassifier import MultinomialNaiveBayes

class DocumentClassifier:
    """
    Streamlined document classification system for training and prediction.
    """
    
    def __init__(self, alpha: float = 1.0, max_features: int = 5000, 
                 ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize the document classification system."""
        self.classifier = MultinomialNaiveBayes(alpha=alpha)
        self.dataset_manager = DatasetManager()
        self.preprocessor = TextPreprocessor()
        
        self.dataset_manager.feature_extractor = FeatureExtractor(
            max_features=max_features, 
            ngram_range=ngram_range
        )
        
        self.is_trained = False
        self.training_results = {}
    
    def train_from_documents(self, documents: List[Dict], test_size: float = 0.2, 
                           random_state: int = 42) -> Dict:
        """Train the classifier using raw documents."""
        print(f"Training on {len(documents)} documents...")
        
        # Prepare dataset
        dataset = self.dataset_manager.prepare_dataset(
            documents, test_size=test_size, random_state=random_state, encode_labels=False
        )
        
        print(f"Training: {dataset['X_train'].shape[0]} samples, "
              f"Testing: {dataset['X_test'].shape[0]} samples")
        
        # Train classifier
        self.classifier.fit(dataset['X_train'], dataset['y_train'])
        
        # Evaluate
        y_pred = self.classifier.predict(dataset['X_test'])
        accuracy = accuracy_score(dataset['y_test'], y_pred)
        
        self.training_results = {
            'accuracy': accuracy,
            'unique_labels': dataset['unique_labels'],
            'train_samples': len(dataset['y_train']),
            'test_samples': len(dataset['y_test'])
        }
        
        self.is_trained = True
        print(f"Training completed. Accuracy: {accuracy:.4f}")
        
        return self.training_results
    
    def train_from_file(self, filepath: str, test_size: float = 0.2, 
                       random_state: int = 42) -> Dict:
        """Train from JSON file."""
        print(f"filepath: {filepath}")
        documents = self.dataset_manager.load_raw_data(filepath)
        print(f"documents: {len(documents)}")
        return self.train_from_documents(documents, test_size, random_state)
    
    def classify_text(self, text: str) -> Dict:
        """Classify a single text document."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if not text.strip():
            return {'predicted_category': 'Unknown', 'confidence': 0.0}
        
        # Preprocess and extract features
        processed_text = self.preprocessor.preprocess_text(text)
        if not processed_text.strip():
            return {'predicted_category': 'Unknown', 'confidence': 0.0}
        
        features = self.dataset_manager.feature_extractor.transform([processed_text])
        # Predict with confidence
        prediction, confidence = self.classifier.predict_with_confidence(features)
        print(f"prediction: {prediction}, confidence: {confidence}")
        return {
            'predicted_category': prediction[0],
            'confidence': float(confidence[0])
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify multiple texts efficiently."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        results = []
        processed_texts = []
        valid_indices = []
        
        # Preprocess valid texts
        for i, text in enumerate(texts):
            if text.strip():
                processed_text = self.preprocessor.preprocess_text(text)
                if processed_text.strip():
                    processed_texts.append(processed_text)
                    valid_indices.append(i)
        
        # Batch prediction for valid texts
        if processed_texts:
            features = self.dataset_manager.feature_extractor.transform(processed_texts)
            predictions, confidences = self.classifier.predict_with_confidence(features)
        
        # Build results
        processed_idx = 0
        for i, text in enumerate(texts):
            if i in valid_indices:
                results.append({
                    'predicted_category': predictions[processed_idx],
                    'confidence': float(confidences[processed_idx])
                })
                processed_idx += 1
            else:
                results.append({
                    'predicted_category': 'Unknown',
                    'confidence': 0.0
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get basic model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "accuracy": self.training_results['accuracy'],
            "categories": self.training_results['unique_labels'],
            "train_samples": self.training_results['train_samples'],
            "test_samples": self.training_results['test_samples']
        }
    
    def save_model(self, model_dir: str = "models"):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier and vectorizer
        self.classifier.save_model(str(model_path / "naive_bayes_model.json"))
        self.dataset_manager.feature_extractor.save_vectorizer(
            str(model_path / "vectorizer.pkl")
        )
        
        # Save training results
        with open(model_path / "training_results.json", 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        print(f"Model saved to {model_path}/")
    
    def load_model(self, model_dir: str = "models"):
        """Load a trained model."""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load classifier and vectorizer
        self.classifier.load_model(str(model_path / "naive_bayes_model.json"))
        self.dataset_manager.feature_extractor.load_vectorizer(
            str(model_path / "vectorizer.pkl")
        )
        
        # Load training results
        with open(model_path / "training_results.json", 'r') as f:
            self.training_results = json.load(f)
        
        self.is_trained = True
        print(f"Model loaded from {model_path}/")