"""
Optimized Data Preprocessing Module for Document Classification System

This module handles text preprocessing tasks including:
- Text cleaning and normalization using NLTK
- Tokenization and stop word removal with NLTK
- Feature extraction using TF-IDF
- Data preparation for machine learning models
- Enhanced JSON file loading

"""

import re
import json
import pickle
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab')


class TextPreprocessor:
    """
    Optimized text preprocessing class using NLTK.
    """
    
    def __init__(self, language: str = 'english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for better performance
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data using regex patterns.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and emails
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        
        # Remove special characters but keep spaces
        text = self.special_char_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK word_tokenize.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        tokens = word_tokenize(text)
        # Filter out tokens that are too short
        tokens = [token for token in tokens if len(token) >= 3]
        return tokens
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words using NLTK stopwords.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using NLTK WordNetLemmatizer.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text (str): Raw text to preprocess
            remove_stopwords (bool): Whether to remove stop words
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stop words if requested
        if remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Lemmatize tokens
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back into string
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Preprocess a list of documents.
        
        Args:
            documents (List[Dict]): List of document dictionaries
            
        Returns:
            Tuple[List[str], List[str]]: Preprocessed texts and labels
        """
        texts = []
        labels = []
        
        for doc in documents:
            preprocessed_text = self.preprocess_text(doc['text'])
            texts.append(preprocessed_text)
            labels.append(doc['category'])
        
        return texts, labels


class FeatureExtractor:
    """
    Feature extraction using TF-IDF vectorization.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize feature extractor.
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): N-gram range for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            lowercase=False,  # Already handled in preprocessing
            analyzer='word'
        )
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        features = self.vectorizer.transform(texts)
        return features.toarray()
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from fitted vectorizer.
        
        Returns:
            List[str]: List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save_vectorizer(self, filepath: str):
        """
        Save fitted vectorizer to file.
        
        Args:
            filepath (str): Path to save vectorizer
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_vectorizer(self, filepath: str):
        """
        Load vectorizer from file.
        
        Args:
            filepath (str): Path to load vectorizer from
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True


class DatasetManager:
    """
    Enhanced dataset management with improved JSON loading.
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
    
    def load_raw_data(self, filepath: str) -> List[Dict]:
        """
        Load raw documents from JSON file with enhanced error handling.
        
        Args:
            filepath (str): Path to JSON file
            
        Returns:
            List[Dict]: List of document dictionaries
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'documents' in data:
                    return data['documents']
                else:
                    return [data]  # Single document
            else:
                raise ValueError("JSON data must be a list or dictionary")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading JSON file: {e}")
    
    def prepare_dataset(self, documents: List[Dict], test_size: float = 0.2, random_state: int = 42, 
                       encode_labels: bool = True) -> Dict:
        """
        Prepare complete dataset for training and testing.
        
        Args:
            documents (List[Dict]): Raw documents
            test_size (float): Proportion of test set
            random_state (int): Random seed
            encode_labels (bool): Whether to encode labels to integers
            
        Returns:
            Dict: Prepared dataset with train/test splits
        """
        # Preprocess texts and extract labels
        texts, labels = self.preprocessor.preprocess_documents(documents)
        
        # Split into train and test sets
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Extract features
        X_train = self.feature_extractor.fit_transform(X_train_text)
        X_test = self.feature_extractor.transform(X_test_text)
        
        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_text': X_train_text,
            'X_test_text': X_test_text,
            'unique_labels': sorted(list(set(labels))),
            'feature_names': self.feature_extractor.get_feature_names()
        }
        
        # Optional label encoding
        if encode_labels:
            unique_labels = sorted(list(set(labels)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
            
            y_train_idx = [label_to_idx[label] for label in y_train]
            y_test_idx = [label_to_idx[label] for label in y_test]
            
            dataset.update({
                'y_train_encoded': np.array(y_train_idx),
                'y_test_encoded': np.array(y_test_idx),
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label
            })
        
        return dataset
    
    def save_processed_data(self, dataset: Dict, filepath: str):
        """
        Save processed dataset to file.
        
        Args:
            dataset (Dict): Processed dataset
            filepath (str): Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'X_train': dataset['X_train'].tolist(),
            'X_test': dataset['X_test'].tolist(),
            'y_train': dataset['y_train'].tolist(),
            'y_test': dataset['y_test'].tolist(),
            'y_train_labels': dataset['y_train_labels'],
            'y_test_labels': dataset['y_test_labels'],
            'X_train_text': dataset['X_train_text'],
            'X_test_text': dataset['X_test_text'],
            'label_to_idx': dataset['label_to_idx'],
            'idx_to_label': dataset['idx_to_label'],
            'feature_names': dataset['feature_names']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2)


def main():
    """
    Main function to demonstrate preprocessing functionality.
    """
    dataset_manager = DatasetManager()
    
    try:
        documents = dataset_manager.load_raw_data('data/raw_documents.json')
        print(f"Loaded {len(documents)} documents")
        
        # Prepare dataset
        dataset = dataset_manager.prepare_dataset(documents)
        print(f"Training set: {dataset['X_train'].shape}")
        print(f"Test set: {dataset['X_test'].shape}")
        print(f"Labels: {list(dataset['idx_to_label'].values())}")
        
        # Save processed data
        dataset_manager.save_processed_data(dataset, 'data/processed_data.json')
        dataset_manager.feature_extractor.save_vectorizer('models/vectorizer.pkl')
        
        print("Preprocessing completed successfully!")
        
    except FileNotFoundError:
        print("Raw data file not found. Please ensure raw_documents.json exists.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")


if __name__ == "__main__":
    main()