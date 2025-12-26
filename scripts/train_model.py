#!/usr/bin/env python3
"""
================================================================================
 * Sentiment Analysis Dataset - Model Training Script
 * 
 * Project: Sentiment Analysis Dataset
 * Description: Train machine learning models for sentiment classification
 *              using various algorithms including Naive Bayes, SVM, and more.
 * Category: Text Data
 * Difficulty: Intermediate
 * 
 * Author: Molla Samser (Founder)
 * Designer & Tester: Rima Khatun
 * Website: https://rskworld.in
 * Email: help@rskworld.in | support@rskworld.in
 * Phone: +91 93305 39277
 * 
 * © 2026 RSK World - Free Programming Resources & Source Code
 * All rights reserved.
================================================================================

Usage:
    python train_model.py --train ../data/train_data.csv --test ../data/test_data.csv
    python train_model.py --input ../data/sentiment_data.csv --model svm --split 0.8
    python train_model.py --input ../data/sentiment_data.csv --all-models --save
"""

import argparse
import csv
import json
import os
import pickle
import re
import string
from typing import List, Dict, Tuple, Optional
from collections import Counter
from datetime import datetime

# Try to import ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================
# Data Loading and Preprocessing
# ============================================

def load_data(filepath: str) -> List[Dict]:
    """Load data from CSV or JSON file."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line for line in f if not line.strip().startswith('#')]
            reader = csv.DictReader(lines)
            for row in reader:
                data.append(row)
        return data
    
    elif ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                return content['data']
            return content
    
    raise ValueError(f"Unsupported file format: {ext}")


def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def prepare_data(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """Prepare texts and labels from data."""
    texts = []
    labels = []
    
    for sample in data:
        text = sample.get('text', '')
        label = sample.get('sentiment', 'neutral')
        
        if text and label:
            texts.append(preprocess_text(text))
            labels.append(label)
    
    return texts, labels


# ============================================
# Model Training
# ============================================

class SentimentModelTrainer:
    """Train and evaluate sentiment classification models."""
    
    def __init__(self, vectorizer_type: str = 'tfidf'):
        self.vectorizer_type = vectorizer_type
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
        
        # Available models
        if SKLEARN_AVAILABLE:
            self.available_models = {
                'naive_bayes': MultinomialNB(),
                'svm': LinearSVC(max_iter=10000),
                'logistic_regression': LogisticRegression(max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, n_jobs=-1)
            }
    
    def create_vectorizer(self) -> object:
        """Create text vectorizer."""
        if self.vectorizer_type == 'tfidf':
            return TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            return CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
    
    def train(
        self,
        X_train: List[str],
        y_train: List[str],
        model_name: str = 'naive_bayes'
    ) -> object:
        """Train a single model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model training")
        
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create pipeline
        self.vectorizer = self.create_vectorizer()
        model = self.available_models[model_name]
        
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        self.models[model_name] = pipeline
        
        return pipeline
    
    def train_all(
        self,
        X_train: List[str],
        y_train: List[str]
    ) -> Dict[str, object]:
        """Train all available models."""
        results = {}
        
        for model_name in self.available_models:
            print(f"  Training {model_name}...")
            pipeline = self.train(X_train, y_train, model_name)
            results[model_name] = pipeline
        
        return results
    
    def evaluate(
        self,
        X_test: List[str],
        y_test: List[str],
        model_name: Optional[str] = None
    ) -> Dict:
        """Evaluate model(s) on test data."""
        results = {}
        
        models_to_evaluate = {model_name: self.models[model_name]} if model_name else self.models
        
        best_accuracy = 0
        
        for name, pipeline in models_to_evaluate.items():
            y_pred = pipeline.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = pipeline
                self.best_model_name = name
        
        return results
    
    def cross_validate(
        self,
        X: List[str],
        y: List[str],
        model_name: str = 'naive_bayes',
        cv: int = 5
    ) -> Dict:
        """Perform cross-validation."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        vectorizer = self.create_vectorizer()
        model = self.available_models[model_name]
        
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        return {
            'model': model_name,
            'cv_folds': cv,
            'scores': scores.tolist(),
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
    
    def predict(self, texts: List[str], model_name: Optional[str] = None) -> List[str]:
        """Make predictions on new texts."""
        if model_name:
            pipeline = self.models.get(model_name)
        else:
            pipeline = self.best_model
        
        if not pipeline:
            raise ValueError("No trained model available")
        
        # Preprocess texts
        processed_texts = [preprocess_text(t) for t in texts]
        
        return pipeline.predict(processed_texts).tolist()
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """Save trained model to file."""
        if model_name:
            pipeline = self.models.get(model_name)
        else:
            pipeline = self.best_model
            model_name = self.best_model_name
        
        if not pipeline:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': pipeline,
                'model_name': model_name,
                'saved_at': datetime.now().isoformat()
            }, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Tuple[object, str]:
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data['pipeline'], data['model_name']


# ============================================
# Main Function
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Sentiment Classification Models - RSK World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py --input ../data/sentiment_data.csv --model naive_bayes
  python train_model.py --train ../data/train_data.csv --test ../data/test_data.csv --all-models
  python train_model.py --input ../data/sentiment_data.csv --model svm --save --output ./models/

Author: Molla Samser (Founder) - RSK World
Website: https://rskworld.in
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input data file (will be split into train/test)"
    )
    
    parser.add_argument(
        "--train",
        type=str,
        help="Training data file"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        help="Test data file"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=['naive_bayes', 'svm', 'logistic_regression', 'random_forest'],
        default='naive_bayes',
        help="Model to train (default: naive_bayes)"
    )
    
    parser.add_argument(
        "--all-models", "-a",
        action="store_true",
        help="Train all available models"
    )
    
    parser.add_argument(
        "--split", "-s",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)"
    )
    
    parser.add_argument(
        "--vectorizer", "-v",
        type=str,
        choices=['tfidf', 'count'],
        default='tfidf',
        help="Vectorizer type (default: tfidf)"
    )
    
    parser.add_argument(
        "--cross-validate", "-cv",
        type=int,
        default=0,
        help="Number of cross-validation folds (0 = no CV)"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the best model"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models",
        help="Output directory for saved models"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         RSK World - Sentiment Model Trainer                      ║
║                                                                  ║
║  Author: Molla Samser (Founder)                                  ║
║  Website: https://rskworld.in                                    ║
║  © 2026 RSK World - Free Programming Resources & Source Code     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check for scikit-learn
    if not SKLEARN_AVAILABLE:
        print("❌ Error: scikit-learn is required for model training.")
        print("   Install with: pip install scikit-learn")
        return
    
    # Load data
    if args.input:
        print(f"Loading data from {args.input}...")
        data = load_data(args.input)
        texts, labels = prepare_data(data)
        
        print(f"Total samples: {len(texts)}")
        print(f"Splitting data ({args.split:.0%} train, {1-args.split:.0%} test)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, train_size=args.split, random_state=42, stratify=labels
        )
    
    elif args.train and args.test:
        print(f"Loading training data from {args.train}...")
        train_data = load_data(args.train)
        X_train, y_train = prepare_data(train_data)
        
        print(f"Loading test data from {args.test}...")
        test_data = load_data(args.test)
        X_test, y_test = prepare_data(test_data)
    
    else:
        print("❌ Error: Provide either --input or both --train and --test")
        return
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {set(y_train)}")
    print()
    
    # Initialize trainer
    trainer = SentimentModelTrainer(vectorizer_type=args.vectorizer)
    
    # Cross-validation if requested
    if args.cross_validate > 0:
        print(f"Performing {args.cross_validate}-fold cross-validation...")
        cv_results = trainer.cross_validate(X_train, y_train, args.model, args.cross_validate)
        print(f"  Mean accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
        print()
    
    # Train models
    if args.all_models:
        print("Training all models...")
        trainer.train_all(X_train, y_train)
    else:
        print(f"Training {args.model}...")
        trainer.train(X_train, y_train, args.model)
    
    print()
    
    # Evaluate
    print("Evaluating models...")
    print("=" * 60)
    
    results = trainer.evaluate(X_test, y_test)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"Accuracy: {result['accuracy']:.4f} ({result['accuracy']:.2%})")
        print()
        
        report = result['classification_report']
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 48)
        
        for cls in ['positive', 'neutral', 'negative']:
            if cls in report:
                m = report[cls]
                print(f"{cls:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1-score']:<12.4f}")
        
        print("-" * 48)
        print(f"{'Macro Avg':<12} {report['macro avg']['precision']:<12.4f} "
              f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
    
    # Print best model
    print()
    print("=" * 60)
    print(f"🏆 Best Model: {trainer.best_model_name}")
    print(f"   Accuracy: {results[trainer.best_model_name]['accuracy']:.2%}")
    
    # Save model if requested
    if args.save:
        os.makedirs(args.output, exist_ok=True)
        model_path = os.path.join(args.output, f'{trainer.best_model_name}_model.pkl')
        trainer.save_model(model_path)
    
    print()
    print("✓ Training complete!")
    print()


if __name__ == "__main__":
    main()

