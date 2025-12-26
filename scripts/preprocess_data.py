#!/usr/bin/env python3
"""
================================================================================
 * Sentiment Analysis Dataset - Data Preprocessing Script
 * 
 * Project: Sentiment Analysis Dataset
 * Description: Preprocess text data for NLP models - cleaning, tokenization,
 *              normalization, and feature extraction.
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
    python preprocess_data.py --input ./data/sentiment_data.csv --output ./preprocessed/
    python preprocess_data.py --input ./data/sentiment_data.json --lowercase --remove-stopwords
    python preprocess_data.py --input ./data/ --batch --lemmatize
"""

import argparse
import csv
import json
import os
import re
import string
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import Counter

# Try to import optional NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ============================================
# Text Preprocessing Functions
# ============================================

class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        expand_contractions: bool = True,
        lemmatize: bool = False,
        stem: bool = False,
        min_word_length: int = 1,
        max_word_length: int = 50
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.expand_contractions = expand_contractions
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Initialize NLTK components if available
        self.stop_words = set()
        self.stemmer = None
        self.lemmatizer = None
        
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("Downloading NLTK wordnet...")
                nltk.download('wordnet', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        
        # Common contractions
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
            "'ve": " have", "'m": " am", "let's": "let us",
            "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'd": "i would",
            "you'd": "you would", "he'd": "he would", "she'd": "she would",
            "we'd": "we would", "they'd": "they would", "i'll": "i will",
            "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "doesn't": "does not", "don't": "do not", "didn't": "did not",
            "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not", "mustn't": "must not"
        }
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps to text."""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions (@username)
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word, remove #)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Expand contractions
        if self.expand_contractions:
            for contraction, expansion in self.contractions.items():
                text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        else:
            tokens = text.split()
        
        # Filter by word length
        tokens = [t for t in tokens if self.min_word_length <= len(t) <= self.max_word_length]
        
        # Remove stopwords
        if self.remove_stopwords and self.stop_words:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
        
        # Stemming
        if self.stem and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Lemmatization
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def process(self, text: str) -> Dict:
        """Process text and return cleaned version with metadata."""
        original_text = text
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        
        return {
            "original": original_text,
            "cleaned": cleaned_text,
            "tokens": tokens,
            "word_count": len(tokens),
            "char_count": len(cleaned_text),
            "avg_word_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        }


# ============================================
# Feature Extraction
# ============================================

class FeatureExtractor:
    """Extract features from preprocessed text."""
    
    def __init__(self):
        # Sentiment lexicons (simplified)
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'best', 'love', 'perfect', 'happy', 'beautiful', 'nice',
            'brilliant', 'outstanding', 'superb', 'incredible', 'impressive',
            'recommend', 'satisfied', 'delighted', 'pleased', 'exceptional'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
            'disappointing', 'disappointed', 'broken', 'defective', 'useless',
            'waste', 'garbage', 'scam', 'fraud', 'regret', 'avoid', 'never',
            'unhappy', 'frustrated', 'angry', 'annoyed', 'unacceptable'
        }
        
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
            'hardly', 'scarcely', 'barely', "n't", 'cannot', "can't", "won't"
        }
        
        self.intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely', 'totally',
            'highly', 'incredibly', 'remarkably', 'exceptionally', 'super'
        }
    
    def extract_features(self, tokens: List[str]) -> Dict:
        """Extract sentiment-related features from tokens."""
        tokens_lower = [t.lower() for t in tokens]
        
        positive_count = sum(1 for t in tokens_lower if t in self.positive_words)
        negative_count = sum(1 for t in tokens_lower if t in self.negative_words)
        negation_count = sum(1 for t in tokens_lower if t in self.negation_words)
        intensifier_count = sum(1 for t in tokens_lower if t in self.intensifiers)
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / (len(tokens) + 1)
        
        # Adjust for negations
        if negation_count > 0:
            sentiment_score *= -0.5
        
        return {
            "positive_word_count": positive_count,
            "negative_word_count": negative_count,
            "negation_count": negation_count,
            "intensifier_count": intensifier_count,
            "sentiment_score": round(sentiment_score, 4),
            "exclamation_count": sum(1 for t in tokens if '!' in t),
            "question_count": sum(1 for t in tokens if '?' in t),
            "caps_ratio": sum(1 for t in tokens if t.isupper()) / (len(tokens) + 1)
        }


# ============================================
# Data Loading and Saving
# ============================================

def load_data(filepath: str) -> List[Dict]:
    """Load data from CSV or JSON file."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip comment lines
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
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_preprocessed_csv(data: List[Dict], filepath: str):
    """Save preprocessed data to CSV."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        f.write("""# ================================================================================
# Sentiment Analysis Dataset - Preprocessed Data
# 
# Project: Sentiment Analysis Dataset
# Preprocessed by: RSK World Text Preprocessor
# Website: https://rskworld.in
# 
# Author: Molla Samser (Founder)
# Designer & Tester: Rima Khatun
# Email: help@rskworld.in | support@rskworld.in
# 
# © 2026 RSK World - Free Programming Resources & Source Code
# ================================================================================

""")
        
        if data:
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    print(f"✓ Saved preprocessed data to {filepath}")


def save_preprocessed_json(data: List[Dict], filepath: str, metadata: Dict = None):
    """Save preprocessed data to JSON."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    output = {
        "_metadata": {
            "project": "Sentiment Analysis Dataset",
            "description": "Preprocessed sentiment analysis data",
            "preprocessor": "RSK World Text Preprocessor",
            "website": "https://rskworld.in",
            "author": "Molla Samser (Founder)",
            "designer_tester": "Rima Khatun",
            "email": "help@rskworld.in | support@rskworld.in",
            "copyright": "© 2026 RSK World - Free Programming Resources & Source Code",
            "processed_at": datetime.now().isoformat(),
            "total_samples": len(data),
            **(metadata or {})
        },
        "data": data
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved preprocessed data to {filepath}")


# ============================================
# Main Processing Pipeline
# ============================================

def process_dataset(
    input_path: str,
    output_dir: str,
    preprocessor: TextPreprocessor,
    feature_extractor: FeatureExtractor,
    extract_features: bool = True
) -> List[Dict]:
    """Process entire dataset."""
    
    print(f"Loading data from {input_path}...")
    data = load_data(input_path)
    print(f"Loaded {len(data)} samples")
    
    processed_data = []
    cleaned_data = []
    tokenized_data = []
    
    print("Processing...")
    for i, sample in enumerate(data):
        text = sample.get('text', '')
        result = preprocessor.process(text)
        
        # Cleaned data entry
        cleaned_entry = {
            "id": sample.get('id', i + 1),
            "cleaned_text": result['cleaned'],
            "sentiment": sample.get('sentiment', ''),
            "word_count": result['word_count']
        }
        
        # Tokenized data entry
        tokenized_entry = {
            "id": sample.get('id', i + 1),
            "tokens": result['tokens'],
            "sentiment": sample.get('sentiment', ''),
            "label": {"positive": 2, "neutral": 1, "negative": 0}.get(sample.get('sentiment', ''), 1)
        }
        
        # Add features if requested
        if extract_features:
            features = feature_extractor.extract_features(result['tokens'])
            cleaned_entry.update(features)
            tokenized_entry['features'] = features
        
        cleaned_data.append(cleaned_entry)
        tokenized_data.append(tokenized_entry)
        
        # Progress
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples...")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_csv_path = os.path.join(output_dir, 'cleaned_data.csv')
    tokenized_json_path = os.path.join(output_dir, 'tokenized_data.json')
    
    save_preprocessed_csv(cleaned_data, cleaned_csv_path)
    save_preprocessed_json(tokenized_data, tokenized_json_path, {
        "preprocessing": {
            "lowercase": preprocessor.lowercase,
            "remove_punctuation": preprocessor.remove_punctuation,
            "remove_stopwords": preprocessor.remove_stopwords,
            "lemmatize": preprocessor.lemmatize,
            "stem": preprocessor.stem
        },
        "label_mapping": {"0": "negative", "1": "neutral", "2": "positive"}
    })
    
    return cleaned_data, tokenized_data


# ============================================
# Vocabulary Builder
# ============================================

def build_vocabulary(tokenized_data: List[Dict], min_freq: int = 2) -> Dict:
    """Build vocabulary from tokenized data."""
    word_counts = Counter()
    
    for sample in tokenized_data:
        tokens = sample.get('tokens', [])
        word_counts.update(tokens)
    
    # Filter by minimum frequency
    vocab = {word: count for word, count in word_counts.items() if count >= min_freq}
    
    # Create word to index mapping
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted(vocab.keys()):
        word2idx[word] = len(word2idx)
    
    return {
        "vocabulary_size": len(word2idx),
        "word_counts": dict(word_counts.most_common(1000)),
        "word2idx": word2idx
    }


# ============================================
# Main Function
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess sentiment analysis data - RSK World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_data.py --input ../data/sentiment_data.csv
  python preprocess_data.py --input ../data/sentiment_data.json --lowercase --remove-stopwords
  python preprocess_data.py --input ../data/sentiment_data.csv --lemmatize --extract-features

Author: Molla Samser (Founder) - RSK World
Website: https://rskworld.in
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input file path (CSV or JSON)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./preprocessed",
        help="Output directory (default: ./preprocessed)"
    )
    
    parser.add_argument(
        "--lowercase", "-l",
        action="store_true",
        default=True,
        help="Convert text to lowercase (default: True)"
    )
    
    parser.add_argument(
        "--remove-stopwords", "-s",
        action="store_true",
        help="Remove stopwords"
    )
    
    parser.add_argument(
        "--remove-punctuation", "-p",
        action="store_true",
        default=True,
        help="Remove punctuation (default: True)"
    )
    
    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="Apply lemmatization"
    )
    
    parser.add_argument(
        "--stem",
        action="store_true",
        help="Apply stemming"
    )
    
    parser.add_argument(
        "--extract-features", "-f",
        action="store_true",
        default=True,
        help="Extract sentiment features (default: True)"
    )
    
    parser.add_argument(
        "--build-vocab", "-v",
        action="store_true",
        help="Build vocabulary file"
    )
    
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum word frequency for vocabulary (default: 2)"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         RSK World - Sentiment Analysis Data Preprocessor         ║
║                                                                  ║
║  Author: Molla Samser (Founder)                                  ║
║  Website: https://rskworld.in                                    ║
║  © 2026 RSK World - Free Programming Resources & Source Code     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check for NLTK
    if not NLTK_AVAILABLE:
        print("⚠ NLTK not installed. Some features may be limited.")
        print("  Install with: pip install nltk")
        print()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_stopwords=args.remove_stopwords,
        lemmatize=args.lemmatize,
        stem=args.stem
    )
    
    feature_extractor = FeatureExtractor()
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Settings:")
    print(f"  - Lowercase: {args.lowercase}")
    print(f"  - Remove punctuation: {args.remove_punctuation}")
    print(f"  - Remove stopwords: {args.remove_stopwords}")
    print(f"  - Lemmatize: {args.lemmatize}")
    print(f"  - Stem: {args.stem}")
    print(f"  - Extract features: {args.extract_features}")
    print()
    
    # Process dataset
    cleaned_data, tokenized_data = process_dataset(
        args.input,
        args.output,
        preprocessor,
        feature_extractor,
        args.extract_features
    )
    
    # Build vocabulary if requested
    if args.build_vocab:
        print()
        print("Building vocabulary...")
        vocab = build_vocabulary(tokenized_data, args.min_freq)
        
        vocab_path = os.path.join(args.output, 'vocabulary.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2)
        print(f"✓ Saved vocabulary to {vocab_path}")
        print(f"  Vocabulary size: {vocab['vocabulary_size']}")
    
    print()
    print("✓ Preprocessing complete!")
    print()


if __name__ == "__main__":
    main()

