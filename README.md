<!--
================================================================================
 * Sentiment Analysis Dataset - README
 * 
 * Project: Sentiment Analysis Dataset
 * Description: Text sentiment analysis dataset with labeled reviews, comments,
 *              and social media posts for sentiment classification models.
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
-->

# 📊 Sentiment Analysis Dataset

[![RSK World](https://img.shields.io/badge/RSK-World-dc3545)](https://rskworld.in)
[![Category](https://img.shields.io/badge/Category-Text%20Data-blue)](https://rskworld.in)
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow)](https://rskworld.in)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-Educational-green)](https://rskworld.in)

Text sentiment analysis dataset with labeled reviews, comments, and social media posts for sentiment classification models. Includes **Python scripts for data generation, preprocessing, analysis, visualization, and model training**.

## 🌟 Features

- ✅ **Labeled Sentiment Data** - Pre-labeled text with positive, negative, and neutral classifications
- ✅ **Multiple Text Sources** - Diverse collection from product reviews, social media, and comments
- ✅ **Training & Test Sets** - Pre-split datasets ready for ML model development
- ✅ **Preprocessed Versions** - Cleaned and tokenized data ready for NLP pipelines
- ✅ **Ready for NLP Models** - Compatible with NLTK, spaCy, and popular frameworks
- ✅ **Python Scripts** - Generate unlimited data, train models, visualize results
- ✅ **Interactive Demo** - Beautiful web interface to explore the dataset

## 📁 Project Structure

```
sentiment-analysis/
├── 📂 data/
│   ├── sentiment_data.csv      # Main dataset (CSV)
│   ├── sentiment_data.json     # Main dataset (JSON)
│   ├── sentiment_data.txt      # Main dataset (TXT)
│   ├── train_data.csv          # Training set (80%)
│   └── test_data.csv           # Test set (20%)
├── 📂 preprocessed/
│   ├── cleaned_data.csv        # Cleaned/normalized text
│   └── tokenized_data.json     # Tokenized data for NLP
├── 📂 scripts/                  # 🆕 Python Scripts
│   ├── generate_data.py        # Generate unlimited synthetic data
│   ├── preprocess_data.py      # Preprocess and clean text
│   ├── analyze_sentiment.py    # Analyze sentiment with multiple methods
│   ├── visualize_data.py       # Generate charts and visualizations
│   ├── train_model.py          # Train ML models
│   └── requirements.txt        # Python dependencies
├── 📂 css/
│   └── styles.css              # Demo page styles
├── 📂 js/
│   └── script.js               # Demo page scripts
├── index.html                  # Interactive demo page
├── README.md                   # This file
└── LICENSE                     # License information
```

## 🚀 Quick Start

### 1️⃣ Install Python Dependencies

```bash
cd scripts
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2️⃣ Generate Custom Dataset

```bash
# Generate 1000 samples
python generate_data.py --samples 1000

# Generate 5000 balanced samples in all formats
python generate_data.py --samples 5000 --balanced --all-formats

# Generate with train/test split
python generate_data.py --samples 10000 --split 0.8 --output ../data/custom
```

### 3️⃣ Preprocess Data

```bash
# Basic preprocessing
python preprocess_data.py --input ../data/sentiment_data.csv

# Advanced preprocessing with lemmatization
python preprocess_data.py --input ../data/sentiment_data.csv --lemmatize --remove-stopwords

# Build vocabulary
python preprocess_data.py --input ../data/sentiment_data.csv --build-vocab
```

### 4️⃣ Analyze Sentiment

```bash
# Analyze single text
python analyze_sentiment.py --text "I love this product!"

# Interactive mode
python analyze_sentiment.py --interactive

# Evaluate on dataset
python analyze_sentiment.py --file ../data/sentiment_data.csv --evaluate
```

### 5️⃣ Visualize Data

```bash
# Generate all charts
python visualize_data.py --input ../data/sentiment_data.csv --all-charts

# Generate HTML report
python visualize_data.py --input ../data/sentiment_data.csv --html-report
```

### 6️⃣ Train ML Models

```bash
# Train Naive Bayes model
python train_model.py --input ../data/sentiment_data.csv --model naive_bayes

# Train all models and save best
python train_model.py --input ../data/sentiment_data.csv --all-models --save

# Train with custom train/test files
python train_model.py --train ../data/train_data.csv --test ../data/test_data.csv --model svm
```

## 📜 Python Scripts Reference

### `generate_data.py` - Data Generator

Generate synthetic sentiment analysis data with customizable parameters.

```bash
python generate_data.py [OPTIONS]

Options:
  -n, --samples        Number of samples to generate (default: 1000)
  -o, --output         Output file path (without extension)
  -f, --format         Output format: csv, json, txt, all
  -b, --balanced       Generate balanced dataset
  -s, --split          Train/test split ratio (e.g., 0.8)
  -m, --include-metadata  Include metadata in samples
  -a, --all-formats    Export in all formats
  --seed               Random seed for reproducibility
```

### `preprocess_data.py` - Data Preprocessor

Clean and preprocess text data for NLP models.

```bash
python preprocess_data.py [OPTIONS]

Options:
  -i, --input          Input file path (required)
  -o, --output         Output directory
  -l, --lowercase      Convert to lowercase
  -s, --remove-stopwords  Remove stopwords
  -p, --remove-punctuation  Remove punctuation
  --lemmatize          Apply lemmatization
  --stem               Apply stemming
  -f, --extract-features  Extract sentiment features
  -v, --build-vocab    Build vocabulary file
```

### `analyze_sentiment.py` - Sentiment Analyzer

Analyze sentiment using multiple methods (lexicon-based, VADER, TextBlob).

```bash
python analyze_sentiment.py [OPTIONS]

Options:
  -t, --text           Text to analyze
  -f, --file           File to analyze
  -e, --evaluate       Evaluate predictions against labels
  -m, --method         Analysis method: lexicon, vader, textblob, ensemble
  -i, --interactive    Run in interactive mode
  -o, --output         Output file for results
```

### `visualize_data.py` - Data Visualizer

Generate charts, word clouds, and statistical reports.

```bash
python visualize_data.py [OPTIONS]

Options:
  -i, --input          Input file path (required)
  -o, --output         Output directory for charts
  -a, --all-charts     Generate all available charts
  -s, --stats-only     Only print statistics
  -r, --html-report    Generate HTML report
```

### `train_model.py` - Model Trainer

Train and evaluate machine learning models for sentiment classification.

```bash
python train_model.py [OPTIONS]

Options:
  -i, --input          Input data file
  --train              Training data file
  --test               Test data file
  -m, --model          Model: naive_bayes, svm, logistic_regression, random_forest
  -a, --all-models     Train all available models
  -s, --split          Train/test split ratio
  -v, --vectorizer     Vectorizer: tfidf, count
  -cv, --cross-validate  Cross-validation folds
  --save               Save the best model
  -o, --output         Output directory for models
```

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 50+ (expandable with generator) |
| Sentiment Classes | 3 (Positive, Neutral, Negative) |
| Text Sources | 3 (Reviews, Social Media, Comments) |
| Avg. Text Length | ~142 characters |
| File Formats | CSV, JSON, TXT |
| Languages | English |

## 🏷️ Sentiment Distribution

- **Positive**: ~37% of samples
- **Neutral**: ~30% of samples  
- **Negative**: ~33% of samples

## 🛠️ Technologies & Dependencies

### Core Technologies
| Technology | Description |
|------------|-------------|
| CSV | Standard comma-separated values format |
| JSON | JavaScript Object Notation format |
| TXT | Plain text format |
| Python 3.8+ | Programming language |

### Python Libraries
| Library | Purpose |
|---------|---------|
| NLTK | Natural language processing |
| spaCy | Industrial NLP |
| TextBlob | Simple sentiment analysis |
| VADER | Sentiment analysis |
| scikit-learn | Machine learning |
| matplotlib | Data visualization |
| wordcloud | Word cloud generation |

## 📖 Usage Examples

### Loading Data in Python

```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv('data/sentiment_data.csv', comment='#')

# Load JSON
with open('data/sentiment_data.json', 'r') as f:
    data = json.load(f)
    samples = data['data']
```

### Training a Custom Model

```python
from scripts.train_model import SentimentModelTrainer, load_data, prepare_data

# Load data
data = load_data('data/sentiment_data.csv')
texts, labels = prepare_data(data)

# Train model
trainer = SentimentModelTrainer()
trainer.train(texts[:80], labels[:80], 'svm')

# Evaluate
results = trainer.evaluate(texts[80:], labels[80:])
print(f"Accuracy: {results['svm']['accuracy']:.2%}")
```

### Interactive Sentiment Analysis

```python
from scripts.analyze_sentiment import EnsembleSentimentAnalyzer

analyzer = EnsembleSentimentAnalyzer()

# Analyze text
result = analyzer.analyze("This product is absolutely amazing!")
print(f"Sentiment: {result['ensemble']['sentiment']}")
```

## 📜 License

This dataset is provided for **educational purposes only**. 

See the [LICENSE](LICENSE) file for more details.

## 👨‍💻 Author

**Molla Samser** - Founder of RSK World

- 🌐 Website: [https://rskworld.in](https://rskworld.in)
- 📧 Email: help@rskworld.in
- 📞 Phone: +91 93305 39277

### Design & Testing

**Rima Khatun** - Designer & Tester at RSK World

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## 📞 Contact

For questions, suggestions, or support:

- **General Inquiries**: info@rskworld.in
- **Support**: support@rskworld.in
- **Website**: [https://rskworld.in/contact.php](https://rskworld.in/contact.php)

## ⭐ Support

If you find this dataset helpful, please consider:

- ⭐ Starring this repository
- 📢 Sharing with others
- 🔗 Linking back to RSK World

---

<p align="center">
  <strong>© 2026 RSK World - Free Programming Resources & Source Code</strong><br>
  Founded by <strong>Molla Samser</strong> | Designed by <strong>Rima Khatun</strong>
</p>

<p align="center">
  <a href="https://rskworld.in">Website</a> •
  <a href="https://rskworld.in/about.php">About</a> •
  <a href="https://rskworld.in/contact.php">Contact</a>
</p>
