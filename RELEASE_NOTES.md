# 🎉 Sentiment Analysis Dataset v1.0.0

**Release Date:** December 26, 2026  
**Author:** Molla Samser (Founder) - RSK World  
**Website:** [https://rskworld.in](https://rskworld.in)

---

## 📦 What's Included

### 📊 Dataset Files
| File | Description | Format |
|------|-------------|--------|
| `data/sentiment_data.csv` | Main dataset with all samples | CSV |
| `data/sentiment_data.json` | Main dataset with metadata | JSON |
| `data/sentiment_data.txt` | Plain text format | TXT |
| `data/train_data.csv` | Training set (80%) | CSV |
| `data/test_data.csv` | Test set (20%) | CSV |

### 🔧 Preprocessed Data
| File | Description |
|------|-------------|
| `preprocessed/cleaned_data.csv` | Cleaned and normalized text |
| `preprocessed/tokenized_data.json` | Tokenized data for NLP models |

### 🐍 Python Scripts
| Script | Description |
|--------|-------------|
| `generate_data.py` | Generate unlimited synthetic sentiment data |
| `preprocess_data.py` | Clean, tokenize, and prepare text |
| `analyze_sentiment.py` | Multi-method sentiment analysis |
| `visualize_data.py` | Generate charts and HTML reports |
| `train_model.py` | Train ML models (4 algorithms) |

---

## ✨ Key Features

### 🚀 Advanced Capabilities
- ✅ **Unlimited Data Generation** - Generate 1,000 to 1,000,000+ samples
- ✅ **4 ML Algorithms** - Naive Bayes, SVM, Logistic Regression, Random Forest
- ✅ **Ensemble Analysis** - Combine Lexicon, VADER, and TextBlob methods
- ✅ **Interactive Mode** - Real-time sentiment analysis
- ✅ **Auto Reports** - Generate HTML visualizations automatically
- ✅ **Model Persistence** - Save and load trained models

### 📈 Dataset Statistics
- **Total Samples:** 50+ (expandable with generator)
- **Sentiment Classes:** 3 (Positive, Neutral, Negative)
- **Text Sources:** Product Reviews, Social Media, Customer Feedback
- **Languages:** English

---

## 🛠️ Quick Start

```bash
# Clone the repository
git clone https://github.com/rskworld/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
cd scripts
pip install -r requirements.txt

# Generate 5000 samples
python generate_data.py --samples 5000 --balanced

# Train all models
python train_model.py --input ../data/sentiment_data.csv --all-models --save

# Interactive sentiment analysis
python analyze_sentiment.py --interactive
```

---

## 📸 Demo Page

Visit the interactive demo page to explore the dataset:
- Filter by sentiment (Positive/Neutral/Negative)
- View dataset statistics
- Step-by-step usage guide
- Python scripts documentation

---

## 🙏 Credits

| Role | Name |
|------|------|
| **Author/Founder** | Molla Samser |
| **Designer & Tester** | Rima Khatun |
| **Website** | [rskworld.in](https://rskworld.in) |
| **Email** | help@rskworld.in |
| **Phone** | +91 93305 39277 |

---

## 📄 License

This dataset is provided for **educational purposes only**.

---

## 🔗 Links

- 🌐 **Website:** [https://rskworld.in](https://rskworld.in)
- 📧 **Email:** help@rskworld.in
- 📞 **Phone:** +91 93305 39277

---

<p align="center">
  <strong>© 2026 RSK World - Free Programming Resources & Source Code</strong>
</p>

