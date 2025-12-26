#!/usr/bin/env python3
"""
================================================================================
 * Sentiment Analysis Dataset - Sentiment Analyzer Script
 * 
 * Project: Sentiment Analysis Dataset
 * Description: Analyze sentiment of text using multiple methods including
 *              rule-based, lexicon-based, and ML-based approaches.
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
    python analyze_sentiment.py --text "I love this product!"
    python analyze_sentiment.py --file ./data/sentiment_data.csv --evaluate
    python analyze_sentiment.py --interactive
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import List, Dict, Optional, Tuple
from collections import Counter

# Try to import optional libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


# ============================================
# Lexicon-Based Sentiment Analyzer
# ============================================

class LexiconSentimentAnalyzer:
    """Rule-based sentiment analyzer using word lexicons."""
    
    def __init__(self):
        # Positive words with weights
        self.positive_lexicon = {
            'excellent': 3, 'amazing': 3, 'outstanding': 3, 'perfect': 3,
            'fantastic': 3, 'wonderful': 3, 'brilliant': 3, 'superb': 3,
            'great': 2, 'good': 2, 'nice': 2, 'love': 2, 'like': 1,
            'happy': 2, 'satisfied': 2, 'pleased': 2, 'recommend': 2,
            'best': 3, 'awesome': 3, 'incredible': 3, 'impressive': 2,
            'beautiful': 2, 'delighted': 2, 'exceptional': 3, 'phenomenal': 3,
            'remarkable': 2, 'splendid': 2, 'terrific': 2, 'marvelous': 2,
            'enjoy': 1, 'helpful': 1, 'useful': 1, 'valuable': 2
        }
        
        # Negative words with weights
        self.negative_lexicon = {
            'terrible': -3, 'horrible': -3, 'awful': -3, 'worst': -3,
            'bad': -2, 'poor': -2, 'disappointing': -2, 'disappointed': -2,
            'hate': -3, 'dislike': -1, 'unhappy': -2, 'frustrated': -2,
            'broken': -2, 'defective': -2, 'useless': -2, 'waste': -2,
            'garbage': -3, 'trash': -3, 'scam': -3, 'fraud': -3,
            'regret': -2, 'avoid': -2, 'never': -1, 'angry': -2,
            'annoyed': -2, 'unacceptable': -2, 'pathetic': -3, 'dreadful': -3,
            'disgusting': -3, 'disaster': -3, 'nightmare': -3, 'rubbish': -3
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing',
            'nowhere', 'hardly', 'scarcely', 'barely', "n't"
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'absolutely': 2.0,
            'completely': 2.0, 'totally': 1.8, 'highly': 1.5, 'incredibly': 2.0,
            'remarkably': 1.8, 'exceptionally': 2.0, 'super': 1.5, 'so': 1.3
        }
        
        # Diminishers
        self.diminishers = {
            'slightly': 0.5, 'somewhat': 0.6, 'barely': 0.3, 'hardly': 0.3,
            'kind of': 0.5, 'sort of': 0.5, 'a bit': 0.5, 'a little': 0.5
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _check_negation(self, tokens: List[str], index: int, window: int = 3) -> bool:
        """Check if word is negated within window."""
        start = max(0, index - window)
        for i in range(start, index):
            if tokens[i] in self.negation_words:
                return True
        return False
    
    def _get_intensifier(self, tokens: List[str], index: int) -> float:
        """Get intensifier multiplier for word."""
        if index > 0:
            prev_word = tokens[index - 1]
            if prev_word in self.intensifiers:
                return self.intensifiers[prev_word]
            if prev_word in self.diminishers:
                return self.diminishers[prev_word]
        return 1.0
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of text."""
        tokens = self._tokenize(text)
        
        positive_score = 0
        negative_score = 0
        positive_words = []
        negative_words = []
        
        for i, token in enumerate(tokens):
            # Check positive words
            if token in self.positive_lexicon:
                weight = self.positive_lexicon[token]
                multiplier = self._get_intensifier(tokens, i)
                
                if self._check_negation(tokens, i):
                    negative_score += abs(weight) * multiplier
                    negative_words.append(f"not {token}")
                else:
                    positive_score += weight * multiplier
                    positive_words.append(token)
            
            # Check negative words
            elif token in self.negative_lexicon:
                weight = abs(self.negative_lexicon[token])
                multiplier = self._get_intensifier(tokens, i)
                
                if self._check_negation(tokens, i):
                    positive_score += weight * multiplier * 0.5  # Negated negative is weakly positive
                    positive_words.append(f"not {token}")
                else:
                    negative_score += weight * multiplier
                    negative_words.append(token)
        
        # Calculate final scores
        total = positive_score + negative_score + 0.001  # Avoid division by zero
        
        if positive_score > negative_score * 1.2:
            sentiment = "positive"
            confidence = positive_score / total
        elif negative_score > positive_score * 1.2:
            sentiment = "negative"
            confidence = negative_score / total
        else:
            sentiment = "neutral"
            confidence = 1 - abs(positive_score - negative_score) / total
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "positive_score": round(positive_score, 2),
            "negative_score": round(negative_score, 2),
            "positive_words": positive_words,
            "negative_words": negative_words
        }


# ============================================
# Ensemble Sentiment Analyzer
# ============================================

class EnsembleSentimentAnalyzer:
    """Combine multiple sentiment analysis methods."""
    
    def __init__(self):
        self.lexicon_analyzer = LexiconSentimentAnalyzer()
        
        # Initialize VADER if available
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_with_vader(self, text: str) -> Dict:
        """Analyze using VADER."""
        if not self.vader_analyzer:
            return None
        
        scores = self.vader_analyzer.polarity_scores(text)
        
        if scores['compound'] >= 0.05:
            sentiment = "positive"
        elif scores['compound'] <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "compound": scores['compound'],
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu']
        }
    
    def analyze_with_textblob(self, text: str) -> Dict:
        """Analyze using TextBlob."""
        if not TEXTBLOB_AVAILABLE:
            return None
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4)
        }
    
    def analyze(self, text: str, method: str = "ensemble") -> Dict:
        """Analyze sentiment using specified method."""
        results = {}
        
        # Lexicon-based analysis
        lexicon_result = self.lexicon_analyzer.analyze(text)
        results["lexicon"] = lexicon_result
        
        # VADER analysis
        if self.vader_analyzer:
            vader_result = self.analyze_with_vader(text)
            if vader_result:
                results["vader"] = vader_result
        
        # TextBlob analysis
        if TEXTBLOB_AVAILABLE:
            textblob_result = self.analyze_with_textblob(text)
            if textblob_result:
                results["textblob"] = textblob_result
        
        # Ensemble decision (voting)
        if method == "ensemble":
            sentiments = [r.get("sentiment") for r in results.values() if r]
            sentiment_counts = Counter(sentiments)
            final_sentiment = sentiment_counts.most_common(1)[0][0]
            
            results["ensemble"] = {
                "sentiment": final_sentiment,
                "votes": dict(sentiment_counts),
                "methods_used": list(results.keys())
            }
        
        return results


# ============================================
# Evaluation Functions
# ============================================

def evaluate_predictions(predictions: List[str], labels: List[str]) -> Dict:
    """Calculate evaluation metrics."""
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(labels)
    accuracy = correct / total if total > 0 else 0
    
    # Per-class metrics
    classes = ["positive", "neutral", "negative"]
    metrics = {}
    
    for cls in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    
    # Macro average
    macro_precision = sum(m["precision"] for m in metrics.values()) / len(classes)
    macro_recall = sum(m["recall"] for m in metrics.values()) / len(classes)
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(classes)
    
    return {
        "accuracy": round(accuracy, 4),
        "total_samples": total,
        "correct": correct,
        "per_class": metrics,
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4)
        }
    }


def load_data(filepath: str) -> List[Dict]:
    """Load data from file."""
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


# ============================================
# Interactive Mode
# ============================================

def interactive_mode(analyzer: EnsembleSentimentAnalyzer):
    """Run interactive sentiment analysis."""
    print("\n" + "="*60)
    print("Interactive Sentiment Analysis Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            text = input("Enter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                continue
            
            results = analyzer.analyze(text)
            
            print("\n" + "-"*40)
            print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print("-"*40)
            
            # Show lexicon results
            lex = results.get("lexicon", {})
            print(f"\n📊 Lexicon Analysis:")
            print(f"   Sentiment: {lex.get('sentiment', 'N/A').upper()}")
            print(f"   Confidence: {lex.get('confidence', 0):.2%}")
            print(f"   Positive words: {', '.join(lex.get('positive_words', [])) or 'None'}")
            print(f"   Negative words: {', '.join(lex.get('negative_words', [])) or 'None'}")
            
            # Show VADER results if available
            if "vader" in results:
                vader = results["vader"]
                print(f"\n📈 VADER Analysis:")
                print(f"   Sentiment: {vader.get('sentiment', 'N/A').upper()}")
                print(f"   Compound: {vader.get('compound', 0):.4f}")
            
            # Show TextBlob results if available
            if "textblob" in results:
                tb = results["textblob"]
                print(f"\n📝 TextBlob Analysis:")
                print(f"   Sentiment: {tb.get('sentiment', 'N/A').upper()}")
                print(f"   Polarity: {tb.get('polarity', 0):.4f}")
                print(f"   Subjectivity: {tb.get('subjectivity', 0):.4f}")
            
            # Show ensemble result
            if "ensemble" in results:
                ens = results["ensemble"]
                print(f"\n🎯 ENSEMBLE RESULT: {ens.get('sentiment', 'N/A').upper()}")
                print(f"   Votes: {ens.get('votes', {})}")
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


# ============================================
# Main Function
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analyzer - RSK World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sentiment.py --text "I love this product!"
  python analyze_sentiment.py --file ../data/sentiment_data.csv --evaluate
  python analyze_sentiment.py --interactive
  python analyze_sentiment.py --text "Terrible experience" --method vader

Author: Molla Samser (Founder) - RSK World
Website: https://rskworld.in
        """
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to analyze"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="File to analyze (CSV or JSON)"
    )
    
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Evaluate predictions against labels"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["lexicon", "vader", "textblob", "ensemble"],
        default="ensemble",
        help="Analysis method (default: ensemble)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           RSK World - Sentiment Analysis Tool                    ║
║                                                                  ║
║  Author: Molla Samser (Founder)                                  ║
║  Website: https://rskworld.in                                    ║
║  © 2026 RSK World - Free Programming Resources & Source Code     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Show available methods
    print("Available methods:")
    print(f"  ✓ Lexicon-based analyzer")
    print(f"  {'✓' if VADER_AVAILABLE else '✗'} VADER (vaderSentiment)")
    print(f"  {'✓' if TEXTBLOB_AVAILABLE else '✗'} TextBlob")
    print()
    
    # Initialize analyzer
    analyzer = EnsembleSentimentAnalyzer()
    
    # Interactive mode
    if args.interactive:
        interactive_mode(analyzer)
        return
    
    # Single text analysis
    if args.text:
        results = analyzer.analyze(args.text, args.method)
        
        print(f"Text: {args.text}")
        print("-" * 50)
        
        if args.method == "ensemble" or args.method == "lexicon":
            lex = results.get("lexicon", {})
            print(f"Sentiment: {lex.get('sentiment', 'N/A').upper()}")
            print(f"Confidence: {lex.get('confidence', 0):.2%}")
        
        if "ensemble" in results:
            print(f"\nEnsemble Result: {results['ensemble']['sentiment'].upper()}")
        
        return
    
    # File analysis
    if args.file:
        print(f"Loading data from {args.file}...")
        data = load_data(args.file)
        print(f"Loaded {len(data)} samples")
        print()
        
        predictions = []
        labels = []
        results_data = []
        
        print("Analyzing...")
        for i, sample in enumerate(data):
            text = sample.get('text', '')
            label = sample.get('sentiment', '')
            
            result = analyzer.analyze(text, args.method)
            
            if args.method == "ensemble" and "ensemble" in result:
                pred = result["ensemble"]["sentiment"]
            else:
                pred = result.get("lexicon", {}).get("sentiment", "neutral")
            
            predictions.append(pred)
            labels.append(label)
            
            results_data.append({
                "id": sample.get('id', i + 1),
                "text": text[:100],
                "actual": label,
                "predicted": pred,
                "correct": label == pred
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(data)}...")
        
        print()
        
        # Evaluate if requested
        if args.evaluate and labels:
            print("Evaluation Results:")
            print("=" * 50)
            
            metrics = evaluate_predictions(predictions, labels)
            
            print(f"Accuracy: {metrics['accuracy']:.2%}")
            print(f"Correct: {metrics['correct']}/{metrics['total_samples']}")
            print()
            
            print("Per-class metrics:")
            for cls, m in metrics['per_class'].items():
                print(f"  {cls.capitalize()}:")
                print(f"    Precision: {m['precision']:.4f}")
                print(f"    Recall: {m['recall']:.4f}")
                print(f"    F1: {m['f1']:.4f}")
            
            print()
            print("Macro Average:")
            print(f"  Precision: {metrics['macro_avg']['precision']:.4f}")
            print(f"  Recall: {metrics['macro_avg']['recall']:.4f}")
            print(f"  F1: {metrics['macro_avg']['f1']:.4f}")
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            print(f"\n✓ Results saved to {args.output}")
        
        return
    
    # If no arguments, show help
    parser.print_help()


if __name__ == "__main__":
    main()

