#!/usr/bin/env python3
"""
================================================================================
 * Sentiment Analysis Dataset - Data Visualization Script
 * 
 * Project: Sentiment Analysis Dataset
 * Description: Generate visualizations and statistics for sentiment analysis
 *              datasets including distribution charts, word clouds, and more.
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
    python visualize_data.py --input ./data/sentiment_data.csv
    python visualize_data.py --input ./data/sentiment_data.json --output ./charts/
    python visualize_data.py --input ./data/ --all-charts --interactive
"""

import argparse
import csv
import json
import os
import re
from typing import List, Dict, Optional
from collections import Counter
from datetime import datetime

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================
# Data Loading
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


# ============================================
# Statistics Functions
# ============================================

def calculate_statistics(data: List[Dict]) -> Dict:
    """Calculate comprehensive statistics for the dataset."""
    total = len(data)
    
    # Sentiment distribution
    sentiments = [d.get('sentiment', 'unknown') for d in data]
    sentiment_counts = Counter(sentiments)
    
    # Source distribution
    sources = [d.get('source', 'unknown') for d in data]
    source_counts = Counter(sources)
    
    # Text length statistics
    text_lengths = [len(d.get('text', '')) for d in data]
    word_counts = [len(d.get('text', '').split()) for d in data]
    
    avg_text_length = sum(text_lengths) / total if total > 0 else 0
    avg_word_count = sum(word_counts) / total if total > 0 else 0
    min_text_length = min(text_lengths) if text_lengths else 0
    max_text_length = max(text_lengths) if text_lengths else 0
    
    # Date range
    dates = [d.get('date', '') for d in data if d.get('date')]
    
    return {
        "total_samples": total,
        "sentiment_distribution": dict(sentiment_counts),
        "source_distribution": dict(source_counts),
        "text_statistics": {
            "avg_length": round(avg_text_length, 2),
            "avg_word_count": round(avg_word_count, 2),
            "min_length": min_text_length,
            "max_length": max_text_length
        },
        "date_range": {
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None
        }
    }


def get_top_words(data: List[Dict], sentiment: Optional[str] = None, top_n: int = 50) -> List[tuple]:
    """Get most frequent words, optionally filtered by sentiment."""
    words = []
    
    for sample in data:
        if sentiment and sample.get('sentiment') != sentiment:
            continue
        
        text = sample.get('text', '').lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words.extend(text.split())
    
    # Remove common stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'in', 'for', 'on',
        'with', 'as', 'was', 'that', 'this', 'i', 'my', 'but', 'have', 'has',
        'be', 'are', 'been', 'will', 'would', 'could', 'should', 'from', 'at',
        'or', 'by', 'so', 'if', 'just', 'what', 'all', 'were', 'we', 'they'
    }
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(top_n)


# ============================================
# Visualization Functions
# ============================================

def set_style():
    """Set matplotlib style for RSK World branding."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d0d0d',
        'axes.facecolor': '#1a1a1a',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#ffffff',
        'text.color': '#ffffff',
        'xtick.color': '#b3b3b3',
        'ytick.color': '#b3b3b3',
        'grid.color': '#333333',
        'font.family': 'sans-serif',
        'font.size': 10
    })


def plot_sentiment_distribution(data: List[Dict], output_path: str):
    """Create sentiment distribution pie chart."""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ Matplotlib not installed. Skipping chart generation.")
        return
    
    set_style()
    
    sentiments = [d.get('sentiment', 'unknown') for d in data]
    counts = Counter(sentiments)
    
    labels = list(counts.keys())
    values = list(counts.values())
    
    # RSK World color scheme
    colors = {
        'positive': '#28a745',
        'neutral': '#ffc107',
        'negative': '#dc3545',
        'unknown': '#6c757d'
    }
    chart_colors = [colors.get(l, '#6c757d') for l in labels]
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0d0d0d')
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        colors=chart_colors,
        explode=[0.02] * len(labels),
        shadow=True,
        startangle=90
    )
    
    # Style the text
    for text in texts:
        text.set_color('white')
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Sentiment Distribution\nRSK World - Sentiment Analysis Dataset', 
                 fontsize=14, fontweight='bold', color='white', pad=20)
    
    # Add legend
    ax.legend(wedges, [f'{l.capitalize()}: {v}' for l, v in zip(labels, values)],
              loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d0d0d', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved sentiment distribution chart to {output_path}")


def plot_source_distribution(data: List[Dict], output_path: str):
    """Create source distribution bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    set_style()
    
    sources = [d.get('source', 'unknown') for d in data]
    counts = Counter(sources)
    
    labels = list(counts.keys())
    values = list(counts.values())
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d0d0d')
    
    # Create gradient-like colors
    colors = ['#dc3545', '#e35d6a', '#e8838e', '#eda9b2', '#f2ced6'][:len(labels)]
    
    bars = ax.bar(labels, values, color=colors, edgecolor='#333333', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='white', fontweight='bold')
    
    ax.set_xlabel('Data Source', fontsize=12, color='white')
    ax.set_ylabel('Number of Samples', fontsize=12, color='white')
    ax.set_title('Data Source Distribution\nRSK World - Sentiment Analysis Dataset',
                 fontsize=14, fontweight='bold', color='white', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d0d0d', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved source distribution chart to {output_path}")


def plot_text_length_histogram(data: List[Dict], output_path: str):
    """Create text length histogram."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    set_style()
    
    # Separate by sentiment
    lengths_by_sentiment = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
    
    for sample in data:
        sentiment = sample.get('sentiment', 'neutral')
        length = len(sample.get('text', '').split())
        if sentiment in lengths_by_sentiment:
            lengths_by_sentiment[sentiment].append(length)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d0d0d')
    
    colors = {'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'}
    
    for sentiment, lengths in lengths_by_sentiment.items():
        if lengths:
            ax.hist(lengths, bins=20, alpha=0.6, label=sentiment.capitalize(),
                   color=colors[sentiment], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Word Count', fontsize=12, color='white')
    ax.set_ylabel('Frequency', fontsize=12, color='white')
    ax.set_title('Text Length Distribution by Sentiment\nRSK World - Sentiment Analysis Dataset',
                 fontsize=14, fontweight='bold', color='white', pad=20)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d0d0d', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved text length histogram to {output_path}")


def plot_word_frequency(data: List[Dict], output_path: str, top_n: int = 20):
    """Create word frequency bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    set_style()
    
    top_words = get_top_words(data, top_n=top_n)
    
    words = [w for w, c in top_words]
    counts = [c for w, c in top_words]
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0d0d0d')
    
    # Create horizontal bar chart
    y_pos = range(len(words))
    bars = ax.barh(y_pos, counts, color='#dc3545', edgecolor='#333333', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.annotate(f'{count}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    color='white', fontsize=9)
    
    ax.set_xlabel('Frequency', fontsize=12, color='white')
    ax.set_title(f'Top {top_n} Most Frequent Words\nRSK World - Sentiment Analysis Dataset',
                 fontsize=14, fontweight='bold', color='white', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0d0d0d', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved word frequency chart to {output_path}")


def generate_wordcloud(data: List[Dict], output_path: str, sentiment: Optional[str] = None):
    """Generate word cloud."""
    if not WORDCLOUD_AVAILABLE:
        print("⚠ WordCloud not installed. Skipping word cloud generation.")
        return
    
    # Collect text
    texts = []
    for sample in data:
        if sentiment and sample.get('sentiment') != sentiment:
            continue
        texts.append(sample.get('text', ''))
    
    text = ' '.join(texts)
    
    # Color based on sentiment
    if sentiment == 'positive':
        colormap = 'Greens'
    elif sentiment == 'negative':
        colormap = 'Reds'
    else:
        colormap = 'Blues'
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='#0d0d0d',
        colormap=colormap,
        max_words=100,
        min_font_size=10,
        max_font_size=150
    ).generate(text)
    
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d0d0d')
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        title = f'Word Cloud - {sentiment.capitalize() if sentiment else "All"} Sentiment'
        ax.set_title(f'{title}\nRSK World - Sentiment Analysis Dataset',
                     fontsize=14, fontweight='bold', color='white', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='#0d0d0d', edgecolor='none', bbox_inches='tight')
        plt.close()
    else:
        wordcloud.to_file(output_path)
    
    print(f"✓ Saved word cloud to {output_path}")


def generate_html_report(data: List[Dict], stats: Dict, output_path: str, chart_dir: str):
    """Generate an HTML report with embedded statistics."""
    
    html_content = f"""<!DOCTYPE html>
<!--
================================================================================
 * Sentiment Analysis Dataset - Statistics Report
 * 
 * Author: Molla Samser (Founder)
 * Designer & Tester: Rima Khatun
 * Website: https://rskworld.in
 * © 2026 RSK World - Free Programming Resources & Source Code
================================================================================
-->
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Statistics Report - RSK World</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a0a0a 0%, #0d0d0d 50%, #0a0a1a 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 40px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: rgba(220, 53, 69, 0.1);
            border-radius: 15px;
            border: 1px solid rgba(220, 53, 69, 0.3);
        }}
        .header h1 {{ color: #dc3545; font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ color: #b3b3b3; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: #1e1e1e;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #333;
            text-align: center;
        }}
        .stat-card h3 {{ color: #dc3545; font-size: 2em; margin-bottom: 10px; }}
        .stat-card p {{ color: #b3b3b3; }}
        .section {{
            background: #1e1e1e;
            padding: 30px;
            border-radius: 12px;
            border: 1px solid #333;
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #dc3545;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        .chart-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart {{ background: #0d0d0d; padding: 15px; border-radius: 8px; }}
        .chart img {{ width: 100%; height: auto; border-radius: 5px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ color: #dc3545; background: #0d0d0d; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            margin-top: 40px;
        }}
        .footer a {{ color: #dc3545; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dataset Statistics Report</h1>
            <p>Sentiment Analysis Dataset - RSK World</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{stats['total_samples']:,}</h3>
                <p>Total Samples</p>
            </div>
            <div class="stat-card">
                <h3>{stats['sentiment_distribution'].get('positive', 0):,}</h3>
                <p>Positive Samples</p>
            </div>
            <div class="stat-card">
                <h3>{stats['sentiment_distribution'].get('neutral', 0):,}</h3>
                <p>Neutral Samples</p>
            </div>
            <div class="stat-card">
                <h3>{stats['sentiment_distribution'].get('negative', 0):,}</h3>
                <p>Negative Samples</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Text Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Average Text Length</td><td>{stats['text_statistics']['avg_length']:.1f} characters</td></tr>
                <tr><td>Average Word Count</td><td>{stats['text_statistics']['avg_word_count']:.1f} words</td></tr>
                <tr><td>Min Text Length</td><td>{stats['text_statistics']['min_length']} characters</td></tr>
                <tr><td>Max Text Length</td><td>{stats['text_statistics']['max_length']} characters</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📊 Visualizations</h2>
            <div class="chart-container">
                <div class="chart">
                    <img src="sentiment_distribution.png" alt="Sentiment Distribution">
                </div>
                <div class="chart">
                    <img src="source_distribution.png" alt="Source Distribution">
                </div>
                <div class="chart">
                    <img src="text_length_histogram.png" alt="Text Length Histogram">
                </div>
                <div class="chart">
                    <img src="word_frequency.png" alt="Word Frequency">
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2026 RSK World - Free Programming Resources & Source Code</p>
            <p>Author: <strong>Molla Samser</strong> | Designer: <strong>Rima Khatun</strong></p>
            <p><a href="https://rskworld.in">rskworld.in</a> | help@rskworld.in</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Generated HTML report: {output_path}")


# ============================================
# Main Function
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Data Visualization Tool - RSK World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_data.py --input ../data/sentiment_data.csv
  python visualize_data.py --input ../data/sentiment_data.json --output ./charts/
  python visualize_data.py --input ../data/sentiment_data.csv --all-charts

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
        default="./charts",
        help="Output directory for charts (default: ./charts)"
    )
    
    parser.add_argument(
        "--all-charts", "-a",
        action="store_true",
        help="Generate all available charts"
    )
    
    parser.add_argument(
        "--stats-only", "-s",
        action="store_true",
        help="Only print statistics, no charts"
    )
    
    parser.add_argument(
        "--html-report", "-r",
        action="store_true",
        help="Generate HTML report"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         RSK World - Data Visualization Tool                      ║
║                                                                  ║
║  Author: Molla Samser (Founder)                                  ║
║  Website: https://rskworld.in                                    ║
║  © 2026 RSK World - Free Programming Resources & Source Code     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check for required libraries
    print("Available features:")
    print(f"  {'✓' if MATPLOTLIB_AVAILABLE else '✗'} Charts (matplotlib)")
    print(f"  {'✓' if WORDCLOUD_AVAILABLE else '✗'} Word Clouds (wordcloud)")
    print()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"Loaded {len(data)} samples")
    print()
    
    # Calculate statistics
    stats = calculate_statistics(data)
    
    # Print statistics
    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total samples: {stats['total_samples']:,}")
    print()
    print("Sentiment Distribution:")
    for sentiment, count in stats['sentiment_distribution'].items():
        pct = count / stats['total_samples'] * 100
        print(f"  {sentiment.capitalize()}: {count:,} ({pct:.1f}%)")
    print()
    print("Source Distribution:")
    for source, count in stats['source_distribution'].items():
        print(f"  {source}: {count:,}")
    print()
    print("Text Statistics:")
    print(f"  Average length: {stats['text_statistics']['avg_length']:.1f} chars")
    print(f"  Average words: {stats['text_statistics']['avg_word_count']:.1f}")
    print()
    
    if args.stats_only:
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate charts
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating charts...")
        
        plot_sentiment_distribution(
            data, 
            os.path.join(args.output, 'sentiment_distribution.png')
        )
        
        plot_source_distribution(
            data,
            os.path.join(args.output, 'source_distribution.png')
        )
        
        plot_text_length_histogram(
            data,
            os.path.join(args.output, 'text_length_histogram.png')
        )
        
        plot_word_frequency(
            data,
            os.path.join(args.output, 'word_frequency.png')
        )
        
        if args.all_charts and WORDCLOUD_AVAILABLE:
            print("\nGenerating word clouds...")
            generate_wordcloud(data, os.path.join(args.output, 'wordcloud_all.png'))
            generate_wordcloud(data, os.path.join(args.output, 'wordcloud_positive.png'), 'positive')
            generate_wordcloud(data, os.path.join(args.output, 'wordcloud_negative.png'), 'negative')
    
    # Generate HTML report
    if args.html_report:
        generate_html_report(
            data, stats,
            os.path.join(args.output, 'report.html'),
            args.output
        )
    
    print()
    print("✓ Visualization complete!")
    print(f"  Output directory: {args.output}")
    print()


if __name__ == "__main__":
    main()

