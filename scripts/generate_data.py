#!/usr/bin/env python3
"""
================================================================================
 * Sentiment Analysis Dataset - Data Generator Script
 * 
 * Project: Sentiment Analysis Dataset
 * Description: Generate synthetic sentiment analysis data with customizable
 *              parameters for training NLP models.
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
    python generate_data.py --samples 1000 --output ./data/generated_data.csv
    python generate_data.py --samples 5000 --format json --balanced
    python generate_data.py --samples 10000 --include-metadata --split 0.8
"""

import argparse
import csv
import json
import random
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import hashlib

# ============================================
# Sentiment Templates and Vocabulary
# ============================================

POSITIVE_TEMPLATES = [
    "Absolutely love this {product}! It exceeded all my expectations.",
    "Best {product} I've ever purchased. Highly recommend to everyone!",
    "Amazing quality and fast delivery. The {product} is perfect!",
    "Five stars! This {product} changed my life for the better.",
    "Impressed beyond words! The {product} is truly exceptional.",
    "Great value for money! The {product} performs really well.",
    "Outstanding {product}! Can't imagine going back to what I used before.",
    "Perfect in every way! The {product} exceeded my expectations.",
    "Fantastic {product}! My friends were so impressed they ordered one too.",
    "Simply phenomenal! This {product} has improved my daily routine.",
    "Wow! Just wow! This {product} is worth every penny.",
    "Life-changing {product}! Already recommended to all my colleagues.",
    "Brilliant {product}! Such attention to detail and quality.",
    "The {product} arrived quickly and works flawlessly. Love it!",
    "Exceeded expectations! The {product} quality is superb.",
    "I can't believe how good this {product} is! My whole family loves it.",
    "This {product} is everything I hoped for and more!",
    "Super impressed with the {product} quality. Ordering more as gifts!",
    "The {product} looks even better in person than in photos!",
    "Absolutely thrilled with my {product} purchase. 10/10!",
    "This {product} is a game changer! So glad I bought it.",
    "Remarkable {product}! The craftsmanship is top-notch.",
    "Couldn't be happier with this {product}. Exactly what I needed!",
    "The {product} works like a charm! Highly satisfied customer here.",
    "Premium quality {product} at an affordable price. Amazing deal!",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay. Nothing special but does the job.",
    "Received the {product} today. It looks decent, will test more.",
    "Standard {product}, meets basic requirements. Fair price.",
    "The {product} works as expected. Nothing extraordinary.",
    "Average {product}. No complaints but no praises either.",
    "Mixed feelings about this {product}. Some features are good.",
    "The {product} arrived on time. Performance is as described.",
    "It's an average {product}. Does what it's supposed to do.",
    "Got what I paid for with this {product}. Standard experience.",
    "The {product} is functional but unremarkable. Average overall.",
    "Decent {product} for the price. Might buy again in future.",
    "The {product} seems okay. Will need more time to evaluate.",
    "Nothing stood out positively or negatively about this {product}.",
    "Adequate {product} for basic needs. Won't wow you.",
    "The {product} is middle of the road. Serves its purpose.",
    "First impressions of the {product} are neither good nor bad.",
    "The {product} packaging was adequate. Performance is standard.",
    "Just unboxed the {product}. Quality meets basic requirements.",
    "The {product} does the job. Not exceptional but functional.",
    "Standard shipping, standard {product}, standard everything.",
    "The {product} is what you'd expect at this price point.",
    "Neither excited nor disappointed with this {product}.",
    "The {product} performs adequately. Nothing more, nothing less.",
    "Received my {product} yesterday. It's just okay overall.",
    "The {product} doesn't stand out from competition in any way.",
]

NEGATIVE_TEMPLATES = [
    "Terrible experience! The {product} arrived damaged.",
    "Worst {product} I've ever purchased. Complete waste of money!",
    "So disappointed with this {product}. Description was misleading.",
    "Horrible {product}! Customer support was unhelpful.",
    "Don't waste your money on this {product}! Poor quality.",
    "Completely broken {product} on arrival! Demanding refund.",
    "Avoid this {product} at all costs! Stopped working after a week.",
    "Extremely disappointed with the {product}. False advertising!",
    "Total rip-off! The {product} quality is nowhere near promised.",
    "Save your money! This {product} is a complete joke.",
    "Absolutely terrible {product}! Broke within days of purchase.",
    "Regret buying this {product}. Reviews were misleading.",
    "Complete disaster! {product} arrived late and damaged.",
    "Horrible quality control! The {product} had multiple defects.",
    "Never again! This {product} company has lost all credibility.",
    "Disappointed beyond words with this {product}. Reality was different.",
    "The {product} is garbage! Poor materials and horrible design.",
    "Worst customer service ever! My {product} issue is still unresolved.",
    "This {product} is a scam! Falls apart after first use.",
    "Feeling completely cheated by this {product} purchase.",
    "The {product} failed immediately. Support is non-existent.",
    "Massive letdown! The {product} photos were misleading.",
    "Demanded a refund for this {product}. Absolute nightmare!",
    "The {product} is defective. Company refuses to acknowledge issues.",
    "Stay away from this {product}! Waste of time and money.",
]

PRODUCTS = [
    "product", "item", "purchase", "device", "gadget", "tool", "equipment",
    "accessory", "appliance", "machine", "system", "kit", "set", "package",
    "solution", "unit", "model", "version", "edition", "series"
]

SOURCES = ["Product Review", "Customer Feedback", "Social Media", "Survey Response", "Email Feedback"]
SOURCE_ICONS = {
    "Product Review": "shopping-cart",
    "Customer Feedback": "comment",
    "Social Media": "twitter",
    "Survey Response": "poll",
    "Email Feedback": "envelope"
}

# Additional vocabulary for variation
POSITIVE_ADJECTIVES = ["amazing", "excellent", "fantastic", "wonderful", "brilliant", "superb", "outstanding", "perfect", "incredible", "remarkable"]
NEGATIVE_ADJECTIVES = ["terrible", "horrible", "awful", "dreadful", "disappointing", "frustrating", "unacceptable", "poor", "defective", "useless"]
NEUTRAL_ADJECTIVES = ["average", "standard", "typical", "ordinary", "decent", "fair", "acceptable", "moderate", "passable", "adequate"]

# ============================================
# Data Generation Functions
# ============================================

def generate_text(sentiment: str, variation: bool = True) -> str:
    """Generate a sentiment text based on templates."""
    if sentiment == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)
    
    product = random.choice(PRODUCTS)
    text = template.format(product=product)
    
    if variation:
        # Add some random variations
        if random.random() > 0.7:
            if sentiment == "positive":
                adj = random.choice(POSITIVE_ADJECTIVES)
                text = f"{adj.capitalize()}! " + text
            elif sentiment == "negative":
                adj = random.choice(NEGATIVE_ADJECTIVES)
                text = f"{adj.capitalize()}. " + text
    
    return text


def generate_date(start_date: datetime, end_date: datetime) -> str:
    """Generate a random date between start and end."""
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    date = start_date + timedelta(days=random_days)
    return date.strftime("%Y-%m-%d")


def generate_id(text: str, index: int) -> str:
    """Generate a unique ID for a data point."""
    hash_input = f"{text}{index}{datetime.now().isoformat()}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def generate_sample(
    index: int,
    sentiment: str,
    start_date: datetime,
    end_date: datetime,
    include_metadata: bool = False
) -> Dict:
    """Generate a single data sample."""
    text = generate_text(sentiment)
    source = random.choice(SOURCES)
    date = generate_date(start_date, end_date)
    
    sample = {
        "id": index,
        "text": text,
        "sentiment": sentiment,
        "source": source,
        "date": date
    }
    
    if include_metadata:
        sample["metadata"] = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "source_icon": SOURCE_ICONS.get(source, "file"),
            "generated_at": datetime.now().isoformat(),
            "hash": generate_id(text, index)
        }
    
    return sample


def generate_dataset(
    num_samples: int,
    balanced: bool = True,
    sentiment_ratio: Optional[Dict[str, float]] = None,
    include_metadata: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict]:
    """Generate a complete dataset."""
    
    if start_date is None:
        start_date = datetime(2026, 1, 1)
    if end_date is None:
        end_date = datetime(2026, 12, 31)
    
    if balanced:
        ratio = {"positive": 1/3, "neutral": 1/3, "negative": 1/3}
    elif sentiment_ratio:
        ratio = sentiment_ratio
    else:
        # Slight imbalance towards positive
        ratio = {"positive": 0.4, "neutral": 0.3, "negative": 0.3}
    
    dataset = []
    sentiments = []
    
    for sentiment, proportion in ratio.items():
        count = int(num_samples * proportion)
        sentiments.extend([sentiment] * count)
    
    # Add remaining samples to balance
    while len(sentiments) < num_samples:
        sentiments.append(random.choice(["positive", "neutral", "negative"]))
    
    random.shuffle(sentiments)
    
    for i, sentiment in enumerate(sentiments, 1):
        sample = generate_sample(i, sentiment, start_date, end_date, include_metadata)
        dataset.append(sample)
        
        # Progress indicator
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples} samples...")
    
    return dataset


def split_dataset(
    dataset: List[Dict],
    train_ratio: float = 0.8
) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into training and test sets."""
    random.shuffle(dataset)
    split_index = int(len(dataset) * train_ratio)
    return dataset[:split_index], dataset[split_index:]


# ============================================
# Export Functions
# ============================================

def export_csv(dataset: List[Dict], filepath: str, include_header_comment: bool = True):
    """Export dataset to CSV format."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        if include_header_comment:
            f.write("""# ================================================================================
# Sentiment Analysis Dataset - Generated Data
# 
# Project: Sentiment Analysis Dataset
# Generated by: RSK World Data Generator
# Website: https://rskworld.in
# 
# Author: Molla Samser (Founder)
# Designer & Tester: Rima Khatun
# Email: help@rskworld.in | support@rskworld.in
# 
# © 2026 RSK World - Free Programming Resources & Source Code
# ================================================================================

""")
        
        # Determine fields
        fields = ["id", "text", "sentiment", "source", "date"]
        if dataset and "metadata" in dataset[0]:
            fields.extend(["text_length", "word_count"])
        
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for sample in dataset:
            row = {k: sample.get(k) for k in ["id", "text", "sentiment", "source", "date"]}
            if "metadata" in sample:
                row["text_length"] = sample["metadata"]["text_length"]
                row["word_count"] = sample["metadata"]["word_count"]
            writer.writerow(row)
    
    print(f"✓ Exported {len(dataset)} samples to {filepath}")


def export_json(dataset: List[Dict], filepath: str):
    """Export dataset to JSON format."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    output = {
        "_metadata": {
            "project": "Sentiment Analysis Dataset",
            "description": "Generated sentiment analysis data",
            "generator": "RSK World Data Generator",
            "website": "https://rskworld.in",
            "author": "Molla Samser (Founder)",
            "designer_tester": "Rima Khatun",
            "email": "help@rskworld.in | support@rskworld.in",
            "copyright": "© 2026 RSK World - Free Programming Resources & Source Code",
            "generated_at": datetime.now().isoformat(),
            "total_samples": len(dataset),
            "sentiment_distribution": {
                "positive": sum(1 for s in dataset if s["sentiment"] == "positive"),
                "neutral": sum(1 for s in dataset if s["sentiment"] == "neutral"),
                "negative": sum(1 for s in dataset if s["sentiment"] == "negative")
            }
        },
        "data": dataset
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported {len(dataset)} samples to {filepath}")


def export_txt(dataset: List[Dict], filepath: str):
    """Export dataset to TXT format."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("""# ================================================================================
# Sentiment Analysis Dataset - Generated Data
# 
# Project: Sentiment Analysis Dataset
# Generated by: RSK World Data Generator
# Website: https://rskworld.in
# 
# Author: Molla Samser (Founder)
# Designer & Tester: Rima Khatun
# Email: help@rskworld.in | support@rskworld.in
# 
# © 2026 RSK World - Free Programming Resources & Source Code
# 
# Format: SENTIMENT | SOURCE | TEXT
# ================================================================================

""")
        for sample in dataset:
            f.write(f"{sample['sentiment']} | {sample['source']} | {sample['text']}\n\n")
    
    print(f"✓ Exported {len(dataset)} samples to {filepath}")


# ============================================
# Main Function
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate sentiment analysis dataset - RSK World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_data.py --samples 1000
  python generate_data.py --samples 5000 --format json --balanced
  python generate_data.py --samples 10000 --output ./data/custom.csv --split 0.8
  python generate_data.py --samples 2000 --include-metadata --all-formats

Author: Molla Samser (Founder) - RSK World
Website: https://rskworld.in
        """
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/generated_data",
        help="Output file path (without extension)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["csv", "json", "txt", "all"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "--balanced", "-b",
        action="store_true",
        help="Generate balanced dataset (equal samples per class)"
    )
    
    parser.add_argument(
        "--split", "-s",
        type=float,
        default=None,
        help="Train/test split ratio (e.g., 0.8 for 80% train)"
    )
    
    parser.add_argument(
        "--include-metadata", "-m",
        action="store_true",
        help="Include metadata in generated samples"
    )
    
    parser.add_argument(
        "--all-formats", "-a",
        action="store_true",
        help="Export in all formats (CSV, JSON, TXT)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         RSK World - Sentiment Analysis Data Generator            ║
║                                                                  ║
║  Author: Molla Samser (Founder)                                  ║
║  Website: https://rskworld.in                                    ║
║  © 2026 RSK World - Free Programming Resources & Source Code     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Generating {args.samples} samples...")
    print(f"Balanced: {args.balanced}")
    print(f"Include metadata: {args.include_metadata}")
    print()
    
    # Generate dataset
    dataset = generate_dataset(
        num_samples=args.samples,
        balanced=args.balanced,
        include_metadata=args.include_metadata
    )
    
    # Calculate statistics
    pos_count = sum(1 for s in dataset if s["sentiment"] == "positive")
    neu_count = sum(1 for s in dataset if s["sentiment"] == "neutral")
    neg_count = sum(1 for s in dataset if s["sentiment"] == "negative")
    
    print()
    print("Dataset Statistics:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Positive: {pos_count} ({pos_count/len(dataset)*100:.1f}%)")
    print(f"  - Neutral: {neu_count} ({neu_count/len(dataset)*100:.1f}%)")
    print(f"  - Negative: {neg_count} ({neg_count/len(dataset)*100:.1f}%)")
    print()
    
    # Export based on format
    formats_to_export = []
    if args.all_formats or args.format == "all":
        formats_to_export = ["csv", "json", "txt"]
    else:
        formats_to_export = [args.format]
    
    # Handle split if requested
    if args.split:
        train_data, test_data = split_dataset(dataset, args.split)
        print(f"Split dataset: {len(train_data)} train, {len(test_data)} test")
        print()
        
        for fmt in formats_to_export:
            train_path = f"{args.output}_train.{fmt}"
            test_path = f"{args.output}_test.{fmt}"
            
            if fmt == "csv":
                export_csv(train_data, train_path)
                export_csv(test_data, test_path)
            elif fmt == "json":
                export_json(train_data, train_path)
                export_json(test_data, test_path)
            elif fmt == "txt":
                export_txt(train_data, train_path)
                export_txt(test_data, test_path)
    else:
        for fmt in formats_to_export:
            filepath = f"{args.output}.{fmt}"
            
            if fmt == "csv":
                export_csv(dataset, filepath)
            elif fmt == "json":
                export_json(dataset, filepath)
            elif fmt == "txt":
                export_txt(dataset, filepath)
    
    print()
    print("✓ Data generation complete!")
    print()


if __name__ == "__main__":
    main()

