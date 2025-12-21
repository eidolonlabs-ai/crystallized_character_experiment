#!/usr/bin/env python3
"""
Dataset Analysis & Comparison Tool
Shows current data status and training adequacy across all models
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

def count_tokens(data: list) -> int:
    """Estimate tokens in dataset (rough: 1.3 tokens per word)"""
    total = 0
    for sample in data:
        for msg in sample.get('messages', []):
            words = len(msg.get('content', '').split())
            total += int(words * 1.3)
    return total

def load_dataset(path: str) -> Tuple[int, int]:
    """Load dataset and return (sample_count, estimated_tokens)"""
    if not os.path.exists(path):
        return 0, 0
    
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    return len(data), count_tokens(data)

def main():
    print("\n" + "=" * 100)
    print("BASELINE CHARACTER TRAINING FRAMEWORK - DATA ANALYSIS")
    print("=" * 100)
    print()
    
    # Define all datasets
    datasets = {
        'Baseline (Original)': {
            'paths': ['raw_data/baseline/train.jsonl', 'raw_data/baseline/valid.jsonl'],
            'use_case': 'Quick validation only (TOO SMALL for real training)'
        },
        'Baseline (Augmented)': {
            'paths': ['raw_data/baseline_augmented/train.jsonl', 'raw_data/baseline_augmented/valid.jsonl'],
            'use_case': '✅ PRIMARY BASELINE - model comparison'
        }
    }
    
    # Model requirements
    model_specs = {
        'mistral': {'size': '7B', 'min_samples': 100, 'min_tokens': 30000},
        'llama31_8b': {'size': '8B', 'min_samples': 120, 'min_tokens': 40000},
        'llama3_8b': {'size': '8B', 'min_samples': 120, 'min_tokens': 40000},
        'llama2_7b': {'size': '7B', 'min_samples': 100, 'min_tokens': 30000},
    }
    
    # Load all datasets
    loaded_datasets = {}
    for name, config in datasets.items():
        total_samples = 0
        total_tokens = 0
        for path in config['paths']:
            samples, tokens = load_dataset(path)
            total_samples += samples
            total_tokens += tokens
        loaded_datasets[name] = (total_samples, total_tokens)
    
    # Display dataset summary
    print("AVAILABLE DATASETS")
    print("-" * 100)
    print(f"{'Dataset':<30} {'Samples':<15} {'Est. Tokens':<20} {'Use Case':<35}")
    print("-" * 100)
    
    for name, (samples, tokens) in loaded_datasets.items():
        use_case = datasets[name]['use_case']
        print(f"{name:<30} {samples:<15} {tokens:>15,}   {use_case:<35}")
    
    print()
    print("TRAINING ADEQUACY MATRIX")
    print("=" * 100)
    print()
    
    # Check each dataset against each model
    for dataset_name, (total_samples, total_tokens) in loaded_datasets.items():
        print(f"\n{dataset_name}")
        print("-" * 100)
        print(f"{'Model':<20} {'Size':<6} {'Need Samples':<15} {'Have':<10} {'Need Tokens':<15} {'Have':<15} {'Ready?':<10}")
        print("-" * 100)
        
        for model_name in sorted(model_specs.keys(), key=lambda x: model_specs[x]['size']):
            spec = model_specs[model_name]
            min_samp = spec['min_samples']
            min_tok = spec['min_tokens']
            
            samp_ok = total_samples >= min_samp
            tok_ok = total_tokens >= min_tok
            ready = "✅ YES" if (samp_ok and tok_ok) else "❌ NO"
            
            print(f"{model_name:<20} {spec['size']:<6} {min_samp:<15} {total_samples:<10} {min_tok:<15} {total_tokens:<15,} {ready:<10}")
    
    # Recommendations
    print()
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print()
    
    print("✅ USE THESE DATASETS:")
    print("  • Baseline (Augmented): For A/B testing models with same data")
    print()
    
    print("❌ DON'T USE:")
    print("  • Baseline (Original): Only 22 samples - causes complete overfitting")
    print("    └─ Use ONLY for 2-minute validation tests")
    print()
    
    print("📊 TRAINING PLAN:")
    print("  1. Train models with Baseline (Augmented)")
    print("     └─ Establishes performance baseline, validates data adequacy")
    print()
    
    print("⏱️  ESTIMATED TRAINING TIMES (per model, standard tuning):")
    print("  • 3B-4B models: 8-12 minutes")
    print("  • 7B-8B models: 12-18 minutes")
    print("  • 9B models: 15-22 minutes")
    print("  • 12B models: 20-30 minutes")
    print()
    
    print("=" * 100)
    print()

if __name__ == "__main__":
    main()
