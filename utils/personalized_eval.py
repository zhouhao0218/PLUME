#!/usr/bin/env python3
"""
Evaluate inference results using generation_metrics.py

This script loads the converted data and runs evaluation using the metrics from generation_metrics.py

Usage:
    python evaluate_results.py --data evaluation_data.json --name "zero_shot_llama_topic_writing"
"""

import argparse
import json
import sys
from pathlib import Path

from utils.generation_metrics import evaluate_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results using generation metrics")
    parser.add_argument("--data", required=True, help="Path to converted evaluation data JSON file")
    parser.add_argument("--name", required=True, help="Name for this evaluation (will be saved in metrics.txt)")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file does not exist: {args.data}")
        sys.exit(1)
    
    # Load the evaluation data
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples for evaluation")
    
    if len(data) == 0:
        print("Error: No data to evaluate!")
        sys.exit(1)
    
    # Validate data format
    required_fields = ["prompt", "output", "generated_text"]
    sample = data[0]
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        print(f"Error: Missing required fields in data: {missing_fields}")
        print(f"Available fields: {list(sample.keys())}")
        sys.exit(1)
    
    # Count valid examples (both generated_text and output exist)
    valid_examples = sum(1 for d in data if d.get("generated_text") and d.get("output"))
    print(f"Valid examples for evaluation: {valid_examples}/{len(data)}")
    
    if valid_examples == 0:
        print("Error: No valid examples to evaluate (missing generated_text or output)!")
        sys.exit(1)
    
    # Determine output directory (same as results file)
    results_dir = Path(args.data).parent
    metrics_file = results_dir / "metrics.txt"
    
    # Run evaluation
    print(f"Running evaluation for: {args.name}")
    metrics = evaluate_data(data, args.name, str(metrics_file))
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {metrics_file}")
    print(f"\nMetrics:")
    for metric_name, score in metrics.items():
        print(f"  {metric_name}: {score}")


if __name__ == "__main__":
    main()
