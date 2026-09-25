#!/usr/bin/env python3
"""
Main Entrypoint for Business Entity Resolution Pipeline
ML Challenge 2026 - Amazon ML Challenge
Reproduces output/matching_results.tsv and output/candidate_pairs.tsv
"""

import sys
import os
import argparse

# Ensure src/ and current dir are in python path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath("."))

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "infer", "full"], default="infer",
                        help="Execution mode: 'train' to retrain model, 'infer' to generate test outputs, 'full' for end-to-end")
    args = parser.parse_args()

    if args.mode in ["train", "full"]:
        print("\n>>> Running Model Training & Hard Negative Mining Pipeline...")
        import train_model
        train_model.train_and_evaluate()

    if args.mode in ["infer", "full"]:
        print("\n>>> Running Full Test Set Inference Pipeline...")
        import inference_partitioned
        inference_partitioned.run_partitioned_inference()

if __name__ == "__main__":
    main()
