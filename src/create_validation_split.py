#!/usr/bin/env python3
"""
Create Standard Stratified Validation Split
ML Challenge 2026 - Business Entity Resolution
"""

import os
import random
from collections import defaultdict

def create_split(val_size=20000, seed=42):
    print(f"Generating stratified validation split of {val_size:,} S1 entities (seed={seed})...")
    random.seed(seed)

    # 1. Load S1 countries
    s1_meta = {} # s1_id -> country
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1_meta[p[0]] = p[3]

    # 2. Load Ground Truth singleton status
    s1_has_matches = set()
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1].strip():
                s1_has_matches.add(p[0])

    # 3. Stratified buckets: (country, is_singleton)
    buckets = defaultdict(list)
    for s1_id, country in s1_meta.items():
        is_singleton = s1_id not in s1_has_matches
        buckets[(country, is_singleton)].append(s1_id)

    total_s1 = len(s1_meta)
    val_ids = []

    for (c, sing), ids in buckets.items():
        ids.sort() # deterministic order before shuffle
        random.shuffle(ids)
        # Allocate proportionally
        target_count = int(round(len(ids) / total_s1 * val_size))
        chosen = ids[:target_count]
        val_ids.extend(chosen)
        print(f"  Bucket ({c}, singleton={sing}): total={len(ids):,}, sampled={len(chosen):,} ({len(chosen)/len(ids)*100:.2f}%)")

    # If minor rounding difference, adjust
    if len(val_ids) > val_size:
        val_ids = val_ids[:val_size]
    elif len(val_ids) < val_size:
        # fill from largest bucket
        diff = val_size - len(val_ids)
        val_ids.extend(buckets[("US", False)][len(val_ids):len(val_ids) + diff])

    val_ids.sort()
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/val_s1_ids.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for vid in val_ids:
            f.write(f"{vid}\n")

    print(f"Total validation entities selected: {len(val_ids):,}")
    print(f"Saved validation IDs to {out_path}")
    return val_ids

if __name__ == "__main__":
    create_split()
