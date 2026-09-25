#!/usr/bin/env python3
"""
Phase 4: Statistical Noise-Pattern Analysis on True Pairs vs False Pairs
ML Challenge 2026 - Business Entity Resolution
"""

import re
import os
import sys
import random
import unicodedata
from collections import Counter

# Set random seed for exact reproducibility
random.seed(42)

def normalize_text(text):
    if not text:
        return ""
    # Unicode normalize
    text = unicodedata.normalize('NFKD', text)
    # Lowercase
    text = text.lower()
    # Replace punctuation with space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'gmbh', 'sa', 'sarl', 'llp', 'pllc'
}

def strip_legal_suffixes(norm_name):
    tokens = norm_name.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else norm_name

def token_sort_form(text):
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def extract_numbers(text):
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def char_ngrams(text, n=3):
    text = f"  {text}  "
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def dice_ngram(text1, text2, n=3):
    if not text1 or not text2:
        return 0.0
    ng1 = char_ngrams(text1, n)
    ng2 = char_ngrams(text2, n)
    if not ng1 or not ng2:
        return 0.0
    return 2 * len(ng1 & ng2) / (len(ng1) + len(ng2))

def main():
    print("=" * 70)
    print("PHASE 4: NOISE PATTERN ANALYSIS ON GROUND-TRUTH MATCHES")
    print("=" * 70)

    # 1. Sample 50,000 S1 records with matches
    print("\n[Step 1] Loading sample of ground-truth matches...")
    sampled_s1_pairs = [] # (s1_id, match_id)
    s1_needed = set()
    s2_needed = set()
    s3_needed = set()

    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            # Sample ~2.5% of rows
            if random.random() < 0.025:
                p = line.rstrip("\r\n").split("\t")
                s1_id = p[0]
                if len(p) > 1 and p[1].strip():
                    mids = p[1].strip().split(",")
                    for mid in mids:
                        sampled_s1_pairs.append((s1_id, mid))
                        s1_needed.add(s1_id)
                        if mid.startswith("S2-"):
                            s2_needed.add(mid)
                        elif mid.startswith("S3-"):
                            s3_needed.add(mid)
            if len(sampled_s1_pairs) >= 50000:
                break

    print(f"Sampled {len(sampled_s1_pairs):,} true pairs across {len(s1_needed):,} S1 entities.")
    print(f"Target IDs: S2={len(s2_needed):,}, S3={len(s3_needed):,}")

    # 2. Load records for sampled entities
    print("\n[Step 2] Reading sampled entity records...")
    s1_data = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s1_needed:
                s1_data[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

    s2_data = {}
    with open("dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s2_needed:
                s2_data[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

    s3_data = {}
    with open("dataset/train/train_source3.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s3_needed:
                s3_data[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

    print("All sampled entity records loaded.")

    # 3. Analyze True Pairs
    print("\n[Step 3] Analyzing True Pairs...")
    name_raw_exact = 0
    name_norm_exact = 0
    name_legal_stripped_exact = 0
    name_token_sort_exact = 0
    name_jaccard_scores = []
    name_dice_scores = []

    addr_raw_exact = 0
    addr_norm_exact = 0
    addr_missing_in_target = 0
    addr_jaccard_scores = []
    addr_num_overlap_scores = []
    addr_dice_scores = []

    by_country_stats = {"US": Counter(), "India": Counter()}
    by_source_stats = {"S2": Counter(), "S3": Counter()}

    noise_examples = []

    for s1_id, mid in sampled_s1_pairs:
        r1 = s1_data.get(s1_id)
        r2 = s2_data.get(mid) if mid.startswith("S2-") else s3_data.get(mid)
        if not r1 or not r2:
            continue

        c = r1["country"]
        src = "S2" if mid.startswith("S2-") else "S3"

        n1_raw, n2_raw = r1["name"], r2["name"]
        a1_raw, a2_raw = r1["addr"], r2["addr"]

        # Name metrics
        if n1_raw == n2_raw:
            name_raw_exact += 1
            by_country_stats[c]["name_raw_exact"] += 1
            by_source_stats[src]["name_raw_exact"] += 1

        n1_norm, n2_norm = normalize_text(n1_raw), normalize_text(n2_raw)
        if n1_norm == n2_norm:
            name_norm_exact += 1
            by_country_stats[c]["name_norm_exact"] += 1
            by_source_stats[src]["name_norm_exact"] += 1

        n1_strip, n2_strip = strip_legal_suffixes(n1_norm), strip_legal_suffixes(n2_norm)
        if n1_strip == n2_strip:
            name_legal_stripped_exact += 1
            by_country_stats[c]["name_legal_stripped_exact"] += 1

        n1_ts, n2_ts = token_sort_form(n1_norm), token_sort_form(n2_norm)
        if n1_ts == n2_ts:
            name_token_sort_exact += 1

        t1, t2 = set(n1_norm.split()), set(n2_norm.split())
        jac = jaccard_similarity(t1, t2)
        name_jaccard_scores.append(jac)

        dice = dice_ngram(n1_norm, n2_norm, 3)
        name_dice_scores.append(dice)

        # Address metrics
        if not a2_raw.strip():
            addr_missing_in_target += 1
            by_source_stats[src]["addr_missing"] += 1
        else:
            if a1_raw == a2_raw:
                addr_raw_exact += 1
            a1_norm, a2_norm = normalize_text(a1_raw), normalize_text(a2_raw)
            if a1_norm == a2_norm:
                addr_norm_exact += 1

            at1, at2 = set(a1_norm.split()), set(a2_norm.split())
            ajac = jaccard_similarity(at1, at2)
            addr_jaccard_scores.append(ajac)

            num1, num2 = extract_numbers(a1_norm), extract_numbers(a2_norm)
            if num1 and num2:
                addr_num_overlap_scores.append(jaccard_similarity(num1, num2))

            adice = dice_ngram(a1_norm, a2_norm, 3)
            addr_dice_scores.append(adice)

        # Collect interesting corruption examples
        if len(noise_examples) < 15:
            if n1_norm != n2_norm and jac > 0.5:
                noise_examples.append({
                    "s1_id": s1_id,
                    "target_id": mid,
                    "country": c,
                    "s1_name": n1_raw,
                    "target_name": n2_raw,
                    "s1_addr": a1_raw,
                    "target_addr": a2_raw,
                    "name_jaccard": round(jac, 3),
                    "name_ngram_dice": round(dice, 3),
                })

    N = len(sampled_s1_pairs)
    print(f"\n--- TRUE PAIR ANALYSIS SUMMARY (N = {N:,}) ---")
    print(f"Name Raw Exact Match:             {name_raw_exact / N * 100:.2f}%")
    print(f"Name Normalized Exact Match:      {name_norm_exact / N * 100:.2f}%")
    print(f"Name Legal-Stripped Exact:        {name_legal_stripped_exact / N * 100:.2f}%")
    print(f"Name Token-Sort Exact:            {name_token_sort_exact / N * 100:.2f}%")
    print(f"Mean Name Token Jaccard:          {sum(name_jaccard_scores) / len(name_jaccard_scores):.4f}")
    print(f"Mean Name 3-Gram Dice:            {sum(name_dice_scores) / len(name_dice_scores):.4f}")
    print()
    print(f"Target Address Missing:           {addr_missing_in_target / N * 100:.2f}%")
    print(f"Address Raw Exact Match:          {addr_raw_exact / N * 100:.2f}%")
    print(f"Address Normalized Exact Match:   {addr_norm_exact / N * 100:.2f}%")
    print(f"Mean Address Token Jaccard:       {sum(addr_jaccard_scores) / len(addr_jaccard_scores):.4f}")
    print(f"Mean Address 3-Gram Dice:         {sum(addr_dice_scores) / len(addr_dice_scores):.4f}")
    if addr_num_overlap_scores:
        print(f"Mean Address Number Jaccard:      {sum(addr_num_overlap_scores) / len(addr_num_overlap_scores):.4f}")

    print("\n--- SAMPLE REAL-WORLD CORRUPTIONS ---")
    for ex in noise_examples[:8]:
        print(f"[{ex['country']}] S1: {ex['s1_name']} | Target: {ex['target_name']}")
        print(f"     Addr S1: {ex['s1_addr']}")
        print(f"     Addr Tgt: {ex['target_addr']}")
        print(f"     Name Jac: {ex['name_jaccard']} | 3-gram Dice: {ex['name_ngram_dice']}\n")

if __name__ == "__main__":
    main()
