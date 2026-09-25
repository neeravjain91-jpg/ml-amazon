#!/usr/bin/env python3
"""
Diagnose Missed True Matches in Blocking
ML Challenge 2026 - Business Entity Resolution
"""

import re
import os
import sys
import unicodedata
from collections import defaultdict, Counter

def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'gmbh', 'sa', 'sarl', 'llp', 'pllc'
}

def strip_legal(norm_name):
    tokens = norm_name.split()
    f = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(f) if f else norm_name

def token_sort_form(text):
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def extract_numbers(text):
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def extract_significant_tokens(norm_name):
    tokens = norm_name.split()
    return [t for t in tokens if len(t) >= 3 and t not in LEGAL_SUFFIXES]

def main():
    print("Investigating Missed True Matches...")

    # Load 1,000 S1 validation queries
    val_s1_ids = []
    with open("reports/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_ids = [line.strip() for line in f if line.strip()][:1000]
    val_s1_set = set(val_s1_ids)

    gt = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1_set:
                mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
                gt[p[0]] = mids

    s1_data = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1_set:
                s1_data[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

    # All true match targets needed
    needed_targets = set()
    for s1, trues in gt.items():
        needed_targets.update(trues)

    target_data = {}
    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in needed_targets:
                    target_data[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

    print(f"Loaded {len(s1_data):,} S1 queries and {len(target_data):,} target true matches.")

    # Check which ones are missed by A, B, C, E
    missed_examples = []
    for s1, trues in gt.items():
        r1 = s1_data[s1]
        n1 = normalize_text(r1["name"])
        s1_name = strip_legal(n1)
        ts1_name = token_sort_form(s1_name)
        nums1 = extract_numbers(r1["addr"])
        sig1 = set(extract_significant_tokens(s1_name))

        for mid in trues:
            r2 = target_data.get(mid)
            if not r2:
                continue

            n2 = normalize_text(r2["name"])
            s2_name = strip_legal(n2)
            ts2_name = token_sort_form(s2_name)
            nums2 = extract_numbers(r2["addr"])
            sig2 = set(extract_significant_tokens(s2_name))

            # Channel A
            hit_A = (n1 == n2)
            # Channel B
            hit_B = (s1_name == s2_name)
            # Channel C
            hit_C = (ts1_name == ts2_name)
            # Channel E
            hit_E = bool(sig1 & sig2) and bool(nums1 & nums2)

            if not (hit_A or hit_B or hit_C or hit_E):
                shared_tokens = sig1 & sig2
                shared_nums = nums1 & nums2
                addr1_tokens = set(normalize_text(r1["addr"]).split())
                addr2_tokens = set(normalize_text(r2["addr"]).split())
                shared_addr_tokens = addr1_tokens & addr2_tokens
                missed_examples.append({
                    "s1_id": s1,
                    "target_id": mid,
                    "country": r1["country"],
                    "s1_name": r1["name"],
                    "tgt_name": r2["name"],
                    "s1_addr": r1["addr"],
                    "tgt_addr": r2["addr"],
                    "shared_name_tokens": list(shared_tokens),
                    "shared_nums": list(shared_nums),
                    "shared_addr_tokens": list(shared_addr_tokens)[:5],
                })

    print(f"\nTotal missed in sample: {len(missed_examples)} / {sum(len(t) for t in gt.values())} ({len(missed_examples)/sum(len(t) for t in gt.values())*100:.1f}%)")
    print("\n--- SAMPLE 15 MISSED TRUE MATCHES ---")
    for ex in missed_examples[:15]:
        print(f"[{ex['country']}] S1: {ex['s1_name']} | TGT: {ex['tgt_name']}")
        print(f"     S1 Addr: {ex['s1_addr']}")
        print(f"     TGT Addr: {ex['tgt_addr']}")
        print(f"     Shared Name Tokens: {ex['shared_name_tokens']}")
        print(f"     Shared Nums: {ex['shared_nums']}")
        print(f"     Shared Addr Tokens: {ex['shared_addr_tokens']}\n")

if __name__ == "__main__":
    main()
