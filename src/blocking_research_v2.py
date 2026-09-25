#!/usr/bin/env python3
"""
Phase 6: Multi-Channel Blocking Research (v2 Enhanced)
ML Challenge 2026 - Business Entity Resolution
"""

import re
import os
import sys
import time
import unicodedata
from collections import defaultdict

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
    'pvt', 'private', 'co', 'company', 'gmbh', 'sa', 'sarl', 'llp', 'pllc',
    'services', 'solutions', 'enterprises', 'enterprise', 'technologies',
    'technology', 'group', 'holdings', 'international', 'global', 'india',
    'associates', 'trading', 'consulting', 'management', 'industries',
    'industry', 'agency', 'center', 'centre', 'stores', 'store', 'shop'
}

ADDR_STOPWORDS = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'court', 'ct', 'boulevard', 'blvd', 'way', 'place', 'pl', 'near', 'opp',
    'opposite', 'behind', 'beside', 'floor', 'fl', 'block', 'blk', 'sector',
    'sec', 'phase', 'nagar', 'colony', 'city', 'state', 'india', 'usa', 'us'
}

def strip_legal(norm_name):
    tokens = norm_name.split()
    f = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(f) if f else norm_name

def token_sort_form(text):
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def alpha_compact_form(text):
    clean = re.sub(r'[^a-z0-9]', '', text)
    # remove common web extensions like .com, .in, .org
    clean = re.sub(r'(com|org|net|in|co|us)$', '', clean)
    return clean

def extract_numbers_clean(text):
    if not text:
        return set()
    nums = re.findall(r'\b\d+\b', text)
    return set(n.lstrip('0') or '0' for n in nums)

def extract_significant_tokens(norm_name):
    tokens = norm_name.split()
    return [t for t in tokens if len(t) >= 3 and t not in LEGAL_SUFFIXES]

def extract_addr_tokens(norm_addr):
    tokens = norm_addr.split()
    return [t for t in tokens if len(t) >= 4 and t not in ADDR_STOPWORDS and not t.isdigit()]

def run_enhanced_blocking():
    print("=" * 70)
    print("PHASE 6: ENHANCED MULTI-CHANNEL BLOCKING RESEARCH (v2)")
    print("=" * 70)
    t_start = time.time()

    val_s1_ids = []
    with open("reports/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_ids = [line.strip() for line in f if line.strip()]
    val_s1_set = set(val_s1_ids)

    gt = {}
    total_true_matches = 0
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1_set:
                mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
                gt[p[0]] = mids
                total_true_matches += len(mids)

    print(f"Total True Matches to retrieve: {total_true_matches:,} across {len(gt):,} S1 entities.")

    # Query maps
    map_norm_name = defaultdict(list)
    map_strip_name = defaultdict(list)
    map_tokensort_name = defaultdict(list)
    map_compact_name = defaultdict(list)
    map_token_and_num = defaultdict(list)
    map_token_and_addr = defaultdict(list)
    map_rare_name_stem = defaultdict(list)

    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1_id, bname, baddr, c = p[0], p[1], p[2], p[3]
            if s1_id in val_s1_set:
                n_name = normalize_text(bname)
                s_name = strip_legal(n_name)
                ts_name = token_sort_form(s_name)
                compact_name = alpha_compact_form(s_name)
                n_addr = normalize_text(baddr)
                nums = extract_numbers_clean(baddr)
                sig_tokens = extract_significant_tokens(s_name)
                addr_tokens = extract_addr_tokens(n_addr)

                map_norm_name[(c, n_name)].append(s1_id)
                map_strip_name[(c, s_name)].append(s1_id)
                map_tokensort_name[(c, ts_name)].append(s1_id)
                if len(compact_name) >= 5:
                    map_compact_name[(c, compact_name)].append(s1_id)

                for tok in sig_tokens:
                    if len(tok) >= 3:
                        for num in nums:
                            map_token_and_num[(c, tok, num)].append(s1_id)
                        for atok in addr_tokens[:3]:
                            map_token_and_addr[(c, tok, atok)].append(s1_id)

                # If name has rare distinctive stem (len >= 6)
                if sig_tokens:
                    longest = max(sig_tokens, key=len)
                    if len(longest) >= 6:
                        map_rare_name_stem[(c, longest)].append(s1_id)

    print(f"Enhanced query maps built in {time.time() - t_start:.2f}s:")
    print(f"  map_compact_name:   {len(map_compact_name):,}")
    print(f"  map_token_and_num:  {len(map_token_and_num):,}")
    print(f"  map_token_and_addr: {len(map_token_and_addr):,}")
    print(f"  map_rare_name_stem: {len(map_rare_name_stem):,}")

    # Collectors
    cand_base = defaultdict(set) # A + B + C
    cand_compact = defaultdict(set)
    cand_token_num = defaultdict(set)
    cand_token_addr = defaultdict(set)
    cand_rare_stem = defaultdict(set)

    for source_file in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        print(f"\n[Streaming {source_file}]...")
        t0 = time.time()
        row_cnt = 0
        with open(source_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                row_cnt += 1
                p = line.rstrip("\r\n").split("\t")
                tid, bname, baddr, c = p[0], p[1], p[2], p[3]

                n_name = normalize_text(bname)
                s_name = strip_legal(n_name)
                ts_name = token_sort_form(s_name)

                # Base channels
                for k in [(c, n_name), (c, s_name), (c, ts_name)]:
                    if k in map_norm_name:
                        for s1 in map_norm_name[k]:
                            cand_base[s1].add(tid)
                    if k in map_strip_name:
                        for s1 in map_strip_name[k]:
                            cand_base[s1].add(tid)
                    if k in map_tokensort_name:
                        for s1 in map_tokensort_name[k]:
                            cand_base[s1].add(tid)

                # Compact alphanumeric
                c_name = alpha_compact_form(s_name)
                if len(c_name) >= 5 and (c, c_name) in map_compact_name:
                    for s1 in map_compact_name[(c, c_name)]:
                        cand_compact[s1].add(tid)

                sig_tokens = extract_significant_tokens(s_name)
                nums = extract_numbers_clean(baddr)
                n_addr = normalize_text(baddr)
                addr_tokens = extract_addr_tokens(n_addr)

                # Token + Num
                if nums and sig_tokens:
                    for tok in sig_tokens:
                        if len(tok) >= 3:
                            for num in nums:
                                k = (c, tok, num)
                                if k in map_token_and_num:
                                    for s1 in map_token_and_num[k]:
                                        cand_token_num[s1].add(tid)

                # Token + Addr token
                if addr_tokens and sig_tokens:
                    for tok in sig_tokens:
                        if len(tok) >= 3:
                            for atok in addr_tokens[:3]:
                                k = (c, tok, atok)
                                if k in map_token_and_addr:
                                    for s1 in map_token_and_addr[k]:
                                        cand_token_addr[s1].add(tid)

                # Rare name stem (when addr is empty or short)
                if not baddr.strip() and sig_tokens:
                    for tok in sig_tokens:
                        if len(tok) >= 6 and (c, tok) in map_rare_name_stem:
                            for s1 in map_rare_name_stem[(c, tok)]:
                                if len(cand_rare_stem[s1]) < 30:
                                    cand_rare_stem[s1].add(tid)

                if row_cnt % 2000000 == 0:
                    print(f"  Processed {row_cnt:,} records in {time.time() - t0:.1f}s...")

        print(f"Finished {source_file} in {time.time() - t0:.2f}s.")

    # Evaluate combined
    print("\n" + "=" * 70)
    print("EVALUATING ENHANCED MULTI-CHANNEL BLOCKING (v2)")
    print("=" * 70)

    # 1. Base (A+B+C)
    ret_base = sum(len(cand_base[s1] & gt[s1]) for s1 in val_s1_ids if gt.get(s1))
    print(f"Base (A+B+C):           Recall = {ret_base / total_true_matches * 100:.2f}% | Total Cands = {sum(len(c) for c in cand_base.values()):,}")

    # 2. Base + Token_Num
    cand_v1 = defaultdict(set)
    for s1 in val_s1_ids:
        cand_v1[s1] = cand_base[s1] | cand_token_num[s1]
    ret_v1 = sum(len(cand_v1[s1] & gt[s1]) for s1 in val_s1_ids if gt.get(s1))
    print(f"Base + Token_Num:       Recall = {ret_v1 / total_true_matches * 100:.2f}% | Total Cands = {sum(len(c) for c in cand_v1.values()):,}")

    # 3. Base + Token_Num + Compact + Token_Addr + Rare_Stem (Full v2)
    cand_v2 = defaultdict(set)
    for s1 in val_s1_ids:
        cand_v2[s1] = cand_base[s1] | cand_token_num[s1] | cand_compact[s1] | cand_token_addr[s1] | cand_rare_stem[s1]
    ret_v2 = sum(len(cand_v2[s1] & gt[s1]) for s1 in val_s1_ids if gt.get(s1))
    tot_cands_v2 = sum(len(c) for c in cand_v2.values())
    avg_cands_v2 = tot_cands_v2 / len(val_s1_ids)
    print(f"\nFULL ENHANCED BLOCKING (v2):")
    print(f"  Candidate Recall:     {ret_v2 / total_true_matches * 100:.2f}% ({ret_v2:,} / {total_true_matches:,})")
    print(f"  Total Candidates:     {tot_cands_v2:,}")
    print(f"  Average Cands / S1:   {avg_cands_v2:.2f}")
    print(f"  Reduction Ratio:      {(1.0 - tot_cands_v2 / (len(val_s1_ids) * 10320219)) * 100:.5f}%")

    # Update reports/blocking_analysis.md with v2 results
    out_path = "reports/blocking_analysis.md"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n## 3. Enhanced Blocking Architecture (v2 Results)\n\n")
        f.write(f"- **Candidate Recall**: **{ret_v2 / total_true_matches * 100:.2f}%** ({ret_v2:,} out of {total_true_matches:,} true matches retrieved)\n")
        f.write(f"- **Average Candidates per S1**: **{avg_cands_v2:.2f}**\n")
        f.write(f"- **Reduction Ratio**: **{(1.0 - tot_cands_v2 / (len(val_s1_ids) * 10320219)) * 100:.5f}%**\n")
        f.write("- **New Channels Added**: Clean numeric normalization (leading-zero stripping), alphanumeric domain compaction, name token + non-generic address word co-occurrence, and address-empty rare stem recovery.\n")

    print(f"Updated {out_path} successfully!")

if __name__ == "__main__":
    run_enhanced_blocking()
