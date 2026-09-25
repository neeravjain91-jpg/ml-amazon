#!/usr/bin/env python3
"""
Phase 20 (V3 - Mode F): Memory-Bounded Two-Pass Streaming Min-Heap Test Inference Pipeline
ML Challenge 2026 - Business Entity Resolution
Production Implementation of Retrieval V3 with Validated Mode F Canonical Representations
"""

import os
import gc
import sys
import time
import pickle
import heapq
from collections import defaultdict
import numpy as np

sys.path.append('src')
sys.path.append('.')
import features
import canonical_cleaning

# Production decision policy parameters (Strictly Frozen)
TAU = 0.94
DELTA = 0.05
MAX_K_MATCH = 5   # Final multi-match output cap
MAX_HEAP_K = 120   # Candidate retrieval min-heap cap

def compute_cheap_score(s1_meta, t_strip_name, t_tokens, t_nums, channel_count):
    """
    Online multi-signal pre-ranking function for streaming candidate selection.
    Strictly uses only pre-matching lexical and numeric evidence.
    Formula:
      0.50 * name_dice
    + 0.20 * containment
    + 0.20 * num_agree
    + 0.10 * channel_bonus
    """
    # 1. Name 3-gram Dice
    name_dice = features.ngram_dice_similarity(s1_meta["strip_name"], t_strip_name, n=3)

    # 2. Token-level containment
    s1_toks = set(s1_meta["strip_name"].split())
    t_toks = set(t_tokens)
    if s1_toks and t_toks:
        containment = len(s1_toks & t_toks) / min(len(s1_toks), len(t_toks))
    else:
        containment = 0.0

    # 3. Numeric address agreement
    s1_nums = s1_meta["nums"]
    if s1_nums and t_nums:
        num_agree = len(s1_nums & t_nums) / max(len(s1_nums | t_nums), 1)
    elif not s1_nums and not t_nums:
        num_agree = 0.5  # both missing - neutral
    else:
        num_agree = 0.0

    # 4. Channel bonus
    channel_bonus = channel_count / 3.0

    score = (
        0.50 * name_dice
      + 0.20 * containment
      + 0.20 * num_agree
      + 0.10 * channel_bonus
    )
    return score

def run_v3_partition_pipeline(
    country,
    s1_records,        # list of (s1_id, bname, baddr, c)
    source_files,      # list of filepaths to target sources (e.g. [s2, s3])
    model,
    k_ret=MAX_HEAP_K,
    tau=TAU,
    delta=DELTA,
    max_k_match=MAX_K_MATCH,
    f_match_out=None,  # open file handle or None
    f_cand_out=None,   # open file handle or None
):
    """
    Core Two-Pass Partition Execution Engine with Mode F Canonical Representation.
    Executes Pass 1 (heap selection), Pass 2 (target metadata cache), Pass 3 (LightGBM scoring & decision).
    """
    t_part_start = time.time()
    n_s1 = len(s1_records)

    # Step 1: Pre-process S1 metadata using Canonical Cleaning & Mode F Projections
    s1_partition = {}
    s1_order = []
    strip_query = defaultdict(list)
    ts_query = defaultdict(list)
    tok_num_query = defaultdict(list)

    for sid, bname, baddr, c in s1_records:
        s1_order.append(sid)
        rec = canonical_cleaning.canonicalize_record(sid, bname, baddr, c)
        s_name = rec["name_legal_stripped"]
        ts_name = rec["name_token_sorted"]
        nums = set(rec["address_numeric_tokens"].split()) if rec["address_numeric_tokens"] else set()
        sig_tokens = [t for t in s_name.split() if len(t) >= 3 and t not in features.LEGAL_SUFFIXES]

        # Mode F Representation:
        # norm_name uses name_token_sorted to eliminate word-order variance
        # strip_name uses name_legal_stripped to retain natural n-gram character structure
        s1_partition[sid] = {
            "raw_name": rec["name_unicode"],
            "norm_name": rec["name_token_sorted"],
            "strip_name": s_name,
            "token_sort_name": ts_name,
            "raw_addr": rec["address_unicode"],
            "norm_addr": rec["address_clean"],
            "country": rec["country_clean"],
            "nums": nums,
            "sig_tokens": sig_tokens,
        }

        strip_query[s_name].append(sid)
        ts_query[ts_name].append(sid)
        for tok in sig_tokens:
            if len(tok) >= 3:
                for num in nums:
                    tok_num_query[(tok, num)].append(sid)

    # =========================================================================
    # PASS 1: Stream Candidate IDs & Maintain Bounded Top-K Min-Heap
    # NO target metadata stored during Pass 1 to strictly bound memory
    # =========================================================================
    t_pass1_start = time.time()
    heaps = {}  # sid -> list of (score, tid)

    for src_file in source_files:
        with open(src_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                # Fast country filter before full parsing
                line_clean = line.rstrip("\r\n")
                idx_last = line_clean.rfind("\t")
                if idx_last != -1 and line_clean[idx_last+1:] != country:
                    continue

                p = line_clean.split("\t")
                if len(p) != 4:
                    continue
                tid, bname, baddr, c_raw = p[0], p[1], p[2], p[3]

                rec_t = canonical_cleaning.canonicalize_record(tid, bname, baddr, c_raw)
                s_name = rec_t["name_legal_stripped"]
                ts_name = rec_t["name_token_sorted"]
                nums = set(rec_t["address_numeric_tokens"].split()) if rec_t["address_numeric_tokens"] else set()
                sig_tokens = [t for t in s_name.split() if len(t) >= 3 and t not in features.LEGAL_SUFFIXES]

                matched_channels = defaultdict(int)
                if s_name in strip_query:
                    for sid in strip_query[s_name]:
                        matched_channels[sid] += 1

                if ts_name in ts_query:
                    for sid in ts_query[ts_name]:
                        matched_channels[sid] += 1

                if nums and sig_tokens:
                    matched_tok_num = set()
                    for tok in sig_tokens:
                        if len(tok) >= 3:
                            for num in nums:
                                k = (tok, num)
                                if k in tok_num_query:
                                    for sid in tok_num_query[k]:
                                        matched_tok_num.add(sid)
                    for sid in matched_tok_num:
                        matched_channels[sid] += 1

                if not matched_channels:
                    continue

                for sid, ch_cnt in matched_channels.items():
                    sc = compute_cheap_score(s1_partition[sid], s_name, sig_tokens, nums, min(ch_cnt, 3))
                    entry = (sc, tid)

                    if sid not in heaps:
                        heaps[sid] = [entry]
                    else:
                        h = heaps[sid]
                        if len(h) < k_ret:
                            heapq.heappush(h, entry)
                        elif entry > h[0]:
                            heapq.heapreplace(h, entry)

    # Collect surviving unique target IDs across all heaps
    surviving_tids = set()
    for h in heaps.values():
        for _, tid in h:
            surviving_tids.add(tid)

    # Free query indices to liberate RAM before Pass 2
    del strip_query
    del ts_query
    del tok_num_query
    gc.collect()
    t_pass1 = time.time() - t_pass1_start

    # =========================================================================
    # PASS 2: Fast Re-Stream to Cache Metadata ONLY for Surviving Targets
    # Uses O(1) tid lookup to bypass non-retained rows in sub-microseconds
    # =========================================================================
    t_pass2_start = time.time()
    target_cache = {}
    target_target_count = len(surviving_tids)

    for src_file in source_files:
        with open(src_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                idx = line.find("\t")
                if idx == -1:
                    continue
                tid = line[:idx]
                if tid in surviving_tids and tid not in target_cache:
                    p = line.rstrip("\r\n").split("\t")
                    if len(p) != 4:
                        continue
                    bname, baddr, c_raw = p[1], p[2], p[3]
                    rec_t = canonical_cleaning.canonicalize_record(tid, bname, baddr, c_raw)
                    target_cache[tid] = {
                        "raw_name": rec_t["name_unicode"],
                        "norm_name": rec_t["name_token_sorted"],
                        "strip_name": rec_t["name_legal_stripped"],
                        "raw_addr": rec_t["address_unicode"],
                        "norm_addr": rec_t["address_clean"],
                        "country": rec_t["country_clean"],
                        "nums": set(rec_t["address_numeric_tokens"].split()) if rec_t["address_numeric_tokens"] else set(),
                    }
                    if len(target_cache) == target_target_count:
                        break
        if len(target_cache) == target_target_count:
            break
    t_pass2 = time.time() - t_pass2_start

    # =========================================================================
    # PASS 3: Chunked Batch Scoring & Decision Engine
    # =========================================================================
    t_pass3_start = time.time()
    chunk_size = 2000

    partition_cands = {}
    partition_preds = {}

    for i in range(0, n_s1, chunk_size):
        chunk_s1 = s1_order[i:i+chunk_size]
        pairs = []
        keys = []

        for sid in chunk_s1:
            h = heaps.get(sid, [])
            # Sort candidates by (score, tid) descending
            sorted_entries = sorted(h, key=lambda x: (x[0], x[1]), reverse=True)
            cand_tids = [tid for _, tid in sorted_entries]
            partition_cands[sid] = set(cand_tids)

            if f_cand_out:
                f_cand_out.write(f"{sid}\t{','.join(cand_tids)}\n")

            if not cand_tids:
                continue

            r1 = s1_partition[sid]
            for tid in cand_tids:
                r2 = target_cache[tid]
                pairs.append(features.compute_pairwise_features(r1, r2))
                keys.append((sid, tid))

        s1_scores = defaultdict(list)
        if pairs:
            X = np.array(pairs, dtype=np.float32)
            scores = model.predict_proba(X)[:, 1]
            for (sid, tid), sc in zip(keys, scores):
                s1_scores[sid].append((tid, float(sc)))

        for sid in chunk_s1:
            sc_list = s1_scores.get(sid, [])
            if not sc_list:
                partition_preds[sid] = set()
                if f_match_out:
                    f_match_out.write(f"{sid}\t\n")
            else:
                sc_list.sort(key=lambda x: x[1], reverse=True)
                top_score = sc_list[0][1]
                if top_score < tau:
                    partition_preds[sid] = set()
                    if f_match_out:
                        f_match_out.write(f"{sid}\t\n")
                else:
                    selected = [tid for tid, sc in sc_list if sc >= tau and sc >= (top_score - delta)]
                    final_matches = selected[:max_k_match]
                    partition_preds[sid] = set(final_matches)
                    if f_match_out:
                        f_match_out.write(f"{sid}\t{','.join(final_matches)}\n")

    t_pass3 = time.time() - t_pass3_start

    # Clean up structures
    del s1_partition
    del heaps
    del target_cache
    del surviving_tids
    gc.collect()

    timings = {
        "pass1_s": t_pass1,
        "pass2_s": t_pass2,
        "pass3_s": t_pass3,
        "total_part_s": time.time() - t_part_start,
    }

    return partition_cands, partition_preds, timings
