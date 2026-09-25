#!/usr/bin/env python3
"""
Finish inference for the remaining 318,512 India entities.
ML Challenge 2026 - Business Entity Resolution
"""

import os
import sys
import gc
import time
import pickle
from collections import defaultdict
import numpy as np

# Ensure src/ is in python path
sys.path.insert(0, os.path.dirname(__file__))
import features

MAX_CANDS_PER_S1 = 40
TAU = 0.94
DELTA = 0.05
MAX_K = 5

def main():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("FINISHING REMAINING INDIA TEST ENTITIES", flush=True)
    print("=" * 70, flush=True)

    # 1. Identify completed S1 IDs
    existing_s1 = set()
    with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            existing_s1.add(line.split("\t")[0])
    print(f"Loaded {len(existing_s1):,} already completed S1 IDs.", flush=True)

    # 2. Load model
    print("Loading LightGBM model...", flush=True)
    with open("output/lgb_matching_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 3. Read remaining India S1 entities from test_source1.tsv
    print("Loading remaining India S1 entities...", flush=True)
    s1_partition = {}
    s1_order = []
    strip_name_query = defaultdict(list)
    ts_name_query = defaultdict(list)
    tok_num_query = defaultdict(list)

    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[3] != "India":
                continue
            s1_id = p[0]
            if s1_id in existing_s1:
                continue

            bname, baddr = p[1], p[2]
            s1_order.append(s1_id)

            n_name = features.normalize_text(bname)
            s_name = features.strip_legal(n_name)
            ts_name = features.token_sort_form(s_name)
            nums = features.extract_numbers_clean(baddr)
            sig_tokens = features.extract_significant_tokens(s_name)

            s1_partition[s1_id] = {
                "raw_name": bname,
                "norm_name": n_name,
                "strip_name": s_name,
                "raw_addr": baddr,
                "norm_addr": features.normalize_text(baddr),
                "country": "India",
                "nums": nums,
            }

            strip_name_query[s_name].append(s1_id)
            ts_name_query[ts_name].append(s1_id)
            for tok in sig_tokens:
                if len(tok) >= 3:
                    for num in nums:
                        tok_num_query[(tok, num)].append(s1_id)

    n_s1 = len(s1_order)
    print(f"Identified {n_s1:,} remaining India S1 entities to score.", flush=True)
    assert n_s1 == 318512, f"Expected 318,512 remaining entities, got {n_s1:,}"

    # 4. Stream Source 2 and Source 3 for candidates
    cand_map = defaultdict(set)
    target_cache = {}

    for test_source in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
        print(f"Streaming {test_source} for India candidates...", flush=True)
        t_src = time.time()
        cnt = 0
        with open(test_source, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[3] != "India":
                    continue
                cnt += 1
                tid, bname, baddr = p[0], p[1], p[2]

                n_name = features.normalize_text(bname)
                s_name = features.strip_legal(n_name)
                ts_name = features.token_sort_form(s_name)
                nums = features.extract_numbers_clean(baddr)
                sig_tokens = features.extract_significant_tokens(s_name)

                matched_s1 = []
                if s_name in strip_name_query:
                    matched_s1.extend(strip_name_query[s_name])
                if ts_name in ts_name_query:
                    matched_s1.extend(ts_name_query[ts_name])
                if nums and sig_tokens:
                    for tok in sig_tokens:
                        if len(tok) >= 3:
                            for num in nums:
                                k = (tok, num)
                                if k in tok_num_query:
                                    matched_s1.extend(tok_num_query[k])

                if matched_s1:
                    stored = False
                    for sid in matched_s1:
                        if len(cand_map[sid]) < MAX_CANDS_PER_S1:
                            cand_map[sid].add(tid)
                            stored = True
                    if stored and tid not in target_cache:
                        target_cache[tid] = {
                            "raw_name": bname,
                            "norm_name": n_name,
                            "strip_name": s_name,
                            "raw_addr": baddr,
                            "norm_addr": features.normalize_text(baddr),
                            "country": "India",
                            "nums": nums,
                        }

        print(f"  Streamed {cnt:,} India records in {time.time() - t_src:.1f}s.", flush=True)

    print(f"Total target records cached: {len(target_cache):,}", flush=True)
    print(f"Entities with candidates: {len(cand_map):,} / {n_s1:,}", flush=True)

    # Free index maps
    del strip_name_query
    del ts_name_query
    del tok_num_query
    gc.collect()

    # 5. Batch scoring chunk by chunk and write to remaining files
    rem_matching_path = "output/remaining_matching.tsv"
    rem_candidate_path = "output/remaining_candidates.tsv"

    print("Scoring candidate pairs with LightGBM and streaming to remaining files...", flush=True)
    chunk_size = 2000

    matches_written = 0
    singletons_written = 0

    with open(rem_matching_path, "w", encoding="utf-8") as f_match, open(rem_candidate_path, "w", encoding="utf-8") as f_cand:
        for i in range(0, n_s1, chunk_size):
            chunk_s1 = s1_order[i:i+chunk_size]
            pairs = []
            keys = []

            for sid in chunk_s1:
                r1 = s1_partition[sid]
                cands = list(cand_map.get(sid, set()))
                f_cand.write(f"{sid}\t{','.join(cands)}\n")

                if not cands:
                    continue

                for tid in cands:
                    r2 = target_cache[tid]
                    pairs.append(features.compute_pairwise_features(r1, r2))
                    keys.append((sid, tid))

            s1_scores = defaultdict(list)
            if pairs:
                X = np.array(pairs, dtype=np.float32)
                scores = model.predict_proba(X)[:, 1]
                for (sid, tid), sc in zip(keys, scores):
                    s1_scores[sid].append((tid, float(sc)))

            # Write matching results in the EXACT order of chunk_s1
            for sid in chunk_s1:
                if sid not in s1_scores or not s1_scores[sid]:
                    f_match.write(f"{sid}\t\n")
                    singletons_written += 1
                else:
                    cand_scores = s1_scores[sid]
                    cand_scores.sort(key=lambda x: x[1], reverse=True)
                    top_score = cand_scores[0][1]

                    if top_score < TAU:
                        f_match.write(f"{sid}\t\n")
                        singletons_written += 1
                    else:
                        selected = [tid for tid, sc in cand_scores if sc >= TAU and sc >= (top_score - DELTA)]
                        final_sel = selected[:MAX_K]
                        f_match.write(f"{sid}\t{','.join(final_sel)}\n")
                        matches_written += len(final_sel)

            if (i + chunk_size) % 50000 < chunk_size:
                print(f"  Scored {min(i + chunk_size, n_s1):,} / {n_s1:,} entities...", flush=True)

    print(f"Remaining India entities finished in {time.time() - start_time:.1f}s:", flush=True)
    print(f"  Matches: {matches_written:,} | Singletons: {singletons_written:,}", flush=True)

if __name__ == "__main__":
    main()
