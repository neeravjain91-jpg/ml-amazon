#!/usr/bin/env python3
"""
Phase 20: Partitioned, Memory-Bounded Full Test Inference Pipeline
ML Challenge 2026 - Business Entity Resolution
"""

import os
import gc
import sys
import time
import pickle
import numpy as np
from collections import defaultdict

sys.path.append('src')
sys.path.append('.')
import features

# Production decision policy parameters
TAU = 0.94
DELTA = 0.05
MAX_K = 5
MAX_CANDS_PER_S1 = 40

def run_partitioned_inference():
    print("=" * 70)
    print("PHASE 20: MEMORY-BOUNDED FULL TEST INFERENCE PIPELINE")
    print("=" * 70)
    t_global_start = time.time()

    # 1. Load trained production LightGBM model
    model_path = "output/lgb_matching_model.pkl"
    print(f"Loading production model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    os.makedirs("output", exist_ok=True)
    matching_path = "output/matching_results.tsv"
    candidate_path = "output/candidate_pairs.tsv"

    # Initialize output TSV files with headers
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    total_written_s1 = 0
    total_written_matches = 0
    total_written_cands = 0
    total_written_singletons = 0

    # Process one country at a time to strictly guarantee RAM < 600 MB
    countries = ["France", "US", "India"]

    for country in countries:
        print("\n" + "=" * 70)
        print(f"STARTING PARTITION: {country.upper()}")
        print("=" * 70)
        t_country_start = time.time()

        # Step 1: Read ONLY this country's records from test_source1.tsv
        print(f"Loading {country} entities from dataset/test/test_source1.tsv...")
        s1_partition = {}
        s1_order = []

        strip_name_query = defaultdict(list)
        ts_name_query = defaultdict(list)
        tok_num_query = defaultdict(list)

        with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                c = p[3]
                if c != country:
                    continue
                s1_id, bname, baddr = p[0], p[1], p[2]
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
                    "country": c,
                    "nums": nums,
                }

                strip_name_query[s_name].append(s1_id)
                ts_name_query[ts_name].append(s1_id)
                for tok in sig_tokens:
                    if len(tok) >= 3:
                        for num in nums:
                            tok_num_query[(tok, num)].append(s1_id)

        n_s1 = len(s1_order)
        print(f"Loaded {n_s1:,} {country} S1 entities. Query maps ready.")

        # Step 2: Stream through test_source2.tsv and test_source3.tsv for this country
        cand_map = defaultdict(set)
        target_cache = {}

        for test_source in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
            print(f"Streaming {test_source} for {country} candidates...")
            t_src = time.time()
            cnt = 0
            with open(test_source, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    p = line.rstrip("\r\n").split("\t")
                    if p[3] != country:
                        continue
                    cnt += 1
                    tid, bname, baddr, c = p[0], p[1], p[2], p[3]

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
                        for s1_id in matched_s1:
                            if len(cand_map[s1_id]) < MAX_CANDS_PER_S1:
                                cand_map[s1_id].add(tid)
                                stored = True
                        if stored and tid not in target_cache:
                            target_cache[tid] = {
                                "raw_name": bname,
                                "norm_name": n_name,
                                "strip_name": s_name,
                                "raw_addr": baddr,
                                "norm_addr": features.normalize_text(baddr),
                                "country": c,
                                "nums": nums,
                            }

            print(f"  Streamed {cnt:,} {country} records in {time.time() - t_src:.1f}s.")

        print(f"Total target records cached: {len(target_cache):,}")
        print(f"Entities with candidates: {len(cand_map):,} / {n_s1:,}")

        # Free query index maps before scoring
        del strip_name_query
        del ts_name_query
        del tok_num_query
        gc.collect()

        # Step 3: Batch scoring chunk by chunk and directly stream to output files
        print(f"Scoring {country} candidate pairs with LightGBM and streaming to disk...")
        chunk_size = 2000

        country_matches = 0
        country_cands = 0
        country_singletons = 0

        with open(matching_path, "a", encoding="utf-8") as f_match, open(candidate_path, "a", encoding="utf-8") as f_cand:
            for i in range(0, n_s1, chunk_size):
                chunk_s1 = s1_order[i:i+chunk_size]
                pairs = []
                keys = []

                for s1_id in chunk_s1:
                    r1 = s1_partition[s1_id]
                    cands = list(cand_map.get(s1_id, set()))
                    country_cands += len(cands)
                    f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

                    if not cands:
                        continue

                    for tid in cands:
                        r2 = target_cache[tid]
                        pairs.append(features.compute_pairwise_features(r1, r2))
                        keys.append((s1_id, tid))

                s1_scores = defaultdict(list)
                if pairs:
                    X = np.array(pairs, dtype=np.float32)
                    scores = model.predict_proba(X)[:, 1]
                    for (s1_id, tid), sc in zip(keys, scores):
                        s1_scores[s1_id].append((tid, float(sc)))

                for s1_id in chunk_s1:
                    if s1_id not in s1_scores or not s1_scores[s1_id]:
                        f_match.write(f"{s1_id}\t\n")
                        country_singletons += 1
                    else:
                        cand_scores = s1_scores[s1_id]
                        cand_scores.sort(key=lambda x: x[1], reverse=True)

                        top_score = cand_scores[0][1]
                        if top_score < TAU:
                            f_match.write(f"{s1_id}\t\n")
                            country_singletons += 1
                        else:
                            selected = [tid for tid, sc in cand_scores if sc >= TAU and sc >= (top_score - DELTA)]
                            final_sel = selected[:MAX_K]
                            f_match.write(f"{s1_id}\t{','.join(final_sel)}\n")
                            country_matches += len(final_sel)

                if (i + chunk_size) % 50000 < chunk_size:
                    print(f"  Scored {min(i + chunk_size, n_s1):,} / {n_s1:,} {country} S1 entities...")

        total_written_s1 += n_s1
        total_written_matches += country_matches
        total_written_cands += country_cands
        total_written_singletons += country_singletons

        print(f"Partition {country} complete in {time.time() - t_country_start:.2f}s:")
        print(f"  S1 Entities: {n_s1:,} | Matches: {country_matches:,} | Singletons: {country_singletons:,} ({country_singletons/n_s1*100:.2f}%)")

        # Free all partition structures
        del s1_partition
        del s1_order
        del cand_map
        del target_cache
        gc.collect()

    print("\n" + "=" * 70)
    print("FULL TEST INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Total S1 Processed:       {total_written_s1:,} (Target: 1,732,544)")
    print(f"Total Resolved Matches:   {total_written_matches:,}")
    print(f"Total Singletons:         {total_written_singletons:,} ({total_written_singletons/total_written_s1*100:.2f}%)")
    print(f"Total Blocking Cands:     {total_written_cands:,}")
    print(f"Total Execution Runtime:  {time.time() - t_global_start:.2f}s")
    print(f"Outputs generated: {matching_path}, {candidate_path}")

if __name__ == "__main__":
    run_partitioned_inference()
