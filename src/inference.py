#!/usr/bin/env python3
"""
Phase 20: Full Test Set Inference Pipeline
ML Challenge 2026 - Business Entity Resolution
Produces:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
"""

import os
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

def run_test_inference():
    print("=" * 70)
    print("PHASE 20: FULL TEST SET INFERENCE PIPELINE")
    print("=" * 70)
    t_global_start = time.time()

    # 1. Load trained production LightGBM model
    model_path = "output/lgb_matching_model.pkl"
    print(f"Loading production model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    # 2. Read test_source1.tsv and organize by country
    print("\n[Step 1] Loading test_source1.tsv...")
    test_s1_order = []
    test_s1_by_country = defaultdict(dict)

    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1_id, bname, baddr, c = p[0], p[1], p[2], p[3]
            test_s1_order.append(s1_id)
            
            n_name = features.normalize_text(bname)
            s_name = features.strip_legal(n_name)
            ts_name = features.token_sort_form(s_name)
            nums = features.extract_numbers_clean(baddr)
            sig_tokens = features.extract_significant_tokens(s_name)

            test_s1_by_country[c][s1_id] = {
                "raw_name": bname,
                "norm_name": n_name,
                "strip_name": s_name,
                "ts_name": ts_name,
                "raw_addr": baddr,
                "norm_addr": features.normalize_text(baddr),
                "country": c,
                "nums": nums,
                "sig_tokens": sig_tokens,
            }

    total_test_s1 = len(test_s1_order)
    print(f"Total Test S1 entities: {total_test_s1:,}")
    for c, entities in test_s1_by_country.items():
        print(f"  Country '{c}': {len(entities):,} entities ({len(entities)/total_test_s1*100:.2f}%)")

    # Initialize output TSV files with headers
    os.makedirs("output", exist_ok=True)
    matching_path = "output/matching_results.tsv"
    candidate_path = "output/candidate_pairs.tsv"

    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    total_written_matches = 0
    total_written_cands = 0
    total_written_singletons = 0
    total_written_s1 = 0

    # 3. Process each country partition independently
    # Order: France (open-set, smallest ~259k), then US (~663k), then India (~810k)
    countries_to_process = ["France", "US", "India"]
    
    for country in countries_to_process:
        s1_partition = test_s1_by_country.get(country, {})
        if not s1_partition:
            continue

        print("\n" + "=" * 70)
        print(f"PROCESSING PARTITION: {country.upper()} ({len(s1_partition):,} S1 Entities)")
        print("=" * 70)
        t_country_start = time.time()

        # Build partition query indexes
        strip_name_query = defaultdict(list)
        ts_name_query = defaultdict(list)
        tok_num_query = defaultdict(list)

        for s1_id, r in s1_partition.items():
            strip_name_query[r["strip_name"]].append(s1_id)
            ts_name_query[r["ts_name"]].append(s1_id)
            for tok in r["sig_tokens"]:
                if len(tok) >= 3:
                    for num in r["nums"]:
                        tok_num_query[(tok, num)].append(s1_id)

        print(f"Query maps built: strip_name={len(strip_name_query):,}, ts_name={len(ts_name_query):,}, tok_num={len(tok_num_query):,}")

        # Stream candidate collection for this country
        cand_map = defaultdict(set)
        target_cache = {}

        for test_source in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
            print(f"Streaming {test_source} for {country}...")
            t_src = time.time()
            cnt = 0
            with open(test_source, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    p = line.rstrip("\r\n").split("\t")
                    tid, bname, baddr, c = p[0], p[1], p[2], p[3]
                    
                    if c != country:
                        continue
                    cnt += 1

                    n_name = features.normalize_text(bname)
                    s_name = features.strip_legal(n_name)
                    ts_name = features.token_sort_form(s_name)
                    nums = features.extract_numbers_clean(baddr)
                    sig_tokens = features.extract_significant_tokens(s_name)

                    matched_s1 = []
                    # 1. Strip name match
                    if s_name in strip_name_query:
                        matched_s1.extend(strip_name_query[s_name])
                    # 2. Token-sort name match
                    if ts_name in ts_name_query:
                        matched_s1.extend(ts_name_query[ts_name])
                    # 3. Token + Num co-occurrence
                    if nums and sig_tokens:
                        for tok in sig_tokens:
                            if len(tok) >= 3:
                                for num in nums:
                                    k = (tok, num)
                                    if k in tok_num_query:
                                        matched_s1.extend(tok_num_query[k])

                    if matched_s1:
                        if tid not in target_cache:
                            target_cache[tid] = {
                                "raw_name": bname,
                                "norm_name": n_name,
                                "strip_name": s_name,
                                "raw_addr": baddr,
                                "norm_addr": features.normalize_text(baddr),
                                "country": c,
                                "nums": nums,
                            }
                        for s1_id in matched_s1:
                            cand_map[s1_id].add(tid)

            print(f"  Processed {cnt:,} {country} records in {time.time() - t_src:.1f}s.")

        print(f"Candidates generated for {len(cand_map):,} / {len(s1_partition):,} {country} entities.")
        print(f"Total target records cached: {len(target_cache):,}")

        # Batch scoring chunk by chunk and directly stream to output files
        print(f"Scoring {country} candidate pairs with LightGBM and streaming to disk...")
        s1_ids_list = list(s1_partition.keys())
        chunk_size = 2000

        with open(matching_path, "a", encoding="utf-8") as f_match, open(candidate_path, "a", encoding="utf-8") as f_cand:
            for i in range(0, len(s1_ids_list), chunk_size):
                chunk_s1 = s1_ids_list[i:i+chunk_size]
                pairs = []
                keys = []
                
                for s1_id in chunk_s1:
                    r1 = s1_partition[s1_id]
                    cands = list(cand_map.get(s1_id, set()))
                    total_written_cands += len(cands)
                    f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

                    if not cands:
                        f_match.write(f"{s1_id}\t\n")
                        total_written_singletons += 1
                        total_written_s1 += 1
                        continue

                    for tid in cands:
                        r2 = target_cache[tid]
                        pairs.append(features.compute_pairwise_features(r1, r2))
                        keys.append((s1_id, tid))

                if pairs:
                    X = np.array(pairs, dtype=np.float32)
                    scores = model.predict_proba(X)[:, 1]

                    # Group scores by S1
                    s1_scores = defaultdict(list)
                    for (s1_id, tid), sc in zip(keys, scores):
                        s1_scores[s1_id].append((tid, float(sc)))

                    # Apply Decision Engine policy per S1
                    for s1_id in chunk_s1:
                        if s1_id not in s1_scores:
                            continue
                            
                        cand_scores = s1_scores[s1_id]
                        cand_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        top_score = cand_scores[0][1]
                        if top_score < TAU:
                            # Singleton / no-match gating
                            f_match.write(f"{s1_id}\t\n")
                            total_written_singletons += 1
                        else:
                            # Multi-match margin gating
                            selected = [tid for tid, sc in cand_scores if sc >= TAU and sc >= (top_score - DELTA)]
                            final_selected = selected[:MAX_K]
                            f_match.write(f"{s1_id}\t{','.join(final_selected)}\n")
                            total_written_matches += len(final_selected)
                        total_written_s1 += 1

                if (i + chunk_size) % 50000 < chunk_size:
                    print(f"  Processed {min(i + chunk_size, len(s1_ids_list)):,} / {len(s1_ids_list):,} {country} S1 entities...")

        print(f"Completed {country} partition in {time.time() - t_country_start:.2f}s.")

    print("\n" + "=" * 70)
    print("TEST SET INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total Processed S1 Entities: {total_written_s1:,} / {total_test_s1:,}")
    print(f"Total Matches Predicted:     {total_written_matches:,}")
    print(f"Total Singletons Predicted:  {total_written_singletons:,} ({total_written_singletons/total_written_s1*100:.2f}%)")
    print(f"Total Candidate Pairs:       {total_written_cands:,}")
    print(f"Total Execution Runtime:     {time.time() - t_global_start:.2f}s")
    print(f"Output files saved: {matching_path}, {candidate_path}")

if __name__ == "__main__":
    run_test_inference()
