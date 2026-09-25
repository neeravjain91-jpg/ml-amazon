#!/usr/bin/env python3
"""
Assemble and Order Submission Files
ML Challenge 2026 - Business Entity Resolution

Combines the computed partitions and guarantees 100% exact line-by-line
alignment with dataset/test/test_source1.tsv.
"""

import os
import sys
import time

def assemble():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("ASSEMBLING FINAL ORDERED SUBMISSION FILES", flush=True)
    print("=" * 70, flush=True)

    # 1. Load matching mappings from both partial files
    matching_map = {}
    print("Loading matching results from initial and remaining files...", flush=True)
    for p in ["output/matching_results.tsv", "output/remaining_matching.tsv"]:
        if not os.path.exists(p):
            print(f"Error: {p} not found!", flush=True)
            return False
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("source1_entity_id"):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                sid = parts[0]
                mids = parts[1] if len(parts) > 1 else ""
                matching_map[sid] = mids

    print(f"Total matching results loaded: {len(matching_map):,}", flush=True)

    # 2. Load candidate mappings from both partial files
    candidate_map = {}
    print("Loading candidate pairs from initial and remaining files...", flush=True)
    for p in ["output/candidate_pairs.tsv", "output/remaining_candidates.tsv"]:
        if not os.path.exists(p):
            print(f"Error: {p} not found!", flush=True)
            return False
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("source1_entity_id"):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                sid = parts[0]
                cids = parts[1] if len(parts) > 1 else ""
                candidate_map[sid] = cids

    print(f"Total candidate pairs loaded: {len(candidate_map):,}", flush=True)

    # 3. Stream dataset/test/test_source1.tsv and write final ordered files
    out_match = "output/matching_results_ordered.tsv"
    out_cand = "output/candidate_pairs_ordered.tsv"

    print(f"Writing aligned output files according to test_source1.tsv order...", flush=True)
    written_count = 0
    missing_count = 0

    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f_src, \
         open(out_match, "w", encoding="utf-8", newline="\n") as f_m, \
         open(out_cand, "w", encoding="utf-8", newline="\n") as f_c:

        # Write exact required headers
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_src)  # Skip source1 header
        for line in f_src:
            sid = line.split("\t")[0]
            written_count += 1

            # Candidate IDs
            cids = candidate_map.get(sid, "")
            # Matched IDs
            mids = matching_map.get(sid, "")

            if sid not in candidate_map or sid not in matching_map:
                missing_count += 1

            f_c.write(f"{sid}\t{cids}\n")
            f_m.write(f"{sid}\t{mids}\n")

            if written_count % 200000 == 0:
                print(f"  Written {written_count:,} aligned records...", flush=True)

    print(f"Assembly completed in {time.time() - start_time:.2f}s:")
    print(f"  Total rows written: {written_count:,}")
    print(f"  Missing records: {missing_count}")

    if missing_count == 0 and written_count == 1732544:
        print("SUCCESS: Exact line-by-line alignment verified. Replacing primary output files...", flush=True)
        # Atomic rename/replace
        if os.path.exists("output/matching_results.tsv"):
            os.remove("output/matching_results.tsv")
        if os.path.exists("output/candidate_pairs.tsv"):
            os.remove("output/candidate_pairs.tsv")
        os.rename(out_match, "output/matching_results.tsv")
        os.rename(out_cand, "output/candidate_pairs.tsv")
        print("Replaced output/matching_results.tsv and output/candidate_pairs.tsv with ordered files.", flush=True)
        return True
    else:
        print(f"ERROR: Expected 1,732,544 rows with 0 missing, got {written_count:,} rows and {missing_count} missing.", flush=True)
        return False

if __name__ == "__main__":
    assemble()
