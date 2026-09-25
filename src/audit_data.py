#!/usr/bin/env python3
"""
Phase 2: Comprehensive Data Integrity Audit
ML Challenge 2026 - Business Entity Resolution
"""

import os
import sys
import json
import time
from collections import Counter

DELIM = "\t"

TRAIN_FILES = {
    "train_source1": "dataset/train/train_source1.tsv",
    "train_source2": "dataset/train/train_source2.tsv",
    "train_source3": "dataset/train/train_source3.tsv",
}

TEST_FILES = {
    "test_source1": "dataset/test/test_source1.tsv",
    "test_source2": "dataset/test/test_source2.tsv",
    "test_source3": "dataset/test/test_source3.tsv",
}

GT_FILE = "dataset/train/train_ground_truth.tsv"

EXPECTED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLS = ["source1_entity_id", "matched_entity_ids"]

PREFIX_MAP = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}

def audit_source_file(name, filepath, expected_prefix):
    print(f"\n[Auditing {name}] Path: {filepath} ({os.path.getsize(filepath):,} bytes)")
    t0 = time.time()
    
    row_count = 0
    col_counts = Counter()
    headers = []
    
    unique_ids = set()
    dup_ids = set()
    malformed_prefix_ids = []
    
    null_counts = {col: 0 for col in EXPECTED_SOURCE_COLS}
    whitespace_only_counts = {col: 0 for col in EXPECTED_SOURCE_COLS}
    leading_trailing_ws = {col: 0 for col in EXPECTED_SOURCE_COLS}
    unicode_error_count = 0
    
    countries = Counter()
    
    name_lengths = []
    addr_lengths = []
    
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return {"error": "empty file"}
        headers = [c.strip() for c in header_line.rstrip("\r\n").split(DELIM)]
        
        for line_num, line in enumerate(f, start=2):
            row_count += 1
            parts = line.rstrip("\r\n").split(DELIM)
            col_counts[len(parts)] += 1
            
            if len(parts) != 4:
                continue
                
            eid, bname, baddr, country = parts
            
            # ID checks
            eid_clean = eid.strip()
            if eid_clean in unique_ids:
                dup_ids.add(eid_clean)
            unique_ids.add(eid_clean)
            
            if not eid_clean.startswith(expected_prefix):
                if len(malformed_prefix_ids) < 10:
                    malformed_prefix_ids.append((line_num, eid_clean))
            
            # Value checks
            vals = [eid, bname, baddr, country]
            for col_name, val in zip(EXPECTED_SOURCE_COLS, vals):
                if val == "":
                    null_counts[col_name] += 1
                elif val.strip() == "":
                    whitespace_only_counts[col_name] += 1
                elif val != val.strip():
                    leading_trailing_ws[col_name] += 1
                    
            country_clean = country.strip()
            countries[country_clean] += 1
            
            name_lengths.append(len(bname))
            addr_lengths.append(len(baddr))
            
    t1 = time.time()
    
    name_lengths.sort()
    addr_lengths.sort()
    n = len(name_lengths)
    
    def get_stats(arr):
        if not arr:
            return {}
        return {
            "min": arr[0],
            "p25": arr[int(0.25 * n)],
            "median": arr[int(0.50 * n)],
            "p75": arr[int(0.75 * n)],
            "p95": arr[int(0.95 * n)],
            "max": arr[-1],
            "mean": round(sum(arr) / len(arr), 2)
        }
        
    res = {
        "file": filepath,
        "size_bytes": os.path.getsize(filepath),
        "row_count": row_count,
        "columns": headers,
        "col_counts_distribution": dict(col_counts),
        "unique_id_count": len(unique_ids),
        "duplicate_id_count": len(dup_ids),
        "sample_duplicate_ids": list(dup_ids)[:5],
        "malformed_prefix_count": len(malformed_prefix_ids),
        "sample_malformed_prefix": malformed_prefix_ids[:5],
        "null_counts": null_counts,
        "whitespace_only_counts": whitespace_only_counts,
        "leading_trailing_whitespace_counts": leading_trailing_ws,
        "countries": dict(countries),
        "name_length_stats": get_stats(name_lengths),
        "addr_length_stats": get_stats(addr_lengths),
        "audit_time_sec": round(t1 - t0, 2)
    }
    
    print(f"  Rows: {row_count:,} | Unique IDs: {len(unique_ids):,} | Dups: {len(dup_ids)} | Countries: {dict(countries)}")
    print(f"  Done in {res['audit_time_sec']}s")
    return res, unique_ids


def audit_ground_truth(filepath):
    print(f"\n[Auditing Ground Truth] Path: {filepath} ({os.path.getsize(filepath):,} bytes)")
    t0 = time.time()
    
    row_count = 0
    unique_s1_ids = set()
    dup_s1_ids = set()
    col_counts = Counter()
    
    empty_matches_count = 0
    total_match_ids = 0
    s2_match_count = 0
    s3_match_count = 0
    malformed_matches = []
    intra_list_dupes = 0
    
    match_cardinality = Counter()
    s2_s3_overlap_breakdown = Counter() # only_s2, only_s3, both, neither
    
    with open(filepath, "r", encoding="utf-8") as f:
        header_line = f.readline()
        headers = [c.strip() for c in header_line.rstrip("\r\n").split(DELIM)]
        
        for line_num, line in enumerate(f, start=2):
            row_count += 1
            parts = line.rstrip("\r\n").split(DELIM)
            col_counts[len(parts)] += 1
            
            s1_id = parts[0].strip()
            if s1_id in unique_s1_ids:
                dup_s1_ids.add(s1_id)
            unique_s1_ids.add(s1_id)
            
            raw_matches = parts[1].strip() if len(parts) > 1 else ""
            if not raw_matches:
                empty_matches_count += 1
                match_cardinality[0] += 1
                s2_s3_overlap_breakdown["singleton"] += 1
                continue
                
            mids = raw_matches.split(",")
            match_cardinality[len(mids)] += 1
            if len(mids) != len(set(mids)):
                intra_list_dupes += 1
                
            has_s2 = False
            has_s3 = False
            for mid in mids:
                mid = mid.strip()
                total_match_ids += 1
                if mid.startswith("S2-"):
                    s2_match_count += 1
                    has_s2 = True
                elif mid.startswith("S3-"):
                    s3_match_count += 1
                    has_s3 = True
                else:
                    malformed_matches.append((line_num, mid))
                    
            if has_s2 and has_s3:
                s2_s3_overlap_breakdown["both_s2_and_s3"] += 1
            elif has_s2:
                s2_s3_overlap_breakdown["only_s2"] += 1
            elif has_s3:
                s2_s3_overlap_breakdown["only_s3"] += 1
            else:
                s2_s3_overlap_breakdown["other"] += 1
                
    t1 = time.time()
    
    res = {
        "file": filepath,
        "size_bytes": os.path.getsize(filepath),
        "row_count": row_count,
        "columns": headers,
        "col_counts_distribution": dict(col_counts),
        "unique_s1_count": len(unique_s1_ids),
        "duplicate_s1_count": len(dup_s1_ids),
        "empty_matches_count (singletons)": empty_matches_count,
        "singleton_ratio": round(empty_matches_count / row_count, 4) if row_count else 0,
        "total_matched_ids": total_match_ids,
        "s2_match_count": s2_match_count,
        "s3_match_count": s3_match_count,
        "malformed_match_ids_count": len(malformed_matches),
        "sample_malformed_match_ids": malformed_matches[:5],
        "intra_list_duplicates_count": intra_list_dupes,
        "overlap_breakdown": dict(s2_s3_overlap_breakdown),
        "match_cardinality_distribution": {str(k): v for k, v in sorted(match_cardinality.items())},
        "audit_time_sec": round(t1 - t0, 2)
    }
    
    print(f"  Rows: {row_count:,} | Unique S1: {len(unique_s1_ids):,} | Singletons: {empty_matches_count:,} ({res['singleton_ratio']*100:.2f}%)")
    print(f"  Total match IDs: {total_match_ids:,} (S2: {s2_match_count:,}, S3: {s3_match_count:,})")
    print(f"  Breakdown: {dict(s2_s3_overlap_breakdown)}")
    print(f"  Done in {res['audit_time_sec']}s")
    return res, unique_s1_ids


def main():
    print("=" * 70)
    print("STARTING COMPREHENSIVE DATA INTEGRITY AUDIT (PHASE 2 & 3)")
    print("=" * 70)
    
    audit_results = {}
    id_sets = {}
    
    # Audit train sources
    for key, path in TRAIN_FILES.items():
        prefix = PREFIX_MAP[key.split("_")[1]]
        res, ids = audit_source_file(key, path, prefix)
        audit_results[key] = res
        id_sets[key] = ids
        
    # Audit test sources
    for key, path in TEST_FILES.items():
        prefix = PREFIX_MAP[key.split("_")[1]]
        res, ids = audit_source_file(key, path, prefix)
        audit_results[key] = res
        id_sets[key] = ids
        
    # Audit ground truth
    gt_res, gt_s1_ids = audit_ground_truth(GT_FILE)
    audit_results["train_ground_truth"] = gt_res
    
    # Cross-source ID integrity checks
    print("\n[Cross-Source ID Integrity Checks]")
    cross_checks = {}
    
    # 1. Ground truth S1 IDs vs train_source1 IDs
    s1_train_ids = id_sets["train_source1"]
    gt_missing_in_s1 = gt_s1_ids - s1_train_ids
    s1_missing_in_gt = s1_train_ids - gt_s1_ids
    cross_checks["gt_s1_ids_not_in_train_s1"] = len(gt_missing_in_s1)
    cross_checks["train_s1_ids_not_in_gt"] = len(s1_missing_in_gt)
    print(f"  GT S1 IDs not in train_source1: {len(gt_missing_in_s1)}")
    print(f"  train_source1 IDs not in GT: {len(s1_missing_in_gt)}")
    
    # 2. Train vs Test ID overlap (Leakage / partition check)
    train_all_ids = id_sets["train_source1"] | id_sets["train_source2"] | id_sets["train_source3"]
    test_all_ids = id_sets["test_source1"] | id_sets["test_source2"] | id_sets["test_source3"]
    train_test_overlap = train_all_ids & test_all_ids
    cross_checks["train_test_id_overlap_count"] = len(train_test_overlap)
    print(f"  Train/Test Entity ID Overlap: {len(train_test_overlap)}")
    
    # 3. Cross-source within train overlap
    s1_s2_overlap = id_sets["train_source1"] & id_sets["train_source2"]
    s1_s3_overlap = id_sets["train_source1"] & id_sets["train_source3"]
    s2_s3_overlap = id_sets["train_source2"] & id_sets["train_source3"]
    cross_checks["train_s1_s2_id_overlap"] = len(s1_s2_overlap)
    cross_checks["train_s1_s3_id_overlap"] = len(s1_s3_overlap)
    cross_checks["train_s2_s3_id_overlap"] = len(s2_s3_overlap)
    print(f"  Train cross-source ID overlaps: S1-S2={len(s1_s2_overlap)}, S1-S3={len(s1_s3_overlap)}, S2-S3={len(s2_s3_overlap)}")
    
    audit_results["cross_source_checks"] = cross_checks
    
    # Save machine-readable JSON
    os.makedirs("reports", exist_ok=True)
    json_path = "reports/data_audit_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved raw audit data to {json_path}")

if __name__ == "__main__":
    main()
