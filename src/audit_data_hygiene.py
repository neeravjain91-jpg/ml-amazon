#!/usr/bin/env python3
"""
Data Hygiene, Audit, Collision Analysis & Ground-Truth Compatibility Suite
ML Challenge 2026 - Business Entity Resolution
"""

import os
import sys
import re
import json
import time
import unicodedata
from collections import Counter, defaultdict
import numpy as np

sys.path.append('src')
sys.path.append('.')
import canonical_cleaning
import features

def audit_file_hygiene(fpath, max_rows=500000):
    """
    Audits data hygiene, anomalies, and transformations on a source file.
    """
    t0 = time.time()
    total_rows = 0
    unique_ids = set()
    dup_ids = 0
    missing_name = 0
    missing_addr = 0
    missing_country = 0
    control_char_rows = 0
    changed_records = 0

    trans_counts = {
        "whitespace": 0,
        "unicode": 0,
        "punctuation": 0,
        "abbreviation": 0,
        "legal": 0,
        "token_sort": 0,
    }

    raw_names = []
    clean_names = []
    legal_names = []
    sort_names = []
    raw_addrs = []
    clean_addrs = []
    countries = []

    name_lens_raw = []
    name_lens_clean = []
    name_tok_counts = []
    addr_lens_raw = []
    addr_lens_clean = []
    addr_tok_counts = []
    num_tok_counts = []
    has_postal_count = 0

    with open(fpath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            total_rows += 1
            p = line.rstrip("\r\n").split("\t")
            if len(p) != 4:
                continue
            sid, raw_n, raw_a, raw_c = p[0], p[1], p[2], p[3]

            if sid in unique_ids:
                dup_ids += 1
            else:
                unique_ids.add(sid)

            if not raw_n.strip(): missing_name += 1
            if not raw_a.strip(): missing_addr += 1
            if not raw_c.strip(): missing_country += 1

            # Check for control characters
            has_ctrl = any(unicodedata.category(ch).startswith('C') and ch not in ('\t', '\n') for ch in raw_n + raw_a)
            if has_ctrl:
                control_char_rows += 1

            # Canonicalize
            rec = canonical_cleaning.canonicalize_record(sid, raw_n, raw_a, raw_c)

            # Check transformations
            is_changed = False
            if rec["name_unicode"] != raw_n or rec["address_unicode"] != raw_a:
                trans_counts["unicode"] += 1
                is_changed = True

            if "  " in raw_n or "  " in raw_a or raw_n != raw_n.strip() or raw_a != raw_a.strip():
                trans_counts["whitespace"] += 1
                is_changed = True

            if re.search(r'[^\w\s]', raw_n) or re.search(r'[^\w\s]', raw_a):
                trans_counts["punctuation"] += 1

            if "&" in raw_n or any(re.search(rf'\b{abbr}\b', raw_a.lower()) for abbr in canonical_cleaning.ADDR_EXPANSIONS):
                trans_counts["abbreviation"] += 1

            if rec["name_legal_stripped"] != rec["name_clean"]:
                trans_counts["legal"] += 1

            if rec["name_token_sorted"] != rec["name_clean"]:
                trans_counts["token_sort"] += 1

            if is_changed:
                changed_records += 1

            # Accumulate sample for EDA and Collision analysis
            if total_rows <= 100000:
                raw_names.append(raw_n)
                clean_names.append(rec["name_clean"])
                legal_names.append(rec["name_legal_stripped"])
                sort_names.append(rec["name_token_sorted"])
                raw_addrs.append(raw_a)
                clean_addrs.append(rec["address_clean"])
                countries.append(rec["country_clean"])

                name_lens_raw.append(len(raw_n))
                name_lens_clean.append(len(rec["name_clean"]))
                name_tok_counts.append(len(rec["name_clean"].split()))
                addr_lens_raw.append(len(raw_a))
                addr_lens_clean.append(len(rec["address_clean"]))
                addr_tok_counts.append(len(rec["address_clean"].split()))
                nums = rec["address_numeric_tokens"].split() if rec["address_numeric_tokens"] else []
                num_tok_counts.append(len(nums))
                if rec["address_postal_tokens"]:
                    has_postal_count += 1

            if total_rows >= max_rows:
                break

    stats = {
        "file": fpath,
        "rows_audited": total_rows,
        "unique_ids": len(unique_ids),
        "duplicate_ids": dup_ids,
        "missing_name": missing_name,
        "missing_addr": missing_addr,
        "missing_country": missing_country,
        "control_char_rows": control_char_rows,
        "changed_records": changed_records,
        "changed_pct": (changed_records / total_rows * 100) if total_rows > 0 else 0,
        "trans_counts": trans_counts,
        "eda": {
            "sample_size": len(raw_names),
            "name_len_raw_mean": float(np.mean(name_lens_raw)) if name_lens_raw else 0,
            "name_len_clean_mean": float(np.mean(name_lens_clean)) if name_lens_clean else 0,
            "name_tok_mean": float(np.mean(name_tok_counts)) if name_tok_counts else 0,
            "addr_len_raw_mean": float(np.mean(addr_lens_raw)) if addr_lens_raw else 0,
            "addr_len_clean_mean": float(np.mean(addr_lens_clean)) if addr_lens_clean else 0,
            "addr_tok_mean": float(np.mean(addr_tok_counts)) if addr_tok_counts else 0,
            "num_tok_mean": float(np.mean(num_tok_counts)) if num_tok_counts else 0,
            "postal_coverage_pct": (has_postal_count / len(raw_names) * 100) if raw_names else 0,
            "country_dist": dict(Counter(countries).most_common(10)),
        },
        "collision": {
            "sample_size": len(raw_names),
            "raw_name_unique": len(set(raw_names)),
            "clean_name_unique": len(set(clean_names)),
            "legal_name_unique": len(set(legal_names)),
            "sort_name_unique": len(set(sort_names)),
            "raw_addr_unique": len(set(raw_addrs)),
            "clean_addr_unique": len(set(clean_addrs)),
            "top_clean_collisions": Counter(clean_names).most_common(5),
            "top_legal_collisions": Counter(legal_names).most_common(5),
        },
        "elapsed_s": time.time() - t0,
    }
    return stats

def run_ground_truth_compatibility_analysis(n_val=5000):
    """
    Compares representations on true matched pairs vs sampled hard-negatives.
    """
    print("\nRunning Ground-Truth Compatibility Analysis...", flush=True)
    t0 = time.time()

    # Load 5k validation S1 IDs
    with open("reports/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_ids = [line.strip() for line in f if line.strip()][:n_val]
    val_set = set(val_s1_ids)

    # Load S1 records
    s1_dict = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_set:
                s1_dict[p[0]] = canonical_cleaning.canonicalize_record(p[0], p[1], p[2], p[3])

    # Load Ground Truth
    gt_pairs = []
    all_target_ids = set()
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_set and len(p) > 1 and p[1].strip():
                for tid in p[1].strip().split(","):
                    gt_pairs.append((p[0], tid))
                    all_target_ids.add(tid)

    # Load target records for ground truth pairs
    target_dict = {}
    for sf in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(sf, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                idx = line.find("\t")
                if idx != -1:
                    tid = line[:idx]
                    if tid in all_target_ids and tid not in target_dict:
                        p = line.rstrip("\r\n").split("\t")
                        target_dict[tid] = canonical_cleaning.canonicalize_record(p[0], p[1], p[2], p[3])

    # Evaluate True-Pair Coverage
    total_true = len(gt_pairs)
    true_cov = {
        "raw_name_exact": 0,
        "clean_name_exact": 0,
        "legal_name_exact": 0,
        "sort_name_exact": 0,
        "raw_addr_exact": 0,
        "clean_addr_exact": 0,
        "num_agree": 0,
        "postal_agree": 0,
    }

    for s1_id, tid in gt_pairs:
        if s1_id not in s1_dict or tid not in target_dict:
            continue
        r1 = s1_dict[s1_id]
        r2 = target_dict[tid]

        if r1["business_name"].strip() == r2["business_name"].strip():
            true_cov["raw_name_exact"] += 1
        if r1["name_clean"] == r2["name_clean"] and r1["name_clean"]:
            true_cov["clean_name_exact"] += 1
        if r1["name_legal_stripped"] == r2["name_legal_stripped"] and r1["name_legal_stripped"]:
            true_cov["legal_name_exact"] += 1
        if r1["name_token_sorted"] == r2["name_token_sorted"] and r1["name_token_sorted"]:
            true_cov["sort_name_exact"] += 1

        if r1["business_address"].strip() == r2["business_address"].strip():
            true_cov["raw_addr_exact"] += 1
        if r1["address_clean"] == r2["address_clean"] and r1["address_clean"]:
            true_cov["clean_addr_exact"] += 1

        nums1 = set(r1["address_numeric_tokens"].split()) if r1["address_numeric_tokens"] else set()
        nums2 = set(r2["address_numeric_tokens"].split()) if r2["address_numeric_tokens"] else set()
        if nums1 and nums2 and (nums1 & nums2):
            true_cov["num_agree"] += 1

        post1 = set(r1["address_postal_tokens"].split()) if r1["address_postal_tokens"] else set()
        post2 = set(r2["address_postal_tokens"].split()) if r2["address_postal_tokens"] else set()
        if post1 and post2 and (post1 & post2):
            true_cov["postal_agree"] += 1

    # Evaluate False-Pair Collision on sampled negative pairs
    # Generate 50,000 random negative pairs (different entities)
    np.random.seed(42)
    s1_keys = list(s1_dict.keys())
    t_keys = list(target_dict.keys())
    n_neg = 50000

    neg_coll = {
        "raw_name_exact": 0,
        "clean_name_exact": 0,
        "legal_name_exact": 0,
        "sort_name_exact": 0,
        "raw_addr_exact": 0,
        "clean_addr_exact": 0,
        "num_agree": 0,
        "postal_agree": 0,
    }

    gt_pair_set = set(gt_pairs)
    neg_count = 0
    while neg_count < n_neg:
        s = np.random.choice(s1_keys)
        t = np.random.choice(t_keys)
        if (s, t) in gt_pair_set:
            continue
        neg_count += 1
        r1 = s1_dict[s]
        r2 = target_dict[t]

        if r1["business_name"].strip() == r2["business_name"].strip() and r1["business_name"].strip():
            neg_coll["raw_name_exact"] += 1
        if r1["name_clean"] == r2["name_clean"] and r1["name_clean"]:
            neg_coll["clean_name_exact"] += 1
        if r1["name_legal_stripped"] == r2["name_legal_stripped"] and r1["name_legal_stripped"]:
            neg_coll["legal_name_exact"] += 1
        if r1["name_token_sorted"] == r2["name_token_sorted"] and r1["name_token_sorted"]:
            neg_coll["sort_name_exact"] += 1

        if r1["business_address"].strip() == r2["business_address"].strip() and r1["business_address"].strip():
            neg_coll["raw_addr_exact"] += 1
        if r1["address_clean"] == r2["address_clean"] and r1["address_clean"]:
            neg_coll["clean_addr_exact"] += 1

        nums1 = set(r1["address_numeric_tokens"].split()) if r1["address_numeric_tokens"] else set()
        nums2 = set(r2["address_numeric_tokens"].split()) if r2["address_numeric_tokens"] else set()
        if nums1 and nums2 and (nums1 & nums2):
            neg_coll["num_agree"] += 1

        post1 = set(r1["address_postal_tokens"].split()) if r1["address_postal_tokens"] else set()
        post2 = set(r2["address_postal_tokens"].split()) if r2["address_postal_tokens"] else set()
        if post1 and post2 and (post1 & post2):
            neg_coll["postal_agree"] += 1

    gt_results = {
        "total_true_pairs": total_true,
        "true_pair_coverage": {k: v / total_true * 100 for k, v in true_cov.items()},
        "true_pair_counts": true_cov,
        "total_neg_pairs": n_neg,
        "false_pair_collision": {k: v / n_neg * 100 for k, v in neg_coll.items()},
        "false_pair_counts": neg_coll,
        "elapsed_s": time.time() - t0,
    }
    return gt_results

def main():
    print("=" * 70)
    print("STARTING DATA HYGIENE AUDIT, EDA & COMPATIBILITY ANALYSIS")
    print("=" * 70)

    files_to_audit = [
        ("dataset/train/train_source1.tsv", 500000),
        ("dataset/train/train_source2.tsv", 500000),
        ("dataset/train/train_source3.tsv", 500000),
        ("dataset/test/test_source1.tsv", 500000),
        ("dataset/test/test_source2.tsv", 500000),
        ("dataset/test/test_source3.tsv", 500000),
    ]

    all_audits = []
    for fpath, n_max in files_to_audit:
        print(f"Auditing {fpath} (up to {n_max:,} rows)...", flush=True)
        stat = audit_file_hygiene(fpath, max_rows=n_max)
        all_audits.append(stat)
        print(f"  Done in {stat['elapsed_s']:.1f}s | Changed: {stat['changed_pct']:.2f}% | Control Chars: {stat['control_char_rows']}")

    # Ground-truth compatibility
    gt_compat = run_ground_truth_compatibility_analysis(n_val=5000)

    # Save JSON summary
    os.makedirs("reports", exist_ok=True)
    with open("scratch/data_hygiene_audit_data.json", "w", encoding="utf-8") as f:
        json.dump({"audits": all_audits, "gt_compatibility": gt_compat}, f, indent=2)

    # =========================================================================
    # Write reports/data_cleaning_spec.md
    # =========================================================================
    with open("reports/data_cleaning_spec.md", "w", encoding="utf-8") as f:
        f.write("# Canonical Data Cleaning & Representation Specification\n\n")
        f.write("## 1. Principles & Non-Destructive Hygiene\n")
        f.write("- **Preservation of Raw Fields**: Every canonical record strictly retains the original immutable raw fields (`entity_id`, `business_name`, `business_address`, `country`).\n")
        f.write("- **Deterministic Representations**: All representations are pure deterministic mathematical projections without external knowledge bases, geocoders, or web queries.\n")
        f.write("- **Non-Lossy Design**: No common tokens or entity identifiers are discarded. Legal suffixes are preserved in separate representations rather than wiped from the core text.\n")
        f.write("- **Open-Set Country Support**: Country normalization enforces clean casing while remaining strictly open-set (preserving US, India, France, and unseen values).\n\n")

        f.write("## 2. Canonical Column Definitions\n\n")
        f.write("| Column Name | Type | Description | Transformation Logic |\n")
        f.write("|---|:---:|---|---|\n")
        f.write("| `entity_id` | String | Immutable entity identifier | Stripped of extraneous whitespace |\n")
        f.write("| `business_name` | String | Original raw business name | Exact byte-for-byte original |\n")
        f.write("| `name_unicode` | String | Unicode-cleaned name | NFKD normalized, control chars & accents stripped, whitespace collapsed, case preserved |\n")
        f.write("| `name_lower` | String | Lowercase name | `name_unicode.lower()` |\n")
        f.write("| `name_clean` | String | Standardized business name | Lowercase, `&` -> `and`, punctuation to spaces, normalized spaces |\n")
        f.write("| `name_alphanumeric` | String | Alphanumeric representation | Letters and numbers only (`[a-z0-9]`) |\n")
        f.write("| `name_tokenized` | String | Tokenized representation | Whitespace-delimited clean tokens |\n")
        f.write("| `name_token_sorted` | String | Token-sorted name | Tokens sorted alphabetically for word-order invariance |\n")
        f.write("| `name_legal_stripped` | String | Legal corporate suffix stripped | Suffixes (`pvt`, `ltd`, `inc`, `corp`, etc.) removed if non-empty |\n")
        f.write("| `business_address` | String | Original raw address | Exact byte-for-byte original |\n")
        f.write("| `address_unicode` | String | Unicode-cleaned address | NFKD normalized, control chars stripped, case preserved |\n")
        f.write("| `address_lower` | String | Lowercase address | `address_unicode.lower()` |\n")
        f.write("| `address_clean` | String | Standardized address | Standard street/road abbreviations expanded (`rd`->`road`, `st`->`street`), punctuation to spaces |\n")
        f.write("| `address_alphanumeric`| String | Alphanumeric address | Alphanumeric tokens only |\n")
        f.write("| `address_tokens` | String | Tokenized address | Whitespace-delimited clean address tokens |\n")
        f.write("| `address_numeric_tokens`| String | Extracted numeric tokens | Numbers extracted with leading zeros stripped, space-delimited |\n")
        f.write("| `address_postal_tokens` | String | Candidate postal/PIN codes | 5-digit US/French postal codes and 6-digit Indian PIN codes |\n")
        f.write("| `address_possible_house_number`| String | Candidate building/unit | Leading numeric token from street address |\n")
        f.write("| `country` | String | Original raw country | Exact byte-for-byte original |\n")
        f.write("| `country_clean` | String | Normalized country | Superficial whitespace/casing normalization (Open-set) |\n")

    # =========================================================================
    # Write reports/data_cleaning_audit.md
    # =========================================================================
    with open("reports/data_cleaning_audit.md", "w", encoding="utf-8") as f:
        f.write("# Data Cleaning & Hygiene Audit Report\n\n")
        f.write("## 1. Raw vs Clean Audit Summary\n\n")
        f.write("| Source File | Rows Audited | Unique IDs | Duplicate IDs | Missing Name | Missing Addr | Missing Country | Control Char Rows | Records Changed (%) |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for a in all_audits:
            f.write(f"| `{os.path.basename(a['file'])}` | {a['rows_audited']:,} | {a['unique_ids']:,} | {a['duplicate_ids']} | {a['missing_name']} | {a['missing_addr']} | {a['missing_country']} | {a['control_char_rows']} | {a['changed_records']:,} ({a['changed_pct']:.2f}%) |\n")
        f.write("\n")

        f.write("## 2. Transformation Counts by Category\n\n")
        f.write("| Source File | Whitespace Normalization | Unicode / Control Fixes | Punctuation Cleanup | Abbreviation (`&` / Rd / St) | Legal Suffix Stripped | Token Sorting Shift |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for a in all_audits:
            tc = a["trans_counts"]
            f.write(f"| `{os.path.basename(a['file'])}` | {tc['whitespace']:,} | {tc['unicode']:,} | {tc['punctuation']:,} | {tc['abbreviation']:,} | {tc['legal']:,} | {tc['token_sort']:,} |\n")

    # =========================================================================
    # Write reports/normalization_collision_analysis.md
    # =========================================================================
    with open("reports/normalization_collision_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Normalization Collision Analysis\n\n")
        f.write("## 1. Representation Collision Rates (100,000 Sample per Source)\n\n")
        f.write("| Source File | Raw Name Unique | Clean Name Unique | Clean Collision % | Legal Stripped Unique | Legal Collision % | Token Sorted Unique | Sort Collision % |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for a in all_audits:
            c = a["collision"]
            n_tot = c["sample_size"]
            clean_col = (n_tot - c["clean_name_unique"]) / n_tot * 100
            legal_col = (n_tot - c["legal_name_unique"]) / n_tot * 100
            sort_col = (n_tot - c["sort_name_unique"]) / n_tot * 100
            f.write(f"| `{os.path.basename(a['file'])}` | {c['raw_name_unique']:,} | {c['clean_name_unique']:,} | {clean_col:.2f}% | {c['legal_name_unique']:,} | {legal_col:.2f}% | {c['sort_name_unique']:,} | {sort_col:.2f}% |\n")
        f.write("\n")

        f.write("## 2. Ground-Truth Match Coverage vs False-Collision Tradeoff\n\n")
        f.write(f"- **True Matched Pairs Evaluated**: {gt_compat['total_true_pairs']:,}\n")
        f.write(f"- **Sampled False Negative Pairs Evaluated**: {gt_compat['total_neg_pairs']:,}\n\n")
        f.write("| Representation / Feature | True Match Coverage (Recall Ceil) | False Pair Collision (Precision Risk) | Signal-to-Noise Ratio (Coverage / Collision) |\n")
        f.write("|---|:---:|:---:|:---:|\n")
        tc = gt_compat["true_pair_coverage"]
        fc = gt_compat["false_pair_collision"]
        for k in ["raw_name_exact", "clean_name_exact", "legal_name_exact", "sort_name_exact", "raw_addr_exact", "clean_addr_exact", "num_agree", "postal_agree"]:
            snr = (tc[k] / fc[k]) if fc[k] > 0 else 999.0
            f.write(f"| **`{k}`** | **{tc[k]:.2f}%** | {fc[k]:.3f}% | **{snr:.1f}x** |\n")

    # =========================================================================
    # Write reports/raw_vs_clean_eda.md
    # =========================================================================
    with open("reports/raw_vs_clean_eda.md", "w", encoding="utf-8") as f:
        f.write("# Cleaned Data Exploratory Data Analysis (EDA)\n\n")
        f.write("## 1. Text Length and Token Statistics (Raw vs Clean)\n\n")
        f.write("| Source File | Raw Name Len | Clean Name Len | Name Tokens | Raw Addr Len | Clean Addr Len | Addr Tokens | Numeric Tokens | Postal PIN Coverage |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for a in all_audits:
            e = a["eda"]
            f.write(f"| `{os.path.basename(a['file'])}` | {e['name_len_raw_mean']:.1f} | {e['name_len_clean_mean']:.1f} | {e['name_tok_mean']:.2f} | {e['addr_len_raw_mean']:.1f} | {e['addr_len_clean_mean']:.1f} | {e['addr_tok_mean']:.2f} | {e['num_tok_mean']:.2f} | **{e['postal_coverage_pct']:.2f}%** |\n")
        f.write("\n")

        f.write("## 2. Country Distribution Across Sources\n\n")
        for a in all_audits:
            f.write(f"### `{os.path.basename(a['file'])}`\n")
            for c, cnt in a["eda"]["country_dist"].items():
                f.write(f"- **{c}**: {cnt:,} records ({cnt/a['eda']['sample_size']*100:.2f}%)\n")
            f.write("\n")

    print("\nData hygiene audit complete. Reports written.")

if __name__ == "__main__":
    main()
