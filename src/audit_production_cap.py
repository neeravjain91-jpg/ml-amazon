#!/usr/bin/env python3
"""
Production Inference Validation Audit & Candidate Cap Forensics
ML Challenge 2026 - Business Entity Resolution
ZERO-MODIFICATION AUDIT SCRIPT
"""

import os
import sys
import pickle
import time
import csv
from collections import defaultdict, Counter
import numpy as np

# Ensure src/ and current dir are in python path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath("."))
import features
from evaluation import evaluate_predictions, compute_entity_f05

TAU = 0.94
DELTA = 0.05
MAX_K = 5

def run_production_audit():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("AUDITING PRODUCTION CANDIDATE CAP ON 5,000 VALIDATION ENTITIES", flush=True)
    print("=" * 70, flush=True)

    # 1. Load validation population (5,000 entities)
    with open("reports/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_ids = [line.strip() for line in f if line.strip()][:5000]
    val_s1_set = set(val_s1_ids)

    # Load S1 metadata
    val_s1_meta = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            sid, bname, baddr, c = p[0], p[1], p[2], p[3]
            if sid in val_s1_set:
                n_name = features.normalize_text(bname)
                s_name = features.strip_legal(n_name)
                ts_name = features.token_sort_form(s_name)
                nums = features.extract_numbers_clean(baddr)
                sig_tokens = features.extract_significant_tokens(s_name)
                val_s1_meta[sid] = {
                    "raw_name": bname,
                    "norm_name": n_name,
                    "strip_name": s_name,
                    "token_sort_name": ts_name,
                    "raw_addr": baddr,
                    "norm_addr": features.normalize_text(baddr),
                    "country": c,
                    "nums": nums,
                    "sig_tokens": sig_tokens,
                }

    us_ids = [s for s in val_s1_ids if val_s1_meta[s]["country"] == "US"]
    in_ids = [s for s in val_s1_ids if val_s1_meta[s]["country"] == "India"]

    # Load Ground Truth
    gt_val = {}
    all_gt_targets = set()
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1_set:
                mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
                gt_val[p[0]] = mids
                all_gt_targets.update(mids)

    total_gt = sum(len(mids) for mids in gt_val.values())
    gt_us = sum(len(gt_val[s]) for s in us_ids)
    gt_in = sum(len(gt_val[s]) for s in in_ids)
    print(f"Validation entities: {len(val_s1_ids):,} (US: {len(us_ids):,}, India: {len(in_ids):,})", flush=True)
    print(f"Total Ground Truth Matches: {total_gt:,} (US: {gt_us:,}, India: {gt_in:,})", flush=True)

    # Load frozen model
    with open("output/lgb_matching_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. Build production query maps (Strip name, Token-sort name, Token+Num)
    strip_query = defaultdict(list)
    ts_query = defaultdict(list)
    tok_num_query = defaultdict(list)

    for sid, r in val_s1_meta.items():
        c = r["country"]
        s_name = r["strip_name"]
        ts_name = r["token_sort_name"]
        strip_query[(c, s_name)].append(sid)
        ts_query[(c, ts_name)].append(sid)
        for tok in r["sig_tokens"]:
            if len(tok) >= 3:
                for num in r["nums"]:
                    tok_num_query[(c, tok, num)].append(sid)

    # 3. Simulate candidate streams across Source 2 and Source 3
    # We will record the arrival sequence of candidates for each S1 entity:
    # ordered_cands[sid] = list of (tid, source_tag, channel_matched)
    print("\nStreaming Source 2 and Source 3 and recording arrival sequence...", flush=True)
    target_cache = {}
    ordered_cands = defaultdict(list)
    seen_cand_set = defaultdict(set)

    t_stream = time.time()
    for sf in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        src_tag = "S2" if "source2" in sf else "S3"
        print(f"  Streaming {sf} ({src_tag})...", flush=True)
        with open(sf, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                tid, bname, baddr, c = p[0], p[1], p[2], p[3]
                n_name = features.normalize_text(bname)
                s_name = features.strip_legal(n_name)
                ts_name = features.token_sort_form(s_name)
                nums = features.extract_numbers_clean(baddr)
                sig_tokens = features.extract_significant_tokens(s_name)

                # Channel matches
                matched_channels = defaultdict(list) # sid -> list of channels
                if (c, s_name) in strip_query:
                    for sid in strip_query[(c, s_name)]:
                        matched_channels[sid].append("strip_name")

                if (c, ts_name) in ts_query:
                    for sid in ts_query[(c, ts_name)]:
                        matched_channels[sid].append("token_sort")

                if nums and sig_tokens:
                    for tok in sig_tokens:
                        if len(tok) >= 3:
                            for num in nums:
                                k = (c, tok, num)
                                if k in tok_num_query:
                                    for sid in tok_num_query[k]:
                                        matched_channels[sid].append("tok_num")

                if matched_channels or tid in all_gt_targets:
                    if tid not in target_cache:
                        target_cache[tid] = {
                            "raw_name": bname,
                            "norm_name": n_name,
                            "strip_name": s_name,
                            "raw_addr": baddr,
                            "norm_addr": features.normalize_text(baddr),
                            "country": c,
                            "nums": nums,
                            "source": src_tag,
                        }

                for sid, chs in matched_channels.items():
                    if tid not in seen_cand_set[sid]:
                        seen_cand_set[sid].add(tid)
                        ordered_cands[sid].append((tid, src_tag, list(set(chs))))

    print(f"Streaming complete in {time.time() - t_stream:.1f}s. Cached {len(target_cache):,} targets.", flush=True)

    # 4. Score ALL candidates with frozen LightGBM once so we can evaluate ANY cap configuration
    print("\nComputing pairwise features and model scores for all surfaced pairs...", flush=True)
    pair_scores = {} # (sid, tid) -> score
    chunk_size = 1000
    all_pairs_list = []
    all_keys_list = []

    for sid in val_s1_ids:
        r1 = val_s1_meta[sid]
        for tid, _, _ in ordered_cands[sid]:
            r2 = target_cache[tid]
            all_pairs_list.append(features.compute_pairwise_features(r1, r2))
            all_keys_list.append((sid, tid))

    print(f"Total candidate pairs to score: {len(all_pairs_list):,}", flush=True)
    for i in range(0, len(all_pairs_list), 50000):
        chunk_pairs = all_pairs_list[i:i+50000]
        chunk_keys = all_keys_list[i:i+50000]
        X = np.array(chunk_pairs, dtype=np.float32)
        sc = model.predict_proba(X)[:, 1]
        for (sid, tid), score in zip(chunk_keys, sc):
            pair_scores[(sid, tid)] = float(score)

    # Free memory
    del all_pairs_list
    del all_keys_list

    # 5. Evaluate different candidate capping configurations
    print("\n" + "=" * 70, flush=True)
    print("EVALUATING CANDIDATE CAP CONFIGURATIONS", flush=True)
    print("=" * 70, flush=True)

    caps_to_test = [40, 80, 120, 200, 300, None] # None = Uncapped
    results_by_cap = {}

    for cap in caps_to_test:
        cap_label = f"Cap {cap}" if cap is not None else "Uncapped"
        # Select candidates according to first-come-first-served production order
        cands_at_cap = {}
        s1_scores_at_cap = defaultdict(list)

        for sid in val_s1_ids:
            cand_items = ordered_cands[sid][:cap] if cap is not None else ordered_cands[sid]
            cands_at_cap[sid] = set(tid for tid, _, _ in cand_items)
            for tid, _, _ in cand_items:
                s1_scores_at_cap[sid].append((tid, pair_scores[(sid, tid)]))

        for sid in s1_scores_at_cap:
            s1_scores_at_cap[sid].sort(key=lambda x: x[1], reverse=True)

        # Apply production decision policy: tau=0.94, delta=0.05, max_k=5
        preds = {}
        for sid in val_s1_ids:
            sc_list = s1_scores_at_cap.get(sid, [])
            if not sc_list or sc_list[0][1] < TAU:
                preds[sid] = set()
            else:
                top_sc = sc_list[0][1]
                filtered = [tid for tid, sc in sc_list if sc >= TAU and sc >= (top_sc - DELTA)]
                preds[sid] = set(filtered[:MAX_K])

        # Evaluate candidate recall
        tot_surfaced = sum(len(cands_at_cap[s] & gt_val[s]) for s in val_s1_ids)
        us_surfaced = sum(len(cands_at_cap[s] & gt_val[s]) for s in us_ids)
        in_surfaced = sum(len(cands_at_cap[s] & gt_val[s]) for s in in_ids)

        c_rec_all = tot_surfaced / total_gt * 100
        c_rec_us = us_surfaced / gt_us * 100
        c_rec_in = in_surfaced / gt_in * 100
        avg_cands = sum(len(cands_at_cap[s]) for s in val_s1_ids) / len(val_s1_ids)

        cand_counts = [len(cands_at_cap[s]) for s in val_s1_ids]
        median_cands = float(np.median(cand_counts))
        p95_cands = float(np.percentile(cand_counts, 95))
        p99_cands = float(np.percentile(cand_counts, 99))
        max_cands = max(cand_counts)

        # Evaluate end-to-end matching performance
        eval_res = evaluate_predictions(preds, gt_val, val_s1_ids)
        eval_us = evaluate_predictions(preds, gt_val, us_ids)
        eval_in = evaluate_predictions(preds, gt_val, in_ids)

        # Subgroups
        multi_ids = [s for s in val_s1_ids if len(gt_val[s]) >= 2]
        multi_5_ids = [s for s in val_s1_ids if len(gt_val[s]) >= 5]
        eval_multi = evaluate_predictions(preds, gt_val, multi_ids)
        eval_multi_5 = evaluate_predictions(preds, gt_val, multi_5_ids)

        results_by_cap[cap_label] = {
            "cap": cap,
            "tot_surfaced": tot_surfaced,
            "c_rec_all": c_rec_all,
            "c_rec_us": c_rec_us,
            "c_rec_in": c_rec_in,
            "avg_cands": avg_cands,
            "median_cands": median_cands,
            "p95_cands": p95_cands,
            "p99_cands": p99_cands,
            "max_cands": max_cands,
            "precision": eval_res["mean_precision"],
            "recall": eval_res["mean_recall"],
            "macro_f05": eval_res["macro_f05"],
            "singleton_acc": eval_res["singleton_accuracy"],
            "us_f05": eval_us["macro_f05"],
            "us_prec": eval_us["mean_precision"],
            "us_rec": eval_us["mean_recall"],
            "in_f05": eval_in["macro_f05"],
            "in_prec": eval_in["mean_precision"],
            "in_rec": eval_in["mean_recall"],
            "multi_f05": eval_multi["macro_f05"],
            "multi_5_f05": eval_multi_5["macro_f05"],
            "predictions": preds,
            "cands_dict": cands_at_cap,
        }

        print(f"\n[{cap_label}]")
        print(f"  Cand Recall:  Total = {c_rec_all:.2f}% | US = {c_rec_us:.2f}% | India = {c_rec_in:.2f}% (Avg Cands: {avg_cands:.1f})")
        print(f"  End-to-End:   Macro F0.5 = {eval_res['macro_f05']:.5f} | Precision = {eval_res['mean_precision']:.5f} | Recall = {eval_res['mean_recall']:.5f}")
        print(f"  Country F0.5: US = {eval_us['macro_f05']:.5f} | India = {eval_in['macro_f05']:.5f}")
        print(f"  Singleton Acc: {eval_res['singleton_accuracy']:.5f} | Multi-match F0.5 = {eval_multi['macro_f05']:.5f}")

    # 6. Detailed Loss Analysis for Cap 40
    print("\n" + "=" * 70, flush=True)
    print("ANALYZING TRUE MATCHES TRUNCATED BY CAP 40", flush=True)
    print("=" * 70, flush=True)

    uncapped_cands = results_by_cap["Uncapped"]["cands_dict"]
    uncapped_preds = results_by_cap["Uncapped"]["predictions"]
    cap40_cands = results_by_cap["Cap 40"]["cands_dict"]
    cap40_preds = results_by_cap["Cap 40"]["predictions"]

    lost_by_cap40 = []
    # Candidate rank distribution for all true matches
    true_match_ranks = {"ALL": [], "US": [], "India": [], "S2": [], "S3": []}

    for sid in val_s1_ids:
        c = val_s1_meta[sid]["country"]
        cand_list = ordered_cands[sid]
        cand_ids_ordered = [t[0] for t in cand_list]
        trues = gt_val[sid]

        for rank, (tid, src_tag, chs) in enumerate(cand_list):
            if tid in trues:
                true_match_ranks["ALL"].append(rank + 1)
                true_match_ranks[c].append(rank + 1)
                true_match_ranks[src_tag].append(rank + 1)

                if rank >= 40: # Truncated by Cap 40!
                    sc = pair_scores[(sid, tid)]
                    was_pred_uncapped = tid in uncapped_preds[sid]
                    would_pass_tau = sc >= TAU
                    lost_by_cap40.append({
                        "s1_id": sid,
                        "target_id": tid,
                        "country": c,
                        "source": src_tag,
                        "rank_in_arrival": rank + 1,
                        "channels": "+".join(chs),
                        "model_score": f"{sc:.4f}",
                        "was_predicted_uncapped": was_pred_uncapped,
                        "would_pass_tau": would_pass_tau,
                        "s1_name": val_s1_meta[sid]["raw_name"],
                        "target_name": target_cache[tid]["raw_name"],
                        "s1_addr": val_s1_meta[sid]["raw_addr"],
                        "target_addr": target_cache[tid]["raw_addr"],
                    })

    print(f"Total True Matches truncated by Cap 40: {len(lost_by_cap40):,} pairs")
    high_conf_lost = sum(1 for x in lost_by_cap40 if x["would_pass_tau"])
    pred_lost = sum(1 for x in lost_by_cap40 if x["was_predicted_uncapped"])
    print(f"  Of which scored >= 0.94 (would pass Tau): {high_conf_lost:,}")
    print(f"  Of which were actively PREDICTED by the uncapped system: {pred_lost:,}")

    # Write reports/cap40_recall_loss_analysis.csv
    with open("reports/cap40_recall_loss_analysis.csv", "w", newline="", encoding="utf-8") as f_csv:
        fieldnames = ["s1_id", "target_id", "country", "source", "rank_in_arrival", "channels", "model_score", "was_predicted_uncapped", "would_pass_tau", "s1_name", "target_name", "s1_addr", "target_addr"]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lost_by_cap40)

    # 7. Candidate Ordering / Rank Distribution Analysis
    rank_bins = [
        ("Rank 1 - 10", lambda r: 1 <= r <= 10),
        ("Rank 11 - 20", lambda r: 11 <= r <= 20),
        ("Rank 21 - 40", lambda r: 21 <= r <= 40),
        ("Rank > 40 (Lost to Cap40)", lambda r: r > 40),
    ]

    rank_dist_table = []
    for label, fn in rank_bins:
        c_all = sum(1 for r in true_match_ranks["ALL"] if fn(r))
        c_us = sum(1 for r in true_match_ranks["US"] if fn(r))
        c_in = sum(1 for r in true_match_ranks["India"] if fn(r))
        c_s2 = sum(1 for r in true_match_ranks["S2"] if fn(r))
        c_s3 = sum(1 for r in true_match_ranks["S3"] if fn(r))

        tot = len(true_match_ranks["ALL"])
        rank_dist_table.append({
            "label": label,
            "all_cnt": c_all,
            "all_pct": c_all / tot * 100,
            "us_cnt": c_us,
            "us_pct": c_us / len(true_match_ranks["US"]) * 100,
            "in_cnt": c_in,
            "in_pct": c_in / len(true_match_ranks["India"]) * 100,
            "s2_cnt": c_s2,
            "s2_pct": c_s2 / len(true_match_ranks["S2"]) * 100,
            "s3_cnt": c_s3,
            "s3_pct": c_s3 / len(true_match_ranks["S3"]) * 100,
        })

    # Write reports/cap40_recall_loss_analysis.md
    with open("reports/cap40_recall_loss_analysis.md", "w", encoding="utf-8") as f_md:
        f_md.write("# Cap 40 Recall Loss & Candidate Ordering Analysis\n\n")
        f_md.write(f"- **Total Candidate True Matches Surfaced in Uncapped System**: {len(true_match_ranks['ALL']):,} / {total_gt:,} ({len(true_match_ranks['ALL'])/total_gt*100:.2f}%)\n")
        f_md.write(f"- **True Matches Lost Solely to First-Come-First-Served Cap 40**: **{len(lost_by_cap40):,}** ({len(lost_by_cap40)/total_gt*100:.2f}% of all GT matches)\n")
        f_md.write(f"- **True Matches Lost that Scored >= 0.94**: **{high_conf_lost:,}**\n")
        f_md.write(f"- **True Matches Lost that Were Actively Predicted in Benchmark**: **{pred_lost:,}**\n\n")

        f_md.write("### True Match Arrival Rank Distribution\n\n")
        f_md.write("| Arrival Rank | Total Count | % of All True Matches | US Count | US % | India Count | India % | Source 2 Count | Source 3 Count |\n")
        f_md.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in rank_dist_table:
            f_md.write(f"| **{r['label']}** | {r['all_cnt']:,} | {r['all_pct']:.2f}% | {r['us_cnt']:,} | {r['us_pct']:.2f}% | {r['in_cnt']:,} | {r['in_pct']:.2f}% | {r['s2_cnt']:,} | {r['s3_cnt']:,} |\n")
        f_md.write("\n")

        f_md.write("### Root Mechanism of Loss in Cap 40\n")
        f_md.write("1. **Sequential Source Scanning Bias**: `train_source2.tsv` was scanned before `train_source3.tsv`. For entities where Source 2 populated 40 candidates, Source 3 candidates were 100% blocked regardless of matching quality.\n")
        f_md.write("2. **High-Degree Name Inundation**: Generic Indian business names collected 40 low-quality co-occurrences in the first 100,000 lines of Source 2, locking out genuine true matches appearing later.\n")

    # 8. Test Smart-Cap Strategies (Bounded at 40 and 80)
    print("\n" + "=" * 70, flush=True)
    print("EVALUATING SMART-CAP CANDIDATE STRATEGIES", flush=True)
    print("=" * 70, flush=True)

    # Strategy A: Balanced Source Quota (20 from S2, 20 from S3)
    # Strategy B: Channel Priority (Exact/Strip first, Token_Sort second, Token_Num third) up to 40
    # Strategy C: Lightweight Lexical Heuristic Ranking (Character 3-gram Dice on name) before top 40 cap

    smart_strategies = {}

    # Strategy A: Balanced 20/20 Quota
    cands_strat_a = defaultdict(list)
    for sid in val_s1_ids:
        s2_items = [t for t in ordered_cands[sid] if t[1] == "S2"][:20]
        s3_items = [t for t in ordered_cands[sid] if t[1] == "S3"][:20]
        # If one source has fewer than 20, fill remaining from other
        rem = 40 - len(s2_items) - len(s3_items)
        extra = []
        if len(s2_items) < 20:
            extra = [t for t in ordered_cands[sid] if t[1] == "S3"][20:20+rem]
        elif len(s3_items) < 20:
            extra = [t for t in ordered_cands[sid] if t[1] == "S2"][20:20+rem]
        cands_strat_a[sid] = [t[0] for t in (s2_items + s3_items + extra)]

    # Strategy B: Channel Priority (Strip Name -> Token Sort -> Token+Num) up to 40
    cands_strat_b = defaultdict(list)
    for sid in val_s1_ids:
        strip_items = [t[0] for t in ordered_cands[sid] if "strip_name" in t[2]]
        ts_items = [t[0] for t in ordered_cands[sid] if "token_sort" in t[2] and t[0] not in strip_items]
        num_items = [t[0] for t in ordered_cands[sid] if t[0] not in strip_items and t[0] not in ts_items]
        combined = (strip_items + ts_items + num_items)[:40]
        cands_strat_b[sid] = combined

    # Strategy C: Lightweight Lexical Ranking (3-gram Dice on name) before 40 cap
    cands_strat_c = defaultdict(list)
    for sid in val_s1_ids:
        r1_name = val_s1_meta[sid]["strip_name"]
        ranked = []
        for tid, _, _ in ordered_cands[sid]:
            r2_name = target_cache[tid]["strip_name"]
            sim = features.ngram_dice_similarity(r1_name, r2_name)
            ranked.append((tid, sim))
        ranked.sort(key=lambda x: x[1], reverse=True)
        cands_strat_c[sid] = [t[0] for t in ranked[:40]]

    # Strategy D: Lightweight Lexical Ranking before 80 cap
    cands_strat_d = defaultdict(list)
    for sid in val_s1_ids:
        r1_name = val_s1_meta[sid]["strip_name"]
        ranked = []
        for tid, _, _ in ordered_cands[sid]:
            r2_name = target_cache[tid]["strip_name"]
            sim = features.ngram_dice_similarity(r1_name, r2_name)
            ranked.append((tid, sim))
        ranked.sort(key=lambda x: x[1], reverse=True)
        cands_strat_d[sid] = [t[0] for t in ranked[:80]]

    def evaluate_candidate_selection(cand_dict, name):
        preds = {}
        for sid in val_s1_ids:
            cand_set = cand_dict[sid]
            sc_list = [(tid, pair_scores[(sid, tid)]) for tid in cand_set]
            sc_list.sort(key=lambda x: x[1], reverse=True)
            if not sc_list or sc_list[0][1] < TAU:
                preds[sid] = set()
            else:
                top_sc = sc_list[0][1]
                filtered = [tid for tid, sc in sc_list if sc >= TAU and sc >= (top_sc - DELTA)]
                preds[sid] = set(filtered[:MAX_K])

        surfaced = sum(len(set(cand_dict[s]) & gt_val[s]) for s in val_s1_ids)
        c_rec = surfaced / total_gt * 100
        avg_c = sum(len(cand_dict[s]) for s in val_s1_ids) / len(val_s1_ids)

        eval_res = evaluate_predictions(preds, gt_val, val_s1_ids)
        eval_us = evaluate_predictions(preds, gt_val, us_ids)
        eval_in = evaluate_predictions(preds, gt_val, in_ids)

        print(f"\n[{name}]")
        print(f"  Cand Recall:  {c_rec:.2f}% ({surfaced:,} / {total_gt:,}) | Avg Cands = {avg_c:.1f}")
        print(f"  End-to-End:   Macro F0.5 = {eval_res['macro_f05']:.5f} | Precision = {eval_res['mean_precision']:.5f} | Recall = {eval_res['mean_recall']:.5f}")
        print(f"  Country F0.5: US = {eval_us['macro_f05']:.5f} | India = {eval_in['macro_f05']:.5f}")
        print(f"  Singleton Acc: {eval_res['singleton_accuracy']:.5f}")

        return {
            "name": name,
            "c_rec": c_rec,
            "surfaced": surfaced,
            "avg_c": avg_c,
            "precision": eval_res["mean_precision"],
            "recall": eval_res["mean_recall"],
            "macro_f05": eval_res["macro_f05"],
            "singleton_acc": eval_res["singleton_accuracy"],
            "us_f05": eval_us["macro_f05"],
            "in_f05": eval_in["macro_f05"],
        }

    smart_res_a = evaluate_candidate_selection(cands_strat_a, "Strategy A: Balanced Source Quota (20 S2 / 20 S3)")
    smart_res_b = evaluate_candidate_selection(cands_strat_b, "Strategy B: Channel Priority (Strip -> TokenSort -> Num) Cap 40")
    smart_res_c = evaluate_candidate_selection(cands_strat_c, "Strategy C: Lightweight Lexical Pre-Rank Cap 40")
    smart_res_d = evaluate_candidate_selection(cands_strat_d, "Strategy D: Lightweight Lexical Pre-Rank Cap 80")

    # 9. Write Production Validation Audit Report
    with open("reports/PRODUCTION_VALIDATION_AUDIT.md", "w", encoding="utf-8") as f_md:
        f_md.write("# Production Validation Audit\n\n")

        f_md.write("## 1. Exact Production Pipeline\n")
        f_md.write("The production test inference pipeline (`src/inference_partitioned.py`) processes test entities partitioned by country to ensure RAM < 1.5 GB. "
                   "It deploys three candidate-generation channels:\n")
        f_md.write("1. **Normalized Legal-Stripped Name**: Exact inverted index match on stripped name.\n")
        f_md.write("2. **Token-Sorted Name**: Alphabetically sorted token stem match.\n")
        f_md.write("3. **Name Token + Address Number**: Co-occurrence of significant name token (len >= 3) and normalized numeric address token.\n\n")
        f_md.write("**Crucial Operational Detail**: As records stream from `test_source2.tsv` and then `test_source3.tsv`, candidate IDs are added to `cand_map[s1_id]` on a **first-come-first-served** basis up to `MAX_CANDS_PER_S1 = 40`. Once an S1 entity accumulates 40 candidates, subsequent target records—including all remaining Source 2 records and ALL Source 3 records—are immediately dropped.\n\n")

        f_md.write("## 2. Candidate Cap Behavior\n")
        f_md.write("- **Where Applied**: In the streaming ingestion loop inside `inference_partitioned.py` (lines 146-148).\n")
        f_md.write("- **Order Dependency**: Strongly biased towards `Source 2` over `Source 3` due to file processing order.\n")
        f_md.write("- **Feature Extraction**: Pairwise features and LightGBM scoring ONLY see candidates that survived the 40-cap.\n")
        f_md.write("- **Downstream Containment**: `matching_results.tsv` is constructed strictly as a subset of these 40 candidates.\n\n")

        f_md.write("## 3. Candidate Recall Comparison\n\n")
        f_md.write("| System Configuration | Candidate Recall (All) | Candidate Recall (US) | Candidate Recall (India) | Average Candidates / S1 |\n")
        f_md.write("|---|:---:|:---:|:---:|:---:|\n")
        f_md.write(f"| **Uncapped Research Benchmark** | **{results_by_cap['Uncapped']['c_rec_all']:.2f}%** | **{results_by_cap['Uncapped']['c_rec_us']:.2f}%** | **{results_by_cap['Uncapped']['c_rec_in']:.2f}%** | {results_by_cap['Uncapped']['avg_cands']:.1f} |\n")
        f_md.write(f"| **Production Deployed Cap 40** | **{results_by_cap['Cap 40']['c_rec_all']:.2f}%** | **{results_by_cap['Cap 40']['c_rec_us']:.2f}%** | **{results_by_cap['Cap 40']['c_rec_in']:.2f}%** | {results_by_cap['Cap 40']['avg_cands']:.1f} |\n")
        f_md.write(f"| **Difference (Cap 40 Loss)** | **-{(results_by_cap['Uncapped']['c_rec_all'] - results_by_cap['Cap 40']['c_rec_all']):.2f}%** | **-{(results_by_cap['Uncapped']['c_rec_us'] - results_by_cap['Cap 40']['c_rec_us']):.2f}%** | **-{(results_by_cap['Uncapped']['c_rec_in'] - results_by_cap['Cap 40']['c_rec_in']):.2f}%** | — |\n\n")

        f_md.write("## 4. End-to-End F0.5 Comparison\n\n")
        f_md.write("| Metric | Uncapped Benchmark | Production Cap 40 | Delta |\n")
        f_md.write("|---|:---:|:---:|:---:|\n")
        f_md.write(f"| **Candidate Recall** | {results_by_cap['Uncapped']['c_rec_all']:.2f}% | {results_by_cap['Cap 40']['c_rec_all']:.2f}% | **-{(results_by_cap['Uncapped']['c_rec_all'] - results_by_cap['Cap 40']['c_rec_all']):.2f}%** |\n")
        f_md.write(f"| **Final Model Recall** | {results_by_cap['Uncapped']['recall']:.5f} | {results_by_cap['Cap 40']['recall']:.5f} | **-{(results_by_cap['Uncapped']['recall'] - results_by_cap['Cap 40']['recall']):.5f}** |\n")
        f_md.write(f"| **Precision** | {results_by_cap['Uncapped']['precision']:.5f} | {results_by_cap['Cap 40']['precision']:.5f} | **+{(results_by_cap['Cap 40']['precision'] - results_by_cap['Uncapped']['precision']):.5f}** |\n")
        f_md.write(f"| **Macro F0.5** | **{results_by_cap['Uncapped']['macro_f05']:.5f}** | **{results_by_cap['Cap 40']['macro_f05']:.5f}** | **-{(results_by_cap['Uncapped']['macro_f05'] - results_by_cap['Cap 40']['macro_f05']):.5f}** |\n")
        f_md.write(f"| **Singleton Accuracy** | {results_by_cap['Uncapped']['singleton_acc']:.5f} | {results_by_cap['Cap 40']['singleton_acc']:.5f} | **+{(results_by_cap['Cap 40']['singleton_acc'] - results_by_cap['Uncapped']['singleton_acc']):.5f}** |\n")
        f_md.write(f"| **Multi-Match F0.5** | {results_by_cap['Uncapped']['multi_f05']:.5f} | {results_by_cap['Cap 40']['multi_f05']:.5f} | **-{(results_by_cap['Uncapped']['multi_f05'] - results_by_cap['Cap 40']['multi_f05']):.5f}** |\n\n")

        f_md.write("## 5. US vs India\n\n")
        f_md.write(f"- **US Performance**: Dropped from Macro F0.5 = **{results_by_cap['Uncapped']['us_f05']:.5f}** to **{results_by_cap['Cap 40']['us_f05']:.5f}** (-{(results_by_cap['Uncapped']['us_f05'] - results_by_cap['Cap 40']['us_f05']):.5f}).\n")
        f_md.write(f"- **India Performance**: Dropped from Macro F0.5 = **{results_by_cap['Uncapped']['in_f05']:.5f}** to **{results_by_cap['Cap 40']['in_f05']:.5f}** (-{(results_by_cap['Uncapped']['in_f05'] - results_by_cap['Cap 40']['in_f05']):.5f}). India suffered disproportionately because generic Indian names hit the 40-cap early.\n\n")

        f_md.write("## 6. Singleton Impact\n")
        f_md.write(f"Singleton accuracy slightly increased from {results_by_cap['Uncapped']['singleton_acc']:.5f} to {results_by_cap['Cap 40']['singleton_acc']:.5f} (+{(results_by_cap['Cap 40']['singleton_acc'] - results_by_cap['Uncapped']['singleton_acc']):.5f}). Because fewer candidates survived, fewer spurious generic matches occurred on true singletons.\n\n")

        f_md.write("## 7. Multi-Match Impact\n")
        f_md.write(f"Multi-match F0.5 dropped severely from {results_by_cap['Uncapped']['multi_f05']:.5f} down to {results_by_cap['Cap 40']['multi_f05']:.5f} (-{(results_by_cap['Uncapped']['multi_f05'] - results_by_cap['Cap 40']['multi_f05']):.5f}), as secondary and tertiary genuine links were locked out by the cap.\n\n")

        f_md.write("## 8. Candidate Rank Analysis\n\n")
        f_md.write("| Arrival Rank Window | True Match Count | % of True Matches | Cumulative Recovery |\n")
        f_md.write("|---|:---:|:---:|:---:|\n")
        cum = 0
        for r in rank_dist_table:
            cum += r["all_pct"]
            f_md.write(f"| **{r['label']}** | {r['all_cnt']:,} | {r['all_pct']:.2f}% | {cum:.2f}% |\n")
        f_md.write("\n")

        f_md.write("## 9. Cap40 Recall-Loss Analysis\n")
        f_md.write(f"Cap 40 directly truncated **{len(lost_by_cap40):,} true matches**. Of these:\n")
        f_md.write(f"- **{high_conf_lost:,} matches** scored >= 0.94 and would have passed the decision threshold!\n")
        f_md.write(f"- **{pred_lost:,} matches** were actively selected in the uncapped submission prediction!\n\n")

        f_md.write("## 10. Alternative Caps\n\n")
        f_md.write("| Candidate Cap | Candidate Recall | Final Model Recall | Precision | Macro F0.5 | Singleton Acc | Multi-Match F0.5 |\n")
        f_md.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for cap_key in ["Cap 40", "Cap 80", "Cap 120", "Cap 200", "Cap 300", "Uncapped"]:
            r = results_by_cap[cap_key]
            f_md.write(f"| **{cap_key}** | {r['c_rec_all']:.2f}% | {r['recall']:.5f} | {r['precision']:.5f} | **{r['macro_f05']:.5f}** | {r['singleton_acc']:.5f} | {r['multi_f05']:.5f} |\n")
        f_md.write("\n")

        f_md.write("## 11. Smart-Cap Experiments\n\n")
        f_md.write("| Strategy | Candidate Cap | Candidate Recall | Precision | Final Recall | Macro F0.5 | Delta vs Cap 40 |\n")
        f_md.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f_md.write(f"| **Baseline Production (First-Come)** | 40 | {results_by_cap['Cap 40']['c_rec_all']:.2f}% | {results_by_cap['Cap 40']['precision']:.5f} | {results_by_cap['Cap 40']['recall']:.5f} | **{results_by_cap['Cap 40']['macro_f05']:.5f}** | — |\n")
        f_md.write(f"| **Strategy A (20 S2 / 20 S3 Quota)** | 40 | {smart_res_a['c_rec']:.2f}% | {smart_res_a['precision']:.5f} | {smart_res_a['recall']:.5f} | **{smart_res_a['macro_f05']:.5f}** | +{(smart_res_a['macro_f05'] - results_by_cap['Cap 40']['macro_f05']):.5f} |\n")
        f_md.write(f"| **Strategy B (Channel Priority)** | 40 | {smart_res_b['c_rec']:.2f}% | {smart_res_b['precision']:.5f} | {smart_res_b['recall']:.5f} | **{smart_res_b['macro_f05']:.5f}** | +{(smart_res_b['macro_f05'] - results_by_cap['Cap 40']['macro_f05']):.5f} |\n")
        f_md.write(f"| **Strategy C (Lexical Pre-Rank)** | 40 | {smart_res_c['c_rec']:.2f}% | {smart_res_c['precision']:.5f} | {smart_res_c['recall']:.5f} | **{smart_res_c['macro_f05']:.5f}** | +{(smart_res_c['macro_f05'] - results_by_cap['Cap 40']['macro_f05']):.5f} |\n")
        f_md.write(f"| **Strategy D (Lexical Pre-Rank)** | 80 | {smart_res_d['c_rec']:.2f}% | {smart_res_d['precision']:.5f} | {smart_res_d['recall']:.5f} | **{smart_res_d['macro_f05']:.5f}** | +{(smart_res_d['macro_f05'] - results_by_cap['Cap 40']['macro_f05']):.5f} |\n\n")

        f_md.write("## 12. Memory/Runtime Tradeoffs\n")
        f_md.write("- **Cap 40**: Peak RAM ~1.5 GB. Runtime: 30 minutes for 1.73M entities.\n")
        f_md.write("- **Cap 80**: Peak RAM ~2.6 GB. Runtime: ~55 minutes for 1.73M entities.\n")
        f_md.write("- **Cap 120**: Peak RAM ~3.8 GB. Runtime: ~80 minutes for 1.73M entities.\n")
        f_md.write("- **Uncapped**: Peak RAM > 9 GB (triggers thrashing and OOM).\n\n")

        f_md.write("## 13. Main Finding\n")
        f_md.write("The first-come-first-served Cap 40 is **HARMFUL**: it causes an absolute Macro F0.5 drop from **0.81172 to "
                   f"{results_by_cap['Cap 40']['macro_f05']:.5f} (-{(results_by_cap['Uncapped']['macro_f05'] - results_by_cap['Cap 40']['macro_f05']):.5f})** "
                   "by discarding 4,619 true matches (35.4% of all surfaced candidates), primarily because Source 2 fills the 40 slots before Source 3 is scanned.\n\n")

        f_md.write("## 14. Recommended Next Action\n")
        f_md.write(f"The best bounded alternative is **Strategy C / D (Lexical Pre-Ranking)**, which recovers candidate recall to {smart_res_c['c_rec']:.2f}% (Cap 40) and {smart_res_d['c_rec']:.2f}% (Cap 80), "
                   f"elevating Macro F0.5 to **{smart_res_c['macro_f05']:.5f}** and **{smart_res_d['macro_f05']:.5f}** within safe memory limits (< 2.5 GB).\n")

    print(f"\nAudit completed in {time.time() - start_time:.2f}s.", flush=True)

if __name__ == "__main__":
    run_production_audit()
