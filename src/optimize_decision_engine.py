#!/usr/bin/env python3
"""
Phase 11-13: Entity-Level Decision Optimization, Singleton Gating & Multi-Match Tuning
ML Challenge 2026 - Business Entity Resolution
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict

sys.path.append('src')
sys.path.append('.')
import features
from evaluation import evaluate_predictions

def optimize_decision_engine():
    print("=" * 70)
    print("PHASES 11-13: DECISION ENGINE OPTIMIZATION & SINGLETON GATING")
    print("=" * 70)

    # 1. Load trained model
    model_path = "output/lgb_matching_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded trained model from {model_path}.")

    # 2. Load 5,000 validation entities & GT
    val_s1_ids = []
    with open("reports/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_ids = [line.strip() for line in f if line.strip()][:5000]
    val_s1_set = set(val_s1_ids)

    gt_val = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1_set:
                mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
                gt_val[p[0]] = mids

    val_s1_meta = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in val_s1_set:
                val_s1_meta[p[0]] = {
                    "raw_name": p[1],
                    "norm_name": features.normalize_text(p[1]),
                    "strip_name": features.strip_legal(features.normalize_text(p[1])),
                    "raw_addr": p[2],
                    "norm_addr": features.normalize_text(p[2]),
                    "country": p[3],
                    "nums": features.extract_numbers_clean(p[2]),
                }

    val_strip_map = defaultdict(list)
    val_tok_num_map = defaultdict(list)
    for s1, r in val_s1_meta.items():
        c = r["country"]
        s_name = r["strip_name"]
        val_strip_map[(c, s_name)].append(s1)
        sig = features.extract_significant_tokens(s_name)
        for tok in sig:
            if len(tok) >= 3:
                for num in r["nums"]:
                    val_tok_num_map[(c, tok, num)].append(s1)

    val_candidates = defaultdict(set)
    val_target_cache = {}

    print("Streaming validation candidates...")
    for sf in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(sf, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                tid, bname, baddr, c = p[0], p[1], p[2], p[3]
                n_name = features.normalize_text(bname)
                s_name = features.strip_legal(n_name)
                nums = features.extract_numbers_clean(baddr)

                matched = []
                if (c, s_name) in val_strip_map:
                    matched.extend(val_strip_map[(c, s_name)])

                sig = features.extract_significant_tokens(s_name)
                if nums and sig:
                    for tok in sig:
                        if len(tok) >= 3:
                            for num in nums:
                                k = (c, tok, num)
                                if k in val_tok_num_map:
                                    matched.extend(val_tok_num_map[k])

                if matched:
                    if tid not in val_target_cache:
                        val_target_cache[tid] = {
                            "raw_name": bname,
                            "norm_name": n_name,
                            "strip_name": s_name,
                            "raw_addr": baddr,
                            "norm_addr": features.normalize_text(baddr),
                            "country": c,
                            "nums": nums,
                        }
                    for s1 in matched:
                        val_candidates[s1].add(tid)

    # Score in batch
    print(f"Scoring candidate pairs...")
    s1_scored = defaultdict(list)
    
    # Process S1 by S1 to keep memory minimal
    chunk_size = 500
    for i in range(0, len(val_s1_ids), chunk_size):
        chunk_s1 = val_s1_ids[i:i+chunk_size]
        pairs = []
        keys = []
        for s1 in chunk_s1:
            r1 = val_s1_meta[s1]
            for tid in val_candidates.get(s1, set()):
                r2 = val_target_cache[tid]
                pairs.append(features.compute_pairwise_features(r1, r2))
                keys.append((s1, tid))
        if pairs:
            X = np.array(pairs, dtype=np.float32)
            sc = model.predict_proba(X)[:, 1]
            for (s1, tid), score in zip(keys, sc):
                s1_scored[s1].append((tid, float(score)))

    # Sort candidates by score descending
    for s1 in s1_scored:
        s1_scored[s1].sort(key=lambda x: x[1], reverse=True)

    # Experiment 1: High threshold grid [0.80, 0.95]
    print("\n--- EXPERIMENT 1: HIGH THRESHOLD SWEEP ---")
    best_tau = 0.85
    best_f05 = 0.0
    for tau in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]:
        preds = {}
        for s1 in val_s1_ids:
            cand_list = s1_scored.get(s1, [])
            preds[s1] = {tid for tid, score in cand_list if score >= tau}
        res = evaluate_predictions(preds, gt_val, val_s1_ids)
        print(f"  tau={tau:4.2f} | Macro F0.5: {res['macro_f05']:.5f} | P: {res['mean_precision']:.5f} | R: {res['mean_recall']:.5f} | SingAcc: {res['singleton_accuracy']:.5f}")
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_tau = tau

    # Experiment 2: Score Margin / Relative Gap Policy
    # Match candidate if score >= tau AND score >= (best_score - delta)
    print("\n--- EXPERIMENT 2: RELATIVE SCORE MARGIN POLICY (tau + delta) ---")
    best_delta = None
    for delta in [0.05, 0.10, 0.15, 0.20, 0.30, 1.0]:
        preds = {}
        for s1 in val_s1_ids:
            cand_list = s1_scored.get(s1, [])
            if not cand_list or cand_list[0][1] < best_tau:
                preds[s1] = set()
            else:
                top_score = cand_list[0][1]
                preds[s1] = {tid for tid, score in cand_list if score >= best_tau and score >= (top_score - delta)}
        res = evaluate_predictions(preds, gt_val, val_s1_ids)
        print(f"  tau={best_tau:4.2f}, delta={delta:4.2f} | Macro F0.5: {res['macro_f05']:.5f} | P: {res['mean_precision']:.5f} | R: {res['mean_recall']:.5f} | SingAcc: {res['singleton_accuracy']:.5f}")

    # Experiment 3: Max matches cap (top-k cap)
    print("\n--- EXPERIMENT 3: MAXIMUM MATCHES CAP (top-K) ---")
    for max_k in [1, 2, 3, 4, 5, 8, 12]:
        preds = {}
        for s1 in val_s1_ids:
            cand_list = s1_scored.get(s1, [])
            filtered = [tid for tid, score in cand_list if score >= best_tau]
            preds[s1] = set(filtered[:max_k])
        res = evaluate_predictions(preds, gt_val, val_s1_ids)
        print(f"  tau={best_tau:4.2f}, max_k={max_k:2d} | Macro F0.5: {res['macro_f05']:.5f} | P: {res['mean_precision']:.5f} | R: {res['mean_recall']:.5f} | SingAcc: {res['singleton_accuracy']:.5f}")

    print("\n" + "=" * 70)
    print(f"OPTIMAL DECISION CONFIGURATION DETERMINED!")
    print(f"Best Base Threshold: tau = {best_tau:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    optimize_decision_engine()
