#!/usr/bin/env python3
"""
Phase 8-11: Hard-Negative Mining, GBDT Model Training, Calibration & Entity Decision Optimization
ML Challenge 2026 - Business Entity Resolution
"""

import os
import sys
import time
import random
import pickle
import numpy as np
import lightgbm as lgb
from collections import defaultdict

import sys
sys.path.append('src')
sys.path.append('.')
import features
from evaluation import evaluate_predictions, compute_entity_f05

# Fixed seed
random.seed(42)
np.random.seed(42)

def train_and_evaluate():
    print("=" * 70)
    print("PHASES 8-11: HARD NEGATIVE MINING, GBDT MODELING & MACRO F0.5 OPTIMIZATION")
    print("=" * 70)
    t_start = time.time()

    # 1. Load validation set to guarantee ZERO leakage
    val_s1_ids = []
    with open("reports/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_ids = [line.strip() for line in f if line.strip()]
    val_s1_set = set(val_s1_ids)
    print(f"Loaded {len(val_s1_ids):,} validation S1 entities (isolated).")

    # 2. Sample 40,000 Training S1 entities (strictly excluding validation)
    print("\n[Step 1] Sampling 40,000 Training S1 Entities...")
    train_s1_ids = []
    train_s1_meta = {}
    train_s1_trues = defaultdict(set)

    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1_id = p[0]
            if s1_id not in val_s1_set:
                train_s1_meta[s1_id] = {
                    "raw_name": p[1],
                    "norm_name": features.normalize_text(p[1]),
                    "strip_name": features.strip_legal(features.normalize_text(p[1])),
                    "raw_addr": p[2],
                    "norm_addr": features.normalize_text(p[2]),
                    "country": p[3],
                    "nums": features.extract_numbers_clean(p[2]),
                }
                train_s1_ids.append(s1_id)
            if len(train_s1_ids) >= 40000:
                break

    train_s1_set = set(train_s1_ids)

    # Load ground truth for sampled training entities
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in train_s1_set:
                if len(p) > 1 and p[1].strip():
                    train_s1_trues[p[0]] = set(p[1].strip().split(","))

    pos_pair_count = sum(len(v) for v in train_s1_trues.values())
    print(f"Training S1 queries: {len(train_s1_ids):,} with {pos_pair_count:,} positive links.")

    # 3. Mined Hard Negatives & Positive Pair Construction
    print("\n[Step 2] Building Training Pairs (Positives + Mined Hard Negatives)...")
    # All target records needed for training (positives + candidate negatives)
    all_needed_targets = set()
    for trues in train_s1_trues.values():
        all_needed_targets.update(trues)

    # Build query lookup for candidate generation on training queries
    train_strip_map = defaultdict(list)
    train_token_num_map = defaultdict(list)
    for s1_id, r in train_s1_meta.items():
        c = r["country"]
        s_name = r["strip_name"]
        train_strip_map[(c, s_name)].append(s1_id)
        sig = features.extract_significant_tokens(s_name)
        for tok in sig:
            if len(tok) >= 3:
                for num in r["nums"]:
                    train_token_num_map[(c, tok, num)].append(s1_id)

    # Target data cache
    target_data = {}
    candidate_negatives = defaultdict(list)

    print("Streaming through Source 2 and Source 3 for Training Pairs...")
    for sf in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(sf, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                tid, bname, baddr, c = p[0], p[1], p[2], p[3]
                
                is_needed = tid in all_needed_targets
                n_name = features.normalize_text(bname)
                s_name = features.strip_legal(n_name)
                nums = features.extract_numbers_clean(baddr)
                
                # Check strip name match
                k_strip = (c, s_name)
                matched_s1_list = train_strip_map.get(k_strip, [])
                
                # Check token + num match
                sig = features.extract_significant_tokens(s_name)
                if nums and sig:
                    for tok in sig:
                        if len(tok) >= 3:
                            for num in nums:
                                k = (c, tok, num)
                                if k in train_token_num_map:
                                    matched_s1_list.extend(train_token_num_map[k])

                if matched_s1_list or is_needed:
                    target_data[tid] = {
                        "raw_name": bname,
                        "norm_name": n_name,
                        "strip_name": s_name,
                        "raw_addr": baddr,
                        "norm_addr": features.normalize_text(baddr),
                        "country": c,
                        "nums": nums,
                    }

                # Record candidate negatives (up to 4 hard negatives per S1)
                for s1_id in matched_s1_list:
                    if tid not in train_s1_trues[s1_id]:
                        if len(candidate_negatives[s1_id]) < 4:
                            candidate_negatives[s1_id].append(tid)

    print(f"Loaded {len(target_data):,} target records.")
    
    # 4. Generate Feature Matrix X, y
    print("\n[Step 3] Computing Dense Pairwise Feature Matrix...")
    X_train = []
    y_train = []

    for s1_id in train_s1_ids:
        r1 = train_s1_meta[s1_id]
        
        # Positives
        for pos_id in train_s1_trues.get(s1_id, []):
            r2 = target_data.get(pos_id)
            if r2:
                feat = features.compute_pairwise_features(r1, r2)
                X_train.append(feat)
                y_train.append(1)

        # Mined Hard Negatives
        for neg_id in candidate_negatives.get(s1_id, []):
            r2 = target_data.get(neg_id)
            if r2:
                feat = features.compute_pairwise_features(r1, r2)
                X_train.append(feat)
                y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"X_train shape: {X_train.shape} | Positives: {np.sum(y_train==1):,} | Negatives: {np.sum(y_train==0):,}")

    # 5. Train LightGBM Classifier (MIT Licensed, < 8B parameters)
    print("\n[Step 4] Training LightGBM Pairwise Classifier...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.08,
        'num_leaves': 63,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train)

    # Feature Importance Analysis
    importances = model.feature_importances_
    feat_imp = sorted(zip(features.FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("\n--- TOP 12 MOST DISCRIMINATIVE FEATURES ---")
    for fname, imp in feat_imp[:12]:
        print(f"  {fname:25s}: {imp:6d}")

    # Save model
    os.makedirs("output", exist_ok=True)
    model_path = "output/lgb_matching_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained model to {model_path}")

    # 6. Evaluate on 5,000 Validation Entities & Optimize Threshold
    print("\n[Step 5] Evaluating & Calibrating Decision Threshold on Validation Split...")
    # Load 5,000 validation S1 entities for rapid calibration
    eval_val_ids = val_s1_ids[:5000]
    eval_val_set = set(eval_val_ids)

    gt_val = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in eval_val_set:
                mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
                gt_val[p[0]] = mids

    val_s1_meta = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in eval_val_set:
                val_s1_meta[p[0]] = {
                    "raw_name": p[1],
                    "norm_name": features.normalize_text(p[1]),
                    "strip_name": features.strip_legal(features.normalize_text(p[1])),
                    "raw_addr": p[2],
                    "norm_addr": features.normalize_text(p[2]),
                    "country": p[3],
                    "nums": features.extract_numbers_clean(p[2]),
                }

    # Generate blocking candidates for validation S1
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

    print("Collecting validation candidates from S2 and S3...")
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

    print(f"Validation candidates collected for {len(val_candidates):,} entities.")

    # Score all candidate pairs with trained LightGBM
    val_pairs = []
    val_pair_keys = []
    for s1 in eval_val_ids:
        r1 = val_s1_meta[s1]
        for tid in val_candidates.get(s1, set()):
            r2 = val_target_cache[tid]
            feat = features.compute_pairwise_features(r1, r2)
            val_pairs.append(feat)
            val_pair_keys.append((s1, tid))

    print(f"Scoring {len(val_pairs):,} validation candidate pairs...")
    X_val = np.array(val_pairs, dtype=np.float32)
    scores = model.predict_proba(X_val)[:, 1]

    # Map scores back to S1 entities: s1 -> list of (tid, score)
    s1_scored_candidates = defaultdict(list)
    for (s1, tid), score in zip(val_pair_keys, scores):
        s1_scored_candidates[s1].append((tid, score))

    # Grid search probability thresholds
    print("\n--- THRESHOLD RESEARCH & MACRO F0.5 OPTIMIZATION ---")
    print("Threshold | Macro F0.5 | Precision | Recall | Singleton Acc")
    print("-" * 55)

    best_thresh = 0.5
    best_f05 = 0.0
    best_res = None

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    for tau in thresholds:
        preds = {}
        for s1 in eval_val_ids:
            cand_list = s1_scored_candidates.get(s1, [])
            matched = {tid for tid, sc in cand_list if sc >= tau}
            preds[s1] = matched

        res = evaluate_predictions(preds, gt_val, eval_val_ids)
        print(f"  tau={tau:4.2f} |  {res['macro_f05']:.5f}  |  {res['mean_precision']:.5f}  | {res['mean_recall']:.5f} |    {res['singleton_accuracy']:.5f}")
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_thresh = tau
            best_res = res

    print("\n" + "=" * 70)
    print(f"OPTIMAL DECISION THRESHOLD: tau = {best_thresh:.2f}")
    print(f"PEAK MACRO F0.5:           {best_f05:.5f}")
    print(f"Mean Precision:            {best_res['mean_precision']:.5f}")
    print(f"Mean Recall:               {best_res['mean_recall']:.5f}")
    print(f"Singleton Accuracy:        {best_res['singleton_accuracy']:.5f}")
    print(f"Total Pipeline Runtime:    {time.time() - t_start:.2f}s")
    print("=" * 70)

    # Save to report
    out_md = "reports/model_research.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Model Comparison & Calibration Report (Phases 8–11)\n\n")
        f.write(f"**Model**: LightGBM Pairwise Classifier (MIT Licensed, < 8B params)\n")
        f.write(f"**Training Set**: 40,000 S1 Entities with Mined Hard Negatives\n")
        f.write(f"**Optimal Threshold**: tau = {best_thresh:.2f}\n")
        f.write(f"**Validation Macro F0.5**: **{best_f05:.5f}**\n")
        f.write(f"**Precision**: {best_res['mean_precision']:.5f} | **Recall**: {best_res['mean_recall']:.5f}\n")
        f.write(f"**Singleton Accuracy**: {best_res['singleton_accuracy']:.5f}\n\n")
        f.write("## Top 10 Most Discriminative Features\n\n")
        for fname, imp in feat_imp[:10]:
            f.write(f"- `{fname}`: {imp}\n")

    print(f"Saved research report to {out_md}")

if __name__ == "__main__":
    train_and_evaluate()
