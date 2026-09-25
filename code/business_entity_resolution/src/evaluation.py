#!/usr/bin/env python3
"""
Official Macro-Averaged F0.5 Metric Engine
ML Challenge 2026 - Business Entity Resolution
"""

def compute_entity_f05(pred_ids, true_ids):
    """
    Compute F0.5 score for a single Source 1 entity.
    pred_ids: iterable of predicted entity IDs (set or list)
    true_ids: iterable of ground truth entity IDs (set or list)
    """
    pred_set = set(pred_ids) if pred_ids else set()
    true_set = set(true_ids) if true_ids else set()

    # Case 1: True singleton (no true matches)
    if not true_set:
        return 1.0 if not pred_set else 0.0

    # Case 2: True entity has matches, but model predicted empty
    if not pred_set:
        return 0.0

    # Case 3: Both true and predicted have IDs
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05


def evaluate_predictions(predictions, ground_truth, s1_entity_ids=None):
    """
    Compute macro-averaged F0.5 across all evaluated S1 entities.
    predictions: dict of {s1_id: set/list of predicted IDs}
    ground_truth: dict of {s1_id: set/list of true IDs}
    s1_entity_ids: optional list/set of specific S1 IDs to evaluate (defaults to union)
    
    Returns:
        dict with macro_f05, mean_precision, mean_recall, singleton_accuracy,
        n_entities, n_singletons, n_multi_match
    """
    if s1_entity_ids is None:
        s1_entity_ids = list(ground_truth.keys())

    f05_scores = []
    precisions = []
    recalls = []

    singleton_correct = 0
    singleton_total = 0
    non_singleton_total = 0

    for s1 in s1_entity_ids:
        preds = predictions.get(s1, set())
        trues = ground_truth.get(s1, set())

        pred_set = set(preds) if preds else set()
        true_set = set(trues) if trues else set()

        score = compute_entity_f05(pred_set, true_set)
        f05_scores.append(score)

        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_total += 1
            if pred_set:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                precisions.append(p)
                recalls.append(r)
            else:
                precisions.append(0.0)
                recalls.append(0.0)

    macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
    mean_p = sum(precisions) / len(precisions) if precisions else 0.0
    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    sing_acc = singleton_correct / singleton_total if singleton_total else 1.0

    return {
        "macro_f05": round(macro_f05, 5),
        "mean_precision": round(mean_p, 5),
        "mean_recall": round(mean_r, 5),
        "singleton_accuracy": round(sing_acc, 5),
        "total_evaluated_s1": len(s1_entity_ids),
        "total_singletons": singleton_total,
        "singleton_correct": singleton_correct,
        "total_non_singletons": non_singleton_total,
    }


if __name__ == "__main__":
    # Unit test with example from PDF (page 6)
    pred_ex = {"S2-00047", "S2-00193", "S3-00812"}
    true_ex = {"S2-00047", "S3-00812"}
    score = compute_entity_f05(pred_ex, true_ex)
    print(f"PDF Page 6 test case: Expected ~0.714, Got: {score:.4f}")
    assert abs(score - 0.7142857) < 1e-4, "Test case failed!"

    # Singleton test cases
    assert compute_entity_f05(set(), set()) == 1.0, "Singleton empty prediction should be 1.0"
    assert compute_entity_f05({"S2-00001"}, set()) == 0.0, "Singleton false positive should be 0.0"
    assert compute_entity_f05(set(), {"S2-00001"}) == 0.0, "Missed match should be 0.0"
    print("All unit tests PASSED successfully!")
