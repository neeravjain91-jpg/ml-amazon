# Retrieval V3: Streaming Min-Heap with Multi-Signal Pre-Ranking

## Design Specification for Gemini Implementation Agent

---

## 0. Motivation & Measured Evidence

The current production pipeline (`src/inference_partitioned.py`) uses a first-come-first-served `MAX_CANDS_PER_S1 = 40` cap applied inline during Source 2 → Source 3 sequential streaming. This causes catastrophic recall loss:

| System | Candidate Recall | Final Recall | Precision | Macro F0.5 |
|---|:---:|:---:|:---:|:---:|
| **Uncapped** | 76.05% | 0.66342 | 0.88015 | **0.80414** |
| **Production Cap 40 (FCFS)** | 49.14% | 0.43013 | 0.64938 | **0.58549** |
| **Lexical Pre-Rank Cap 40** | 63.30% | 0.55509 | 0.77662 | **0.70598** |
| **Lexical Pre-Rank Cap 80** | 67.34% | 0.58996 | 0.82178 | **0.74451** |

The root cause is threefold:
1. **Source starvation**: Source 2 fills 40 slots before Source 3 is scanned. 3,097 Source 3 true matches are permanently lost.
2. **Noise flooding**: Generic Indian business names accumulate 40 spurious tok_num co-occurrences from early Source 2 rows, blocking genuine matches that arrive later.
3. **Quality-blind truncation**: The cap is applied without any ranking — arrival order has zero correlation with match quality.

The fix is NOT simply raising the cap (even Cap 300 only reaches F0.5 = 0.72637 with FCFS). The fix is **quality-aware candidate selection** within a memory-bounded streaming architecture.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Per-Country Partition                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  PHASE 1 (Load):  Load S1 partition metadata        │
│                   Build inverted-index query maps    │
│                   Memory: O(N_s1)                    │
│                                                     │
│  PHASE 2 (Heap):  Stream Source 2 + Source 3         │
│                   For each matched candidate:        │
│                     • Compute cheap_score on the fly  │
│                     • Push to per-S1 min-heap(K)     │
│                     • Evict lowest if heap full      │
│                   Store only (tid, cheap_score) in   │
│                   heap. NO target metadata stored.   │
│                   Record set of surviving tids.      │
│                   Memory: O(N_s1 × K × 20 bytes)    │
│                                                     │
│  PHASE 3 (Load):  Re-stream Source 2 + Source 3      │
│                   Load metadata ONLY for tids in     │
│                   the surviving set.                 │
│                   Memory: O(|surviving_targets|)     │
│                                                     │
│  PHASE 4 (Score): Compute 30-dim pairwise features   │
│                   Run frozen LightGBM predict_proba  │
│                   Apply frozen τ/Δ/K decision engine │
│                   Stream results to disk             │
│                   Memory: O(chunk_size × 30)         │
│                                                     │
│  PHASE 5 (Free):  Delete all partition structures    │
│                   gc.collect()                       │
│                   Proceed to next country            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Why Two-Pass Streaming?

The key insight is that **computing the cheap pre-ranking score requires only S1 features (already loaded) and target features (computed inline from the raw TSV line)**. No target metadata needs to be stored during Phase 2 — only the heap entry `(tid, cheap_score)` at 20 bytes per entry.

This decouples **candidate selection** (Phase 2) from **metadata loading** (Phase 3), enabling:
- Phase 2 memory: O(N_s1 × K × 20) — deterministic, predictable
- Phase 3 memory: O(|unique surviving targets|) — bounded by N_s1 × K with heavy deduplication

A single-pass approach would require storing all target metadata eagerly (growing unboundedly during streaming), which was exactly the OOM problem that motivated Cap 40 in the first place.

---

## 2. Cheap Pre-Ranking Score Design

### Available Signals (No Labels, No External Data)

At streaming time, for each `(S1, candidate)` pair, we have:

| Signal | Source | Computation Cost | Discriminative Value |
|---|---|---|---|
| **Name 3-gram Dice** | `ngram_dice_similarity(s1.strip_name, t.strip_name)` | O(n) one-time ngram set per target | **HIGH** — proven: sole signal recovers +12 F0.5 points |
| **Name Token Jaccard** | `token_jaccard(s1_tokens, t_tokens)` | O(k) set intersection | MODERATE — captures word-level overlap |
| **Token Containment** | `token_containment(s1_tokens, t_tokens)` | O(k) set intersection | MODERATE — captures abbreviations/subsets |
| **Numeric Address Agreement** | `|s1.nums ∩ t.nums| / max(|s1.nums ∪ t.nums|, 1)` | O(k) set intersection | MODERATE — strong when addresses are structured |
| **Exact Name Match** | `1.0 if s1.strip_name == t.strip_name else 0.0` | O(1) string comparison | HIGH — these are almost always true matches |
| **Channel Count** | Number of independent blocking channels that fired (1, 2, or 3) | O(1) counter | LOW-MODERATE — convergent evidence |

### Proposed Scoring Function

```python
def compute_cheap_score(s1_meta, t_strip_name, t_tokens, t_nums, channel_count):
    """
    Returns a float in [0.0, 1.0+] representing pre-ranking quality.
    Higher = more likely to be a true match.
    """
    # Primary signal: character 3-gram Dice on legal-stripped name
    name_dice = ngram_dice_similarity(s1_meta["strip_name"], t_strip_name, n=3)

    # Secondary signal: token-level containment (captures abbreviation relationships)
    s1_toks = set(s1_meta["strip_name"].split())
    t_toks = set(t_tokens)
    if s1_toks and t_toks:
        containment = len(s1_toks & t_toks) / min(len(s1_toks), len(t_toks))
    else:
        containment = 0.0

    # Tertiary signal: numeric address agreement
    s1_nums = s1_meta["nums"]
    if s1_nums and t_nums:
        num_agree = len(s1_nums & t_nums) / max(len(s1_nums | t_nums), 1)
    elif not s1_nums and not t_nums:
        num_agree = 0.5  # both missing — neutral
    else:
        num_agree = 0.0

    # Channel convergence bonus
    channel_bonus = channel_count / 3.0

    # Weighted combination
    cheap_score = (
        0.50 * name_dice
      + 0.20 * containment
      + 0.20 * num_agree
      + 0.10 * channel_bonus
    )

    return cheap_score
```

### Design Rationale for Weights

1. **Name Dice at 0.50**: The strongest single signal. Strategy C (Dice-only at Cap 40) recovered candidate recall from 49.14% to 63.30%, proving that lexical name similarity alone captures the majority of true match ranking. Weight must dominate.

2. **Token Containment at 0.20**: Captures cases where one entity's name is a substring of the other (e.g., "ABC Corp" vs "ABC Corporation of India"), which Dice partially misses due to different character contexts. Also helps with abbreviation patterns.

3. **Numeric Agreement at 0.20**: Provides orthogonal address evidence that helps when two entities have similar names but different addresses (should rank lower) or different-looking names at the same address (should rank higher). This specifically targets the Indian address-only-linkage pattern.

4. **Channel Count at 0.10**: Small bonus for candidates retrieved by multiple independent channels (e.g., both strip_name AND tok_num), which represents convergent evidence from independent features.

### Why NOT Fit Weights on Validation?

The weights above are **principled defaults** based on measured signal strength. However, the implementation agent SHOULD benchmark 2–3 alternative weight vectors on the validation split to verify optimality:

- **Variant A (Dice-heavy)**: (0.65, 0.15, 0.15, 0.05)
- **Variant B (Balanced)**: (0.50, 0.20, 0.20, 0.10) ← proposed default
- **Variant C (Address-boosted)**: (0.40, 0.15, 0.35, 0.10)

This is standard hyperparameter selection (analogous to τ/Δ/K tuning) and does NOT constitute label leakage, because:
- Labels are used only for **evaluation** (computing F0.5 on validation predictions)
- Labels never enter the scoring function itself
- The scoring function operates identically on test data (where no labels exist)

---

## 3. Heap Algorithm

### Data Structure: Per-S1 Min-Heap

```python
import heapq

# For each S1 entity, maintain:
heaps = {}  # s1_id -> list of (cheap_score, tid)

# Streaming insertion:
def heap_push(s1_id, tid, cheap_score, K):
    if s1_id not in heaps:
        heaps[s1_id] = []

    h = heaps[s1_id]
    if len(h) < K:
        heapq.heappush(h, (cheap_score, tid))
    elif cheap_score > h[0][0]:  # new score beats current minimum
        heapq.heapreplace(h, (cheap_score, tid))
    # else: discard candidate (worse than all K current survivors)
```

### Complexity Analysis

- **Per candidate insertion**: O(log K) for heap push/replace
- **Total Phase 2**: O(N_matched × log K) where N_matched = total (s1, tid) pairs across all blocking channels
- **Space**: O(N_s1 × K) heap entries

For the largest partition (India: 810K S1 entities, K=120):
- Heap memory: 810,000 × 120 × 24 bytes ≈ **2.33 GB** — within budget
- At K=80: 810,000 × 80 × 24 ≈ **1.56 GB** — comfortable

### Tie-Breaking

When `cheap_score` values are identical (which happens frequently for exact name matches), use **tid string** as a deterministic tiebreaker:

```python
heapq.heappush(h, (cheap_score, tid))
# Python's heapq compares tuples lexicographically, so ties on score
# are broken by tid string comparison — deterministic and source-agnostic.
```

This eliminates any Source 2 vs Source 3 ordering bias.

---

## 4. Memory Budget Analysis

### Per-Partition Peak Memory (Worst Case: India, 810K S1 Entities)

| Component | K=80 | K=120 | K=200 |
|---|:---:|:---:|:---:|
| **S1 Metadata** (pre-computed) | 450 MB | 450 MB | 450 MB |
| **Query Maps** (inverted indices) | 300 MB | 300 MB | 300 MB |
| **Phase 2 Heaps** (tid + score) | 1.56 GB | 2.33 GB | 3.89 GB |
| **Phase 2 Total** | **2.31 GB** | **3.08 GB** | **4.64 GB** |
| | | | |
| **Phase 3 Target Cache** (unique tids × 200B) | ~300 MB | ~400 MB | ~600 MB |
| **Phase 4 Scoring Chunks** (2000 × K × 30 × 4B) | ~19 MB | ~29 MB | ~48 MB |
| **Phase 3+4 Total** | **1.05 GB** | **1.15 GB** | **1.35 GB** |

**Critical observation**: Phases 2 and 3 are sequential, NOT concurrent. Query maps are freed after Phase 2. So peak memory is:

| K | Peak Memory (India Partition) | Verdict |
|---|:---:|---|
| 80 | ~2.3 GB | ✅ Safe on 8 GB system |
| 120 | ~3.1 GB | ✅ Safe on 8 GB system |
| 200 | ~4.6 GB | ⚠️ Tight on 8 GB system |
| 300 | ~6.9 GB | ❌ Risk of thrashing |

**Recommendation: K=120 as primary target, K=80 as safe fallback, K=200 as stretch if memory permits.**

---

## 5. Source Starvation Elimination

The two-pass architecture **completely eliminates** source starvation:

1. Both Source 2 and Source 3 are streamed in full during Phase 2.
2. Every candidate from every source competes for heap slots on equal footing based on `cheap_score`.
3. A Source 3 candidate with `cheap_score = 0.85` will evict a Source 2 candidate with `cheap_score = 0.30` regardless of arrival order.
4. The heap is order-agnostic by construction — only the score determines survival.

**Verification**: In Phase 3 re-streaming, the implementation should log the source distribution of surviving targets. This should show approximately proportional representation (not Source 2 dominated).

---

## 6. Diversity Constraints Assessment

### Question: Is a per-channel or per-source quota needed?

**Answer: NO, based on measured evidence.**

The pre-ranking score naturally provides diversity because:

1. **Strip-name channel candidates** have `name_dice ≈ 1.0` (names are identical after stripping) → they always rank highest, guaranteed survival.
2. **Token-sort channel candidates** have `name_dice ≈ 0.95+` (same tokens, different order) → rank just below exact matches.
3. **Tok-num channel candidates** have variable name similarity → some rank high (genuine partial matches), others rank low (spurious token co-occurrence). The pre-ranker correctly separates them.

**Explicit diversity would be HARMFUL** if it reserved slots for low-quality tok_num candidates at the expense of high-quality strip_name candidates. The pre-ranker's quality ranking already achieves implicit diversity through quality stratification.

### One Exception — Address-Only Linkage

True matches where names share zero tokens (DBA/trade-name aliases) will have `name_dice ≈ 0.0` and `containment = 0.0`. These will always rank below false candidates with partial name overlap. The `num_agree` signal provides partial rescue but cannot fully overcome zero name evidence.

**These matches are fundamentally irrecoverable at any bounded K without a name-independent similarity signal** (e.g., address embedding, phonetic encoding). They represent ~51.5% of blocking failures and are beyond the scope of Retrieval V3. Accept this as a known limitation.

---

## 7. K Sweep Specification

The implementation agent must benchmark the following K values on the exact 5,000-entity validation split (same population as the 0.80414 uncapped baseline):

| K | Purpose |
|---|---|
| 40 | Direct comparison to production Cap 40 |
| 60 | Intermediate point |
| 80 | Match previous lexical-only benchmark |
| 120 | Primary target (best memory/recall tradeoff) |
| 160 | Stretch target |
| 200 | Upper bound for 8 GB system |
| 300 | Theoretical maximum (if memory permits) |

For each K, report:
- Candidate recall (micro, macro, US, India)
- Average / median / P95 / max candidates per S1
- Precision, Recall, Macro F0.5
- Singleton accuracy
- Multi-match F0.5 (k ≥ 2)
- Extreme multi-match F0.5 (k ≥ 5)
- US F0.5, India F0.5
- Phase 2 wall-clock time
- Phase 3 wall-clock time
- Phase 4 wall-clock time
- Estimated peak memory

### Should K Vary by Entity?

**No, for V3.** Fixed K simplifies implementation, guarantees deterministic memory bounds, and the pre-ranker handles per-entity density variance through quality-based eviction. An entity with 1,000 candidates simply has more eviction rounds, but the top-K survivors are the best-quality candidates regardless of density.

If V3 results show entities with GT > K being systematically under-served, consider adaptive K in V4.

---

## 8. Validation Experiment Design

### Script: `src/validate_retrieval_v3.py`

**Inputs** (all frozen, read-only):
- `reports/val_s1_ids.txt` — first 5,000 entity IDs
- `dataset/train/train_source1.tsv` — S1 metadata
- `dataset/train/train_source2.tsv` — Source 2 targets
- `dataset/train/train_source3.tsv` — Source 3 targets
- `dataset/train/train_ground_truth.tsv` — validation labels (evaluation only)
- `output/lgb_matching_model.pkl` — frozen LightGBM model
- `src/features.py` — frozen feature extraction
- `src/evaluation.py` — frozen evaluation functions

**Process**:
1. Load 5,000 validation S1 entities and pre-compute metadata + query maps
2. For each K in sweep and each weight variant:
   a. Stream S2+S3, maintaining per-S1 min-heaps with cheap pre-ranking
   b. Collect surviving target IDs
   c. Re-stream S2+S3, loading metadata for survivors only
   d. Compute 30-dim features + LightGBM scores
   e. Apply frozen decision engine (τ=0.94, Δ=0.05, MAX_K=5)
   f. Evaluate against ground truth
3. Output comparison tables

**Outputs**:
- `reports/retrieval_v3_validation_results.md` — full K sweep and weight variant comparison
- Console log with timing and memory estimates

### Computational Budget

Per (K, weight_variant) configuration:
- 2 full passes through S2+S3: ~100s × 2 = 200s
- Feature computation + LightGBM scoring: ~60s (depends on K)
- Total per config: ~260s

K sweep (7 values) × weight variants (3): 21 runs × 260s = **~91 minutes total**

This is feasible in a single session. If time-constrained, run the 3 weight variants at K=120 only first (~13 min), select the best weight, then sweep all K values with the winner.

---

## 9. Success Criteria

### Primary Success Criterion (MUST ACHIEVE)

> At **K ≤ 120**, Retrieval V3 must achieve **Macro F0.5 ≥ 0.76** on the 5,000-entity validation split.

Rationale: This represents a material improvement over the best previous bounded system (Lexical Pre-Rank Cap 80: F0.5 = 0.74451) and demonstrates that the multi-signal pre-ranker outperforms name-Dice-only pre-ranking.

### Stretch Success Criterion (TARGET)

> At **K ≤ 200**, Retrieval V3 must achieve **Macro F0.5 ≥ 0.78** on the 5,000-entity validation split.

Rationale: This closes more than 60% of the gap between Cap 80 Lexical (0.74451) and Uncapped (0.80414), within practical memory constraints.

### Hard Constraint (MUST NOT VIOLATE)

> India Macro F0.5 must not fall below **0.60** at the selected K.

Rationale: India represents 46.7% of test entities. Any V3 configuration that sacrifices India performance is unacceptable.

---

## 10. Failure Criteria

### Configuration Failure

If ALL K values at ALL weight variants produce Macro F0.5 < 0.74, the multi-signal pre-ranker is not meaningfully better than simple name-Dice pre-ranking. In this case:
- Fall back to name-Dice-only pre-ranking (Strategy C/D from the audit)
- Proceed directly to implementation with Cap 120 + Dice-only ranking
- This still massively outperforms production Cap 40

### Architecture Failure

If Phase 2 (heap construction) takes > 300s per partition on validation (indicating O(n²) blowup), or Phase 3 re-streaming fails to find metadata for > 1% of surviving target IDs, the two-pass architecture has an implementation bug. Debug before proceeding.

---

## 11. Rollback Strategy

The frozen submission (`deepresolve_er_submission.zip`, `output/matching_results.tsv`, `output/candidate_pairs.tsv`) is **never touched** during V3 development.

If V3 validation succeeds but test re-inference fails (OOM, file corruption, runtime exceeding competition deadline):
1. **Immediately rollback** to the existing frozen ZIP
2. The frozen submission, despite its Cap 40 degradation, is a valid, passing submission
3. V3 test inference would be a strictly additive improvement — never a replacement of the safety net

The rollback is automatic: the frozen ZIP is never deleted or overwritten.

---

## 12. Implementation Checklist for Gemini Agent

### Files to Create (NEW)

1. **`src/inference_v3.py`** — Production-ready two-pass inference script
   - Must process countries sequentially (France → US → India) like current `inference_partitioned.py`
   - Must write output TSVs in the EXACT same format and entity order as current submission
   - Must include `MAX_HEAP_K` as a configurable constant (default: 120)
   - Must include weight vector as a configurable tuple

2. **`src/validate_retrieval_v3.py`** — Validation-only benchmark script
   - Runs on 5,000 validation entities
   - Sweeps K values and weight variants
   - Reports full performance matrix
   - Does NOT write to `output/` directory

### Files to READ (FROZEN — DO NOT MODIFY)

- `src/features.py` — Import `normalize_text`, `strip_legal`, `token_sort_form`, `extract_numbers_clean`, `extract_significant_tokens`, `ngram_dice_similarity`, `compute_pairwise_features`
- `src/evaluation.py` — Import `evaluate_predictions`
- `output/lgb_matching_model.pkl` — Load with pickle, call `predict_proba`

### Files to NEVER TOUCH

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `deepresolve_er_submission.zip`
- `src/inference_partitioned.py` (preserve for reference)

---

## 13. Pseudocode for Core Two-Pass Loop

```python
for country in ["France", "US", "India"]:

    # PHASE 1: Load S1 partition
    s1_partition, s1_order = load_s1_for_country(country)
    query_maps = build_inverted_indices(s1_partition)

    # PHASE 2: Streaming heap construction (NO metadata stored)
    heaps = {}  # s1_id -> min-heap of (cheap_score, tid)

    for source_file in [source2_path, source3_path]:
        for line in stream_file(source_file):
            tid, bname, baddr, c = parse_line(line)
            if c != country:
                continue

            # Compute target features inline (temporary — not stored)
            t_strip = strip_legal(normalize_text(bname))
            t_tokens = t_strip.split()
            t_nums = extract_numbers_clean(baddr)

            # Match against query maps
            matched_s1, channel_counts = match_candidate(t_strip, t_tokens, t_nums, query_maps)

            for s1_id in matched_s1:
                score = compute_cheap_score(s1_partition[s1_id], t_strip, t_tokens, t_nums, channel_counts[s1_id])
                heap_push(heaps, s1_id, tid, score, K=MAX_HEAP_K)

    # Collect surviving target IDs
    surviving_tids = set()
    for h in heaps.values():
        for score, tid in h:
            surviving_tids.add(tid)

    # Free query maps (no longer needed)
    del query_maps
    gc.collect()

    # PHASE 3: Re-stream to load metadata for survivors only
    target_cache = {}
    for source_file in [source2_path, source3_path]:
        for line in stream_file(source_file):
            tid, bname, baddr, c = parse_line(line)
            if tid in surviving_tids and tid not in target_cache:
                target_cache[tid] = build_target_metadata(bname, baddr, c)

    # PHASE 4: Score with frozen LightGBM + decision engine
    for chunk in chunks(s1_order, 2000):
        for s1_id in chunk:
            cands = [(tid, score) for score, tid in heaps.get(s1_id, [])]
            # Write candidate_pairs.tsv
            # Compute pairwise features for all cands
            # LightGBM predict_proba
            # Apply τ=0.94, Δ=0.05, K≤5
            # Write matching_results.tsv

    # PHASE 5: Cleanup
    del s1_partition, heaps, target_cache
    gc.collect()
```

---

## 14. Critical Implementation Notes

1. **Candidate deduplication**: A target ID may match the same S1 through multiple channels. The heap naturally handles this — if `tid` already exists in the heap with a lower score, the new (higher) score should replace it. Use a parallel `set` per S1 to track which tids are currently in the heap:
   ```python
   if tid in heap_tid_set[s1_id]:
       # Update score if new score is higher (requires heap rebuild or lazy deletion)
       pass
   ```
   **Simplest correct approach**: Allow duplicate tids in the heap. After Phase 2, deduplicate by keeping the highest-scoring entry per tid. This wastes a small amount of heap space but avoids complex bookkeeping.

2. **Query map key format**: The current production code uses `strip_name_query[s_name]` (no country prefix) because it processes one country at a time. V3 must maintain this convention.

3. **Output ordering**: The output TSVs must be written in the exact order of `s1_order` (same order as `test_source1.tsv` for the partition). This is already how the current code works.

4. **Candidate pairs superset**: `candidate_pairs.tsv` should list ALL surviving heap candidates (up to K per S1), and `matching_results.tsv` should be a subset selected by the decision engine.

5. **Empty heaps**: S1 entities with zero blocking matches have empty heaps. They must still appear in both output files (with empty match/candidate lists).

---

## 15. Expected Performance Projections

Based on measured data from the audit, with the multi-signal pre-ranker:

| K | Expected Cand Recall | Expected Macro F0.5 | Confidence |
|---|:---:|:---:|---|
| 40 | 64–66% | 0.71–0.73 | HIGH (extrapolating from Dice-only at 63.30% / 0.70598) |
| 80 | 69–72% | 0.75–0.77 | MODERATE (multi-signal should outperform Dice-only at 67.34% / 0.74451) |
| 120 | 72–74% | 0.77–0.79 | MODERATE |
| 200 | 74–76% | 0.78–0.80 | LOW-MODERATE (approaching uncapped ceiling of 76.05% / 0.80414) |

These are **projections**, not guarantees. The validation experiment (Section 8) will produce exact measurements.

---

## 16. Summary of Decisions

| Decision Point | Choice | Rationale |
|---|---|---|
| Architecture | Two-pass streaming min-heap | Decouples selection from metadata; predictable memory |
| Pre-ranking signal | Multi-signal (Dice + Containment + NumAgree + Channel) | Dice proven dominant; orthogonal signals provide marginal lift |
| Diversity constraint | None (implicit via quality ranking) | Pre-ranker naturally stratifies channels; explicit quotas would be harmful |
| K selection | Fixed K per run; sweep 40–300 | Simpler than adaptive; pre-ranker handles density variance |
| Primary K target | K=120 | Best memory/recall tradeoff for 8 GB system |
| Source starvation fix | Eliminated by architecture | Heap is order-agnostic; both sources compete equally |
| Label usage | Evaluation only; never in scoring function | Standard hyperparameter selection; no leakage |
| Rollback | Frozen ZIP preserved; V3 is additive | Zero risk to existing submission |
