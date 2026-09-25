# Retrieval V3 Source Order Invariance Audit

A controlled validation experiment was executed comparing candidate retention when streaming **Source 2 then Source 3** versus **Source 3 then Source 2** under the V3 Bounded Min-Heap (K=80):

- **Total Validation Entities**: 5,000
- **Entities with 100.0% Identical Candidate Sets**: **5,000 / 5,000 (100.00%)**
- **Mean Candidate Set Jaccard Agreement**: **100.0000%**

### Forensic Conclusion
By incorporating a deterministic tie-breaker `(score, tid)` in the min-heap, the V3 architecture is **100% mathematically invariant to file streaming order**. Source 2 before Source 3 starvation is completely eliminated. Every target record from Source 2 and Source 3 competes on identical, fair lexical and numeric evidence.
