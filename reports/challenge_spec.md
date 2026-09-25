# Amazon ML Challenge — Business Entity Resolution Specification

**Authoritative Source Document**: `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (Archived in `docs/problem_statement.pdf`)  
**Specification Date**: September 2026  
**Auditor**: Senior ML Engineering & Research Team  

---

## 1. Executive Summary & Objective

The challenge objective is to build an offline Machine Learning and algorithmic solution for **Business Entity Resolution (ER)** across three independent, heterogeneous data sources containing noisy and inconsistent records.

* **Reference Source**: **Source 1** is the deduplicated reference source.
* **Target Objective**: For every Source 1 entity, identify **ALL** corresponding records from **Source 2** and **Source 3**.
* **Cardinality**: Matching is **NOT** forced one-to-one ($1:1$). A Source 1 entity may link to:
  * Zero records (**Singleton / No-match**)
  * Exactly one record ($1:1$)
  * Multiple records ($1:M$) across Source 2 and/or Source 3.

---

## 2. Page-by-Page Specification & Rules Breakdown

### Page 1: Problem Definition & Data Schemas
* **Source 1 Reference Role** (Page 1): Source 1 entities serve as queries/anchors. For each Source 1 entity, the system must discover all matching records in Source 2 and Source 3.
* **File Format Requirement** (Page 1):
  * All dataset and output files are strictly Tab-Separated Values (`.tsv`).
  * Tabs are required because addresses and entity ID lists frequently contain commas.
  * Loading must strictly use `sep="\t"` (e.g., `pd.read_csv(filepath, sep="\t")`).
* **Source Data Attributes** (Page 1):
  Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) consists of 4 columns:
  1. `entity_id`: Unique identifier formatted with a source prefix (`S1-`, `S2-`, or `S3-`).
  2. `business_name`: Free-text name of business, subject to abbreviations, legal suffixes, typographical errors, and transliterations.
  3. `business_address`: Free-text business address, subject to omissions, reordering, landmark references, and postal formatting variations.
  4. `country`: Country designation.

### Page 1–2: Country Open-Set Constraint & Schema Nuances
* **Open-Set Country Requirement** (Pages 1–2):
  * Training dataset covers: **US** and **India**.
  * Test dataset additionally introduces a third country: **France**, which is completely unobserved during training.
  * **Strict Requirement**: Pipelines must treat `country` as an open set of string labels.
  * **Prohibited**: Hard-coding, filtering, or one-hot encoding restricted strictly to `{'US', 'India'}`.
  * Every test entity (including France) must be processed and represented in the final submission.
* **Source Attribution** (Page 2):
  * No separate `source` column exists; source origin is inferred solely by file identity and `entity_id` prefix (`S1-`, `S2-`, `S3-`).
* **Ground Truth Schema** (`train_ground_truth.tsv`) (Page 2):
  * `source1_entity_id`: S1 record identifier.
  * `matched_entity_ids`: Comma-separated list of matching entity IDs from Source 2 and/or Source 3 (e.g., `S2-00047,S3-00812`).
  * If the Source 1 entity is a singleton (has no matches), this field is empty.

### Page 2: Noise Taxonomy & Real-World Variations
The problem statement documents real-world data corruptions:
* **Business Name Noise**:
  * Corporate/Legal abbreviations (e.g., `Corp` vs. `Corporation`, `Pvt` vs. `Private`, `Ltd` vs. `Limited`).
  * Legal suffix inconsistencies and omissions.
  * DBA / Trade names ("Doing Business As").
  * Punctuation variations (`&` vs. `and`, hyphens, periods).
  * Word-order permutations and transpositions.
  * Typographical errors and phonetic/orthographic transliterations.
* **Address Noise**:
  * Thoroughfare abbreviations (e.g., `Rd` vs. `Road`, `St` vs. `Street`, `Ave` vs. `Avenue`).
  * Transliteration variants across regional naming.
  * Missing address components (omitted postal PIN codes, omitted states or cities).
  * Landmark-based references (e.g., `Near SBI ATM`, `Opposite Central Park`).
  * Municipal numbering conventions and diverse formatting.
  * Component order permutations (e.g., `City, State, Street` vs. `Street, City, State`).

### Page 3–4: Evaluation Protocol & Output Specification
* **Validation Split Requirement** (Page 3): Test set ground truth is withheld. Offline validation must hold out a split from the training dataset and evaluate using the macro $F_{0.5}$ metric.
* **Required Output Artifacts** (Pages 3–4):
  Two files placed in the `output/` directory:
  1. `output/matching_results.tsv`: Final resolved matches (leaderboard scored).
  2. `output/candidate_pairs.tsv`: Final candidate set from the blocking/candidate-generation stage fed to the scoring model.
* **Format Specification for `matching_results.tsv`** (Page 3):
  * Column 1: `source1_entity_id`
  * Column 2: `matched_entity_ids` (comma-separated, no quotes, no spaces after commas).
  * Exactly one row per test Source 1 entity.
  * Leave `matched_entity_ids` empty for singletons.
  * No duplicate IDs within any ID list.
  * ID lists must contain only S2 or S3 IDs that exist in the test set.
  * Tab-delimited (`\t`).
* **Format Specification for `candidate_pairs.tsv`** (Page 4):
  * Column 1: `source1_entity_id`
  * Column 2: `candidate_entity_ids` (comma-separated, no quotes).
  * Represents the candidate set fed into the ML model.
  * Every ID in `matching_results.tsv` must strictly appear in `candidate_pairs.tsv` ($M \subseteq C$). A match that is not in the candidate set indicates a pipeline bug.
  * One row per Source 1 entity.
* **Verification Utility** (Page 4):
  * `utils/validate_submission.py` checks both output files against structural, format, and consistency rules.
  * Must exit with code 0 (`PASS`).

### Page 5: Packaging & Model Constraints
* **Submission Package Structure**:
  ```
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
  ```
* **Model Constraints** (Page 5, Constraint 5):
  * **License**: Permitted open-source license: **MIT** or **Apache 2.0**.
  * **Parameter Limit**: Maximum **8 Billion parameters**. Ambiguous or proprietary non-commercial licenses are rejected.

### Page 6: Evaluation Metric — Macro-Averaged $F_{0.5}$
* **Metric Formula**:
  $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
* **Macro-Averaging**: Calculated individually for each Source 1 entity, then arithmetic mean across all Source 1 entities in evaluation set.
* **Precision Weighting**: Precision is weighted $2\times$ more heavily than recall ($\beta=0.5$). False merges (linking two non-identical businesses) incur high penalty.
* **Singleton Evaluation**:
  * An S1 entity with no true matches (empty set) receives $F_{0.5} = 1.0$ if and only if the model predicts an empty set.
  * If the model predicts any candidate for a true singleton, precision $= 0$, recall $= 0 \implies F_{0.5} = 0.0$.
  * Accurate identification of singletons is critical to macro $F_{0.5}$.

### Page 7: Academic Integrity, Fair Play & Technical Guidance
* **Strictly Prohibited: External Data Lookup**:
  * No commercial entity resolution APIs.
  * No government registry lookups (e.g., MCA India, US SEC/state registries).
  * No geocoding APIs (Google Maps, OpenStreetMap, Nominatim, Mapbox, etc.).
  * No external web searching or online business enrichment.
  * Immediate disqualification upon evidence of external network lookup.
* **Permitted Resources**:
  * Supplied training data.
  * Supplied test data.
  * Offline algorithmic transformations and locally trained models.
  * Permitted MIT/Apache 2.0 open-source libraries.

---

## 3. Engineering Checklist Derived from Specifications

| ID | Requirement | Authoritative Source | Verification Check |
|---|---|---|---|
| REQ-01 | Read/write TSV with `sep="\t"` | PDF Page 1, 3 | All I/O explicitly tab-delimited |
| REQ-02 | Dedup Source 1 reference queries | PDF Page 1 | S1 entities unique, evaluated as queries |
| REQ-03 | Support $0, 1, M$ matches | PDF Page 1, 3 | Set-based decision layer |
| REQ-04 | Open-set country (US, India, France) | PDF Page 1-2 | Zero hardcoding of country labels |
| REQ-05 | No external network lookup | PDF Page 7 | Isolated local offline execution |
| REQ-06 | MIT/Apache 2.0 license, $\le 8\text{B}$ params | PDF Page 5 | License and size audit for all components |
| REQ-07 | Generate `matching_results.tsv` and `candidate_pairs.tsv` | PDF Page 3-4 | Output generator & validator |
| REQ-08 | Candidate containment ($M \subseteq C$) | PDF Page 4 | Every match in candidate set |
| REQ-09 | Macro-averaged $F_{0.5}$ metric engine | PDF Page 6 | Exact implementation matching specification |
| REQ-10 | Validation runner `utils/validate_submission.py` | PDF Page 4 | Script execution passes with exit code 0 |
| REQ-11 | Complete package structure & documentation | PDF Page 5 | ZIP bundle matches directory specification |
