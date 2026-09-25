# Canonical Data Cleaning & Representation Specification

## 1. Principles & Non-Destructive Hygiene
- **Preservation of Raw Fields**: Every canonical record strictly retains the original immutable raw fields (`entity_id`, `business_name`, `business_address`, `country`).
- **Deterministic Representations**: All representations are pure deterministic mathematical projections without external knowledge bases, geocoders, or web queries.
- **Non-Lossy Design**: No common tokens or entity identifiers are discarded. Legal suffixes are preserved in separate representations rather than wiped from the core text.
- **Open-Set Country Support**: Country normalization enforces clean casing while remaining strictly open-set (preserving US, India, France, and unseen values).

## 2. Canonical Column Definitions

| Column Name | Type | Description | Transformation Logic |
|---|:---:|---|---|
| `entity_id` | String | Immutable entity identifier | Stripped of extraneous whitespace |
| `business_name` | String | Original raw business name | Exact byte-for-byte original |
| `name_unicode` | String | Unicode-cleaned name | NFKD normalized, control chars & accents stripped, whitespace collapsed, case preserved |
| `name_lower` | String | Lowercase name | `name_unicode.lower()` |
| `name_clean` | String | Standardized business name | Lowercase, `&` -> `and`, punctuation to spaces, normalized spaces |
| `name_alphanumeric` | String | Alphanumeric representation | Letters and numbers only (`[a-z0-9]`) |
| `name_tokenized` | String | Tokenized representation | Whitespace-delimited clean tokens |
| `name_token_sorted` | String | Token-sorted name | Tokens sorted alphabetically for word-order invariance |
| `name_legal_stripped` | String | Legal corporate suffix stripped | Suffixes (`pvt`, `ltd`, `inc`, `corp`, etc.) removed if non-empty |
| `business_address` | String | Original raw address | Exact byte-for-byte original |
| `address_unicode` | String | Unicode-cleaned address | NFKD normalized, control chars stripped, case preserved |
| `address_lower` | String | Lowercase address | `address_unicode.lower()` |
| `address_clean` | String | Standardized address | Standard street/road abbreviations expanded (`rd`->`road`, `st`->`street`), punctuation to spaces |
| `address_alphanumeric`| String | Alphanumeric address | Alphanumeric tokens only |
| `address_tokens` | String | Tokenized address | Whitespace-delimited clean address tokens |
| `address_numeric_tokens`| String | Extracted numeric tokens | Numbers extracted with leading zeros stripped, space-delimited |
| `address_postal_tokens` | String | Candidate postal/PIN codes | 5-digit US/French postal codes and 6-digit Indian PIN codes |
| `address_possible_house_number`| String | Candidate building/unit | Leading numeric token from street address |
| `country` | String | Original raw country | Exact byte-for-byte original |
| `country_clean` | String | Normalized country | Superficial whitespace/casing normalization (Open-set) |
