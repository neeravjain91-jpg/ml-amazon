#!/usr/bin/env python3
"""
Phase 7: Pairwise Feature Engineering Module
ML Challenge 2026 - Business Entity Resolution
"""

import re
import unicodedata
from collections import Counter
import math

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'gmbh', 'sa', 'sarl', 'llp', 'pllc',
    'services', 'solutions', 'enterprises', 'enterprise', 'technologies',
    'technology', 'group', 'holdings', 'international', 'global', 'india',
    'associates', 'trading', 'consulting', 'management', 'industries',
    'industry', 'agency', 'center', 'centre', 'stores', 'store', 'shop'
}

ADDR_STOPWORDS = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'court', 'ct', 'boulevard', 'blvd', 'way', 'place', 'pl', 'near', 'opp',
    'opposite', 'behind', 'beside', 'floor', 'fl', 'block', 'blk', 'sector',
    'sec', 'phase', 'nagar', 'colony', 'city', 'state', 'india', 'usa', 'us'
}

def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def strip_legal(norm_name):
    tokens = norm_name.split()
    f = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(f) if f else norm_name

def token_sort_form(text):
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def extract_numbers_clean(text):
    if not text:
        return set()
    nums = re.findall(r'\b\d+\b', text)
    return set(n.lstrip('0') or '0' for n in nums)

def extract_significant_tokens(norm_name):
    tokens = norm_name.split()
    return [t for t in tokens if len(t) >= 3 and t not in LEGAL_SUFFIXES]

def char_ngrams(text, n=3):
    if not text:
        return set()
    padded = f"  {text}  "
    return set(padded[i:i+n] for i in range(len(padded) - n + 1))

def ngram_dice_similarity(text1, text2, n=3):
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0
    ng1 = char_ngrams(text1, n)
    ng2 = char_ngrams(text2, n)
    if not ng1 or not ng2:
        return 0.0
    return 2.0 * len(ng1 & ng2) / (len(ng1) + len(ng2))

def token_jaccard(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0

def token_containment(tokens1, tokens2):
    """How much of the smaller entity is contained in the larger entity."""
    if not tokens1 or not tokens2:
        return 0.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    inter = len(s1 & s2)
    min_len = min(len(s1), len(s2))
    return inter / min_len if min_len > 0 else 0.0

def prefix_match_len(s1, s2):
    min_len = min(len(s1), len(s2))
    cnt = 0
    for i in range(min_len):
        if s1[i] == s2[i]:
            cnt += 1
        else:
            break
    return cnt / max(len(s1), len(s2), 1)

FEATURE_NAMES = [
    # Name Features
    "name_raw_exact",
    "name_norm_exact",
    "name_strip_exact",
    "name_tokensort_exact",
    "name_dice_3gram",
    "name_dice_2gram",
    "name_token_jaccard",
    "name_token_containment",
    "name_prefix_ratio",
    "name_len_diff",
    "name_len_ratio",
    "name_shared_token_count",
    
    # Address Features
    "addr_missing",
    "addr_raw_exact",
    "addr_norm_exact",
    "addr_dice_3gram",
    "addr_token_jaccard",
    "addr_token_containment",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_shared_token_count",
    
    # Numeric / Postal Features
    "num_shared_count",
    "num_jaccard",
    "num_exact_match",
    
    # Country / Cross-Field Features
    "country_match",
    "name_addr_dice_prod",
    "strong_name_flag",
    "strong_addr_flag",
    "both_strong_flag",
    "evidence_channel_count",
]

def compute_pairwise_features(r1, r2):
    """
    Compute dense feature vector for pair (r1, r2).
    r1: dict of S1 entity {raw_name, norm_name, strip_name, raw_addr, norm_addr, country, nums, ...}
    r2: dict of target entity {raw_name, norm_name, strip_name, raw_addr, norm_addr, country, nums, ...}
    
    Returns:
        list of float feature values corresponding to FEATURE_NAMES
    """
    n1_raw, n2_raw = r1["raw_name"], r2["raw_name"]
    n1_norm, n2_norm = r1["norm_name"], r2["norm_name"]
    n1_strip, n2_strip = r1["strip_name"], r2["strip_name"]
    
    a1_raw, a2_raw = r1["raw_addr"], r2["raw_addr"]
    a1_norm, a2_norm = r1["norm_addr"], r2["norm_addr"]
    
    # 1. Name Features
    name_raw_exact = 1.0 if n1_raw == n2_raw else 0.0
    name_norm_exact = 1.0 if n1_norm == n2_norm else 0.0
    name_strip_exact = 1.0 if n1_strip == n2_strip else 0.0
    
    ts1 = token_sort_form(n1_strip)
    ts2 = token_sort_form(n2_strip)
    name_tokensort_exact = 1.0 if ts1 == ts2 else 0.0
    
    name_dice_3 = ngram_dice_similarity(n1_strip, n2_strip, 3)
    name_dice_2 = ngram_dice_similarity(n1_strip, n2_strip, 2)
    
    tok1 = n1_strip.split()
    tok2 = n2_strip.split()
    name_jac = token_jaccard(tok1, tok2)
    name_contain = token_containment(tok1, tok2)
    name_prefix = prefix_match_len(n1_strip, n2_strip)
    
    len1, len2 = len(n1_strip), len(n2_strip)
    name_len_diff = abs(len1 - len2)
    name_len_ratio = min(len1, len2) / max(len1, len2, 1)
    name_shared_tok = len(set(tok1) & set(tok2))
    
    # 2. Address Features
    addr_missing = 1.0 if not a2_norm else 0.0
    if addr_missing:
        addr_raw_exact = 0.0
        addr_norm_exact = 0.0
        addr_dice_3 = 0.0
        addr_jac = 0.0
        addr_contain = 0.0
        addr_len_diff = len(a1_norm)
        addr_len_ratio = 0.0
        addr_shared_tok = 0
    else:
        addr_raw_exact = 1.0 if a1_raw == a2_raw else 0.0
        addr_norm_exact = 1.0 if a1_norm == a2_norm else 0.0
        addr_dice_3 = ngram_dice_similarity(a1_norm, a2_norm, 3)
        
        atok1 = a1_norm.split()
        atok2 = a2_norm.split()
        addr_jac = token_jaccard(atok1, atok2)
        addr_contain = token_containment(atok1, atok2)
        
        alen1, alen2 = len(a1_norm), len(a2_norm)
        addr_len_diff = abs(alen1 - alen2)
        addr_len_ratio = min(alen1, alen2) / max(alen1, alen2, 1)
        addr_shared_tok = len(set(atok1) & set(atok2))
        
    # 3. Numeric / Postal Features
    nums1 = r1.get("nums")
    if nums1 is None:
        nums1 = extract_numbers_clean(a1_raw)
    nums2 = r2.get("nums")
    if nums2 is None:
        nums2 = extract_numbers_clean(a2_raw)
        
    shared_nums = nums1 & nums2
    num_shared_count = len(shared_nums)
    num_jac = len(shared_nums) / len(nums1 | nums2) if (nums1 or nums2) else 0.0
    num_exact = 1.0 if (nums1 and nums2 and nums1 == nums2) else 0.0
    
    # 4. Country & Cross-Field
    country_match = 1.0 if r1["country"] == r2["country"] else 0.0
    name_addr_dice_prod = name_dice_3 * addr_dice_3
    
    strong_name = 1.0 if (name_dice_3 >= 0.85 or name_strip_exact) else 0.0
    strong_addr = 1.0 if (addr_dice_3 >= 0.70 or num_shared_count >= 1) else 0.0
    both_strong = 1.0 if (strong_name and strong_addr) else 0.0
    
    evidence_channels = sum([
        name_strip_exact,
        1.0 if name_dice_3 >= 0.75 else 0.0,
        1.0 if addr_dice_3 >= 0.60 else 0.0,
        1.0 if num_shared_count >= 1 else 0.0,
        1.0 if name_jac >= 0.50 else 0.0,
    ])
    
    return [
        name_raw_exact,
        name_norm_exact,
        name_strip_exact,
        name_tokensort_exact,
        name_dice_3,
        name_dice_2,
        name_jac,
        name_contain,
        name_prefix,
        name_len_diff,
        name_len_ratio,
        name_shared_tok,
        addr_missing,
        addr_raw_exact,
        addr_norm_exact,
        addr_dice_3,
        addr_jac,
        addr_contain,
        addr_len_diff,
        addr_len_ratio,
        addr_shared_tok,
        num_shared_count,
        num_jac,
        num_exact,
        country_match,
        name_addr_dice_prod,
        strong_name,
        strong_addr,
        both_strong,
        evidence_channels,
    ]
