#!/usr/bin/env python3
"""
Canonical Data Cleaning & Deterministic Normalization Layer
ML Challenge 2026 - Business Entity Resolution
Rigorous, Immutable Pre-Processing with Zero Identity-Bearing Information Loss
"""

import re
import unicodedata
from collections import Counter

# Standard Legal corporate designator tokens across US, India, UK, France, Germany
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'gmbh', 'sa', 'sarl', 'llp', 'pllc',
    'services', 'solutions', 'enterprises', 'enterprise', 'technologies',
    'technology', 'group', 'holdings', 'international', 'global', 'india',
    'associates', 'trading', 'consulting', 'management', 'industries',
    'industry', 'agency', 'center', 'centre', 'stores', 'store', 'shop'
}

# Controlled address expansion dictionary for standardized abbreviations
ADDR_EXPANSIONS = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'ln': 'lane',
    'dr': 'drive', 'blvd': 'boulevard', 'ct': 'court', 'fl': 'floor',
    'blk': 'block', 'sec': 'sector', 'dept': 'department', 'apt': 'apartment',
    'ste': 'suite', 'hwy': 'highway', 'pkwy': 'parkway', 'pl': 'place'
}

def clean_unicode(text):
    """
    Unicode normalization & hygiene:
    - Normalizes via NFKD decomposition.
    - Eliminates non-printable control characters and corrupted surrogate/replacement bytes.
    - Strips combining diacritical marks while preserving the underlying alphanumeric glyphs.
    - Normalizes all whitespace characters (non-breaking space, tabs, etc.) to single spaces.
    - Preserves case and punctuation.
    """
    if not text:
        return ""
    # Normalize unicode
    decomposed = unicodedata.normalize('NFKD', str(text))
    # Filter combining marks and non-printable control characters (except standard spaces)
    chars = []
    for c in decomposed:
        if unicodedata.combining(c):
            continue
        cat = unicodedata.category(c)
        if cat.startswith('C'): # Control characters (\x80-\x9f, \x00-\x1f, etc.)
            continue
        chars.append(c)
    cleaned = "".join(chars)
    # Collapse whitespace
    return " ".join(cleaned.split()).strip()

def canonicalize_name(raw_name):
    """
    Produces deterministic canonical name representations.
    Retains:
    - name_unicode: unicode normalized, case & punctuation preserved
    - name_lower: lowercase version of name_unicode
    - name_clean: lowercase, controlled &->and, punctuation to spaces, normalized spaces
    - name_alphanumeric: alphanumeric tokens only
    - name_tokenized: space-separated clean tokens
    - name_token_sorted: alphabetically sorted tokens
    - name_legal_stripped: corporate designators removed from name_clean
    """
    if not raw_name:
        return {
            'name_unicode': "",
            'name_lower': "",
            'name_clean': "",
            'name_alphanumeric': "",
            'name_tokenized': "",
            'name_token_sorted': "",
            'name_legal_stripped': "",
        }
    
    u = clean_unicode(raw_name)
    l = u.lower()
    
    # Controlled '&' -> ' and ' normalization
    c = re.sub(r'&', ' and ', l)
    # Replace non-alphanumeric punctuation with spaces
    c = re.sub(r'[^\w\s]', ' ', c)
    c = " ".join(c.split()).strip()
    
    alnum = " ".join(re.findall(r'[a-z0-9]+', c))
    toks = c.split()
    tok_sorted = " ".join(sorted(toks))
    
    # Legal stripped
    non_legal = [t for t in toks if t not in LEGAL_SUFFIXES]
    legal_stripped = " ".join(non_legal) if non_legal else c
    
    return {
        'name_unicode': u,
        'name_lower': l,
        'name_clean': c,
        'name_alphanumeric': alnum,
        'name_tokenized': c,
        'name_token_sorted': tok_sorted,
        'name_legal_stripped': legal_stripped,
    }

def canonicalize_address(raw_addr):
    """
    Produces deterministic canonical address representations.
    Retains:
    - address_unicode: unicode normalized, case & punctuation preserved
    - address_lower: lowercase version of address_unicode
    - address_clean: standardized abbreviations (rd->road, st->street), whitespace normalized
    - address_alphanumeric: alphanumeric tokens only
    - address_tokens: tokenized address string
    - address_numeric_tokens: extracted numbers clean
    - address_postal_tokens: extracted postal/PIN candidates
    - address_possible_house_number: leading / first numeric token
    """
    if not raw_addr:
        return {
            'address_unicode': "",
            'address_lower': "",
            'address_clean': "",
            'address_alphanumeric': "",
            'address_tokens': "",
            'address_numeric_tokens': "",
            'address_postal_tokens': "",
            'address_possible_house_number': "",
        }
    
    u = clean_unicode(raw_addr)
    l = u.lower()
    
    # Controlled expansion of standard street/road abbreviations
    words = re.findall(r'[a-z0-9]+', l)
    expanded = [ADDR_EXPANSIONS.get(w, w) for w in words]
    c = " ".join(expanded)
    alnum = c
    toks = c.split()
    
    # Extracted numbers (leading zeros stripped)
    nums = [n.lstrip('0') or '0' for n in re.findall(r'\b\d+\b', c)]
    numeric_tokens_str = " ".join(nums)
    
    # Postal / PIN candidates (5-digit or 6-digit integers, or 5+4 US ZIPs)
    postal_matches = re.findall(r'\b(?:\d{5,6}|\d{5}-\d{4})\b', u)
    postal_tokens_str = " ".join(postal_matches)
    
    # Leading house / building number candidate
    house_num = ""
    # Check if address starts with a number or letter-number (e.g. 123, A/203, A-212)
    leading_num_match = re.search(r'^\s*(?:[a-zA-Z]/|[a-zA-Z]-)?(\d+)\b', u)
    if leading_num_match:
        house_num = leading_num_match.group(1).lstrip('0') or '0'
    elif nums:
        house_num = nums[0]
        
    return {
        'address_unicode': u,
        'address_lower': l,
        'address_clean': c,
        'address_alphanumeric': alnum,
        'address_tokens': " ".join(toks),
        'address_numeric_tokens': numeric_tokens_str,
        'address_postal_tokens': postal_tokens_str,
        'address_possible_house_number': house_num,
    }

def canonicalize_country(raw_country):
    """
    Open-set country normalization:
    - Strips whitespace.
    - Preserves arbitrary unseen country values (US, India, France, etc.).
    """
    if not raw_country:
        return ""
    c = clean_unicode(raw_country).strip()
    return c.title() if c.islower() else c

def canonicalize_record(entity_id, business_name, business_address, country):
    """
    Canonicalizes a single record and returns a dictionary with all original and canonical fields.
    """
    name_dict = canonicalize_name(business_name)
    addr_dict = canonicalize_address(business_address)
    c_clean = canonicalize_country(country)
    
    rec = {
        'entity_id': str(entity_id).strip(),
        'business_name': str(business_name) if business_name is not None else "",
        'business_address': str(business_address) if business_address is not None else "",
        'country': str(country) if country is not None else "",
        'country_clean': c_clean,
    }
    rec.update(name_dict)
    rec.update(addr_dict)
    return rec
