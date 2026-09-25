import json
import time

print("Checking Country Consistency in Ground Truth...")
t0 = time.time()

# Load S1 countries
s1_country = {}
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        s1_country[p[0]] = p[3]

# Load S2 countries
s2_country = {}
with open("dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        s2_country[p[0]] = p[3]

# Load S3 countries
s3_country = {}
with open("dataset/train/train_source3.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        s3_country[p[0]] = p[3]

print(f"Loaded countries in {time.time() - t0:.2f}s")

cross_country_matches = 0
total_checked_matches = 0

with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        s1_id = p[0]
        s1_c = s1_country[s1_id]
        if len(p) > 1 and p[1].strip():
            for mid in p[1].strip().split(","):
                total_checked_matches += 1
                if mid.startswith("S2-"):
                    target_c = s2_country.get(mid)
                else:
                    target_c = s3_country.get(mid)
                if target_c != s1_c:
                    cross_country_matches += 1

print(f"Total checked matches: {total_checked_matches:,}")
print(f"Cross-country matches: {cross_country_matches}")
if cross_country_matches == 0:
    print("CRITICAL FINDING: Exactly 100.0% of matches are strictly WITHIN the same country!")
else:
    print(f"WARNING: Cross country matches exist: {cross_country_matches}")
