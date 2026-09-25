import time
from collections import Counter

print("Computing country-specific ground truth distribution...")
t0 = time.time()

s1_country = {}
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        s1_country[p[0]] = p[3]

gt_by_country = {"US": Counter(), "India": Counter()}
s2_s3_by_country = {
    "US": {"singletons": 0, "only_s2": 0, "only_s3": 0, "both": 0, "total_matches": 0, "s2_matches": 0, "s3_matches": 0},
    "India": {"singletons": 0, "only_s2": 0, "only_s3": 0, "both": 0, "total_matches": 0, "s2_matches": 0, "s3_matches": 0},
}

with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        s1_id = p[0]
        c = s1_country[s1_id]
        raw_m = p[1].strip() if len(p) > 1 else ""
        if not raw_m:
            gt_by_country[c][0] += 1
            s2_s3_by_country[c]["singletons"] += 1
            continue
            
        mids = raw_m.split(",")
        n_m = len(mids)
        gt_by_country[c][n_m] += 1
        s2_s3_by_country[c]["total_matches"] += n_m
        
        has_s2 = any(m.startswith("S2-") for m in mids)
        has_s3 = any(m.startswith("S3-") for m in mids)
        s2_count = sum(1 for m in mids if m.startswith("S2-"))
        s3_count = sum(1 for m in mids if m.startswith("S3-"))
        s2_s3_by_country[c]["s2_matches"] += s2_count
        s2_s3_by_country[c]["s3_matches"] += s3_count
        
        if has_s2 and has_s3:
            s2_s3_by_country[c]["both"] += 1
        elif has_s2:
            s2_s3_by_country[c]["only_s2"] += 1
        else:
            s2_s3_by_country[c]["only_s3"] += 1

print("Completed in", round(time.time() - t0, 2), "s")
print("US breakdown:", s2_s3_by_country["US"])
print("India breakdown:", s2_s3_by_country["India"])

for c in ["US", "India"]:
    tot_s1 = sum(gt_by_country[c].values())
    sing = gt_by_country[c][0]
    print(f"{c}: Total S1 = {tot_s1:,}, Singletons = {sing:,} ({sing/tot_s1*100:.2f}%), Matches = {s2_s3_by_country[c]['total_matches']:,}, Mean = {s2_s3_by_country[c]['total_matches']/tot_s1:.2f}")
