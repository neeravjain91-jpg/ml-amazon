#!/usr/bin/env python3
"""
Canonical Data Layer Builder
ML Challenge 2026 - Business Entity Resolution
Converts Raw Sources into Optimized Columnar Parquet Tables in data/clean/
"""

import os
import sys
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append('src')
sys.path.append('.')
import canonical_cleaning

def convert_tsv_to_canonical_parquet(src_path, out_path, chunk_size=100000, max_rows=None):
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Converting {src_path} -> {out_path}...", flush=True)

    writer = None
    total_written = 0

    with open(src_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        chunk_rows = []
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) != 4:
                continue
            rec = canonical_cleaning.canonicalize_record(p[0], p[1], p[2], p[3])
            chunk_rows.append(rec)
            total_written += 1

            if len(chunk_rows) >= chunk_size:
                df = pd.DataFrame(chunk_rows)
                table = pa.Table.from_pandas(df)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                writer.write_table(table)
                chunk_rows = []
                print(f"  Written {total_written:,} rows...", flush=True)

            if max_rows and total_written >= max_rows:
                break

        if chunk_rows:
            df = pd.DataFrame(chunk_rows)
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
            writer.write_table(table)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    raw_size = os.path.getsize(src_path)
    clean_size = os.path.getsize(out_path)
    print(f"  Complete: {total_written:,} rows in {elapsed:.1f}s | Raw: {raw_size/(1024*1024):.1f} MB -> Parquet: {clean_size/(1024*1024):.1f} MB ({(clean_size/raw_size)*100:.1f}%)", flush=True)

    return {
        "source": src_path,
        "target": out_path,
        "rows": total_written,
        "raw_size_mb": raw_size / (1024*1024),
        "clean_size_mb": clean_size / (1024*1024),
        "elapsed_s": elapsed,
    }

def main():
    print("=" * 70)
    print("BUILDING CANONICAL PARQUET DATASETS (data/clean/)")
    print("=" * 70)

    # Convert S1 reference sources (full)
    s1_files = [
        ("dataset/train/train_source1.tsv", "data/clean/train_source1.parquet"),
        ("dataset/test/test_source1.tsv", "data/clean/test_source1.parquet"),
    ]

    metrics = []
    for src, out in s1_files:
        res = convert_tsv_to_canonical_parquet(src, out)
        metrics.append(res)

    print("\nCanonical S1 Reference Datasets Built Successfully.")

if __name__ == "__main__":
    main()
