#!/usr/bin/env python3
"""
Full Canonical Dataset Builder
ML Challenge 2026 - Business Entity Resolution
Converts all remaining raw sources into data/clean/*.parquet
"""

import os
import sys
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.append('src')
sys.path.append('.')
import canonical_cleaning

def convert_single_file(args):
    src_path, out_path, chunk_size = args
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100 * 1024 * 1024:
        print(f"[SKIP] {out_path} already exists ({os.path.getsize(out_path)/(1024*1024):.1f} MB)", flush=True)
        return out_path, os.path.getsize(out_path), 0.0

    t0 = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[START] {src_path} -> {out_path}...", flush=True)

    writer = None
    total_written = 0
    chunk_rows = []

    with open(src_path, "r", encoding="utf-8") as f:
        header = f.readline()
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
                if total_written % 1000000 == 0:
                    print(f"  [{os.path.basename(out_path)}] Written {total_written:,} rows...", flush=True)

        if chunk_rows:
            df = pd.DataFrame(chunk_rows)
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
            writer.write_table(table)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[DONE] {out_path}: {total_written:,} rows in {elapsed:.1f}s ({size_mb:.1f} MB)", flush=True)
    return out_path, os.path.getsize(out_path), elapsed

def main():
    tasks = [
        ("dataset/train/train_source2.tsv", "data/clean/train_source2.parquet", 100000),
        ("dataset/train/train_source3.tsv", "data/clean/train_source3.parquet", 100000),
        ("dataset/test/test_source2.tsv", "data/clean/test_source2.parquet", 100000),
        ("dataset/test/test_source3.tsv", "data/clean/test_source3.parquet", 100000),
    ]

    print("=" * 70, flush=True)
    print("PARALLEL CANONICAL PARQUET DATASET BUILDER (4 WORKERS)", flush=True)
    print("=" * 70, flush=True)

    with Pool(processes=4) as pool:
        results = pool.map(convert_single_file, tasks)

    print("\nAll canonical datasets built successfully:", flush=True)
    for path, sz, el in results:
        print(f"  {path}: {sz/(1024*1024):.1f} MB in {el:.1f}s")

if __name__ == "__main__":
    main()
