#!/usr/bin/env python3
"""
Convert Canonical Parquet Datasets to TSV Format
ML Challenge 2026 - Business Entity Resolution
Converts all data/clean/*.parquet into data/clean/*.tsv
Streaming execution with strictly bounded memory (< 100MB RAM)
"""

import os
import sys
import time
import pyarrow.parquet as pq
import pyarrow.csv as pcsv

DATASETS = [
    ("data/clean/train_source1.parquet", "data/clean/train_source1.tsv"),
    ("data/clean/train_source2.parquet", "data/clean/train_source2.tsv"),
    ("data/clean/train_source3.parquet", "data/clean/train_source3.tsv"),
    ("data/clean/test_source1.parquet", "data/clean/test_source1.tsv"),
    ("data/clean/test_source2.parquet", "data/clean/test_source2.tsv"),
    ("data/clean/test_source3.parquet", "data/clean/test_source3.tsv"),
]

def convert_parquet_to_tsv(src_parquet, dst_tsv, batch_size=100000):
    t0 = time.time()
    print(f"Converting {src_parquet} -> {dst_tsv}...", flush=True)

    if not os.path.exists(src_parquet):
        raise FileNotFoundError(f"Source parquet file not found: {src_parquet}")

    pf = pq.ParquetFile(src_parquet)
    num_rows = pf.metadata.num_rows
    col_names = pf.schema_arrow.names

    opts = pcsv.WriteOptions(
        delimiter='\t',
        quoting_style='needed',
        include_header=False
    )

    rows_written = 0
    with open(dst_tsv, 'wb') as f:
        # Write clean tab-separated header
        header_line = ('\t'.join(col_names) + '\n').encode('utf-8')
        f.write(header_line)

        # Stream batches to preserve memory
        for batch in pf.iter_batches(batch_size=batch_size):
            pcsv.write_csv(batch, f, write_options=opts)
            rows_written += batch.num_rows

    elapsed = time.time() - t0
    tsv_size_mb = os.path.getsize(dst_tsv) / (1024 * 1024)
    print(f"  Complete: {rows_written:,} rows in {elapsed:.2f}s | Size: {tsv_size_mb:.1f} MB", flush=True)
    return {
        "parquet": src_parquet,
        "tsv": dst_tsv,
        "rows": rows_written,
        "size_mb": tsv_size_mb,
        "elapsed_s": elapsed,
        "columns": len(col_names),
    }

def main():
    print("=" * 70, flush=True)
    print("CONVERTING CANONICAL DATASETS TO TSV FORMAT (data/clean/*.tsv)")
    print("=" * 70, flush=True)

    t_total_start = time.time()
    results = []

    for src_parquet, dst_tsv in DATASETS:
        res = convert_parquet_to_tsv(src_parquet, dst_tsv)
        results.append(res)

    t_total = time.time() - t_total_start
    total_size_mb = sum(r["size_mb"] for r in results)
    total_rows = sum(r["rows"] for r in results)

    print("\n" + "=" * 70, flush=True)
    print(f"ALL 6 CANONICAL TSVs CREATED SUCCESSFULLY IN {t_total:.1f}s")
    print(f"Total Rows: {total_rows:,} | Total Size: {total_size_mb:,.1f} MB ({total_size_mb/1024:.2f} GB)")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
