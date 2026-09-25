#!/usr/bin/env python3
"""
Raw Data Layer Manifest & Discovery
ML Challenge 2026 - Business Entity Resolution
"""

import os
import json
import hashlib

def generate_raw_manifest():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/clean", exist_ok=True)

    raw_files = [
        ("dataset/train/train_source1.tsv", "train_source1"),
        ("dataset/train/train_source2.tsv", "train_source2"),
        ("dataset/train/train_source3.tsv", "train_source3"),
        ("dataset/train/train_ground_truth.tsv", "train_ground_truth"),
        ("dataset/test/test_source1.tsv", "test_source1"),
        ("dataset/test/test_source2.tsv", "test_source2"),
        ("dataset/test/test_source3.tsv", "test_source3"),
    ]

    manifest = {"raw_layer_timestamp": "2026-09-25", "files": {}}

    for fpath, name in raw_files:
        if not os.path.exists(fpath):
            print(f"ERROR: {fpath} does not exist!")
            continue
        
        size = os.path.getsize(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            line_count = 1 + sum(1 for _ in f)

        manifest["files"][name] = {
            "source_path": fpath,
            "size_bytes": size,
            "total_lines": line_count,
            "record_count": line_count - 1,
            "columns": header,
            "status": "IMMUTABLE_ORIGINAL"
        }
        print(f"Registered {name}: {size:,} bytes | {line_count - 1:,} records | cols: {header}")

    with open("data/raw/raw_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme_content = """# Immutable Raw Data Layer

This directory represents the immutable raw data reference for the Amazon ML Challenge 2026 Business Entity Resolution project.

All source TSV files are hosted in `dataset/train/` and `dataset/test/` and are strictly read-only and immutable.

### Source Files Summary

| File Key | Source Filepath | Records | Columns | File Size |
|---|---|:---:|---|:---:|
"""
    for name, info in manifest["files"].items():
        readme_content += f"| `{name}` | `{info['source_path']}` | {info['record_count']:,} | `{', '.join(info['columns'])}` | {info['size_bytes'] / (1024*1024):.1f} MB |\n"

    readme_content += "\n**Rule**: No file in this layer may ever be modified, overwritten, or truncated.\n"

    with open("data/raw/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\nRaw data layer manifest and README created successfully.")

if __name__ == "__main__":
    generate_raw_manifest()
