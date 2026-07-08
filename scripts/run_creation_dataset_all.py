"""
Batch-build train/val/test datasets across several compression configurations.

Mirrors run_compression_all.py: each (method, parameter) combination is run as its
own subprocess against run_creation_dataset.py, so one bad combination doesn't abort
the rest of the sweep.

Edit `species` and `jobs` below, then run:

    python scripts/run_creation_dataset_all.py

The baseline (uncompressed) dataset is built first for each species, since every
compressed variant reuses the baseline's label (Y) pickle — see run_creation_dataset.py.
"""

import subprocess
import sys

species = "thyolo"

# method -> list of parameters. Use "none" for the uncompressed baseline (built
# automatically below; you don't need to list it here).
jobs = {
    "mp3":  ["32k", "56k", "96k"],
    "opus": ["8k", "48k", "112k"],
    "aac":  ["8k", "40k", "96k"],
    "ogg":  ["0", "4", "8"],
    "flac": ["0", "2", "8"],
    # "cs":   ["0.1", "0.15", "0.3"],
}

print(f"=== baseline (no compression) ===")
subprocess.run([
    sys.executable,
    "scripts/run_creation_dataset.py",
    "--species", species,
], check=False)

for method, params in jobs.items():
    for param in params:
        print(f"=== {method} @ {param} ===")
        result = subprocess.run([
            sys.executable,
            "scripts/run_creation_dataset.py",
            "--species", species,
            "--method-compression", method,
            "--parameter-compression", param,
        ])
        if result.returncode != 0:
            print(f"WARNING: {method} @ {param} failed (exit {result.returncode}) - continuing")
