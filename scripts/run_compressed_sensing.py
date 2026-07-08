"""
Terminal entry point for compressed-sensing compression / reconstruction.

Usage examples
--------------
# Compress (default settings for bats):
python scripts/run_compression.py --mode compression

# Reconstruct with IHT:
python scripts/run_compression.py --mode reconstruction

# Override paths and rate:
python scripts/run_compression.py --mode compression \
    --folder-audio "C:/path/to/Audio" \
    --folder-saved "C:/path/to/Compressed_Audio" \
    --compression-rate 0.15

# Show all options:
python scripts/run_compression.py --help
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from compress import CS


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, mode: str, level: str = "INFO") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CS compression or reconstruction on a folder of WAV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--folder-audio",
        default="C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Bats/Audio",
        help="Folder containing raw .wav files.",
    )
    parser.add_argument(
        "--folder-saved",
        default="C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Bats/Compressed_Audio",
        help="Root output folder (sub-folders are created automatically).",
    )
    parser.add_argument(
        "--folder-tracking",
        default="C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Bats/tracking",
        help="Folder where execution-time logs are written.",
    )

    # CS parameters
    parser.add_argument("--sample-rate", type=int, default=256000,
                        help="Recording sample rate in Hz.")
    parser.add_argument("--frame-size", type=int, default=1024,
                        help="Frame length in samples.")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Fractional frame overlap [0, 1).")
    parser.add_argument("--compression-rate", type=float, default=0.1,
                        help="Fraction of measurements to keep (M = rate * N).")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() - 1),
                        help="Parallel workers (-1 = all cores).")

    # Mode
    parser.add_argument(
        "--mode",
        choices=["compression", "reconstruction"],
        default="reconstruction",
        help="Whether to compress or reconstruct.",
    )


    # Reconstruction-only options
    parser.add_argument("--alpha", type=float, default=1e-7,
                        help="Regularisation / tolerance parameter for reconstruction.")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum solver iterations (reconstruction only).")
    parser.add_argument("--solver", choices=["iht", "lasso", "omp"], default="lasso",
                        help="Reconstruction solver (used only with --mode reconstruction).")
    parser.add_argument("--save-wav", action="store_true", default=False,
                        help="Save reconstructed signal as .wav instead of .npy.")

    # Logging
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console/file log verbosity.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cumulative run ledger
# ---------------------------------------------------------------------------

def _load_ledger(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"runs": [], "total_seconds": 0.0}


def _save_ledger(path: Path, ledger: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)


def record_run(tracking_dir: Path, mode: str, compression_rate: float,
               elapsed: float, status: str) -> None:
    ledger_path = tracking_dir / "run_ledger.json"
    ledger = _load_ledger(ledger_path)

    ledger["runs"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "compression_rate": compression_rate,
        "elapsed_seconds": round(elapsed, 2),
        "status": status,          # "completed" or "crashed"
    })
    ledger["total_seconds"] = round(
        sum(r["elapsed_seconds"] for r in ledger["runs"]), 2
    )

    _save_ledger(ledger_path, ledger)

    total_h = ledger["total_seconds"] / 3600
    log = logging.getLogger("run_compression")
    log.info(
        "Run ledger updated — this run: %.1f s | cumulative total: %.2f h (%d runs) | %s",
        elapsed, total_h, len(ledger["runs"]), ledger_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    tracking_dir = Path(args.folder_tracking)
    log_file = setup_logging(tracking_dir, args.mode, args.log_level)

    log = logging.getLogger("run_compression")
    log.info("Log file: %s", log_file)
    log.info("Mode: %s", args.mode)
    log.info(
        "Parameters — sample_rate=%d  frame_size=%d  overlap=%.2f  "
        "compression_rate=%.2f  n_jobs=%d",
        args.sample_rate, args.frame_size, args.overlap,
        args.compression_rate, args.n_jobs,
    )

    cs = CS(
        folder_audio=args.folder_audio,
        folder_saved=args.folder_saved,
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        overlap=args.overlap,
        compression_rate=args.compression_rate,
        n_jobs=args.n_jobs,
    )

    t0 = time.time()
    status = "crashed"
    try:
        if args.mode == "compression":
            cs.compress_folder_legacy()
        else:
            cs.reconstruction_legacy(solver=args.solver, alpha=args.alpha,
                                     saved_in_wav=args.save_wav)
        status = "completed"
    finally:
        elapsed = time.time() - t0
        log.info("Finished in %.2f seconds (status: %s).", elapsed, status)
        record_run(tracking_dir, args.mode, args.compression_rate, elapsed, status)

        save_path = tracking_dir / f"time_execution_{args.mode}_{args.compression_rate}.txt"
        with open(save_path, "w") as f:
            f.write(f"time execution: {elapsed}\n")


if __name__ == "__main__":
    main()
