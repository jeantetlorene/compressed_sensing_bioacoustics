"""
Terminal entry point for EnCodec-based audio compression / reconstruction
(mirrors notebooks/encodec_pipeline_v2.ipynb, ignoring its final dataset-creation cell).

Usage examples
--------------
# Compress Gibbon audio at bandwidth 6.0 (default block/window/batch settings):
python scripts/run_encodec.py compress --species gibbon

# Compress with custom block/window/batch settings:
python scripts/run_encodec.py compress --species gibbon --parameter-compression 6.0 \
    --block-duration-sec 300 --window-duration-sec 60 --batch-size 5

# Reconstruct previously compressed audio back to .wav:
python scripts/run_encodec.py reconstruct --species gibbon --parameter-compression 6.0

# Override species folder:
python scripts/run_encodec.py compress --species ptw --species-folder "D:/Data/Ptw"

# Show all options:
python scripts/run_encodec.py --help
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from encodec_compress_vs2 import EncodecCompression


# ---------------------------------------------------------------------------
# Per-species default data folders (override with --species-folder)
# ---------------------------------------------------------------------------

SPECIES_FOLDER = {
    "gibbon": "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Gibbon",
    "thyolo": "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Thyolo",
    "ptw":    "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Ptw",
    "bats":   "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Bats",
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, action: str, parameter: str, level: str = "INFO") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"encodec_{action}_{parameter}_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
        force=True,
    )
    return log_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress or reconstruct a folder of WAV files with EnCodec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "action",
        choices=["compress", "reconstruct"],
        help="Whether to run EncodecCompression.compress() or .reconstruct().",
    )

    # Paths
    parser.add_argument(
        "--species",
        required=True,
        choices=sorted(SPECIES_FOLDER.keys()),
        help="Target species. Determines the default data folder "
             "({species_folder}/Audio, /Compressed_Audio, /tracking).",
    )
    parser.add_argument(
        "--species-folder",
        default=None,
        help="Override the default data folder for this species.",
    )

    # EnCodec parameters
    parser.add_argument("--parameter-compression", default="24.0",
                        help="Target bandwidth (kbps) passed to EncodecModel.set_target_bandwidth.")
    parser.add_argument("--model-name", choices=["24khz", "48khz"], default="24khz",
                        help="Pretrained EnCodec model to use.")
    parser.add_argument("--block-duration-sec", type=float, default=300,
                        help="Length of each audio block processed at a time (seconds).")
    parser.add_argument("--window-duration-sec", type=float, default=60,
                        help="If set, splits each block into fixed-length windows and encodes "
                             "them in batches instead of encoding the whole block at once.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of windows encoded per batch (only used with --window-duration-sec).")

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


def record_run(tracking_dir: Path, action: str, parameter: str,
               elapsed: float, status: str) -> None:
    ledger_path = tracking_dir / "encodec_run_ledger.json"
    ledger = _load_ledger(ledger_path)

    ledger["runs"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "parameter_compression": parameter,
        "elapsed_seconds": round(elapsed, 2),
        "status": status,          # "completed" or "crashed"
    })
    ledger["total_seconds"] = round(
        sum(r["elapsed_seconds"] for r in ledger["runs"]), 2
    )

    _save_ledger(ledger_path, ledger)

    total_h = ledger["total_seconds"] / 3600
    log = logging.getLogger("run_encodec")
    log.info(
        "Run ledger updated — this run: %.1f s | cumulative total: %.2f h (%d runs) | %s",
        elapsed, total_h, len(ledger["runs"]), ledger_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    species_folder = Path(args.species_folder or SPECIES_FOLDER[args.species])
    folder_audio = species_folder / "Audio"
    folder_compress = species_folder / "Compressed_Audio"
    tracking_dir = species_folder / "tracking"

    log_file = setup_logging(tracking_dir, args.action, args.parameter_compression, args.log_level)

    log = logging.getLogger("run_encodec")
    log.info("Log file: %s", log_file)
    log.info("Species: %s | Species folder: %s", args.species, species_folder)
    log.info("Action: %s | Bandwidth: %s | Model: %s | Block: %ss | Window: %s | Batch size: %s",
              args.action, args.parameter_compression, args.model_name,
              args.block_duration_sec, args.window_duration_sec, args.batch_size)

    compression = EncodecCompression(
        str(folder_audio),
        str(folder_compress),
        parameter_compression=args.parameter_compression,
        model_name=args.model_name,
        block_duration_sec=args.block_duration_sec,
        window_duration_sec=args.window_duration_sec,
        batch_size=args.batch_size,
    )

    t0 = time.time()
    status = "crashed"
    try:
        if args.action == "compress":
            compression.compress()
        else:
            compression.reconstruct()
        status = "completed"
    finally:
        elapsed = time.time() - t0
        log.info("Finished in %.2f seconds (status: %s).", elapsed, status)
        record_run(tracking_dir, args.action, args.parameter_compression, elapsed, status)


if __name__ == "__main__":
    main()
