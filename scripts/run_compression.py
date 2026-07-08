"""
Terminal entry point for codec-based audio compression (mirrors notebooks/compression.ipynb).

By default only the execution time is recorded. Pass --monitor to additionally
sample CPU and memory usage of the process while compression runs.

Usage examples
--------------
# Compress with default (opus, 40k) settings for Gibbon:
python scripts/run_compression.py --species gibbon

# Choose a different codec/bitrate:
python scripts/run_compression.py --species thyolo --method-compression mp3 --parameter-compression 96k

# Enable CPU/memory monitoring:
python scripts/run_compression.py --species gibbon --monitor

# Override species folder:
python scripts/run_compression.py --species ptw --species-folder "D:/Data/Ptw"

# Show all options:
python scripts/run_compression.py --help

for a loop on a powershell : 

$jobs = @{
    "mp3"  = @("8k", "32k", "64k")
    "opus" = @("8k", "40k", "112k")
    "aac"  = @("8k", "40k", "96k")
    "ogg"  = @("0", "5", "10" )
    "flac" = @("0", "6", "12")
}

foreach ($method in $jobs.Keys) {
    foreach ($param in $jobs[$method]) {
        Write-Host "=== $method @ $param ==="
        python scripts/run_compression.py --species gibbon --parameter-compression $param
    }
}

Each combination runs as its own process, gets its own times and appends a row to the shared run_ledger.json — so you end up with one file tracking every run's elapsed time across all codecs/parameters. If a run fails partway through, the loop continues to the next
combination (it doesn't exit on error) since run_compression in its finally block and records status: "crashed" beforere-raising... actually let me check that it does re-raise, otherwise the loop wouldn't know.

  Read 1 file

Good — a bare try/finally re-raises after the finally block runs, so a crash is logged to the ledger and the process exits non-zero. If you want the loop to keep going even when one combination fails, add -Errant here (that's for cmdlets) — instead wrap the call:

foreach ($method in $jobs.Keys) {
    foreach ($param in $jobs[$method]) {
        Write-Host "=== $method @ $param ==="
        & python scripts/run_compression.py --species gibbon --method-compression $method --parameter-compression $param
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$method @ $param failed (exit $LASTEXITCODE) - continuing"
        }
    }
}

This way one bad codec/bitrate combo won't abort the whole brash recorded in run_ledger.json for that run.

"""

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Allow running from project root without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from compress import Compression


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

def setup_logging(log_dir: Path, method: str, parameter: str, level: str = "INFO") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"compression_{method}_{parameter}_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
# Resource monitoring (optional)
# ---------------------------------------------------------------------------

def monitor_resources(stop_event, cpu_usage, mem_usage, sampling_interval=0.1):
    """Sample CPU (%) and memory (MB) of the current process at regular intervals."""
    import psutil

    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_usage.append(process.cpu_percent(interval=None))
        mem_usage.append(process.memory_info().rss / (1024 ** 2))
        time.sleep(sampling_interval)


def save_resource_usage(tracking_dir: Path, method: str, parameter: str, init_time: str,
                        cpu_usage: list, mem_usage: list, sampling_interval: float) -> None:
    import pandas as pd

    log = logging.getLogger("run_compression")
    if not mem_usage:
        log.warning("No resource usage data collected.")
        return

    avg_cpu = sum(cpu_usage) / len(cpu_usage)
    log.info("Average CPU usage: %.2f%%", avg_cpu)
    log.info("Memory usage — min: %.2f MB, max: %.2f MB, avg: %.2f MB",
              min(mem_usage), max(mem_usage), sum(mem_usage) / len(mem_usage))

    start_time = datetime.datetime.strptime(init_time, "%Y-%m-%d %H:%M:%S.%f")
    timestamps = [start_time + datetime.timedelta(seconds=i * sampling_interval)
                  for i in range(len(mem_usage))]
    usage_df = pd.DataFrame({
        "Timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for ts in timestamps],
        "Memory usage": mem_usage,
        "CPU usage": cpu_usage,
    })

    save_path = tracking_dir / f"Resource_usage_{method}_{parameter}.csv"
    usage_df.to_csv(save_path, index=False)
    log.info("Resource usage saved to %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run codec-based compression on a folder of WAV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument(
        "--converter-path",
        default="c:/Users/loren/Documents/Postdoc/Compressed_sensing/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe",
        help="Path to the ffmpeg executable used by pydub.",
    )

    # Compression parameters
    parser.add_argument("--method-compression", choices=["mp3", "aac", "ogg", "flac", "opus"],
                        default="opus", help="Codec used to compress the audio.")
    parser.add_argument("--parameter-compression", default="40k",
                        help="Codec parameter (bitrate for mp3/aac/opus, "
                             "0-12 for flac, 0-10 for ogg).")

    # Resource monitoring
    parser.add_argument("--monitor", action="store_true", default=False,
                        help="Track CPU/memory usage during compression (adds psutil sampling overhead).")
    parser.add_argument("--sampling-interval", type=float, default=0.1,
                        help="Sampling interval in seconds when --monitor is set.")

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


def record_run(tracking_dir: Path, method: str, parameter: str,
               elapsed: float, status: str) -> None:
    ledger_path = tracking_dir / "run_ledger.json"
    ledger = _load_ledger(ledger_path)

    ledger["runs"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method_compression": method,
        "parameter_compression": parameter,
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

    species_folder = Path(args.species_folder or SPECIES_FOLDER[args.species])
    folder_audio = species_folder / "Audio"
    folder_compress = species_folder / "Compressed_Audio"
    tracking_dir = species_folder / "tracking"

    log_file = setup_logging(tracking_dir, args.method_compression, args.parameter_compression, args.log_level)

    log = logging.getLogger("run_compression")
    log.info("Log file: %s", log_file)
    log.info("Species: %s | Species folder: %s", args.species, species_folder)
    log.info("Method: %s | Parameter: %s | Monitor: %s",
              args.method_compression, args.parameter_compression, args.monitor)

    if args.monitor:
        try:
            import psutil  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "psutil is required for --monitor. Install it with `pip install psutil`."
            ) from exc

    compression = Compression(
        str(folder_audio),
        str(folder_compress),
        args.method_compression,
        args.parameter_compression,
        args.converter_path,
    )

    cpu_usage, mem_usage = [], []
    stop_event = threading.Event()
    monitor_thread = None
    init_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]

    if args.monitor:
        monitor_thread = threading.Thread(
            target=monitor_resources,
            args=(stop_event, cpu_usage, mem_usage, args.sampling_interval),
            daemon=True,
        )
        monitor_thread.start()

    t0 = time.time()
    status = "crashed"
    try:
        compression.compress()
        status = "completed"
    finally:
        elapsed = time.time() - t0

        if monitor_thread is not None:
            stop_event.set()
            monitor_thread.join()

        log.info("Finished in %.2f seconds (status: %s).", elapsed, status)
        record_run(tracking_dir, args.method_compression, args.parameter_compression, elapsed, status)

        tracking_dir.mkdir(parents=True, exist_ok=True)
        save_path = tracking_dir / f"time_execution_{args.method_compression}_{args.parameter_compression}.txt"
        with open(save_path, "w") as f:
            f.write(f"time execution: {elapsed}\n")

        if args.monitor:
            save_resource_usage(tracking_dir, args.method_compression, args.parameter_compression,
                                init_time, cpu_usage, mem_usage, args.sampling_interval)


if __name__ == "__main__":
    main()
