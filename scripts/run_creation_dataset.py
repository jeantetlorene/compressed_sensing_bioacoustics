"""
Terminal entry point for train/val/test dataset creation (mirrors notebooks/creation_dataset.ipynb).

Reads audio (original, codec-compressed, or CS-reconstructed), builds mel-spectrogram
segments for train/val/test splits and pickles them to `{species_folder}/{dataset_type}/`.

Usage examples
--------------
# Baseline (uncompressed) dataset for thyolo:
python scripts/run_creation_dataset.py --species thyolo

# Dataset built from opus-compressed audio:
python scripts/run_creation_dataset.py --species thyolo --method-compression opus --parameter-compression 6k

# Dataset built from compressed-sensing reconstructions (expects
# {species_folder}/Compressed_Audio/cs_reconstructed_{parameter}/*.npy):
python scripts/run_creation_dataset.py --species bats --method-compression cs --parameter-compression 0.15

# Only rebuild the validation split:
python scripts/run_creation_dataset.py --species thyolo --dataset-types val

# Show all options:
python scripts/run_creation_dataset.py --help

Note: the Y (label) pickle is only ever written by the baseline (no compression) run —
labels come from segment timing, not audio content, so every compression variant of a
dataset_type reuses the baseline's `{positive_class}_Y_{dataset_type}.pkl`. Run the
baseline once before any compressed variant for a given species.

For a batch sweep across several codecs/parameters, see run_creation_dataset_all.py.
"""

import argparse
import json
import logging
import sys
import time
import pickle
from pathlib import Path

# Allow running from project root without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from preprocess import Preprocessing
from settings import Config
from config_species import get_settings


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
    tag = f"{method}_{parameter}" if method is not None else "baseline"
    log_file = log_dir / f"creation_dataset_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
        description="Build train/val/test mel-spectrogram datasets from (optionally compressed) audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--species",
        required=True,
        choices=sorted(SPECIES_FOLDER.keys()),
        help="Target species. Determines the default data folder and preprocessing/model settings.",
    )
    parser.add_argument(
        "--species-folder",
        default=None,
        help="Override the default data folder for this species.",
    )

    # Compression parameters (identify which audio variant to read)
    parser.add_argument(
        "--method-compression",
        default="none",
        choices=["none", "mp3", "aac", "ogg", "flac", "opus", "cs"],
        help="Audio variant to build the dataset from. 'none' uses the raw Audio/ folder; "
             "'cs' reads compressed-sensing reconstructions (.npy).",
    )
    parser.add_argument(
        "--parameter-compression",
        default=None,
        help="Codec parameter (bitrate for mp3/aac/opus, 0-12 for flac, 0-10 for ogg) "
             "or CS compression rate (e.g. 0.15). Required unless --method-compression none.",
    )
    parser.add_argument(
        "--audio-extension",
        default=None,
        help="Override the audio file extension used to locate raw files. Defaults to the "
             "species preset, except for --method-compression cs which defaults to '.npy'.",
    )

    # Dataset build options
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to (re)build.",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true", default=False,
        help="Disable data augmentation on the train split (augmentation is applied to "
             "train only, and only when this flag is absent).",
    )
    parser.add_argument(
        "--noise-reduction",
        action="store_true", default=False,
        help="Apply noise reduction while building segments.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true", default=False,
        help="Skip a dataset_type if its output pickle(s) already exist.",
    )

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
               dataset_types: list, elapsed: float, status: str) -> None:
    ledger_path = tracking_dir / "dataset_run_ledger.json"
    ledger = _load_ledger(ledger_path)

    ledger["runs"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method_compression": method,
        "parameter_compression": parameter,
        "dataset_types": dataset_types,
        "elapsed_seconds": round(elapsed, 2),
        "status": status,          # "completed" or "crashed"
    })
    ledger["total_seconds"] = round(
        sum(r["elapsed_seconds"] for r in ledger["runs"]), 2
    )

    _save_ledger(ledger_path, ledger)

    total_h = ledger["total_seconds"] / 3600
    log = logging.getLogger("run_creation_dataset")
    log.info(
        "Run ledger updated — this run: %.1f s | cumulative total: %.2f h (%d runs) | %s",
        elapsed, total_h, len(ledger["runs"]), ledger_path,
    )


# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------

def output_paths(species_folder: Path, dataset_type: str, positive_class: str,
                  method_compression: str, parameter_compression: str):
    saving_path = Path(species_folder, dataset_type)
    if method_compression is not None:
        x_path = saving_path / f"{positive_class}_X_{dataset_type}_{method_compression}_{parameter_compression}.pkl"
    else:
        x_path = saving_path / f"{positive_class}_X_{dataset_type}.pkl"
    y_path = saving_path / f"{positive_class}_Y_{dataset_type}.pkl"
    return saving_path, x_path, y_path


def build_dataset(preprocess: Preprocessing, dataset_type: str, positive_class: str,
                   method_compression: str, parameter_compression: str,
                   augment_train: bool, noise_reduction: bool, skip_existing: bool) -> None:
    log = logging.getLogger("run_creation_dataset")

    saving_path, x_path, y_path = output_paths(
        preprocess.species_folder, dataset_type, positive_class,
        method_compression, parameter_compression,
    )

    if skip_existing and x_path.exists() and (method_compression is not None or y_path.exists()):
        log.info("Skipping %s — %s already exists.", dataset_type, x_path)
        return

    data_augmentation = augment_train if dataset_type == "train" else False

    log.info("Building %s split (augmentation=%s, noise_reduction=%s)...",
              dataset_type, data_augmentation, noise_reduction)
    X_calls, Y_calls = preprocess.create_dataset(
        dataset_type,
        method_compression=method_compression,
        parameter_compression=parameter_compression,
        preprocessing=True,
        data_augmentation=data_augmentation,
        noise_reduction=noise_reduction,
    )
    Y = preprocess._one_hot_encode(Y_calls)

    saving_path.mkdir(parents=True, exist_ok=True)

    with open(x_path, "wb") as f:
        pickle.dump(X_calls, f)
    log.info("Saved X (%d segments) to %s", len(X_calls), x_path)

    if method_compression is None:
        with open(y_path, "wb") as f:
            pickle.dump(Y, f)
        log.info("Saved Y to %s", y_path)
    elif not y_path.exists():
        log.warning(
            "Y file %s does not exist yet — run with --method-compression none first "
            "for this species/dataset_type so labels are available.", y_path,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    method_compression = None if args.method_compression == "none" else args.method_compression
    parameter_compression = args.parameter_compression

    if method_compression is not None and parameter_compression is None:
        raise SystemExit("--parameter-compression is required unless --method-compression is 'none'.")

    species_folder = Path(args.species_folder or SPECIES_FOLDER[args.species])
    tracking_dir = species_folder / "tracking"

    log_file = setup_logging(tracking_dir, method_compression, parameter_compression, args.log_level)
    log = logging.getLogger("run_creation_dataset")
    log.info("Log file: %s", log_file)
    log.info("Species: %s | Species folder: %s", args.species, species_folder)
    log.info("Method: %s | Parameter: %s | Dataset types: %s",
              method_compression, parameter_compression, args.dataset_types)

    settings = get_settings(args.species)
    config = Config(settings)
    config.data.species_folder = species_folder

    audio_extension = args.audio_extension
    if audio_extension is None and method_compression == "cs":
        audio_extension = ".npy"
    if audio_extension is not None:
        config.preprocessing.audio_extension = audio_extension

    preprocess = Preprocessing(
        **config.preprocessing.dict(),
        species_folder=config.data.species_folder,
        positive_class=config.data.positive_class,
        negative_class=config.data.negative_class,
    )

    t0 = time.time()
    status = "crashed"
    try:
        for dataset_type in args.dataset_types:
            build_dataset(
                preprocess, dataset_type, config.data.positive_class,
                method_compression, parameter_compression,
                augment_train=not args.no_augmentation,
                noise_reduction=args.noise_reduction,
                skip_existing=args.skip_existing,
            )
        status = "completed"
    finally:
        elapsed = time.time() - t0
        log.info("Finished in %.2f seconds (status: %s).", elapsed, status)
        record_run(tracking_dir, method_compression, parameter_compression,
                   args.dataset_types, elapsed, status)


if __name__ == "__main__":
    main()
