"""
prepare_data.py
---------------
Create raw-audio pickle files for EnCodec fine-tuning from existing
.wav + .svl annotation data.

The main preprocessing pipeline saves mel-spectrograms; EnCodec needs raw
waveforms. This script applies the same lowpass filter + downsampling but
stops before the spectrogram conversion, saving:

  <species>_X_audio.pkl : (N, T) float32 — waveform segments at the original
                                            recording sample rate, normalised to [-1, 1]
  <species>_Y_audio.pkl : (N,)   int64   — 1 = positive class (call), 0 = background

Usage (run from the project root):

  # Combine all splits into one file (recommended for EnCodec fine-tuning):
  python encodec/prepare_data.py --species gibbon --split all

  # Or a single split:
  python encodec/prepare_data.py --species gibbon --split train

After running, update NATIVE_SR in encodec/dataset.py to match the
original sample_rate of your species (9600 for gibbon).
"""

from __future__ import annotations

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from src/ regardless of where the script is called from
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocess import Preprocessing
from AnnotationReader import AnnotationReader
from config_species import get_settings


DATA_ROOT = Path("C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data")

# Annotation file extension (Sonic Visualiser XML)
ANNOTATION_EXT = ".svl"


# ---------------------------------------------------------------------------
# Core extraction (same as Preprocessing.create_dataset but no spectrogram)
# ---------------------------------------------------------------------------

def extract_audio_segments(
    prep:  Preprocessing,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract raw waveform segments for one data split.

    Parameters
    ----------
    prep  : configured Preprocessing instance
    split : "train", "test", or "validation"

    Returns
    -------
    X : (N, T) float32 — normalised waveform segments
    Y : (N,)   int64  — 1 = positive class, 0 = negative class
    """
    # Accept both "validation.txt" and "val.txt"
    for candidate in (split, split.replace("validation", "val")):
        files_path = Path(prep.species_folder, "DataFiles", f"{candidate}.txt")
        if files_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"Split file not found for '{split}' in "
            f"{Path(prep.species_folder, 'DataFiles')} "
            f"(tried {split}.txt and val.txt)"
        )

    files = pd.read_csv(files_path, header=None)
    X_all: list[np.ndarray] = []
    Y_all: list[str] = []

    for row in files.values:
        file_name = row[0]
        print(f"  {file_name}")

        audio_amps, original_sr = prep.read_audio_file(file_name, None, None)
        amplitudes, sample_rate = audio_amps, original_sr
        del audio_amps

        reader = AnnotationReader(
            prep.species_folder, file_name,
            ANNOTATION_EXT, prep.audio_extension,
            prep.positive_class,
        )
        df, _ = reader.get_annotation_information()

        for _, ann in df.iterrows():
            segs, labels = prep.getXY(
                amplitudes, sample_rate,
                ann["Start"],
                ann["End"] - ann["Start"],
                ann["Label"],
                file_name,
                verbose=False,
            )
            X_all.extend(segs)
            Y_all.extend(labels)

    # Stack into arrays
    X = np.stack(X_all, axis=0).astype(np.float32)

    # Normalise each window to [-1, 1]
    peak = np.abs(X).max(axis=1, keepdims=True)
    peak = np.where(peak < 1e-8, 1.0, peak)
    X = X / peak

    # String labels → binary integers
    Y = np.array(
        [1 if y == prep.positive_class else 0 for y in Y_all],
        dtype=np.int64,
    )

    n_calls = int((Y == 1).sum())
    print(f"  -> {len(X):,} windows  ({n_calls:,} calls, "
          f"{len(X)-n_calls:,} background, "
          f"{100*n_calls/max(len(X),1):.1f}% positive)")
    return X, Y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build raw-audio pickle files for EnCodec fine-tuning"
    )
    parser.add_argument(
        "--species", default="gibbon",
        help="Species key as defined in src/config_species.py"
    )
    parser.add_argument(
        "--split", default="all",
        choices=["train", "test", "validation", "all"],
        help="Which split to process. 'all' combines train+test+validation."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to Data/<Species>/"
    )
    args = parser.parse_args()

    cfg = get_settings(args.species)
    p   = cfg["preprocessing"]

    # Override the hard-coded E:/ path in config_species with the real path
    species_folder = DATA_ROOT / args.species.capitalize()
    if not species_folder.exists():
        raise FileNotFoundError(f"Species folder not found: {species_folder}")

    prep = Preprocessing(
        species_folder      = species_folder,
        sample_rate         = p["sample_rate"],
        lowpass_cutoff      = p["lowpass_cutoff"],
        downsample_rate     = p["downsample_rate"],
        nyquist_rate        = p["nyquist_rate"],
        segment_duration    = p["segment_duration"],
        positive_class      = cfg["data"]["positive_class"],
        negative_class      = cfg["data"]["negative_class"],
        nb_negative_class   = p["nb_negative_class"],
        n_fft               = p["n_fft"],
        hop_length          = p["hop_length"],
        n_mels              = p["n_mels"],
        f_min               = p["f_min"],
        f_max               = p["f_max"],
        annotation_extension= ANNOTATION_EXT,
        audio_extension     = p["audio_extension"],
    )

    out_dir = Path(args.output_dir) if args.output_dir else species_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "validation"] if args.split == "all" else [args.split]

    X_parts, Y_parts = [], []

    for split in splits:
        print(f"\n--- {split} ---")
        X, Y = extract_audio_segments(prep, split)
        X_parts.append(X)
        Y_parts.append(Y)

    X_final = np.concatenate(X_parts, axis=0)
    Y_final = np.concatenate(Y_parts, axis=0)

    # File names reflect whether this is a combined or single-split file
    suffix   = "audio" if args.split == "all" else f"{args.split}_audio"
    x_out    = out_dir / f"{args.species}_X_{suffix}.pkl"
    y_out    = out_dir / f"{args.species}_Y_{suffix}.pkl"

    with open(x_out, "wb") as f:
        pickle.dump(X_final, f, protocol=4)
    with open(y_out, "wb") as f:
        pickle.dump(Y_final, f, protocol=4)

    n_calls = int((Y_final == 1).sum())
    print(f"\nSaved X -> {x_out}")
    print(f"       shape={X_final.shape}  dtype={X_final.dtype}")
    print(f"Saved Y -> {y_out}")
    print(f"       shape={Y_final.shape}  dtype={Y_final.dtype}")
    print(f"  {n_calls:,} calls / {len(X_final):,} total "
          f"({100*n_calls/max(len(X_final),1):.1f}% positive)")

    print(f"\n--- Next step ---")
    print(f"Update NATIVE_SR in encodec/dataset.py:")
    print(f"  NATIVE_SR = {p['sample_rate']}  # {args.species} original sample rate")
    print(f"\nThen run:")
    print(f"  python encodec/train.py --x \"{x_out}\" --y \"{y_out}\"")


if __name__ == "__main__":
    main()
