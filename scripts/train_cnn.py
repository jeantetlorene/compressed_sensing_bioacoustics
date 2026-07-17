"""
Train a CNN classifier N times for gibbon, thyolo, or ptw, and save per-run losses and metrics.

Usage examples
--------------
# Thyolo, CS compression rate 0.2 (default):
python scripts/train_cnn.py --species thyolo

# Gibbon, baseline (no compression):
python scripts/train_cnn.py --species gibbon --method-compression baseline

# PTW, custom compression and 5 runs:
python scripts/train_cnn.py --species ptw --method-compression cs --parameter-compression 0.15 --n-runs 5

# Override species folder:
python scripts/train_cnn.py --species thyolo --species-folder "D:/Data/Thyolo"

# Show all options:
python scripts/train_cnn.py --help
"""

import argparse
import gc
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for terminal use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config_species import get_settings
from evaluation import Evaluation
from model import Model
from settings import Config


# ---------------------------------------------------------------------------
# Per-species architecture and training hyperparameters (from notebook)
# ---------------------------------------------------------------------------

SPECIES_ARCH = {
    "thyolo": dict(
        conv_layers=2, fc_layers=2, conv_kernel=8, conv_filters=16,
        dropout_rate=0.3, fc_units=64, max_pooling_size=2,
        learning_rate=0.0001, num_epochs=50, batch_size=64,
    ),
    "ptw": dict(
        conv_layers=1, fc_layers=2, conv_kernel=8, conv_filters=16,
        dropout_rate=0.3, fc_units=64, max_pooling_size=4,
        learning_rate=0.001, num_epochs=50, batch_size=64,
    ),
    "gibbon": dict(
        conv_layers=1, fc_layers=2, conv_kernel=8, conv_filters=8,
        dropout_rate=0.5, fc_units=32, max_pooling_size=4,
        learning_rate=0.0001, num_epochs=50, batch_size=128,
    ),
}

SPECIES_FOLDER = {
    "thyolo": "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Thyolo",
    "ptw":    "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Ptw",
    "gibbon": "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Gibbon",
}

SPECIES_EVAL = {
    "thyolo": dict(overlap=0.10, nb_to_group=0, threshold=0.8, step_size=1),
    "ptw":    dict(overlap=0.25, nb_to_group=2, threshold=0.8, step_size=1),
    "gibbon": dict(overlap=0.25, nb_to_group=2, threshold=0.8, step_size=1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, species: str, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{species}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.info("Log file: %s", log_file)


def load_dataset(species_folder, species, dataset_type="train",
                 method_compression=None, parameter_compression=None):
    if method_compression is not None:
        x_path = Path(species_folder, dataset_type,
                      f"{species}_X_{dataset_type}_{method_compression}_{parameter_compression}.pkl")
    else:
        x_path = Path(species_folder, dataset_type, f"{species}_X_{dataset_type}.pkl")
    y_path = Path(species_folder, dataset_type, f"{species}_Y_{dataset_type}.pkl")

    with open(x_path, "rb") as f:
        X = pickle.load(f)
    with open(y_path, "rb") as f:
        Y = pickle.load(f)

    logging.info("Loaded %s set (%d samples) from %s", dataset_type, len(X), x_path)
    return X, Y



def save_loss_plot(train_losses, val_losses, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_loss_plot(all_train, all_val, out_path: Path, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i, (tl, vl) in enumerate(zip(all_train, all_val)):
        axes[0].plot(tl, alpha=0.7, label=f"Run {i}")
        axes[1].plot(vl, alpha=0.7, label=f"Run {i}")
    for ax, title in zip(axes, ["Training Loss - All Runs", "Validation Loss - All Runs"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN 10 times for gibbon / thyolo / ptw, saving losses and F1 scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--species",
        required=True,
        choices=["gibbon", "thyolo", "ptw"],
        help="Target species.",
    )
    parser.add_argument(
        "--species-folder",
        default=None,
        help="Override the default data folder for this species.",
    )
    parser.add_argument(
        "--method-compression",
        default="cs",
        help="Compression method (e.g. cs, mp3, aac). Use 'baseline' for no compression.",
    )
    parser.add_argument(
        "--parameter-compression",
        default="0.2",
        help="Compression parameter (e.g. CS rate, codec bitrate).",
    )
    parser.add_argument("--n-runs", type=int, default=10, help="Number of training runs.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    species = args.species

    # Compression config
    is_baseline = args.method_compression.lower() == "baseline"
    method_compression = None if is_baseline else args.method_compression
    parameter_compression = None if is_baseline else args.parameter_compression
    tag = f"{method_compression}_{parameter_compression}" if not is_baseline else "baseline"

    # Species config
    settings = get_settings(species)
    config = Config(settings)
    config.data.species_folder = args.species_folder or SPECIES_FOLDER[species]
    if method_compression == "cs":
        config.preprocessing.audio_extension = ".npy"
    elif method_compression in ("mp3", "aac", "opus", "ogg", "flac"):
        config.preprocessing.audio_extension = f".{method_compression}"
    else:
        config.preprocessing.audio_extension = ".wav"

    # Architecture and training hyperparameters (override defaults with notebook-tuned values)
    arch = SPECIES_ARCH[species]
    config.cnn_architecture.conv_layers    = arch["conv_layers"]
    config.cnn_architecture.fc_layers      = arch["fc_layers"]
    config.cnn_architecture.conv_kernel    = arch["conv_kernel"]
    config.cnn_architecture.conv_filters   = arch["conv_filters"]
    config.cnn_architecture.dropout_rate   = arch["dropout_rate"]
    config.cnn_architecture.fc_units       = arch["fc_units"]
    config.cnn_architecture.max_pooling_size = arch["max_pooling_size"]
    config.model.learning_rate = arch["learning_rate"]
    config.model.num_epochs    = arch["num_epochs"]
    config.model.batch_size    = arch["batch_size"]

    # Output folder
    save_path = Path(config.data.species_folder, "results",
                     f"{config.data.positive_class}_{tag}")
    save_path.mkdir(parents=True, exist_ok=True)

    setup_logging(save_path, species, args.log_level)
    logging.info("Species        : %s", species)
    logging.info("Species folder : %s", config.data.species_folder)
    logging.info("Compression    : %s", tag)
    logging.info("Runs           : %d", args.n_runs)
    logging.info("Epochs         : %d  Batch: %d  LR: %g",
                 arch["num_epochs"], arch["batch_size"], arch["learning_rate"])

    # Load datasets once
    X_train, Y_train = load_dataset(
        config.data.species_folder, config.data.positive_class,
        method_compression=method_compression, parameter_compression=parameter_compression)
    X_val, Y_val = load_dataset(
        config.data.species_folder, config.data.positive_class, dataset_type="val",
        method_compression=method_compression, parameter_compression=parameter_compression)

    all_train_losses, all_val_losses = [], []
    f1_scores, f1_scores_full, precision_scores, recall_scores = [], [], [], []
    model_filename = f"{tag}_cnn_state.pth"

    eval_params = SPECIES_EVAL[species]
    evaluation = Evaluation(
        species_folder=config.data.species_folder,
        settings=config,
        overlap=eval_params["overlap"],
        nb_to_group=eval_params["nb_to_group"],
        threshold=eval_params["threshold"],
        step_size=eval_params["step_size"],
        method_compression=method_compression,
        parameter_compression=parameter_compression,
        force_calc_amplitudes=False,
        audio_extension=config.preprocessing.audio_extension,
    )

    for i in range(args.n_runs):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("=== Run %d / %d ===", i + 1, args.n_runs)

        # --- Training ---
        model = Model(
            save_path,
            input_shape=(1, X_train.shape[1], X_train.shape[2]),
            architecture_args=config.cnn_architecture.dict(),
            **config.model.dict(),
        )
        train_losses, val_losses = model.train(
            X_train=X_train, Y_train=Y_train,
            X_val=X_val, Y_val=Y_val,
            model_name=tag,
            early_stopping=True, patience=10, min_delta=0.005,
        )
        del model

        # Save raw losses
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        logging.info("Run %d — %d epochs  train_loss=%.4f  val_loss=%.4f",
                     i, len(train_losses), train_losses[-1], val_losses[-1])

        # Per-run loss plot
        save_loss_plot(
            train_losses, val_losses,
            save_path / f"loss_run{i}.png",
            f"Training and Validation Loss — Run {i}  ({species} / {tag})",
        )

        # --- Evaluation ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn = Model.load_cnn(str(save_path / model_filename), device)

        result_dataset = evaluation.run(cnn, test_type="testing_dataset")
        f1 = result_dataset[0]
        logging.info("Run %d — F1-score (testing_dataset) = %.4f", i, f1)
        f1_scores.append(f1)

        result_full = evaluation.run(cnn, test_type="entire_files", preprocessing_arg=True)
        f1_full, _, _, precision, recall = result_full
        logging.info("Run %d — F1-score (entire_files) = %.4f  precision=%.4f  recall=%.4f",
                     i, f1_full, precision, recall)
        f1_scores_full.append(f1_full)
        precision_scores.append(precision)
        recall_scores.append(recall)

        del cnn

    # --- Summary ---
    save_summary_loss_plot(
        all_train_losses, all_val_losses,
        save_path / f"loss_all_runs_{tag}.png",
        f"{species} — all runs — {tag}",
    )

    df = pd.DataFrame({
        "run": range(args.n_runs),
        "f1_score_dataset": f1_scores,
        "f1_score_full": f1_scores_full,
        "precision_full": precision_scores,
        "recall_full": recall_scores,
        "train_loss_final": [tl[-1] for tl in all_train_losses],
        "val_loss_final": [vl[-1] for vl in all_val_losses],
        "n_epochs": [len(tl) for tl in all_train_losses],
    })
    csv_path = save_path / f"{config.data.positive_class}_{tag}_results.csv"
    df.to_csv(csv_path, index=False)

    logging.info("F1-score (testing_dataset) mean=%.4f  std=%.4f", np.mean(f1_scores), np.std(f1_scores))
    logging.info("F1-score (entire_files)    mean=%.4f  std=%.4f", np.mean(f1_scores_full), np.std(f1_scores_full))
    logging.info("Results saved to: %s", save_path)


if __name__ == "__main__":
    main()
