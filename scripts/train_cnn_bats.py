"""
Train the bat CNN classifier N times and save per-run losses and metrics.

Usage examples
--------------
# CS compression, rate 0.2 (default):
python scripts/train_cnn_bats.py

# Baseline (no compression):
python scripts/train_cnn_bats.py --method-compression baseline

# Custom compression and 5 runs:
python scripts/train_cnn_bats.py --method-compression cs --parameter-compression 0.15 --n-runs 5

# Different species folder:
python scripts/train_cnn_bats.py --species-folder "D:/Data/Bats"

# Show all options:
python scripts/train_cnn_bats.py --help
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
from sklearn.metrics import confusion_matrix, f1_score

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config_species import get_settings
from model import Model
from settings import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_bats_{time.strftime('%Y%m%d_%H%M%S')}.log"
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


def batch_predict(model, inputs, batch_size=62):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    X_tensor = torch.from_numpy(inputs).float()
    if X_tensor.ndim == 3:
        X_tensor = X_tensor.unsqueeze(1)
    loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_preds.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(all_preds).numpy()


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
        description="Train bat CNN 10 times, saving per-run losses and F1 scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--species-folder",
        default="C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Bats",
        help="Root folder for the bats dataset.",
    )
    parser.add_argument(
        "--method-compression",
        default="cs",
        help="Compression method (e.g. cs, mp3, aac). Use 'baseline' for no compression.",
    )
    parser.add_argument(
        "--parameter-compression",
        default="0.1",
        help="Compression parameter (e.g. CS rate, codec bitrate).",
    )
    parser.add_argument("--n-runs", type=int, default=10, help="Number of training runs.")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Compression config
    is_baseline = args.method_compression.lower() == "baseline"
    method_compression = None if is_baseline else args.method_compression
    parameter_compression = None if is_baseline else args.parameter_compression

    # Species config
    settings = get_settings("bats")
    config = Config(settings)
    config.data.species_folder = args.species_folder
    config.preprocessing.audio_extension = ".npy"

    # CNN architecture for bats (from notebook)
    config.cnn_architecture.conv_layers = 2
    config.cnn_architecture.fc_layers = 2
    config.cnn_architecture.conv_kernel = 16
    config.cnn_architecture.conv_filters = 16
    config.cnn_architecture.dropout_rate = 0.5
    config.cnn_architecture.fc_units = 32
    config.cnn_architecture.max_pooling_size = 4

    config.model.learning_rate = args.learning_rate
    config.model.num_epochs = args.num_epochs
    config.model.batch_size = args.batch_size

    # Output folder
    tag = f"{method_compression}_{parameter_compression}" if not is_baseline else "baseline"
    save_path = Path(config.data.species_folder, "results",
                     f"{config.data.positive_class}_{tag}")
    save_path.mkdir(parents=True, exist_ok=True)

    setup_logging(save_path, args.log_level)
    logging.info("Species folder : %s", config.data.species_folder)
    logging.info("Compression    : %s", tag)
    logging.info("Runs           : %d", args.n_runs)
    logging.info("Epochs         : %d  Batch: %d  LR: %g",
                 args.num_epochs, args.batch_size, args.learning_rate)

    # Load datasets once
    X_train, Y_train = load_dataset(
        config.data.species_folder, config.data.positive_class,
        method_compression=method_compression, parameter_compression=parameter_compression)
    X_test, Y_test = load_dataset(
        config.data.species_folder, config.data.positive_class, dataset_type="test",
        method_compression=method_compression, parameter_compression=parameter_compression)

    all_train_losses, all_val_losses, f1_scores = [], [], []
    model_filename = f"{tag}_cnn_state.pth"

    for i in range(args.n_runs):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("=== Run %d / %d ===", i + 1, args.n_runs)

        # --- Training ---
        model = Model(
            save_path,
            input_shape=(1, X_train.shape[1], X_train.shape[2]),
            architecture_args=config.cnn_architecture.dict(),
            task="classification",
            **config.model.dict(),
        )
        train_losses, val_losses = model.train(
            X_train=X_train, Y_train=Y_train,
            model_name=tag,
            early_stopping=True, patience=15, min_delta=0.0005,
        )
        del model

        # Save raw losses
        np.save(save_path / f"train_losses_run{i}.npy", np.array(train_losses))
        np.save(save_path / f"val_losses_run{i}.npy", np.array(val_losses))
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        logging.info("Run %d — %d epochs  train_loss=%.4f  val_loss=%.4f",
                     i, len(train_losses), train_losses[-1], val_losses[-1])

        # Per-run loss plot
        save_loss_plot(
            train_losses, val_losses,
            save_path / f"loss_run{i}.png",
            f"Training and Validation Loss — Run {i}  ({tag})",
        )

        # --- Evaluation ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn = Model.load_cnn(str(save_path / model_filename), device)
        preds = batch_predict(cnn, X_test)
        del cnn

        f1 = f1_score(Y_test, preds, average="macro")
        logging.info("Run %d — F1-score (macro) = %.4f", i, f1)
        f1_scores.append(f1)

    # --- Summary ---
    save_summary_loss_plot(
        all_train_losses, all_val_losses,
        save_path / f"loss_all_runs_{tag}.png",
        f"All runs — {tag}",
    )

    df = pd.DataFrame({
        "run": range(args.n_runs),
        "f1_score": f1_scores,
        "train_loss_final": [tl[-1] for tl in all_train_losses],
        "val_loss_final": [vl[-1] for vl in all_val_losses],
        "n_epochs": [len(tl) for tl in all_train_losses],
    })
    csv_path = save_path / f"bats_{tag}_results.csv"
    df.to_csv(csv_path, index=False)

    logging.info("F1-score  mean=%.4f  std=%.4f", np.mean(f1_scores), np.std(f1_scores))
    logging.info("Results saved to: %s", save_path)


if __name__ == "__main__":
    main()
