"""
train.py
--------
Full 3-stage EnCodec fine-tuning loop for bioacoustic signals.

Memory optimisations for <16 GB GPU
------------------------------------
  - torch.cuda.amp (automatic mixed precision): fp16 activations, fp32 params
  - Gradient checkpointing in stage 3 (enabled in model.py)
  - Small default batch size (4). Increase to 8 if VRAM allows.
  - GPU memory cleared between stages via torch.cuda.empty_cache()

Quickstart
----------
  python train.py --x X.pkl --y Y.pkl

Or import and call train() from a notebook with a config dict.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from dataset import (
    load_data,
    make_dataloaders,
    resample_batch,
    NATIVE_SR,
    TARGET_SR,
)
from model import BioacousticEnCodec, BioacousticSpectralLoss
from evaluate import (
    evaluate_codec,
    build_cnn_dataset,
    CodebookMonitor,
)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CFG = dict(
    # ── Paths ──────────────────────────────────────────────────────────
    x_path          = "X.pkl",
    y_path          = "Y.pkl",
    checkpoint_dir  = "checkpoints",
    cnn_output_dir  = "cnn_data",

    # ── Data split ─────────────────────────────────────────────────────
    codec_frac      = 0.70,
    cnn_frac        = 0.15,
    split_seed      = 42,

    # ── Model ──────────────────────────────────────────────────────────
    bandwidth       = 3.0,

    # ── Stage schedule  (epoch -> stage to advance to) ─────────────────
    stage_schedule  = {0: 1, 15: 2, 35: 3},
    n_upper_blocks  = 2,

    # ── Loss ───────────────────────────────────────────────────────────
    fft_sizes       = (1024, 2048, 4096),
    loss_fmin       = 850.0,
    loss_fmax       = 2500.0,

    # ── Optimiser ──────────────────────────────────────────────────────
    epochs          = 60,
    batch_size      = 32,
    lr              = 3e-4,
    weight_decay    = 1e-4,
    grad_clip       = 1.0,
    num_workers     = 8,

    # ── Misc ───────────────────────────────────────────────────────────
    seed            = 42,
)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model:     BioacousticEnCodec,
    loader,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    loss_fn:   BioacousticSpectralLoss,
    cb_mon:    CodebookMonitor,
    device:    torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss, n = 0.0, 0

    for batch in loader:
        wav_native   = batch["waveform"].to(device)
        loss_weights = batch["loss_weight"].to(device)

        wav_24k = resample_batch(wav_native, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=device.type == "cuda"):
            recon, codes = model(wav_24k)
            L = min(recon.shape[-1], wav_24k.shape[-1])
            loss = loss_fn(recon[..., :L], wav_24k[..., :L], loss_weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        cb_mon.update(codes)
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def train(cfg: dict = DEFAULT_CFG):
    torch.manual_seed(cfg.get("seed", 42))

    # ── Device ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("WARNING: no GPU found, training on CPU (very slow)")

    # ── Data ───────────────────────────────────────────────────────────────
    X, Y = load_data(cfg["x_path"], cfg["y_path"])

    codec_loader, cnn_loader, test_loader = make_dataloaders(
        X, Y,
        codec_frac  = cfg["codec_frac"],
        cnn_frac    = cfg["cnn_frac"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        seed        = cfg.get("split_seed", 42),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = BioacousticEnCodec(
        bandwidth           = cfg["bandwidth"],
        use_grad_checkpoint = True,
    ).to(device)

    loss_fn = BioacousticSpectralLoss(
        sample_rate = TARGET_SR,
        fft_sizes   = cfg["fft_sizes"],
        fmin        = cfg["loss_fmin"],
        fmax        = cfg["loss_fmax"],
    )

    scaler  = GradScaler('cuda', enabled=device.type == "cuda")
    cb_mon  = CodebookMonitor(n_codebooks=4, codebook_size=1024)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage_schedule = {int(k): int(v) for k, v in cfg["stage_schedule"].items()}
    current_stage  = 0

    # ── Helper: build optimizer from currently trainable params ────────────
    def _build_optimizer(stage: int, remaining_epochs: int, lr_override: float = None):
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                f"Stage {stage}: no trainable parameters found. "
                "Check ENCODER_BLOCKS paths in model.py match your EnCodec "
                "version (run: list(model.codec.named_modules()))."
            )
        lr = lr_override if lr_override is not None else cfg["lr"]
        opt = optim.AdamW(
            trainable, lr=lr, weight_decay=cfg["weight_decay"]
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=remaining_epochs, eta_min=cfg["lr"] * 0.01
        )
        return opt, sch

    # ── Advance to stage 1 before the epoch loop ───────────────────────────
    initial_stage = stage_schedule.get(0, 1)
    model.advance_to_stage(initial_stage, cfg.get("n_upper_blocks", 2))
    current_stage = initial_stage
    optimizer, scheduler = _build_optimizer(initial_stage, cfg["epochs"])

    best_val_loss = float("inf")

    # ── Epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg["epochs"]):

        # Stage advancement — skip epoch 0, already handled above
        if epoch > 0 and epoch in stage_schedule:
            new_stage = stage_schedule[epoch]
            if new_stage != current_stage:
                model.advance_to_stage(new_stage, cfg.get("n_upper_blocks", 2))
                current_stage = new_stage
                remaining = cfg["epochs"] - epoch
                # Halve lr at stage 2+ to prevent instability from newly
                # unfrozen layers starting with speech-pretrained weights
                stage_lr = cfg["lr"] if new_stage == 1 else cfg["lr"] * 0.5
                optimizer, scheduler = _build_optimizer(
                    new_stage, remaining, lr_override=stage_lr
                )
                torch.cuda.empty_cache()
                print(f"  Rebuilt optimizer — lr={stage_lr:.2e}, "
                      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
                      f"trainable params")

        # ── Train ──────────────────────────────────────────────────────────
        cb_mon.reset()
        t0 = time.time()

        train_loss = _train_epoch(
            model, codec_loader, optimizer, scaler,
            loss_fn, cb_mon, device, cfg["grad_clip"],
        )
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        val_metrics = evaluate_codec(model, cnn_loader, loss_fn, device)
        elapsed = time.time() - t0

        # ── Console log ────────────────────────────────────────────────────
        print(
            f"Epoch {epoch:03d} | stage {current_stage} | "
            f"train {train_loss:.4f} | val {val_metrics['val_loss']:.4f} | "
            f"f0_err {val_metrics.get('f0_error_semitones', float('nan')):.3f} st | "
            f"{elapsed:.0f}s"
        )
        print(f"  {cb_mon.report()}")

        # ── Codebook collapse warning ───────────────────────────────────────
        min_util = min(cb_mon.utilization())
        if min_util < 0.30:
            print(
                f"  Codebook collapse risk: min utilization {min_util:.1%}.\n"
                "    Options: lower bandwidth, reduce codebook_size, or apply "
                "EMA reset."
            )

        # ── GPU memory snapshot ─────────────────────────────────────────────
        if device.type == "cuda":
            mem_gb = torch.cuda.memory_reserved(device) / 1024**3
            print(f"  GPU reserved: {mem_gb:.1f} GB")

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            _save_checkpoint(model, optimizer, epoch, val_metrics,
                             current_stage, cfg, ckpt_dir / "best.pt")
            print(f"  New best ({best_val_loss:.4f}) -> {ckpt_dir/'best.pt'}")

        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            _save_checkpoint(model, optimizer, epoch, val_metrics,
                             current_stage, cfg,
                             ckpt_dir / f"epoch_{epoch:03d}.pt")

    # ── Post-training: build CNN dataset from best checkpoint ──────────────
    print("\nTraining complete. Building CNN training data from best checkpoint...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    cnn_out = Path(cfg["cnn_output_dir"])
    cnn_out.mkdir(parents=True, exist_ok=True)

    print("Encoding CNN sites...")
    build_cnn_dataset(model, cnn_loader, device, cnn_out / "cnn")
    print("Encoding test sites...")
    build_cnn_dataset(model, test_loader, device, cnn_out / "test")

    print(f"\nDone. CNN arrays saved to {cnn_out}/")
    print("Next step: train your CNN detector on cnn_data/cnn_X_recon.npy + cnn_Y.npy")
    print("           evaluate it on               cnn_data/test_X_recon.npy + test_Y.npy")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, epoch, metrics, stage, cfg, path):
    torch.save({
        "epoch":            epoch,
        "stage":            stage,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "metrics":          metrics,
        "cfg":              cfg,
    }, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bioacoustic EnCodec fine-tuning")
    parser.add_argument("--x",              default="X.pkl",      help="Path to X pickle file")
    parser.add_argument("--y",              default="Y.pkl",      help="Path to Y pickle file")
    parser.add_argument("--codec-frac",     type=float, default=0.70)
    parser.add_argument("--cnn-frac",       type=float, default=0.15)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--epochs",         type=int,   default=60)
    parser.add_argument("--bandwidth",      type=float, default=3.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--cnn-output-dir", default="cnn_data")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg.update({
        "x_path":         args.x,
        "y_path":         args.y,
        "codec_frac":     args.codec_frac,
        "cnn_frac":       args.cnn_frac,
        "batch_size":     args.batch_size,
        "epochs":         args.epochs,
        "bandwidth":      args.bandwidth,
        "checkpoint_dir": args.checkpoint_dir,
        "cnn_output_dir": args.cnn_output_dir,
    })
    train(cfg)
