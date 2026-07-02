"""
quickstart.py
-------------
Self-contained demo you can run immediately to verify the pipeline works
before plugging in your real data.

Generates synthetic gibbon-like audio (sinusoidal FM sweeps + noise),
runs the full 3-stage training loop for 3 epochs per stage, and saves
the CNN dataset. Real training should use 15–35 epochs per stage.

Run:
    python quickstart.py

Expected output on a laptop GPU (~8 GB):
    GPU: NVIDIA GeForce RTX 3070 (8 GB)
    Random split (seed=42): ...
    Stage 1 — unfroze: quantizer (RVQ)
    Epoch 000 | stage 1 | train 0.xxxx | val 0.xxxx | ...
    ...
    CNN arrays saved to cnn_data/
"""

import pickle
import numpy as np
import torch
from train import train, DEFAULT_CFG


# ---------------------------------------------------------------------------
# Synthetic data — replace with your real pickle files
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_windows:  int = 800,
    window_sec: float = 3.0,
    native_sr:  int = 9_600,
    call_ratio: float = 0.08,   # 8% positive — typical for passive monitoring
    seed:       int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce plausible synthetic data matching your array format.

    X : (N, T) float32  — audio at 9600 Hz, normalised [-1, 1]
    Y : (N,)   int      — 1 = call present
    """
    rng = np.random.default_rng(seed)
    T   = int(window_sec * native_sr)
    t   = np.linspace(0, window_sec, T, dtype=np.float32)

    X = np.zeros((n_windows, T), dtype=np.float32)
    Y = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        # Background: pink-ish noise
        noise = rng.standard_normal(T).astype(np.float32) * 0.05

        if rng.random() < call_ratio:
            # Synthetic gibbon-like call: FM sweep 800 Hz -> 1600 Hz
            # duration 0.3–0.8 s, appears at random position in window
            call_dur   = rng.uniform(0.3, 0.8)
            call_start = rng.uniform(0.2, window_sec - call_dur - 0.2)
            call_end   = call_start + call_dur

            mask = (t >= call_start) & (t < call_end)
            tc   = t[mask] - call_start                 # local time
            f0   = 800.0 + 800.0 * tc / call_dur       # linear FM 800->1600 Hz
            call = np.sin(2 * np.pi * f0 * tc).astype(np.float32) * 0.4

            noise[mask] += call
            Y[i] = 1

        # Normalise
        peak  = np.abs(noise).max()
        X[i]  = noise / max(peak, 1e-6)

    n_calls = int(Y.sum())
    print(
        f"Synthetic dataset: {n_windows} windows, "
        f"{n_calls} calls ({100*n_calls/n_windows:.1f}%)"
    )
    return X, Y


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, Y = make_synthetic_data(n_windows=800)

    # Save to disk so train() can load them
    with open("X.pkl", "wb") as f:
        pickle.dump(X, f)
    with open("Y.pkl", "wb") as f:
        pickle.dump(Y, f)

    cfg = dict(DEFAULT_CFG)
    cfg.update({
        "x_path":        "X.pkl",
        "y_path":        "Y.pkl",
        "codec_frac":    0.70,
        "cnn_frac":      0.15,

        # Short demo run — use 15/35/55 for real training
        "stage_schedule": {0: 1, 3: 2, 6: 3},
        "epochs":         9,
        "batch_size":     4,
        "bandwidth":      3.0,
        "checkpoint_dir": "checkpoints_demo",
        "cnn_output_dir": "cnn_data_demo",
    })

    train(cfg)
