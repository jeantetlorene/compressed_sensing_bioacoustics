"""
dataset.py
----------
Dataset for pre-windowed numpy arrays.

Expected on-disk layout
-----------------------
  X.pkl  : pickle of (N, T) float32 — audio windows at NATIVE_SR (9600 Hz),
                                       normalised to [-1, 1]
  Y.pkl  : pickle of (N,)   int     — binary labels (1 = call, 0 = background)

The arrays are loaded once at startup and kept in RAM.

Resampling 9600 -> 24000 Hz happens inside the training loop on the GPU
so nothing larger than the native-rate arrays ever lives on disk or in
pinned memory.

Splitting
---------
Random split by index into codec / cnn / test fractions.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


NATIVE_SR   = 9_600     # gibbon original sample rate — update for other species
                        # (thyolo=32000, ptw=48000, or whatever librosa.load returns)
TARGET_SR   = 24_000    # EnCodec 24 kHz model input rate
CALL_WEIGHT = 4.0       # per-sample loss upweight for call windows


# ---------------------------------------------------------------------------
# Resampler — built once, moved to device on first use
# ---------------------------------------------------------------------------

_resampler: torchaudio.transforms.Resample | None = None


def get_resampler(device: torch.device) -> torchaudio.transforms.Resample:
    """Return a cached resampler on the correct device."""
    global _resampler
    if _resampler is None:
        _resampler = torchaudio.transforms.Resample(
            orig_freq=NATIVE_SR,
            new_freq=TARGET_SR,
            resampling_method="sinc_interp_kaiser",
        )
    return _resampler.to(device)


def resample_batch(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Resample a batch of waveforms from NATIVE_SR to TARGET_SR on the GPU.

    Parameters
    ----------
    x : (B, 1, T_native) — already on `device`

    Returns
    -------
    (B, 1, T_target)
    """
    return get_resampler(device)(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowedDataset(Dataset):
    """
    Thin wrapper around pre-windowed numpy arrays.

    Parameters
    ----------
    X       : (N, T) float32 at NATIVE_SR, normalised [-1, 1]
    Y       : (N,)   int, binary labels
    augment : if True, apply ±3 dB gain jitter on call windows
    """

    def __init__(
        self,
        X:       np.ndarray,
        Y:       np.ndarray,
        augment: bool = False,
    ):
        assert X.ndim == 2, f"Expected X shape (N, T), got {X.shape}"
        assert Y.shape == (X.shape[0],), \
            f"Y shape {Y.shape} does not match X rows {X.shape[0]}"

        # Store as float32 in RAM — contiguous for fast pin_memory transfer
        self.X       = np.ascontiguousarray(X, dtype=np.float32)
        self.Y       = np.ascontiguousarray(Y, dtype=np.int64)
        self.augment = augment

        n_calls = int((Y == 1).sum())
        print(
            f"    {len(X):,} windows  "
            f"({n_calls:,} calls, {len(X) - n_calls:,} background, "
            f"{100 * n_calls / max(len(X), 1):.1f}% positive)"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        x = torch.from_numpy(self.X[idx])      # (T,)  at NATIVE_SR
        y = int(self.Y[idx])

        # Augmentation: small gain jitter on call windows only
        if self.augment and y == 1:
            gain_db = (torch.rand(1).item() * 6.0) - 3.0   # uniform [-3, +3]
            x = (x * 10 ** (gain_db / 20.0)).clamp(-1.0, 1.0)

        return {
            "waveform":    x.unsqueeze(0),                              # (1, T)
            "label":       torch.tensor(y,                dtype=torch.long),
            "loss_weight": torch.tensor(CALL_WEIGHT if y else 1.0),
        }


# ---------------------------------------------------------------------------
# Weighted sampler — oversample call windows so each batch is ~25% positive
# ---------------------------------------------------------------------------

def make_sampler(Y: np.ndarray) -> WeightedRandomSampler:
    weights = np.where(Y == 1, float(CALL_WEIGHT), 1.0)
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    """Module-level so Windows spawn can pickle it."""
    warnings.filterwarnings("ignore", category=FutureWarning)


def make_dataloaders(
    X:           np.ndarray,
    Y:           np.ndarray,
    codec_frac:  float = 0.70,
    cnn_frac:    float = 0.15,
    batch_size:  int = 4,
    num_workers: int = 2,
    seed:        int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build three DataLoaders with random splits.

    codec_loader : weighted sampler, augmentation ON  (codec_frac of data)
    cnn_loader   : sequential, no augmentation        (cnn_frac of data)
    test_loader  : sequential, no augmentation        (remainder)
    """
    n   = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_codec = int(n * codec_frac)
    n_cnn   = int(n * cnn_frac)

    codec_idx = idx[:n_codec]
    cnn_idx   = idx[n_codec:n_codec + n_cnn]
    test_idx  = idx[n_codec + n_cnn:]

    test_frac = 1.0 - codec_frac - cnn_frac
    print(f"\nRandom split (seed={seed}):")
    print(f"  Codec : {len(codec_idx):>6,} windows  ({codec_frac:.0%})")
    print(f"  CNN   : {len(cnn_idx):>6,} windows  ({cnn_frac:.0%})")
    print(f"  Test  : {len(test_idx):>6,} windows  ({test_frac:.0%})")

    print("\nCodec training set:")
    codec_ds = WindowedDataset(X[codec_idx], Y[codec_idx], augment=True)
    print("CNN training set:")
    cnn_ds   = WindowedDataset(X[cnn_idx],   Y[cnn_idx],   augment=False)
    print("Test set:")
    test_ds  = WindowedDataset(X[test_idx],  Y[test_idx],  augment=False)

    common = dict(num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn)

    codec_loader = DataLoader(
        codec_ds, batch_size=batch_size,
        sampler=make_sampler(Y[codec_idx]),
        drop_last=True, **common,
    )
    cnn_loader = DataLoader(
        cnn_ds, batch_size=batch_size,
        shuffle=False, **common,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, **common,
    )
    return codec_loader, cnn_loader, test_loader


# ---------------------------------------------------------------------------
# Convenience: load arrays from pickle files
# ---------------------------------------------------------------------------

def load_data(
    x_path: str,
    y_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load X, Y from pickle files."""
    with open(x_path, "rb") as f:
        X = pickle.load(f)
    with open(y_path, "rb") as f:
        Y = pickle.load(f)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y)
    print(f"Loaded X={X.shape}, Y={Y.shape}")
    return X, Y
