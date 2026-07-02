"""
evaluate.py
-----------
Three things:

1. CodebookMonitor  — tracks RVQ code usage per epoch to detect collapse
2. evaluate_codec   — reconstruction loss + acoustic feature metrics
3. build_cnn_dataset— encode all windows, save X_recon.npy + Y.npy for CNN

Acoustic metrics
----------------
  val_loss               : spectral loss (same as training objective)
  f0_error_semitones     : median |F0_orig - F0_recon| in semitones
                           (most important metric for gibbon call fidelity)
  spectral_centroid_err  : mean |centroid_orig - centroid_recon| in Hz

Both acoustic metrics are computed on call-containing windows only,
because background audio has no meaningful F0 to compare.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[evaluate] librosa not found — acoustic metrics will be skipped.\n"
          "           pip install librosa")

from dataset import resample_batch, TARGET_SR
from model import BioacousticEnCodec, BioacousticSpectralLoss


# ---------------------------------------------------------------------------
# 1. Codebook monitor
# ---------------------------------------------------------------------------

class CodebookMonitor:
    """
    Tracks which discrete codes the RVQ actually uses across an epoch.

    A healthy codec uses most of its codebook. Usage below 30% on any
    quantizer means collapse is starting and the codebook is wasting capacity.
    """

    def __init__(self, n_codebooks: int = 4, codebook_size: int = 1024):
        self.n_codebooks   = n_codebooks
        self.codebook_size = codebook_size
        self.reset()

    def reset(self):
        self._usage = [
            torch.zeros(self.codebook_size, dtype=torch.long)
            for _ in range(self.n_codebooks)
        ]

    def update(self, codes: torch.Tensor):
        """
        codes : (B, n_codebooks, T_latent) int tensor
        """
        B, Q, T = codes.shape
        for q in range(min(Q, self.n_codebooks)):
            flat = codes[:, q, :].reshape(-1).cpu()
            valid = flat[(flat >= 0) & (flat < self.codebook_size)]
            self._usage[q].scatter_add_(
                0, valid, torch.ones_like(valid, dtype=torch.long)
            )

    def utilization(self) -> list[float]:
        """Fraction of codebook entries used per quantizer."""
        return [(u > 0).float().mean().item() for u in self._usage]

    def report(self) -> str:
        parts = [f"Q{i}: {u:.0%}" for i, u in enumerate(self.utilization())]
        return "Codebook util — " + " | ".join(parts)


# ---------------------------------------------------------------------------
# 2. Codec evaluation
# ---------------------------------------------------------------------------

def _extract_f0(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """pyin F0 extraction. Returns array with NaN where unvoiced."""
    if not LIBROSA_AVAILABLE:
        return np.array([])
    # Guard against NaN/inf in reconstructed audio (can occur at stage transitions)
    if not np.all(np.isfinite(y)):
        return np.array([])
    y = np.clip(y, -1.0, 1.0)
    try:
        f0, voiced, _ = librosa.pyin(
            y, fmin=200.0, fmax=4_000.0, sr=sr, hop_length=256
        )
        f0[~voiced] = np.nan
        return f0
    except Exception:
        return np.array([])



def _f0_error_semitones(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """Median |difference| in semitones between two F0 arrays."""
    both = ~np.isnan(f0_a) & ~np.isnan(f0_b)
    if both.sum() < 3:
        return float("nan")
    ratio = np.clip(f0_b[both] / (f0_a[both] + 1e-8), 1e-4, 1e4)
    return float(np.median(np.abs(12.0 * np.log2(ratio))))


def _centroid_error(a: np.ndarray, b: np.ndarray, sr: int = TARGET_SR) -> float:
    if not LIBROSA_AVAILABLE:
        return float("nan")
    ca = librosa.feature.spectral_centroid(y=a, sr=sr)[0]
    cb = librosa.feature.spectral_centroid(y=b, sr=sr)[0]
    L  = min(len(ca), len(cb))
    return float(np.mean(np.abs(ca[:L] - cb[:L])))


@torch.no_grad()
def evaluate_codec(
    model:     BioacousticEnCodec,
    loader,
    loss_fn:   BioacousticSpectralLoss,
    device:    torch.device,
    max_batches: int = 40,    # cap evaluation time; ~2 min on a typical GPU
) -> dict:
    """
    Compute validation metrics over `loader`.

    Acoustic metrics are computed on a maximum of 20 call windows to keep
    evaluation fast (pyin is slow). Loss is computed on all batches up to
    `max_batches`.
    """
    model.eval()

    total_loss = 0.0
    n_batches  = 0
    f0_errors, cent_errors = [], []
    acoustic_budget = 20   # number of call windows to run pyin on

    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break

        wav_native   = batch["waveform"].to(device)
        loss_weights = batch["loss_weight"].to(device)
        labels       = batch["label"]

        wav_24k = resample_batch(wav_native, device)
        recon   = model.reconstruct(wav_24k)

        L    = min(recon.shape[-1], wav_24k.shape[-1])
        loss = loss_fn(recon[..., :L], wav_24k[..., :L], loss_weights)
        total_loss += loss.item()
        n_batches  += 1

        # Acoustic metrics — only on call windows, only while budget remains
        if LIBROSA_AVAILABLE and acoustic_budget > 0:
            for i in range(wav_24k.shape[0]):
                if labels[i].item() != 1:
                    continue
                if acoustic_budget <= 0:
                    break

                orig_np  = wav_24k[i, 0, :L].cpu().float().numpy()
                recon_np = recon[i,  0, :L].cpu().float().numpy()

                f0_o = _extract_f0(orig_np)
                f0_r = _extract_f0(recon_np)
                err  = _f0_error_semitones(f0_o, f0_r)
                if not np.isnan(err):
                    f0_errors.append(err)

                cent_errors.append(_centroid_error(orig_np, recon_np))
                acoustic_budget -= 1

    return {
        "val_loss":              total_loss / max(n_batches, 1),
        "f0_error_semitones":    float(np.median(f0_errors))  if f0_errors   else float("nan"),
        "centroid_error_hz":     float(np.mean(cent_errors))  if cent_errors else float("nan"),
    }


# ---------------------------------------------------------------------------
# 3. Build CNN dataset from encoded reconstructions
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_cnn_dataset(
    model:      BioacousticEnCodec,
    loader,
    device:     torch.device,
    output_dir: Path,
    prefix:     str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode every window in `loader` through the trained codec and save
    the reconstructed waveforms as numpy arrays ready for CNN training.

    Saved files
    -----------
    {output_dir}/{prefix}X_recon.npy  — (N, T_24k) float32 reconstructed audio
    {output_dir}/{prefix}Y.npy        — (N,)        int labels (unchanged)

    The labels come directly from the original Y array — they are unaffected
    by the codec and require no re-annotation.

    Returns
    -------
    X_recon, Y  (also saved to disk)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_recon, all_labels = [], []

    for batch in loader:
        wav_native   = batch["waveform"].to(device)
        labels       = batch["label"]

        wav_24k = resample_batch(wav_native, device)
        recon   = model.reconstruct(wav_24k)

        L = min(recon.shape[-1], wav_24k.shape[-1])
        # Back to CPU numpy, squeeze channel dim: (B, T)
        all_recon.append(recon[:, 0, :L].cpu().float().numpy())
        all_labels.append(labels.numpy())

    X_recon = np.concatenate(all_recon,  axis=0)   # (N, T_24k)
    Y       = np.concatenate(all_labels, axis=0)   # (N,)

    x_path = output_dir / f"{prefix}X_recon.npy"
    y_path = output_dir / f"{prefix}Y.npy"
    np.save(x_path, X_recon)
    np.save(y_path, Y)

    n_calls = int((Y == 1).sum())
    print(
        f"  Saved {len(X_recon):,} windows to {output_dir}/ "
        f"({n_calls:,} calls)"
    )
    print(f"    X: {x_path}  shape {X_recon.shape}")
    print(f"    Y: {y_path}  shape {Y.shape}")

    return X_recon, Y
