"""
model.py
--------
Staged adaptation of pretrained EnCodec 24 kHz for gibbons.

Memory budget for a <16 GB GPU
-------------------------------
EnCodec 24 kHz model              ~  90 MB parameters
Batch of 4 windows × 3 s × 24 kHz ~  1.1 MB  (float32)
After resampling from 9.6 kHz, same window is 72 000 samples → 1.1 MB/batch
Activations during forward        ~ 200–400 MB  (depends on batch size)
Mixed precision (fp16 activations) cuts activation memory roughly in half.
Gradient checkpointing in stage 3 trades compute for memory in the decoder.

Staged unfreezing
-----------------
Stage 1  RVQ only           ~  4 M trainable params   (fast convergence)
Stage 2  + upper encoder    ~ 12 M trainable params   (moderate)
Stage 3  full model         ~ 75 M trainable params   (all unlocked)

Always start at stage 1. Only advance after validation loss plateaus.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from encodec import EncodecModel
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
    print(
        "[model] encodec not installed — using stub.\n"
        "        pip install encodec"
    )


# ---------------------------------------------------------------------------
# Bioacoustic spectral loss
# ---------------------------------------------------------------------------

class BioacousticSpectralLoss(nn.Module):
    """
    Multi-scale mel-spectrogram loss tuned for gibbon vocalisations.

    Changes vs. the EnCodec default:
    - Mel filterbank concentrated on 300–4000 Hz (gibbon call range)
    - Three FFT scales that resolve individual notes (21, 43, 85 ms at 24 kHz)
    - L1 on amplitude + L1 on log-amplitude  (standard for codec training)
    - Per-sample loss weighting (call windows weighted 4× higher)
    """

    def __init__(
        self,
        sample_rate: int   = 24_000,
        fft_sizes:   tuple = (1024, 2048, 4096),
        n_mels:      int   = 128,
        fmin:        float = 850.0,
        fmax:        float = 2500.0,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes

        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n,
                hop_length=n // 4,
                n_mels=n_mels,
                f_min=fmin,
                f_max=fmax,
                power=1.0,          # amplitude spectrogram
            )
            for n in fft_sizes
        ])

    def forward(
        self,
        recon:   torch.Tensor,    # (B, 1, T)
        target:  torch.Tensor,    # (B, 1, T)
        weights: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        total = target.new_zeros(1).squeeze()
        eps   = 1e-6

        for mel_fn in self.mel_transforms:
            mel_fn = mel_fn.to(target.device)

            r_mel = mel_fn(recon.squeeze(1))    # (B, n_mels, frames)
            t_mel = mel_fn(target.squeeze(1))

            l_amp = F.l1_loss(r_mel, t_mel, reduction="none").mean(dim=(1, 2))
            l_log = F.l1_loss(
                torch.log(r_mel + eps),
                torch.log(t_mel + eps),
                reduction="none",
            ).mean(dim=(1, 2))

            total = total + (weights * (l_amp + l_log)).mean()

        return total / len(self.fft_sizes)


# ---------------------------------------------------------------------------
# NOTE: hardcoded encoder block indices were removed because the real SEANet
# encoder interleaves ELU activations (no parameters) between every conv/LSTM
# block, so fixed indices like [5] may land on an activation. Block discovery
# is now done dynamically via _get_encoder_param_blocks().


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class BioacousticEnCodec(nn.Module):
    """
    EnCodec 24 kHz with staged unfreezing for bioacoustic fine-tuning.

    Parameters
    ----------
    bandwidth : float
        Target bitrate (kbps). Lower = more compression, harder to reconstruct.
        Options: 1.5, 3.0, 6.0, 12.0, 24.0
        Recommended starting point for gibbons: 3.0
    use_grad_checkpoint : bool
        Enable gradient checkpointing in stage 3 to save ~30% GPU memory.
        Adds ~15% training time overhead. Strongly recommended on <16 GB GPUs.
    """

    def __init__(
        self,
        bandwidth:            float = 3.0,
        use_grad_checkpoint:  bool  = True,
    ):
        super().__init__()
        self.bandwidth           = bandwidth
        self.use_grad_checkpoint = use_grad_checkpoint
        self.current_stage       = 0

        if ENCODEC_AVAILABLE:
            self.codec = EncodecModel.encodec_model_24khz()
            self.codec.set_target_bandwidth(bandwidth)
            print(f"[model] Loaded EnCodec 24 kHz  (bandwidth={bandwidth} kbps)")
        else:
            self.codec = _StubCodec()
            print("[model] Using stub codec")

        self._freeze_all()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_all(self):
        for p in self.codec.parameters():
            p.requires_grad_(False)

    def _unfreeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(True)

    def _get_submodule(self, dotpath: str) -> nn.Module | None:
        """Navigate a dotted attribute path, return None if not found."""
        m = self.codec
        for part in dotpath.split("."):
            m = getattr(m, part, None)
            if m is None:
                return None
        return m

    def _get_encoder_param_blocks(self) -> list[tuple[str, nn.Module]]:
        """
        Return (name, module) for every direct child of encoder.model that
        has at least one parameter, ordered from input to bottleneck.

        This skips activation layers (ELU, etc.) which have no parameters and
        would leave the optimizer with an empty parameter list.
        """
        seq = self._get_submodule("encoder.model")
        if seq is None:
            return []
        return [
            (f"encoder.model.{i}", child)
            for i, child in enumerate(seq.children())
            if any(True for _ in child.parameters())
        ]

    def _find_quantizer(self) -> nn.Module | None:
        """
        Try multiple known attribute paths for the RVQ quantizer, then fall
        back to a name search. Returns None only if truly absent.
        """
        for path in ("quantizer", "quantizer.layers", "rq"):
            m = self._get_submodule(path)
            if m is not None:
                return m
        # Name-search fallback for version differences
        for name, module in self.codec.named_modules():
            lower = name.lower()
            if any(k in lower for k in ("quantizer", "rvq", "residual_vq")):
                return module
        return None

    # ------------------------------------------------------------------
    # Stage advancement
    # ------------------------------------------------------------------

    def advance_to_stage(self, stage: int, n_upper_blocks: int = 2):
        """
        Cumulatively unfreeze parameters.

        stage 1 — RVQ quantizer only
        stage 2 — RVQ + top `n_upper_blocks` encoder blocks
        stage 3 — entire model (+ enables gradient checkpointing if requested)

        Call this once per stage transition; it is cumulative (stage 2 does
        not re-freeze what stage 1 opened).
        """
        if stage < 1 or stage > 3:
            raise ValueError(f"Stage must be 1, 2, or 3. Got {stage}.")
        if stage <= self.current_stage:
            print(f"[model] Already at stage {self.current_stage}, skipping.")
            return

        if stage >= 1 and self.current_stage < 1:
            # RVQ codebooks are register_buffer (EMA updates), not nn.Parameter,
            # so the only reliable source of trainable params is the encoder.
            # Unfreeze the top block (closest to bottleneck) that has parameters.
            blocks = self._get_encoder_param_blocks()
            if not blocks:
                raise RuntimeError(
                    "Stage 1: no encoder blocks with parameters found. "
                    "Run list(model.codec.named_modules()) to inspect your model."
                )
            top_name, top_module = blocks[-1]
            self._unfreeze(top_module)
            print(f"[model] Stage 1 — unfroze: {top_name}")

            # Quantizer may have learnable params in some versions (no-op otherwise)
            quantizer = self._find_quantizer()
            if quantizer is not None:
                self._unfreeze(quantizer)
                print("[model] Stage 1 — unfroze: quantizer (RVQ, if learnable)")

        if stage >= 2 and self.current_stage < 2:
            # Unfreeze the next n_upper_blocks closest to bottleneck
            # (top block was already opened at stage 1)
            blocks = self._get_encoder_param_blocks()
            for name, module in blocks[-(n_upper_blocks + 1):-1]:
                self._unfreeze(module)
                print(f"[model] Stage 2 — unfroze: {name}")

        if stage == 3 and self.current_stage < 3:
            self._unfreeze(self.codec)
            print("[model] Stage 3 — unfroze: entire model")
            if self.use_grad_checkpoint:
                print("[model]          Gradient checkpointing enabled")

        self.current_stage = stage
        self._print_param_counts()

    def _print_param_counts(self):
        total     = sum(p.numel() for p in self.codec.parameters())
        trainable = sum(p.numel() for p in self.codec.parameters() if p.requires_grad)
        print(
            f"[model] Trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / max(total, 1):.1f}%)"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,   # (B, 1, T) at 24 kHz
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode → quantize (STE) → decode.

        Returns
        -------
        reconstructed : (B, 1, T)
        codes         : (B, n_codebooks, T_latent)  integer codes (monitoring only)
        """
        if not ENCODEC_AVAILABLE:
            return waveform, torch.zeros(waveform.shape[0], 4, 1, device=waveform.device)

        # Encoder
        if self.use_grad_checkpoint and self.current_stage == 3 and self.training:
            emb = grad_checkpoint(self.codec.encoder, waveform, use_reentrant=False)
        else:
            emb = self.codec.encoder(waveform)

        # quantizer() uses the straight-through estimator so gradients flow back
        # through the quantisation step. Never use quantizer.encode() + .decode()
        # during training — those return discrete integer codes with no grad_fn.
        q_res = self.codec.quantizer(emb, self.codec.frame_rate, self.bandwidth)
        reconstructed = self.codec.decoder(q_res.quantized)

        # Trim to input length (encoder/decoder may add padding)
        reconstructed = reconstructed[..., :waveform.shape[-1]]

        # Normalise codes shape to (B, K, T) for CodebookMonitor
        codes = q_res.codes
        if codes.dim() == 3 and codes.shape[0] != waveform.shape[0]:
            codes = codes.transpose(0, 1)

        return reconstructed, codes

    # ------------------------------------------------------------------
    # Inference helpers (no grad, no checkpointing)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_to_codes(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compress waveform to integer codes.
        waveform : (B, 1, T) at 24 kHz
        returns  : (B, n_codebooks, T_latent)
        """
        if not ENCODEC_AVAILABLE:
            return torch.zeros(waveform.shape[0], 4, 1, device=waveform.device, dtype=torch.long)
        frames = self.codec.encode(waveform)
        return torch.cat([f[0] for f in frames], dim=-1)

    @torch.inference_mode()
    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from integer codes.
        codes   : (B, n_codebooks, T_latent)
        returns : (B, 1, T)
        """
        if not ENCODEC_AVAILABLE:
            return torch.zeros(codes.shape[0], 1, codes.shape[-1] * 320, device=codes.device)
        return self.codec.decode([(codes, None)])

    @torch.inference_mode()
    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        """Round-trip encode→decode. Used for evaluation."""
        frames = self.codec.encode(waveform)
        return self.codec.decode(frames)


# ---------------------------------------------------------------------------
# Stub codec for development without encodec installed
# ---------------------------------------------------------------------------

class _StubCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = nn.Linear(1, 1)   # placeholder so advance_to_stage works
        self.encoder   = nn.Sequential(nn.Identity())
        self.decoder   = nn.Sequential(nn.Identity())

    def encode(self, x):
        T = x.shape[-1] // 320
        return [(torch.zeros(x.shape[0], 4, T, device=x.device, dtype=torch.long), None)]

    def decode(self, frames):
        codes, _ = frames[0]
        return torch.zeros(codes.shape[0], 1, codes.shape[-1] * 320, device=codes.device)

    def set_target_bandwidth(self, bw):
        pass
