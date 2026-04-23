# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying the effect of audio compression (compressed sensing and standard codecs) on bioacoustic species classification accuracy. The pipeline compresses `.wav` audio files, generates mel-spectrograms, trains a CNN, and exports predictions in Sonic Visualiser `.svl` format.

## Running the Code

There is no build system or test suite. The primary workflow runs through Jupyter notebooks:

```bash
jupyter notebook notebooks/main_vs2.ipynb
```

Modules in `src/` are imported directly within notebooks. There is no package install step.

**System dependency:** FFmpeg or libav must be installed for `pydub` to handle MP3/AAC/OGG/OPUS compression. Without it, codec-based compression will fail silently or raise errors.

**GPU:** PyTorch will use CUDA if available; no explicit device selection is needed beyond what PyTorch detects.

## Architecture

### Pipeline (in execution order)

1. **Compression** (`src/compress.py`) — Compress raw `.wav` files before any ML work.
   - `Compression_vs2`: wraps pydub to encode/decode via standard codecs (MP3, AAC, OGG, FLAC, OPUS). Output is a re-decoded `.wav`.
   - `CS`: compressed sensing via DCT random sampling. Saves measurement vector as `.npy`. Reconstruction uses IHT, LASSO, or OMP solvers.

2. **Preprocessing** (`src/preprocess.py`) — `Preprocessing` class reads audio (original, codec-compressed, or CS-reconstructed), applies a Butterworth lowpass filter, downsamples, converts to mel-spectrogram, extracts fixed-length segments from annotated regions, and augments (time shift, Gaussian noise, audio blending).

3. **Annotation parsing** (`src/AnnotationReader.py`) — Reads Sonic Visualiser `.svl` XML to get temporal boundaries and labels. Filters by confidence (default: 10 = high confidence).

4. **Training** (`src/model.py`, `src/cnn.py`) — `Model` wraps `BaseCNN` (configurable Conv2d stack + FC layers) with a standard PyTorch training loop. Uses Adam, CrossEntropy loss, ReduceLROnPlateau scheduler, and early stopping. Saves state dict as `{name}_cnn_state.pth`.

5. **Evaluation** (`src/evaluation.py`) — `Evaluation` runs sliding-window inference on audio files, groups consecutive confident predictions, and exports results as `.svl` XML for inspection in Sonic Visualiser.

### Configuration

All hyperparameters flow through dataclasses in `src/settings.py` (`DataConfig`, `PreprocessingConfig`, `ModelConfig`, `ArchitectureConfig`). Species-specific presets live in `src/config_species.py` — currently three species: `"gibbon"`, `"thyolo"`, `"ptw"`. Hard-coded base paths in that file reference an `E:/` drive; update these for your environment.

### Expected Directory Layout

```
<species_root>/
├── Audio/                        # raw .wav files
├── Annotations/                  # .svl XML files
├── DataFiles/train.txt, test.txt, validation.txt
├── Compressed_Audio/
│   ├── cs_0.15/                  # CS measurement vectors (.npy)
│   ├── cs_reconstructed_0.15/    # reconstructed .wav files
│   └── {codec}_{bitrate}/        # codec-compressed .wav files
├── Saved_Data/                   # cached numpy datasets
└── Predictions/                  # output .svl files
```

### Key Design Notes

- Segment duration, sample rate, mel-spectrogram parameters, and CNN architecture all vary per species — always trace settings through the species config rather than assuming defaults.
- `metrics.py` is empty and reserved for future use.
- Several legacy/commented-out functions exist in `compress.py` (e.g., `compress_one_file_legacy`) — do not remove them without confirming they are truly dead.
-Several legacy/commented-out functions are the originals functions that we want to keep while the associated ones were created by codex to improve the original ones. We want to figure out which one is better or even improve these functions. 
- The CNN's `_get_min_input_size()` method is used to validate spectrogram dimensions before training; if architecture parameters change, re-check that input shapes are compatible.
