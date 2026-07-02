# Bioacoustic EnCodec — Gibbon pilot (numpy arrays edition)

Adapts pretrained EnCodec 24 kHz to compress gibbon vocalisations,
starting from pre-windowed numpy arrays on a single consumer GPU (<16 GB).

## File structure

```
bioacoustic_encodec/
├── dataset.py      site-split dataset, GPU resampling, weighted sampler
├── model.py        staged-unfreezing EnCodec wrapper + bioacoustic loss
├── train.py        full 3-stage training loop, checkpointing
├── evaluate.py     codebook monitor, F0 metrics, CNN dataset builder
├── quickstart.py   synthetic-data demo — run this first
└── requirements.txt
```

## Input format

```python
X.npy    : float32  (N, T)    # audio windows at 9600 Hz, range [-1, 1]
Y.npy    : int      (N,)      # 1 = gibbon call present, 0 = background
sites.npy: str/int  (N,)      # site ID per window, e.g. "SiteA"
```

Resampling from 9600 → 24 000 Hz happens inside the training loop on the
GPU. Nothing larger than the native-rate arrays ever lives on disk.

## Site naming

Site IDs can be any string or integer. The split is configured in `train.py`:

```python
cfg["codec_sites"] = ["SiteA", "SiteB", "SiteC"]   # ~60%
cfg["cnn_sites"]   = ["SiteD", "SiteE"]             # ~20%
cfg["test_sites"]  = ["SiteF", "SiteG"]             # ~20%  ← never touch until end
```

## Quickstart

```bash
pip install -r requirements.txt
python quickstart.py          # synthetic data demo, verifies everything works
```

Then with your real data:

```bash
python train.py --x X.npy --y Y.npy --sites sites.npy
```

## Stage schedule

Controlled by `stage_schedule` in the config dict:

| Stage | What trains             | Trainable params | When to advance         |
|-------|-------------------------|------------------|-------------------------|
| 1     | RVQ only                | ~4 M             | Start here, always      |
| 2     | RVQ + upper encoder     | ~12 M            | After val_loss plateaus |
| 3     | Full model              | ~75 M            | After stage 2 plateaus  |

Default schedule: `{0: 1, 15: 2, 35: 3}` (epochs).

## Memory tips for <16 GB GPUs

| Technique              | Saving   | Where              |
|------------------------|----------|--------------------|
| Mixed precision (AMP)  | ~40%     | auto, always on    |
| Gradient checkpointing | ~30%     | auto in stage 3    |
| Batch size 4 → 2       | ~50%     | `cfg["batch_size"]`|
| `num_workers=0`        | avoids   | if DataLoader OOMs |

If you hit OOM in stage 3, reduce batch size to 2.

## After training: CNN dataset

After training completes, `build_cnn_dataset()` is called automatically.
It saves:

```
cnn_data/cnn_X_recon.npy   (N_cnn,  T_24k)   # codec-compressed CNN training audio
cnn_data/cnn_Y.npy         (N_cnn,)           # original labels — unchanged
cnn_data/test_X_recon.npy  (N_test, T_24k)   # codec-compressed test audio
cnn_data/test_Y.npy        (N_test,)          # original labels — unchanged
```

Train your CNN detector on `cnn_*` files, evaluate on `test_*` files.
The test labels were never seen by the codec — no annotation bias.

## Codebook collapse warning

Each epoch prints per-quantizer utilisation:
```
Codebook util — Q0: 84% | Q1: 71% | Q2: 45% | Q3: 38%
```
If any quantizer drops below **30%**, collapse is starting. Remedies:
1. Lower `bandwidth` (fewer active quantizers)
2. Reduce learning rate by 5×
3. Add codebook EMA reset (see EnCodec paper Appendix B)

## Key metric to watch

`f0_error_semitones` — the median absolute error in fundamental frequency
between original and reconstructed call windows, measured in semitones.

- < 0.5 st  : excellent (sub-perceptual for most listeners)
- 0.5–1.5 st: good (FM trajectory largely preserved)
- 1.5–3.0 st: marginal (species detection may still work)
- > 3.0 st  : poor (call structure significantly distorted)
