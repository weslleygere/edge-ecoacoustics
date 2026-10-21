# Student model — 48-kHz input

The student will predict three independent continuous values: `bio`, `geo`, and
`antro`. Inputs and labels are aligned by `audio_id`, `segment_index` and
`start_s`; one recording yields ten non-overlapping 6-second examples.

## Prepare features

Run from the repository root, with the original 48-kHz WAVs accessible via the
paths listed in `data/recordings_1000.csv`:

```bash
python -m unittest discover -s tests -v   # check offline vs 20-ms streaming
python model/prepare_data.py --limit 10 --out data/processed_smoke
python model/prepare_data.py              # all 10,000 windows
```

If the archive is mounted somewhere other than `/escutado-audios`, use:

```bash
python model/prepare_data.py --audio-root /path/to/new/audio/root
```

This replaces the recorded prefix `/escutado-audios` with your new root.

The full feature cache under `data/processed/` contains:

| File | Description |
| --- | --- |
| `logmel.npy` | `[N, 299, 40]` float32, ~479 MB at N=10,000 |
| `mfcc.npy` | `[N, 299, 13]` float32, ~156 MB at N=10,000 |
| `metadata.csv` | Row-aligned labels, original IDs, sites, and train/val/test split |
| `feature_config.json` | Exact DSP specification for later firmware implementation |

The large `.npy` files are intentionally ignored by Git. You can open either
array without loading it all into RAM:

```python
import numpy as np
import pandas as pd

x = np.load('data/processed/logmel.npy', mmap_mode='r')
meta = pd.read_csv('data/processed/metadata.csv')
y = meta[['bio', 'geo', 'antro']].to_numpy(dtype='float32')
train_rows = (meta['split'] == 'train').to_numpy()
```

## Audio/DSP contract

- Sample rate: **48,000 Hz mono** (original stereo WAVs are averaged to mono).
- Segment: **6 seconds = 288,000 samples**, independently framed.
- Window: **1,200 samples (25 ms)**, periodic Hann.
- Hop: **960 samples (20 ms)**, yielding **299 frames** per segment.
- FFT: **2,048-point RFFT** with zero padding **to the right**.
- Power spectrum: `abs(rfft)**2 / 2048`.
- Mel: **40 unit-height triangular filters**, HTK frequency scale,
  from 60 to 20,000 Hz, applied to power.
- Log-Mel: natural logarithm of Mel energy, floored at `1e-10`.
- MFCC: **13** coefficients (including C0), orthonormal DCT-II of Log-Mel.
- **No centered frames or implicit edge padding.**
- No feature standardization yet: train-set-only normalization belongs to
  the later training pipeline, and will be exported with the model.

`model/features.py` implements both vectorized extraction and a
`StreamingFeatures` reference, which accepts successive 20-ms PCM chunks
and retains only the small unconsumed tail. Its tests verify the two paths
produce the same 299 frames and values within floating-point tolerances.

The future ESP-IDF implementation will use I²S DMA + ring buffer, and reproduce
this contract without buffering an entire 6-second PCM waveform. Mel filter
coefficients can be stored compactly in Flash. A provisional 16-kHz input can
be resampled to 48 kHz **before** feeding the identical feature pipeline.

## Data split

A deterministic seed (`42`) assigns approximately 70/15/15 of recordings to
training, validation and test, grouping recordings by **station and calendar
date** to avoid leaking near-neighbor recordings between splits. The exact
assigned set for every window is in `metadata.csv`; the percentages are only
approximate because groups must remain intact.

## Next step

Build and compare MLP, compact CNN, and DS-CNN models for both feature types,
with three continuous outputs; then evaluate INT8 deployment on ESP32-S3.
