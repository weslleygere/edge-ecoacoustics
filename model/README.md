
# Student model and audio features

This directory contains feature extraction, training and export code for the ecoacoustic student model.

## Selected deployment

The selected model is a compact CNN trained on Log-Mel features:

- Model: `cnn_logmel`.
- Input shape: `[1, 299, 40, 1]`.
- Output shape: `[1, 3]`.
- Precision: float32.
- Parameters: 5,403.
- Validation macro MAE: 0.068134.
- TFLite size: 25,540 bytes.

Output order:

```text
bio, geo, antro
```

The outputs are independent continuous indices in [0, 1], not probabilities constrained to sum to one.

The ESP32-S3 firmware implements the selected DSP pipeline and runs the exported CNN using TensorFlow Lite Micro.

## Dataset

The dataset contains 50,000 six-second windows derived from 5,000 recordings.

Original recordings are stereo WAV files sampled at 48 kHz. The right channel is used for offline feature preparation.

Each recording contributes ten non-overlapping six-second windows.

## Feature extraction contract

- Sample rate: 48,000 Hz mono.
- Segment duration: 6 seconds (288,000 samples).
- Frame length: 1,200 samples (25 ms).
- Hop length: 960 samples (20 ms).
- Window function: periodic Hann.
- FFT size: 2,048, right-zero-padded.
- Power spectrum: `abs(rfft)**2 / 2048`.
- Mel filters: 40 unit-height triangular filters.
- Mel scale: HTK.
- Frequency range: 60 to 20,000 Hz.
- Log-Mel: natural logarithm with floor at `1e-10`.
- Frames per segment: 299.
- MFCC baseline: 13 coefficients using orthonormal DCT-II.

Frames are not centered and no implicit edge padding is applied.

The same DSP contract is implemented in Python and in the ESP32-S3 firmware.

## Preparing features

From the repository root, with original WAV files accessible:

```bash
python -m unittest discover -s tests -v
python model/prepare_data.py --limit 10 --out data/processed_smoke
python model/prepare_data.py
```

If the audio archive is mounted at a different location:

```bash
python model/prepare_data.py --audio-root /path/to/audio/root
```

The generated feature cache contains:

| File | Content |
| --- | --- |
| `logmel.npy` | `[N, 299, 40]` float32 |
| `mfcc.npy` | `[N, 299, 13]` float32 |
| `metadata.csv` | Labels, IDs, sites and data splits |
| `feature_config.json` | DSP configuration |

Generated feature arrays are intentionally excluded from Git.

## Normalization

Neural model normalization is calculated using training data only.

The ESP32-S3 applies per-Mel-band normalization:

```python
normalized = (logmel - mean) / scale
```

The deployed normalization artifact contains 40 float32 means followed by 40 float32 scales.

## Training and evaluation

Training compares Ridge, MLP, CNN and DS-CNN architectures using MFCC and Log-Mel features.

The selected CNN was evaluated against CLAP-derived pseudo-labels, not independent ecological ground truth.

See `TRAINING.md` for experiments, validation and deployment details.

## Firmware

The ESP32-S3 implementation is available in:

```text
firmware/esp32s3/
```

The firmware processes incoming PCM continuously, extracts streaming Log-Mel features and executes one CNN inference per six-second segment.

The device has been tested with real microphone input and publishes its predictions over MQTT/TLS.
