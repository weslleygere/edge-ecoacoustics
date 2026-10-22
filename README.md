# Edge Ecoacoustics

**Ecoacoustic monitoring at the edge using an ESP32-S3 and a compact student model.**

## Dataset

The current dataset contains **5,000 terrestrial recordings**, with one 60-second interval per recording and ten non-overlapping 6-second windows each (**50,000 labeled windows**). Original recordings are stereo WAVs sampled at 48 kHz. Only the **right channel** is used for pseudo-label generation and student features.

Recording candidates are balanced across the **30 sites** and **24 hours**, with randomized 60-second offsets. Clipping is screened on the right channel; silence is retained. The identifiers for recording stations are not used for sampling or splitting.

- `data/recordings.csv`: selected 5,000-recording manifest.
- `data/recordings_clipping.csv`: clipping audit of examined candidates.
- `data/cei_6s.csv`: 50,000 CLAP-derived labels (`bio`, `geo`, `antro`).

The 48-kHz original WAVs are stored outside Git and referenced via `source_path` in the manifest.

## Pipeline

1. **Select recordings:** `python dataset/select_recordings.py --metadata /path/to/audio_quality_integrated.csv` (requires original WAVs and reproduces the selection only with the same source inventory).
2. **Generate labels:** `python dataset/generate_labels.py` (CLAP 2023, right-channel only, Continuous Top-2).
3. **Visualize:** `python dataset/visualize.py` (figures and summary tables in `results/figures/`).
4. **Prepare features:** `python model/prepare_data.py` (48-kHz MFCC/Log-Mel; output in `data/processed/`).
5. **Train models:** `python model/train.py --model all --feature all` (Ridge, MLP, CNN, DS-CNN × MFCC/Log-Mel; outputs in `results/training/`).

For dependencies: `python -m pip install -r requirements.txt`. Tests: `python -m unittest discover -s tests -v`.

The three targets are independent continuous indices in [0,1], **not calibrated probabilities** and not constrained to sum to 1. The teacher uses prompts in `dataset/prompts.txt`, originating from Teytaut et al. (2026), and pools anthropophony and technophony prompts together. Labels use `q = clip((similarity-5)/15, 0, 1)` followed by `0.75*top1 + 0.25*top2` for each category.

## Model and firmware

MFCC and Log-Mel features are generated from six-second frames at 48 kHz. The CNN + Log-Mel pipeline is the selected initial deployment candidate. Planned ESP32-S3 firmware uses I²S + DMA + ring buffer, streaming Log-Mel extraction, one inference per 6-second window, transmission after 60 seconds of acquisition, and deep sleep between ten-minute cycles.

See [`model/README.md`](model/README.md) and [`model/TRAINING.md`](model/TRAINING.md).

## References

- [Teytaut et al. (2026), Contrastive Ecoacoustic Indices](https://doi.org/10.1111/2041-210x.70369)
- [Microsoft CLAP](https://github.com/microsoft/CLAP)
