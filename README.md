# Edge Ecoacoustics

**Edge-based passive acoustic monitoring using an ESP32-S3 and a compact student model.**

The current dataset uses **1,000 terrestrial recordings** sampled from the Caatinga acoustic collection. A 60-second excerpt of each recording is partitioned into ten consecutive **6-second windows** (up to 10,000 training targets). CLAP 2023 serves as a teacher for three continuous indices: biophony (`bio`), geophony (`geo`), and a combined anthropophony/technophony (`antro`).

## Repository

```text
dataset/            Sampling, teacher label generation, visualizations, prompts
model/              Future student-model code
firmware/esp32s3/   Future embedded implementation
data/               Selected recording manifest and final labels
results/            Figures and model/hardware metrics
```

## Dataset

`data/recordings_1000.csv` is the **actual selected manifest**, containing IDs, original source paths, recording sites and stations, timestamps, offsets, and sample rates. It is committed so the same selected segments can be reused. Source WAVs are **not included**; they must exist at the paths recorded in `source_path` or the paths must be remapped to your local mount. Some files use absolute paths specific to the original server.

`dataset/select_recordings.py` can create a **new selection** from the full curated source metadata. It is not an exact re-enactment of the historical selection; use the committed CSV for reproducibility of this project.

### Install

Tested workflow uses a Python environment with CUDA-enabled PyTorch (GPU optional). Install compatible PyTorch for your platform first, then:

```bash
pip install -r requirements.txt
```

### Generate continuous labels

From the repository root:

```bash
python dataset/generate_labels.py --limit 10 --out /tmp/cei_smoke.csv  # smoke test
python dataset/generate_labels.py              # all 1,000 recordings
```

You may set `--cpu` or `--batch-size 16` as needed. The script reads each 60-second excerpt directly from the existing WAV, mixes channels to mono, extracts 6-second clips, and obtains CLAP text/audio similarities. Missing/unreadable audio is reported and skipped.

The final labels are produced with:

```text
q(s)    = clip((s - 5) / 15, 0, 1)
index_G = 0.75 * highest(q in G) + 0.25 * second_highest(q in G)
```

Prompts are in `dataset/prompts.txt`, originally drawn from [Teytaut et al., Contrastive Ecoacoustic Indices](https://doi.org/10.1111/2041-210x.70369). Anthropophony and technophony descriptions are pooled into the single `antro` target. **These are adapted, CLAP-derived pseudo-labels—not calibrated probabilities or physical sound-intensity measurements.** No baseline-comparison scripts are included in this final-project repository.

Expected output: `data/cei_6s.csv`, with `audio_id`, `source_path`, `segment_index`, `start_s`, `end_s`, `bio`, `geo`, and `antro` columns. Optional `--similarities-out` preserves CLAP scores outside version control. Validate that the output contains 10,000 rows before model training.

### Visualize

```bash
python dataset/visualize.py --csv data/cei_6s.csv --sample data/recordings_1000.csv
```

Figures and summary tables are written under the output directory selected by the visualization script (see `--help`).

The current verified figures and summary tables from the continuous-label run are included under `results/figures/` and `results/metrics/`. **The complete 10,000-row continuous-label CSV is not stored in the provided source branch or available attachments, so `data/cei_6s.csv` is intentionally not fabricated.** Copy your already generated CSV to that path, or recreate it using the generator.

## Roadmap

1. Implement embedded-compatible Log-Mel/MFCC preprocessing.
2. Train and evaluate a compact three-output regression model, splitting datasets by recording (not window).
3. Deploy to the ESP32-S3 with I²S DMA recording, inference, transmission, and deep sleep.

## References

- Teytaut et al. (2026), *Contrastive Ecoacoustic Indices*, [article](https://doi.org/10.1111/2041-210x.70369) and [code](https://github.com/ear-team/cei_package).
- [CLAP / msclap](https://github.com/microsoft/CLAP).
