# Edge Ecoacoustics

Ecoacoustic monitoring on an ESP32-S3 using **tinyCLAP**, a quantized INT8 audio encoder, and acoustic indices inspired by the **Contrastive Ecoacoustic Indices (CEI)** methodology.

## Overview

Edge Ecoacoustics captures environmental audio and estimates three indices:

- **Bio:** biophony (sounds produced by living organisms).
- **Geo:** geophony (sounds produced by natural abiotic sources).
- **Antro:** anthropophony and technophony (human and technological sounds).

The ESP32-S3 processes audio locally and sends the results through MQTT/TLS to EMQX, Telegraf, InfluxDB, and Grafana.

## Methodology

The project uses the pretrained **PhiNet audio encoder from [tinyCLAP](https://github.com/fpaissan/tinyCLAP)** (Paissan & Farella, 2024), converted to INT8 TFLite for execution on the ESP32-S3.

The indices are inspired by **[Contrastive Ecoacoustic Indices (CEI)](https://github.com/ear-team/cei_package)** (Teytaut et al., 2026), which uses CLAP audio-text similarities to characterize soundscapes.

Unlike the original CEI, this implementation:

- Combines anthropophony and technophony into one index (Antro).
- Uses a weighted combination of the two highest prompt scores per category.
- Calculates indices independently for each six-second segment, without temporal aggregation.

The resulting indices are therefore **CEI-inspired, not identical to the original CEI**.

### Index calculation

For each segment, the audio embedding is compared with 41 text embeddings generated using CLAP 2022.

Cosine similarities are scaled and converted to scores:

```python
score = clip((cosine * logit_scale - 5) / 15, 0, 1)
```

Each category index is calculated from its two highest scores:

```python
index = 0.75 * top1 + 0.25 * top2
```

The indices range from 0 to 1 and are not calibrated probabilities. The firmware also reports the three highest-ranked prompts for each category.

## Architecture

```text
I2S microphone (44.1 kHz)
           |
     6 s audio capture
           |
      Log-Mel (827x64)
           |
      PhiNet INT8
           |
   Audio embedding (1024)
           |
   41 text similarities
           |
     Bio / Geo / Antro
           |
       MQTT/TLS
           |
       EMQX Cloud
           |
       Telegraf
           |
       InfluxDB
           |
        Grafana
```

The firmware captures six seconds of audio, pauses acquisition during processing, and resumes capture after inference.

### Hardware

- ESP32-S3 N16R8 (16 MB Flash, 8 MB PSRAM).
- I2S microphone: BCLK GPIO 4, WS GPIO 5, DIN GPIO 6.
- ESP-IDF 6.0.2.
- TensorFlow Lite Micro with a 6 MiB tensor arena in PSRAM.

## Model generation

The pipeline uses three scripts in `model/`:

- `build_audio.py`: exports and quantizes the PhiNet audio encoder.
- `generate_text.py`: generates embeddings for the 41 text prompts.
- `package_firmware.py`: prepares the model assets for the firmware.

### Requirements

Python 3.12.10 and the dependencies listed in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the pretrained checkpoints and place them in `model/checkpoints/`:

```text
CLAP_weights_2022.pth
phinet_alpha_1.50_beta_0.75_t0_6_N_7.ckpt
```

The PhiNet checkpoint is available from the [tinyCLAP pretrained models](https://huggingface.co/fpaissan/tinyCLAP).

### Generate the model

```powershell
python model/build_audio.py --audio-dir "path/to/wavs"
```

Calibration uses stereo WAV recordings sampled at 48 kHz, with at least 60 seconds per recording. The script uses the right audio channel and selects 40 recordings by default.

### Generate text embeddings

```powershell
python model/generate_text.py
```

Prompts are defined in `model/text/prompts.txt`.

### Prepare firmware assets

```powershell
python model/package_firmware.py --install
```

This generates or installs the required firmware assets:

```text
firmware/esp32s3/model/
├── tinyclap_phinet_int8.tflite
├── tinyclap_text_assets.bin
└── tinyclap_mel_filters.bin
```

The model files and pretrained checkpoints are not included in Git.

## Firmware

The firmware requires ESP-IDF 6.0.2, the three model assets, and the MQTT/TLS CA certificate.

From the repository root:

```powershell
.\firmware\esp32s3\build.ps1 -Action Prepare
.\firmware\esp32s3\build.ps1
```

Wi-Fi and MQTT credentials must be configured locally using ESP-IDF `menuconfig`.

See [firmware/esp32s3/README.md](firmware/esp32s3/README.md) for build, configuration, and flashing instructions.

## Monitoring

The ESP32-S3 publishes measurements through MQTT/TLS with QoS 1.

**Topic:** `eco/esp32s3_01/indices`

Each message contains the Bio, Geo, and Antro indices, timestamp, RMS, inference time, error counters, and Top-3 prompt information.

The complete pipeline **ESP32-S3 → EMQX → Telegraf → InfluxDB → Grafana** has been tested with real device measurements.

## References

- Paissan, F., & Farella, E. (2024). [tinyCLAP: Distilling Contrastive Language-Audio Pretrained Models](https://doi.org/10.21437/Interspeech.2024-193). *Interspeech 2024*, 1685–1689. [Code](https://github.com/fpaissan/tinyCLAP).
- Teytaut, Y., et al. (2026). [Contrastive ecoacoustic indices: Large-scale global soundscape characterization with contrastive inference](https://doi.org/10.1111/2041-210x.70369). *Methods in Ecology and Evolution*. [Code](https://github.com/ear-team/cei_package).
- [Microsoft CLAP](https://github.com/microsoft/CLAP).
