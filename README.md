
# Edge Ecoacoustics

Ecoacoustic monitoring at the edge using an ESP32-S3, streaming audio processing and a compact CNN.

## Overview

The system captures environmental audio and estimates three continuous ecoacoustic indices:

- **Bio:** biophony
- **Geo:** geophony
- **Antro:** anthropophony, including technophony

The indices are independent values in [0, 1]. They are not calibrated probabilities and are not constrained to sum to 1.

The current firmware processes audio continuously and produces one set of indices for each non-overlapping six-second window.

## System architecture

```text
I2S microphone (48 kHz)
        |
      DMA
        |
  FreeRTOS ring buffer
        |
 Streaming Log-Mel DSP
        |
  CNN float32 / TFLite Micro
        |
 BIO / GEO / ANTRO
        |
   Wi-Fi + MQTT/TLS
        |
     EMQX Cloud
        |
 Telegraf -> InfluxDB -> Grafana
```

The ESP32-S3 acquisition, inference and MQTT publication stages have been validated on hardware. Telegraf, InfluxDB and Grafana are the next integration stage and are not yet operational.

## Dataset

The dataset contains 5,000 terrestrial recordings, each contributing one 60-second interval divided into ten non-overlapping six-second windows, for 50,000 labeled examples.

Original recordings are stereo WAV files sampled at 48 kHz. The right channel is used for training-data preparation and pseudo-label generation.

The dataset selection covers 30 recording sites and 24 hours of the day.

Main files:

- `data/recordings.csv`: selected recording manifest.
- `data/recordings_clipping.csv`: clipping audit.
- `data/cei_6s.csv`: CLAP-derived labels for the 50,000 windows.

Original WAV files are not stored in Git.

## Pseudo-label generation

Pseudo-labels are derived from CLAP 2023 using prompts adapted from the Contrastive Ecoacoustic Indices methodology of Teytaut et al. (2026).

For each category, similarity scores are rescaled using:

```python
q = clip((similarity - 5) / 15, 0, 1)
index = 0.75 * top1 + 0.25 * top2
```

Anthropophony and technophony prompts are combined into the Antro category.

The three indices are independently computed for each six-second window. No temporal aggregation is performed on the device.

## Student model

The selected model is `cnn_logmel`:

- Input: 299 Log-Mel frames x 40 Mel bands.
- Output: three float32 indices (`bio`, `geo`, `antro`).
- Parameters: 5,403.
- TFLite model: 25,540 bytes.
- Validation macro MAE: 0.068134 against CLAP-derived pseudo-labels.

The ESP32-S3 uses TensorFlow Lite Micro, with the tensor arena allocated in PSRAM.

Observed on-device inference latency is approximately 490 ms per six-second window.

See `model/README.md` and `model/TRAINING.md`.

## ESP32-S3 firmware

Hardware:

- ESP32-S3 with 16 MB Flash and 8 MB PSRAM.
- I2S microphone at 48 kHz.
- BCLK: GPIO 4.
- WS: GPIO 5.
- DIN: GPIO 6.
- Left I2S slot.

Firmware:

- ESP-IDF 6.0.2.
- Continuous I2S acquisition through DMA.
- FreeRTOS ring buffer.
- Streaming Log-Mel feature extraction.
- CNN float32 inference every six seconds.
- Wi-Fi connectivity and NTP clock synchronization.
- MQTT/TLS publication to EMQX Cloud.

Each MQTT message includes the three indices, a UTC timestamp, sequence number, RMS, inference duration and error counters.

The device does not currently use deep sleep or periodic wake/sleep acquisition cycles.

See `firmware/esp32s3/README.md` for build and configuration instructions.

## Cloud communication

Broker: EMQX Cloud Serverless.

MQTT topic:

```text
eco/esp32s3_01/indices
```

Transport: MQTT over TLS, port 8883, QoS 1.

The ESP32-S3 publishes one JSON message approximately every six seconds. Connectivity has been verified using Mosquitto clients on Windows.

The remaining monitoring stack will use Telegraf, InfluxDB and Grafana on Windows, without Docker.

## Project structure

```text
data/                  Dataset manifests and pseudo-labels
dataset/               Recording selection and CLAP labeling
firmware/esp32s3/      ESP32-S3 firmware and build tools
model/                 Feature extraction, training and export
results/               Figures and experiment results
tests/                 Python tests
```

## Reproducibility

Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Tests:

```bash
python -m unittest discover -s tests -v
```

Model and firmware artifacts are intentionally not all stored in Git. See the firmware README for required local files.

## References

- [Teytaut et al. (2026), Contrastive Ecoacoustic Indices](https://doi.org/10.1111/2041-210x.70369)
- [Microsoft CLAP](https://github.com/microsoft/CLAP)
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/)
