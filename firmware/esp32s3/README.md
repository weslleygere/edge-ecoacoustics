
# ESP32-S3 firmware

ESP-IDF firmware for real-time ecoacoustic monitoring with tinyCLAP INT8.

## Hardware

The validated hardware configuration is:

- ESP32-S3 N16R8.
- 16 MB Flash.
- 8 MB octal PSRAM.
- I2S microphone with 32-bit stereo slots.
- Active microphone channel: left I2S slot.

GPIO configuration:

| Signal | GPIO |
|--------|------|
| BCLK   | 4    |
| WS     | 5    |
| DIN    | 6    |

The ESP32-S3 captures at 44,100 Hz and extracts signed 16-bit PCM from the upper 16 bits of each 32-bit left-slot sample.

## Processing pipeline

```text
I2S microphone
      |
6-second PCM capture
      |
Log-Mel frontend
    827 x 64
      |
PhiNet INT8
      |
1024-D audio embedding
      |
41 text embeddings
      |
Cosine similarity
      |
Bio / Geo / Antro
      |
MQTT/TLS
```

Audio acquisition and processing are sequential.

For every measurement:

1. Enable I2S and capture six seconds of audio.
2. Disable I2S.
3. Compute the Log-Mel spectrogram.
4. Execute PhiNet INT8 inference.
5. Compute prompt similarities and ecoacoustic indices.
6. Queue the measurement for MQTT publication.
7. Begin the next capture.

There is no continuous acquisition ring buffer.

The network runs in a separate FreeRTOS task, allowing telemetry to be transmitted independently of audio processing.

## Audio frontend

The frontend is implemented in `main/feature.c`.

Parameters:

| Parameter | Value |
|-----------|-------|
| Sample rate | 44,100 Hz |
| Segment duration | 6 s |
| Samples per segment | 264,600 |
| Audio block | 882 samples (20 ms) |
| FFT length | 1,024 |
| Hop length | 320 |
| Window | Periodic Hann |
| Center padding | Reflect |
| Mel bands | 64 |
| Mel frequency range | 50–14,000 Hz |
| Mel normalization | Slaney |
| Output shape | 827 x 64 |

The spectrogram uses squared FFT magnitudes followed by Mel filtering and logarithmic conversion to decibels.

The Mel filter matrix is generated offline by `model/package_firmware.py` and stored in Flash.

## tinyCLAP inference

The audio encoder is a PhiNet model exported to fully quantized INT8 TFLite.

Model interface:

```text
Input:  int8 [1, 827, 64]
Output: int8 [1, 1024]
```

The firmware runs inference with TensorFlow Lite Micro.

The tensor arena reserves 6 MiB in PSRAM.

The model uses 12 supported TFLite operator types:

```text
ADD
CONV_2D
DEPTHWISE_CONV_2D
FULLY_CONNECTED
LOGISTIC
MEAN
MUL
PAD
RESHAPE
RSQRT
SQUARED_DIFFERENCE
SUB
```

The firmware dequantizes the output embedding and calculates cosine similarities against 41 precomputed CLAP text embeddings.

For each prompt:

```python
score = clip((cosine * logit_scale - 5) / 15, 0, 1)
```

For each category:

```python
index = 0.75 * top1 + 0.25 * top2
```

The resulting indices are:

- `bio`: biophony.
- `geo`: geophony.
- `antro`: anthropophony and technophony.

The indices are independent values in [0, 1] and are not calibrated probabilities.

The firmware also reports the three most similar prompts per category.

## Required local artifacts

The firmware requires these files:

```text
firmware/esp32s3/
├── main/
│   └── emqxsl-ca.crt
└── model/
    ├── tinyclap_phinet_int8.tflite
    ├── tinyclap_text_assets.bin
    └── tinyclap_mel_filters.bin
```

Their roles are:

| File | Purpose |
|------|---------|
| `tinyclap_phinet_int8.tflite` | Quantized audio encoder |
| `tinyclap_text_assets.bin` | Text embeddings, category masks, and logit scale |
| `tinyclap_mel_filters.bin` | Slaney Mel filter matrix |
| `emqxsl-ca.crt` | CA certificate for MQTT/TLS server verification |

The model assets are generated or copied by:

```powershell
python model/package_firmware.py --install
```

Run this command from the repository root after generating the model and text embeddings.

The public CA certificate must correspond to the configured MQTT broker.

Model binaries and certificates are not automatically downloaded during compilation.

## Dependencies

The firmware was validated with ESP-IDF 6.0.2.

The component manifest `main/idf_component.yml` declares:

- `espressif/mqtt`
- `espressif/esp-dsp`
- `espressif/esp-tflite-micro`

The ESP-IDF Component Manager resolves these dependencies during project configuration.

## Building on Windows

The firmware was developed on Windows, with the repository stored under OneDrive.

Because ESP-IDF build tools may have problems with accented characters and spaces in paths, `build.ps1` copies the firmware into an ASCII-only staging directory.

Default locations:

```text
C:\esp\edge-ecoacoustics-build\source
C:\esp\edge-ecoacoustics-build\build
```

The script copies:

- Firmware source files.
- ESP-IDF CMake configuration.
- `sdkconfig.defaults`.
- `partitions.csv`.
- The three tinyCLAP model assets.

### Prepare the staging directory

From the repository root:

```powershell
.\firmware\esp32s3\build.ps1 -Action Prepare
```

This copies the necessary files without compiling the firmware.

### Configure Wi-Fi and MQTT

Open an ESP-IDF 6.0.2 terminal and run:

```powershell
cd C:\esp\edge-ecoacoustics-build\source
idf.py -B ..\build menuconfig
```

Under `Ecoacoustics Network`, configure:

- Wi-Fi SSID.
- Wi-Fi password.
- MQTT password.

Save the configuration before building.

The resulting `sdkconfig` is stored in the staging directory and should not be committed to Git.

The MQTT broker URI, topic, and username are currently defined in `main/network.c`.

To use another broker or device identifier, update those values and provide the appropriate CA certificate.

### Compile

Return to the repository root and run:

```powershell
.\firmware\esp32s3\build.ps1
```

The script configures and builds the firmware.

Compilation does not automatically flash the device.

### Flash the ESP32-S3

From an ESP-IDF terminal:

```powershell
cd C:\esp\edge-ecoacoustics-build\source
idf.py -B ..\build -p COM10 flash monitor
```

Replace `COM10` with the serial port used by your ESP32-S3.

The firmware should initialize the model and Log-Mel frontend, connect to Wi-Fi, synchronize the clock, establish MQTT/TLS connectivity, and begin capturing audio.

## Memory and partitions

The project targets:

- 16 MB Flash.
- 8 MB octal PSRAM at 80 MHz.
- 240 MHz CPU frequency.
- External memory allocation for mbedTLS.
- A custom partition table with a 12 MB factory application partition.

The memory and partition settings are defined in:

```text
sdkconfig.defaults
partitions.csv
```

The large tensor arena and audio buffers are allocated in PSRAM.

External mbedTLS allocation is important because the project previously encountered internal heap limitations during TLS initialization.

The current firmware does not implement OTA updates.

## Network communication

The network implementation is located in `main/network.c`.

It provides:

- Wi-Fi station connectivity.
- Automatic reconnection after Wi-Fi disconnection.
- NTP time synchronization.
- MQTT over TLS.
- Telemetry buffering using a FreeRTOS queue.
- QoS 1 publication.

The current MQTT topic is:

```text
eco/esp32s3_01/indices
```

The default broker configuration uses EMQX Cloud Serverless.

The firmware must have a valid Wi-Fi connection and synchronized system clock before establishing its initial MQTT connection.

## MQTT telemetry

Each measurement is published as a JSON object.

Main fields:

| Field | Description |
|-------|-------------|
| `device` | Device identifier |
| `ts_ms` | UTC Unix timestamp in milliseconds |
| `seq` | Measurement sequence number |
| `bio` | Biophony index |
| `geo` | Geophony index |
| `antro` | Anthropophony/technophony index |
| `rms` | Audio RMS |
| `inference_ms` | PhiNet inference duration |
| `errors` | Acquisition and processing error counter |
| `overflows` | Reserved compatibility counter; currently zero |
| `dropped` | Dropped telemetry counter |

The message also contains nine groups of Top-3 prompt fields, with three prompts for each category.

Naming convention:

```text
bio_top1_id
bio_top1_cosine
bio_top1_score

geo_top1_id
geo_top1_cosine
geo_top1_score

antro_top1_id
antro_top1_cosine
antro_top1_score
```

The same naming pattern is used for `top2` and `top3`.

Prompt IDs are zero-based and correspond to the order of entries in `model/text/prompts.txt`.

The JSON field names and payload structure are preserved by the refactored firmware.

## Telemetry buffering

The network module uses a FreeRTOS queue with capacity for 32 measurements.

Audio processing does not wait for each MQTT publication.

If the queue becomes full, new telemetry may be dropped. The `dropped` counter records such losses and other publishing failures handled by the network module.

MQTT uses QoS 1, which permits duplicate message delivery. Downstream processing should take this into account.

Measurements are not persisted across power failures.

## Monitoring stack

The validated monitoring pipeline is:

```text
ESP32-S3
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

The system has been tested with real measurements published by the ESP32-S3 and visualized in Grafana.

The dashboard can display Bio, Geo, and Antro time series, together with prompt-level information.

## Verified behavior

The following components have been validated on the ESP32-S3:

- Six-second I2S acquisition.
- Log-Mel extraction with 827 frames and 64 bands.
- PhiNet INT8 inference.
- Computation of Bio, Geo, and Antro indices.
- Top-3 prompt scoring.
- Wi-Fi connectivity and NTP synchronization.
- MQTT/TLS publication.
- Reception and storage through EMQX, Telegraf, and InfluxDB.
- Visualization in Grafana.

The model-generation pipeline has also reproduced the validated TFLite artifact with an identical SHA-256 hash using the original calibration data.

## Current limitations

- Audio acquisition is paused during processing.
- Wi-Fi credentials are configured at build time.
- Broker configuration is currently defined in source code.
- Telemetry buffering is limited to 32 measurements.
- Data is not persisted across power failures.
- No OTA firmware update mechanism.
- No deep sleep or power optimization.
- Extended unattended stability testing remains to be completed.
