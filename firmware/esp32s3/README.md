
# ESP32-S3 firmware

Continuous ecoacoustic monitoring with ESP-IDF 6.0.2.

## Hardware

- ESP32-S3 N16R8: 16 MB Flash, 8 MB octal PSRAM.
- I2S microphone: 48 kHz, 32-bit stereo slots.
- BCLK: GPIO 4.
- WS: GPIO 5.
- DIN: GPIO 6.
- Active microphone data: left I2S slot.

The firmware extracts 16-bit PCM from the high 16 bits of each left-slot sample.

## Processing pipeline

```text
I2S -> DMA -> Ring Buffer -> Log-Mel -> CNN float32 -> MQTT/TLS
```

Audio acquisition is continuous.

- Audio block: 960 samples (20 ms).
- Segment: 288,000 samples (6 s).
- Feature representation: 299 x 40 Log-Mel.
- CNN inference: once per segment.
- MQTT publication: once per segment.

Acquisition, DSP and network communication run in separate FreeRTOS tasks.

## Model

Selected student: `cnn_logmel`.

- 5,403 parameters.
- Float32 TFLite model: 25,540 bytes.
- Tensor arena: 256 KiB allocated in PSRAM.
- Actual arena usage: approximately 146 KiB.
- Inference latency: approximately 490 ms.

The model outputs three independent indices in this order:

```text
bio, geo, antro
```

These values are continuous ecoacoustic indices, not class probabilities.

## Required local artifacts

The following files must exist before building:

```text
firmware/esp32s3/
    main/
        emqxsl-ca.crt
    model/
        cnn_logmel_float32.tflite
        normalization.bin
```

The TFLite file contains the trained CNN weights.

`normalization.bin` stores 40 means followed by 40 scale values, as float32 values.

`emqxsl-ca.crt` contains the public CA certificate used to verify the EMQX Cloud server.

These artifacts are excluded from Git and must be supplied locally. Do not place Wi-Fi or MQTT passwords in versioned files.

## Dependencies

The component manifest `main/idf_component.yml` defines:

- `espressif/mqtt`
- `espressif/esp-dsp`
- `espressif/esp-tflite-micro`

The ESP-IDF Component Manager resolves these dependencies during configuration and build.

## Windows / OneDrive build

The repository may be located under OneDrive, where spaces and accented characters in directory names can interfere with ESP-IDF tools.

The existing `build.ps1` script mirrors the firmware into an ASCII-only staging directory:

```text
C:\esp\edge-ecoacoustics-build\source
```

Build outputs are generated in:

```text
C:\esp\edge-ecoacoustics-build\build
```

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File firmware/esp32s3/build.ps1
```

The script synchronizes the source files and runs the build. The build does not flash the device.

The staged source contains its own generated `sdkconfig`, which preserves locally configured options across regular builds.

## Local configuration

Use the ESP-IDF terminal to open the staged firmware project:

```powershell
cd C:\esp\edge-ecoacoustics-build\source
idf.py -B ..\build menuconfig
```

Configure the following options under `Ecoacoustics Network`:

- Wi-Fi SSID.
- Wi-Fi password.
- EMQX MQTT password.

The MQTT username is currently `eco`.

The firmware connects to:

```text
x709811d.ala.eu-central-1.emqxsl.com:8883
```

MQTT transport uses TLS and the bundled CA certificate.

The current firmware uses fixed Wi-Fi credentials. Changing to a different SSID or password requires reconfiguring and rebuilding.

## Memory and partition configuration

The project targets:

- 16 MB Flash.
- 8 MB octal PSRAM at 80 MHz.
- Large factory application partition.
- External PSRAM allocation for the mbedTLS connection.

The mbedTLS memory allocation strategy must be configured to use external SPIRAM, as validated during the Wi-Fi/MQTT integration.

Without this setting, the TLS connection previously failed because internal heap memory was insufficient.

The firmware currently does not use OTA updates.

## MQTT telemetry

Topic:

```text
eco/esp32s3_01/indices
```

QoS: 1.

Messages are JSON objects containing:

- `device`: device identifier.
- `ts_ms`: UTC Unix timestamp in milliseconds.
- `seq`: inference sequence number.
- `bio`, `geo`, `antro`: CNN outputs.
- `rms`: audio RMS.
- `inference_ms`: inference duration.
- `errors`: I2S error counter.
- `overflows`: ring-buffer overflow counter.
- `dropped`: telemetry drop counter.

The network module uses a FreeRTOS queue to avoid blocking the DSP task.

The queue stores up to 32 measurements, corresponding to approximately 3.2 minutes of data. Long network outages may result in dropped measurements.

QoS 1 permits duplicate delivery. The downstream database should account for this when analyzing the time series.

## Verified hardware behavior

The following stages have been validated on the ESP32-S3:

- Continuous I2S acquisition.
- Log-Mel extraction with 299 frames per segment.
- CNN float32 inference.
- Wi-Fi connection.
- NTP synchronization.
- MQTT/TLS connection to EMQX Cloud.
- JSON publication approximately every six seconds.
- Message reception on Windows using Mosquitto.

The initial hardware tests reported zero I2S errors, zero ring-buffer overflows and zero telemetry drops.

Long-duration stability over 24 hours remains to be evaluated.

## Current limitations

- Wi-Fi credentials are fixed at build time.
- Telemetry buffering is limited to 32 measurements.
- MQTT delivery is not persisted across power failures.
- No OTA update mechanism.
- No deep sleep.
- The 24-hour unattended deployment has not yet been validated.
