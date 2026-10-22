# ESP32-S3 firmware — stage 1

This firmware builds with ESP-IDF 6.0.2 and implements **48-kHz I²S + DMA + FreeRTOS ring buffer** using the **right I²S slot**. It reads exactly 60 s in 20-ms blocks, measures losses and logs RMS. It does **not** yet calculate MFCC/Log-Mel, run TFLite Micro, publish indices, or enter deep sleep.

## Windows / OneDrive build

From the repository root, press **Ctrl+Shift+B** in VS Code and run
**Build ESP32-S3 firmware**, or run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File firmware/esp32s3/build.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File firmware/esp32s3/build.ps1 -Action CheckChip -Port COM10
```

The script uses the installed ESP-IDF 6.0.2 tools and sets `IDF_TARGET=esp32s3`.
It synchronizes the firmware sources to `C:/esp/edge-ecoacoustics-build/source`
and builds in `C:/esp/edge-ecoacoustics-build/build`. These stable paths avoid
the spaces and accented characters in the OneDrive path. Edit the files in
this repository; `main/` is mirrored before every build, including removal
of obsolete staged files. Generated `sdkconfig` and build outputs remain in
the staging directory. The build task does not flash firmware.

Open the repository root in VS Code. The standard **ESP-IDF Build**
command uses the same staging setup: `idf.preBuildTask`
synchronizes the source files, `idf.cmakeCompilerArgs` selects the staged
project, and `idf.buildPath` selects its build folder. Reload the VS Code
window after changing these settings. The firmware folder also has settings
for opening it directly. Its `.clangd` configuration is inherited from the
repository root.

## Microphone setup

Before building, edit `MIC_BCLK_GPIO`, `MIC_WS_GPIO`, `MIC_DIN_GPIO` in `main/main.c` to match your microphone wiring. Current 4/5/6 are **placeholders**. This code assumes 32-bit Philips I²S stereo slot framing and the right slot as `raw[2*i+1]`. Confirm microphone output format, L/R selection, bit alignment and actual sample rate (some mics support 48 kHz but others do not). Current interim 16-kHz sensor needs a separate resampler; do not use it at 48 kHz if unsupported.

`build.ps1` initializes the installed ESP-IDF tools; an ESP-IDF terminal is not required for the Windows commands above. Firmware execution is **not verified on the real ESP32-S3**. A valid capture test records all 3000 blocks with 0 overflows and 0 read errors. `RING_BYTES` reserves about 96 KiB of internal heap; adjust after memory profiling. Avoid arbitrary pin assumptions.

## Planned next steps

1. Port `model/features.py` DSP exactly: 1200-sample Hann, 960 hop, right-zero-padded FFT2048, HTK Mel 40 bins, Log-Mel natural logarithm, 299 frames per six seconds; validate against Python.
2. Export shortlisted `cnn_logmel/model.keras` with its `normalization.npz` to TFLite INT8; verify operators and output agreement, then add TFLite Micro. The predictions CSV alone **cannot** reconstruct the network weights.
3. Ten 6-second inferences; 60-second wake/transmit/deep sleep cycle. Later validate temporary 16->48 kHz microphone testing separately.
