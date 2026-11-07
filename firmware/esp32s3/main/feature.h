#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

// Exact six-second input shape used by tinyCLAP / torchlibrosa.
#define ECO_SAMPLE_RATE          44100
#define ECO_SEGMENT_SECONDS      6
#define ECO_SEGMENT_SAMPLES      (ECO_SAMPLE_RATE * ECO_SEGMENT_SECONDS)
#define ECO_BLOCK_SAMPLES        882       // 20 ms at 44.1 kHz
#define ECO_FFT_LENGTH           1024
#define ECO_HOP_LENGTH           320
#define ECO_MEL_BANDS            64
#define ECO_FRAMES_PER_SEGMENT   (ECO_SEGMENT_SAMPLES / ECO_HOP_LENGTH + 1)

typedef struct {
    // time-major, exactly [827, 64], as required by the TFLite input
    float logmel[ECO_FRAMES_PER_SEGMENT][ECO_MEL_BANDS];
} eco_features_t;

// Initialize ESP-DSP FFT and verify precomputed Slaney mel matrix.
esp_err_t eco_features_init(void);

// Compute 827 centered/reflect-padded log-mel frames from 6 s of PCM16.
esp_err_t eco_features_compute(
    const int16_t *pcm,
    size_t samples,
    eco_features_t *output
);
