
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include <stdbool.h>

#define ECO_SAMPLE_RATE       48000
#define ECO_BLOCK_SAMPLES     960
#define ECO_FRAME_LENGTH      1200
#define ECO_HOP_LENGTH        960
#define ECO_FFT_LENGTH        2048
#define ECO_MEL_BANDS         40
#define ECO_SEGMENT_SAMPLES   288000
#define ECO_FRAMES_PER_SEGMENT 299

typedef struct {
    float pcm[ECO_FRAME_LENGTH];
    float logmel[ECO_FRAMES_PER_SEGMENT][ECO_MEL_BANDS];

    size_t buffered;
    size_t samples_received;
    size_t frames_generated;
} eco_features_t;

esp_err_t eco_features_init(void);

void eco_features_reset(eco_features_t *ctx);

esp_err_t eco_features_push(
    eco_features_t *ctx,
    const int16_t *samples,
    size_t count
);

bool eco_features_complete(const eco_features_t *ctx);
