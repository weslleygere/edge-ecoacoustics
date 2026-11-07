
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "inference.h"

typedef struct {
    int64_t end_us;
    uint32_t seq;
    float bio, geo, antro, rms, inference_ms;
    uint32_t errors, overflows;
    eco_prompt_rank_t top[ECO_PROMPT_GROUPS][ECO_TOP_K];
} eco_telemetry_t;

esp_err_t eco_network_init(void);
bool eco_network_publish(const eco_telemetry_t *data);
