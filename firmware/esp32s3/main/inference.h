
#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t eco_inference_init(void);
esp_err_t eco_inference_run(const float *logmel, float indices[3], int64_t *elapsed_us);

#ifdef __cplusplus
}
#endif
