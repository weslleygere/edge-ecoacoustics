#pragma once
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Persistent tinyCLAP INT8 interpreter, with 6 MiB tensor arena in PSRAM.
esp_err_t eco_inference_init(void);

// Input: [827,64] float32 torchlibrosa-compatible log-mel in dB.
// Output indices: bio, geo, antro; all in [0,1].
esp_err_t eco_inference_run(const float *logmel, float indices[3],
                            int64_t *inference_us);

#ifdef __cplusplus
}
#endif
