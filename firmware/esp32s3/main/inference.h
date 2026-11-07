#pragma once
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// IDs are zero-based indices into model/prompts.txt.
#define ECO_PROMPT_GROUPS 3
#define ECO_TOP_K 3

typedef struct {
    uint8_t id;
    float cosine;  // Raw cosine used to rank prompts, including zero-score ties.
    float score;   // Original clipped score in [0,1].
} eco_prompt_rank_t;

// Persistent tinyCLAP INT8 interpreter, with 6 MiB tensor arena in PSRAM.
esp_err_t eco_inference_init(void);

// Input: [827,64] float32 torchlibrosa-compatible log-mel in dB.
// Output indices: bio, geo, antro; all in [0,1].
esp_err_t eco_inference_run(const float *logmel, float indices[3],
                            int64_t *inference_us);

// Same inference and indices, also exposes top-3 prompts ranked by raw cosine.
esp_err_t eco_inference_run_with_top(
    const float *logmel, float indices[3], int64_t *inference_us,
    eco_prompt_rank_t top[ECO_PROMPT_GROUPS][ECO_TOP_K]);

#ifdef __cplusplus
}
#endif
