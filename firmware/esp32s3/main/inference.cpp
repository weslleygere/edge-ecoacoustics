/* tinyCLAP INT8 audio encoder and Bio/Geo/Antro scoring. */
#include "inference.h"
#include "feature.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace {
constexpr size_t kArenaSize = 6 * 1024 * 1024;
constexpr size_t kEmbeddingSize = 1024;
constexpr size_t kPrompts = 41;
constexpr uint8_t kBio = 0x01;
constexpr uint8_t kGeo = 0x02;
constexpr uint8_t kAntro = 0x04;
static const char *TAG = "eco_tinyclap";

extern const uint8_t model_start[] asm("_binary_tinyclap_phinet_int8_tflite_start");
extern const uint8_t model_end[]   asm("_binary_tinyclap_phinet_int8_tflite_end");
extern const uint8_t text_start[]  asm("_binary_tinyclap_text_assets_bin_start");
extern const uint8_t text_end[]    asm("_binary_tinyclap_text_assets_bin_end");

static uint8_t *arena = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static const uint8_t *prompt_masks = nullptr;
static const float *text_embeddings = nullptr;
static float logit_scale = 0.0f;

struct PromptRank {
    int id;
    float score;
};

static bool load_text_assets()
{
    // magic[4] + uint32 N + uint32 DIM + float scale
    // + N masks + 4-byte alignment + float32 N*DIM normalized vectors.
    const size_t header_bytes = 16;
    const size_t offset = (header_bytes + kPrompts + 3u) & ~size_t(3u);
    const size_t expected = offset + kPrompts * kEmbeddingSize * sizeof(float);
    if ((size_t)(text_end - text_start) != expected ||
        std::memcmp(text_start, "ECO1", 4) != 0) {
        ESP_LOGE(TAG, "Invalid text asset size/magic");
        return false;
    }
    uint32_t n = 0, dim = 0;
    std::memcpy(&n, text_start + 4, 4);
    std::memcpy(&dim, text_start + 8, 4);
    std::memcpy(&logit_scale, text_start + 12, 4);
    if (n != kPrompts || dim != kEmbeddingSize ||
        !std::isfinite(logit_scale) || logit_scale <= 0.0f ||
        ((uintptr_t)(text_start + offset) % alignof(float)) != 0) {
        ESP_LOGE(TAG, "Invalid text asset metadata");
        return false;
    }
    prompt_masks = text_start + header_bytes;
    text_embeddings = reinterpret_cast<const float *>(text_start + offset);
    int groups[3] = {0, 0, 0};
    for (size_t p = 0; p < kPrompts; ++p) {
        if (prompt_masks[p] & kBio) ++groups[0];
        if (prompt_masks[p] & kGeo) ++groups[1];
        if (prompt_masks[p] & kAntro) ++groups[2];
    }
    if (groups[0] < 3 || groups[1] < 3 || groups[2] < 3) {
        ESP_LOGE(TAG, "Not enough prompts for top 3 in a category");
        return false;
    }
    ESP_LOGI(TAG, "Text assets ready | prompts=%u | scale=%.4f | B/G/A=%d/%d/%d",
             (unsigned)n, (double)logit_scale,
             groups[0], groups[1], groups[2]);
    return true;
}

static float clip01(float x)
{
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static void insert_rank(PromptRank ranks[3], int prompt_id, float score)
{
    for (size_t pos = 0; pos < 3; ++pos) {
        if (score > ranks[pos].score) {
            for (size_t shift = 2; shift > pos; --shift) {
                ranks[shift] = ranks[shift - 1];
            }
            ranks[pos] = {prompt_id, score};
            break;
        }
    }
}

static bool score_embedding(const TfLiteTensor *output,
                            float indices[3],
                            PromptRank ranks[3][3],
                            float *embedding_norm,
                            PromptRank cosine_ranks[3][3] = nullptr)
{
    const float out_scale = output->params.scale;
    const int out_zero = output->params.zero_point;
    // One processing task; static storage avoids exhausting its small stack.
    static float audio[kEmbeddingSize];
    double sum_squares = 0.0;
    for (size_t j = 0; j < kEmbeddingSize; ++j) {
        audio[j] = (static_cast<int>(output->data.int8[j]) - out_zero) * out_scale;
        sum_squares += (double)audio[j] * (double)audio[j];
    }
    if (!std::isfinite(sum_squares) || sum_squares <= 1e-12) {
        ESP_LOGE(TAG, "Invalid audio embedding");
        return false;
    }
    *embedding_norm = static_cast<float>(std::sqrt(sum_squares));
    const float inv_norm = 1.0f / *embedding_norm;
    const uint8_t bits[3] = {kBio, kGeo, kAntro};
    for (int group = 0; group < 3; ++group) {
        for (int pos = 0; pos < 3; ++pos) {
            ranks[group][pos] = {-1, -1.0f};
            if (cosine_ranks != nullptr) cosine_ranks[group][pos] = {-1, -2.0f};
        }
    }

    for (size_t p = 0; p < kPrompts; ++p) {
        const float *text = text_embeddings + p * kEmbeddingSize;
        double dot = 0.0;
        for (size_t j = 0; j < kEmbeddingSize; ++j) {
            dot += static_cast<double>(audio[j]) * static_cast<double>(text[j]);
        }
        const float cosine = static_cast<float>(dot) * inv_norm;
        const float similarity = clip01((cosine * logit_scale - 5.0f) / 15.0f);
        if (!std::isfinite(similarity)) return false;
        for (int group = 0; group < 3; ++group) {
            if (prompt_masks[p] & bits[group]) {
                insert_rank(ranks[group], static_cast<int>(p), similarity);
                if (cosine_ranks != nullptr) {
                    insert_rank(cosine_ranks[group], static_cast<int>(p), cosine);
                }
            }
        }
    }
    for (int group = 0; group < 3; ++group) {
        if (ranks[group][1].id < 0) return false;
        indices[group] = 0.75f * ranks[group][0].score +
                         0.25f * ranks[group][1].score;
    }
    return true;
}

}  // namespace

extern "C" esp_err_t eco_inference_init(void)
{
    if (interpreter != nullptr) return ESP_OK;
    if (!load_text_assets()) return ESP_ERR_INVALID_SIZE;

    const tflite::Model *model = tflite::GetModel(model_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "TFLite schema mismatch");
        return ESP_FAIL;
    }

    static tflite::MicroMutableOpResolver<12> resolver;
    if (resolver.AddAdd() != kTfLiteOk ||
        resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddLogistic() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddMul() != kTfLiteOk ||
        resolver.AddPad() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddRsqrt() != kTfLiteOk ||
        resolver.AddSquaredDifference() != kTfLiteOk ||
        resolver.AddSub() != kTfLiteOk) {
        ESP_LOGE(TAG, "Operator registration failed");
        return ESP_FAIL;
    }

    arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate 6 MiB tensor arena");
        return ESP_ERR_NO_MEM;
    }

    interpreter = new (std::nothrow) tflite::MicroInterpreter(
        model, resolver, arena, kArenaSize);
    if (interpreter == nullptr || interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        delete interpreter;
        interpreter = nullptr;
        heap_caps_free(arena);
        arena = nullptr;
        return ESP_FAIL;
    }

    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);
    if (!input || !output || !input->dims || !output->dims ||
        input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        input->dims->size != 3 || input->dims->data[0] != 1 ||
        input->dims->data[1] != ECO_FRAMES_PER_SEGMENT ||
        input->dims->data[2] != ECO_MEL_BANDS ||
        output->dims->size != 2 || output->dims->data[0] != 1 ||
        output->dims->data[1] != kEmbeddingSize ||
        input->bytes != ECO_FRAMES_PER_SEGMENT * ECO_MEL_BANDS ||
        output->bytes != kEmbeddingSize ||
        input->data.int8 == nullptr || output->data.int8 == nullptr ||
        input->params.scale <= 0 || output->params.scale <= 0) {
        ESP_LOGE(TAG, "Unexpected model I/O tensor shape/type");
        delete interpreter;
        interpreter = nullptr;
        heap_caps_free(arena);
        arena = nullptr;
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "tinyCLAP INT8 ready | model=%u | arena_used=%u bytes",
             (unsigned)(model_end - model_start),
             (unsigned)interpreter->arena_used_bytes());
    return ESP_OK;
}

extern "C" esp_err_t eco_inference_run_with_top(
    const float *logmel, float indices[3], int64_t *inference_us,
    eco_prompt_rank_t top[ECO_PROMPT_GROUPS][ECO_TOP_K])
{
    if (!interpreter || !logmel || !indices || !inference_us) {
        return ESP_ERR_INVALID_STATE;
    }
    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);

    const float in_scale = input->params.scale;
    const int in_zero = input->params.zero_point;
    const size_t n = ECO_FRAMES_PER_SEGMENT * ECO_MEL_BANDS;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(logmel[i])) return ESP_ERR_INVALID_ARG;
        long q = std::lround(logmel[i] / in_scale) + in_zero;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input->data.int8[i] = static_cast<int8_t>(q);
    }

    const int64_t start = esp_timer_get_time();
    const TfLiteStatus status = interpreter->Invoke();
    *inference_us = esp_timer_get_time() - start;
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed after %.2f ms", (double)*inference_us / 1000.0);
        return ESP_FAIL;
    }

    PromptRank ranks[3][3];
    PromptRank cosine_ranks[3][3];
    float norm = 0.0f;
    if (!score_embedding(output, indices, ranks, &norm,
                         top != nullptr ? cosine_ranks : nullptr)) {
        return ESP_FAIL;
    }
    if (top != nullptr) {
        for (int g = 0; g < ECO_PROMPT_GROUPS; ++g) {
            for (int k = 0; k < ECO_TOP_K; ++k) {
                if (cosine_ranks[g][k].id < 0) return ESP_FAIL;
                const float cosine = cosine_ranks[g][k].score;
                top[g][k].id = static_cast<uint8_t>(cosine_ranks[g][k].id);
                top[g][k].cosine = cosine;
                top[g][k].score = clip01((cosine * logit_scale - 5.0f) / 15.0f);
            }
        }
    }

    return ESP_OK;
}

// Preserve the original API for any other existing call sites.
extern "C" esp_err_t eco_inference_run(const float *logmel, float indices[3],
                                       int64_t *inference_us)
{
    return eco_inference_run_with_top(logmel, indices, inference_us, nullptr);
}
