
#include "inference.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

static const char *TAG = "eco_ml";

extern const uint8_t model_start[] asm("_binary_cnn_logmel_float32_tflite_start");
extern const uint8_t model_end[] asm("_binary_cnn_logmel_float32_tflite_end");
extern const uint8_t normalization_start[] asm("_binary_normalization_bin_start");
extern const uint8_t normalization_end[] asm("_binary_normalization_bin_end");

static constexpr size_t ARENA_BYTES = 256 * 1024;
static constexpr size_t MEL_BANDS = 40;
static constexpr size_t INPUT_ELEMENTS = 299 * MEL_BANDS;

static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;
static const float *normalization = nullptr;

extern "C" esp_err_t eco_inference_init(void)
{
    if (interpreter != nullptr) return ESP_OK;

    size_t model_bytes = model_end - model_start;
    size_t normalization_bytes = normalization_end - normalization_start;

    if (model_bytes == 0 || normalization_bytes != 2 * MEL_BANDS * sizeof(float)) {
        ESP_LOGE(TAG, "Invalid model or normalization file");
        return ESP_ERR_INVALID_SIZE;
    }

    const tflite::Model *model = tflite::GetModel(model_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "TFLite schema mismatch: model=%d runtime=%d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    static tflite::MicroMutableOpResolver<4> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddLogistic() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to register TFLite operators");
        return ESP_FAIL;
    }

    uint8_t *arena = static_cast<uint8_t *>(
        heap_caps_aligned_alloc(16, ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (arena == nullptr) {
        ESP_LOGE(TAG, "Cannot allocate %u bytes in PSRAM", (unsigned)ARENA_BYTES);
        return ESP_ERR_NO_MEM;
    }

    static tflite::MicroInterpreter micro_interpreter(model, resolver, arena, ARENA_BYTES);

    if (micro_interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed. Arena=%u bytes", (unsigned)ARENA_BYTES);
        return ESP_FAIL;
    }

    TfLiteTensor *input = micro_interpreter.input(0);
    TfLiteTensor *output = micro_interpreter.output(0);

    if (input == nullptr || output == nullptr ||
        input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32 ||
        input->dims->size != 4 || output->dims->size != 2 ||
        input->dims->data[0] != 1 || input->dims->data[1] != 299 ||
        input->dims->data[2] != 40 || input->dims->data[3] != 1 ||
        output->dims->data[0] != 1 || output->dims->data[1] != 3 ||
        input->bytes != INPUT_ELEMENTS * sizeof(float)) {
        ESP_LOGE(TAG, "Unexpected float32 model input/output");
        return ESP_FAIL;
    }

    normalization = reinterpret_cast<const float *>(normalization_start);

    for (size_t m = 0; m < MEL_BANDS; m++) {
        if (!std::isfinite(normalization[m]) ||
            !std::isfinite(normalization[MEL_BANDS + m]) ||
            normalization[MEL_BANDS + m] <= 0.0f) {
            ESP_LOGE(TAG, "Invalid normalization at band %u", (unsigned)m);
            return ESP_FAIL;
        }
    }

    interpreter = &micro_interpreter;
    input_tensor = input;
    output_tensor = output;

    ESP_LOGI(TAG, "CNN float32 initialized | model=%u bytes | arena_allocated=%u bytes | arena_used=%u bytes",
             (unsigned)model_bytes, (unsigned)ARENA_BYTES,
             (unsigned)interpreter->arena_used_bytes());

    return ESP_OK;
}

extern "C" esp_err_t eco_inference_run(const float *logmel, float indices[3], int64_t *elapsed_us)
{
    if (interpreter == nullptr || logmel == nullptr || indices == nullptr || elapsed_us == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }

    int64_t start_us = esp_timer_get_time();

    for (size_t i = 0; i < INPUT_ELEMENTS; i++) {
        size_t m = i % MEL_BANDS;
        input_tensor->data.f[i] = (logmel[i] - normalization[m]) / normalization[MEL_BANDS + m];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite inference failed");
        return ESP_FAIL;
    }

    *elapsed_us = esp_timer_get_time() - start_us;

    for (int i = 0; i < 3; i++) {
        indices[i] = output_tensor->data.f[i];
        if (!std::isfinite(indices[i])) return ESP_FAIL;
    }

    return ESP_OK;
}
