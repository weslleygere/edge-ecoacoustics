
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

static const char *TAG = "tinyclap_probe";

// Modelo TFLite incorporado ao firmware pelo CMake
extern const uint8_t tinyclap_start[]
    asm("_binary_tinyclap_phinet_int8_tflite_start");

extern const uint8_t tinyclap_end[]
    asm("_binary_tinyclap_phinet_int8_tflite_end");

// Configuracao do tinyCLAP
static constexpr size_t ARENA_BYTES = 6 * 1024 * 1024;
static constexpr size_t INPUT_ELEMENTS = 827 * 64;
static constexpr size_t OUTPUT_ELEMENTS = 1024;

// Valor sintetico para testar a execucao.
// Nao corresponde a uma gravacao real.
static constexpr float TEST_LOGMEL = -50.0f;

extern "C" esp_err_t eco_tinyclap_probe(void)
{
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "tinyCLAP INT8 - ESP32-S3 probe");
    ESP_LOGI(TAG, "================================");

    // --------------------------------------------------
    // 1. Verificar o modelo na Flash
    // --------------------------------------------------

    const size_t model_bytes = tinyclap_end - tinyclap_start;

    ESP_LOGI(TAG, "Model size: %u bytes",
             (unsigned)model_bytes);

    if (model_bytes == 0) {
        ESP_LOGE(TAG, "Invalid model size");
        return ESP_FAIL;
    }

    const tflite::Model *model =
        tflite::GetModel(tinyclap_start);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG,
                 "TFLite schema mismatch: model=%d runtime=%d",
                 model->version(),
                 TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "TFLite schema OK");

    // --------------------------------------------------
    // 2. Registrar os 19 operadores do tinyCLAP
    // --------------------------------------------------

    static tflite::MicroMutableOpResolver<19> resolver;
    static bool operators_registered = false;

    if (!operators_registered) {
        if (resolver.AddAdd() != kTfLiteOk ||
            resolver.AddConv2D() != kTfLiteOk ||
            resolver.AddDepthwiseConv2D() != kTfLiteOk ||
            resolver.AddDequantize() != kTfLiteOk ||
            resolver.AddDiv() != kTfLiteOk ||
            resolver.AddFullyConnected() != kTfLiteOk ||
            resolver.AddLogistic() != kTfLiteOk ||
            resolver.AddMaximum() != kTfLiteOk ||
            resolver.AddMean() != kTfLiteOk ||
            resolver.AddMul() != kTfLiteOk ||
            resolver.AddPad() != kTfLiteOk ||
            resolver.AddQuantize() != kTfLiteOk ||
            resolver.AddReshape() != kTfLiteOk ||
            resolver.AddRsqrt() != kTfLiteOk ||
            resolver.AddSqrt() != kTfLiteOk ||
            resolver.AddSquare() != kTfLiteOk ||
            resolver.AddSquaredDifference() != kTfLiteOk ||
            resolver.AddSub() != kTfLiteOk ||
            resolver.AddSum() != kTfLiteOk) {

            ESP_LOGE(TAG, "Operator registration failed");
            return ESP_FAIL;
        }

        operators_registered = true;
    }

    ESP_LOGI(TAG, "19 operators registered");

    // --------------------------------------------------
    // 3. Verificar memoria PSRAM
    // --------------------------------------------------

    ESP_LOGI(TAG, "Free PSRAM: %u bytes",
             (unsigned)heap_caps_get_free_size(
                 MALLOC_CAP_SPIRAM));

    ESP_LOGI(TAG, "Largest PSRAM block: %u bytes",
             (unsigned)heap_caps_get_largest_free_block(
                 MALLOC_CAP_SPIRAM));

    // --------------------------------------------------
    // 4. Alocar Tensor Arena na PSRAM
    // --------------------------------------------------

    uint8_t *arena = static_cast<uint8_t *>(
        heap_caps_aligned_alloc(
            16,
            ARENA_BYTES,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        )
    );

    if (arena == nullptr) {
        ESP_LOGE(TAG, "Cannot allocate 6 MB Tensor Arena");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "Tensor Arena allocated");

    TfLiteStatus status = kTfLiteError;

    // O interpretador deve ser destruido antes de liberar a arena.
    {
        tflite::MicroInterpreter interpreter(
            model,
            resolver,
            arena,
            ARENA_BYTES
        );

        // ----------------------------------------------
        // 5. Preparar os tensores
        // ----------------------------------------------

        ESP_LOGI(TAG, "Calling AllocateTensors()");

        status = interpreter.AllocateTensors();

        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "AllocateTensors FAILED");
        } else {
            ESP_LOGI(TAG, "AllocateTensors SUCCESS!");

            ESP_LOGI(TAG, "Arena used: %u bytes",
                     (unsigned)interpreter.arena_used_bytes());

            TfLiteTensor *input = interpreter.input(0);
            TfLiteTensor *output = interpreter.output(0);

            // ------------------------------------------
            // 6. Validar entrada e saida
            // ------------------------------------------

            if (input == nullptr || output == nullptr) {
                ESP_LOGE(TAG, "Input or output tensor is null");
                status = kTfLiteError;
            } else {
                ESP_LOGI(TAG, "Input type: %d", input->type);
                ESP_LOGI(TAG, "Output type: %d", output->type);

                ESP_LOGI(TAG, "Input bytes: %u",
                         (unsigned)input->bytes);

                ESP_LOGI(TAG, "Output bytes: %u",
                         (unsigned)output->bytes);

                const bool valid_input =
                    input->type == kTfLiteInt8 &&
                    input->dims != nullptr &&
                    input->dims->size == 3 &&
                    input->dims->data[0] == 1 &&
                    input->dims->data[1] == 827 &&
                    input->dims->data[2] == 64 &&
                    input->bytes == INPUT_ELEMENTS &&
                    input->data.int8 != nullptr &&
                    std::isfinite(input->params.scale) &&
                    input->params.scale > 0.0f;

                const bool valid_output =
                    output->type == kTfLiteInt8 &&
                    output->dims != nullptr &&
                    output->dims->size == 2 &&
                    output->dims->data[0] == 1 &&
                    output->dims->data[1] == 1024 &&
                    output->bytes == OUTPUT_ELEMENTS &&
                    output->data.int8 != nullptr &&
                    std::isfinite(output->params.scale) &&
                    output->params.scale > 0.0f;

                if (!valid_input || !valid_output) {
                    ESP_LOGE(TAG, "Unexpected INT8 tensor format");
                    status = kTfLiteError;
                } else {

                    // ----------------------------------
                    // 7. Preparar entrada sintetica
                    // ----------------------------------

                    const float input_scale =
                        input->params.scale;

                    const int input_zero_point =
                        input->params.zero_point;

                    ESP_LOGI(TAG, "Input scale: %.8f",
                             (double)input_scale);

                    ESP_LOGI(TAG, "Input zero point: %d",
                             input_zero_point);

                    int quantized_value =
                        static_cast<int>(std::lround(
                            TEST_LOGMEL / input_scale +
                            input_zero_point
                        ));

                    if (quantized_value < -128) {
                        quantized_value = -128;
                    }

                    if (quantized_value > 127) {
                        quantized_value = 127;
                    }

                    for (size_t i = 0;
                         i < INPUT_ELEMENTS;
                         i++) {

                        input->data.int8[i] =
                            static_cast<int8_t>(
                                quantized_value
                            );
                    }

                    ESP_LOGI(
                        TAG,
                        "Synthetic Log-Mel: %.2f dB -> INT8 %d",
                        (double)TEST_LOGMEL,
                        quantized_value
                    );

                    // ----------------------------------
                    // 8. Executar inferencia
                    // ----------------------------------

                    ESP_LOGI(
                        TAG,
                        "Starting tinyCLAP INT8 inference"
                    );

                    const int64_t start_us =
                        esp_timer_get_time();

                    status = interpreter.Invoke();

                    const int64_t elapsed_us =
                        esp_timer_get_time() - start_us;

                    if (status != kTfLiteOk) {
                        ESP_LOGE(TAG, "Invoke FAILED");
                    } else {
                        ESP_LOGI(TAG, "Invoke SUCCESS!");

                        ESP_LOGI(
                            TAG,
                            "Inference time: %.2f ms",
                            (double)elapsed_us / 1000.0
                        );

                        // ------------------------------
                        // 9. Inspecionar embedding
                        // ------------------------------

                        ESP_LOGI(
                            TAG,
                            "Output INT8 (first 8 values):"
                        );

                        for (size_t i = 0; i < 8; i++) {
                            ESP_LOGI(
                                TAG,
                                "  embedding[%u] = %d",
                                (unsigned)i,
                                (int)output->data.int8[i]
                            );
                        }

                        const float output_scale =
                            output->params.scale;

                        const int output_zero_point =
                            output->params.zero_point;

                        double sum_squares = 0.0;
                        bool finite_output = true;

                        for (size_t i = 0;
                             i < OUTPUT_ELEMENTS;
                             i++) {

                            const float value =
                                (static_cast<int>(
                                     output->data.int8[i]
                                 ) - output_zero_point) *
                                output_scale;

                            if (!std::isfinite(value)) {
                                finite_output = false;
                                break;
                            }

                            sum_squares +=
                                static_cast<double>(value) *
                                static_cast<double>(value);
                        }

                        if (!finite_output) {
                            ESP_LOGE(TAG, "Non-finite embedding");
                            status = kTfLiteError;
                        } else {
                            ESP_LOGI(
                                TAG,
                                "Embedding L2 norm: %.6f",
                                std::sqrt(sum_squares)
                            );

                            ESP_LOGI(
                                TAG,
                                "Embedding size: %u",
                                (unsigned)OUTPUT_ELEMENTS
                            );
                        }
                    }
                }
            }
        }
    }

    // --------------------------------------------------
    // 10. Liberar memoria temporaria
    // --------------------------------------------------

    heap_caps_free(arena);

    ESP_LOGI(TAG, "Temporary Tensor Arena released");

    ESP_LOGI(TAG, "Free PSRAM after probe: %u bytes",
             (unsigned)heap_caps_get_free_size(
                 MALLOC_CAP_SPIRAM));

    if (status == kTfLiteOk) {
        ESP_LOGI(TAG, "tinyCLAP probe completed SUCCESSFULLY");
        return ESP_OK;
    }

    ESP_LOGE(TAG, "tinyCLAP probe completed with FAILURE");
    return ESP_FAIL;
}
