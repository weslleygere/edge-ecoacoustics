
/*
 * Edge Ecoacoustics - ESP32-S3 tinyCLAP.
 *
 * Captura I2S de 6 s -> Log-Mel -> PhiNet INT8 -> MQTT/TLS.
 * A captura fica pausada durante o processamento.
 */

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "feature.h"
#include "inference.h"
#include "network.h"

#define MIC_BCLK_GPIO    GPIO_NUM_4
#define MIC_WS_GPIO      GPIO_NUM_5
#define MIC_DIN_GPIO     GPIO_NUM_6
#define DMA_FRAME_NUM    441
#define DMA_DESC_NUM     8
#define CAPTURE_STACK    8192
#define READ_TIMEOUT_MS  1000

static const char *TAG = "eco_audio";

static i2s_chan_handle_t i2s_rx = NULL;
static int32_t raw_buffer[ECO_BLOCK_SAMPLES * 2];

static uint32_t audio_errors = 0;
static uint32_t windows = 0;

static int16_t *pcm_buffer = NULL;
static eco_features_t *feature_buffer = NULL;

static esp_err_t capture_segment(int16_t *pcm, double *rms, int64_t *end_us)
{
    esp_err_t err = i2s_channel_enable(i2s_rx);
    if (err != ESP_OK) {
        return err;
    }

    double sum_squares = 0.0;
    size_t collected = 0;

    while (collected < ECO_SEGMENT_SAMPLES) {
        const size_t count =
            (ECO_SEGMENT_SAMPLES - collected > ECO_BLOCK_SAMPLES)
                ? ECO_BLOCK_SAMPLES
                : (ECO_SEGMENT_SAMPLES - collected);

        const size_t need_bytes = count * 2 * sizeof(int32_t);
        size_t received = 0;

        while (received < need_bytes) {
            size_t just_read = 0;

            err = i2s_channel_read(
                i2s_rx,
                (uint8_t *)raw_buffer + received,
                need_bytes - received,
                &just_read,
                pdMS_TO_TICKS(READ_TIMEOUT_MS)
            );

            if (err != ESP_OK || just_read == 0) {
                if (err == ESP_OK) {
                    err = ESP_ERR_TIMEOUT;
                }
                goto capture_exit;
            }

            received += just_read;
        }

        // Microfone no slot esquerdo I2S.
        for (size_t i = 0; i < count; ++i) {
            const int16_t sample =
                (int16_t)(raw_buffer[2 * i] >> 16);

            pcm[collected + i] = sample;

            const double x = (double)sample / 32768.0;
            sum_squares += x * x;
        }

        collected += count;
    }

capture_exit:
    *end_us = esp_timer_get_time();

    esp_err_t disable_err = i2s_channel_disable(i2s_rx);
    if (err == ESP_OK) {
        err = disable_err;
    }

    if (err != ESP_OK) {
        ESP_LOGE(
            TAG,
            "I2S capture failed: %s",
            esp_err_to_name(err)
        );
        return err;
    }

    *rms = sqrt(sum_squares / ECO_SEGMENT_SAMPLES);

    return ESP_OK;
}

static void acquisition_task(void *arg)
{
    (void)arg;

    int16_t *pcm = pcm_buffer;
    eco_features_t *features = feature_buffer;

    while (true) {
        double rms = 0.0;
        int64_t end_us = 0;

        // Captura de 6 segundos.
        esp_err_t err = capture_segment(pcm, &rms, &end_us);

        if (err != ESP_OK) {
            ++audio_errors;
            break;
        }

        // Extracao Log-Mel.
        const int64_t start_dsp = esp_timer_get_time();

        err = eco_features_compute(
            pcm,
            ECO_SEGMENT_SAMPLES,
            features
        );

        const double dsp_ms =
            (double)(esp_timer_get_time() - start_dsp) / 1000.0;

        if (err != ESP_OK) {
            ++audio_errors;
            ESP_LOGE(
                TAG,
                "Log-Mel failed: %s",
                esp_err_to_name(err)
            );
            break;
        }

        // Inferencia tinyCLAP e indices ecoacusticos.
        float indices[3] = {0.0f};
        eco_prompt_rank_t top[ECO_PROMPT_GROUPS][ECO_TOP_K] = {0};
        int64_t inference_us = 0;

        err = eco_inference_run_with_top(
            &features->logmel[0][0],
            indices,
            &inference_us,
            top
        );

        if (err != ESP_OK) {
            ++audio_errors;
            ESP_LOGE(
                TAG,
                "tinyCLAP failed: %s",
                esp_err_to_name(err)
            );
            break;
        }

        ++windows;

        ESP_LOGI(
            TAG,
            "Window %" PRIu32
            " | Bio=%.4f Geo=%.4f Antro=%.4f"
            " | DSP=%.2f ms | PhiNet=%.2f ms",
            windows,
            (double)indices[0],
            (double)indices[1],
            (double)indices[2],
            dsp_ms,
            (double)inference_us / 1000.0
        );

        // Preparar telemetria MQTT.
        eco_telemetry_t telemetry = {
            .end_us = end_us,
            .seq = windows,
            .bio = indices[0],
            .geo = indices[1],
            .antro = indices[2],
            .rms = (float)rms,
            .inference_ms = (float)inference_us / 1000.0f,
            .errors = audio_errors,
            .overflows = 0,
        };

        for (int g = 0; g < ECO_PROMPT_GROUPS; ++g) {
            for (int k = 0; k < ECO_TOP_K; ++k) {
                telemetry.top[g][k] = top[g][k];
            }
        }

        eco_network_publish(&telemetry);
    }

    ESP_LOGE(
        TAG,
        "Acquisition stopped | errors=%" PRIu32,
        audio_errors
    );

    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "Edge Ecoacoustics - tinyCLAP INT8");

    // Inicializar o modelo antes das demais alocacoes em PSRAM.
    ESP_ERROR_CHECK(eco_inference_init());
    ESP_ERROR_CHECK(eco_features_init());

    pcm_buffer = (int16_t *)heap_caps_aligned_alloc(
        16,
        ECO_SEGMENT_SAMPLES * sizeof(int16_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    feature_buffer = (eco_features_t *)heap_caps_aligned_alloc(
        16,
        sizeof(eco_features_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    assert(pcm_buffer != NULL && feature_buffer != NULL);

    // Configuracao I2S.
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_AUTO,
            I2S_ROLE_MASTER
        );

    chan_cfg.dma_desc_num = DMA_DESC_NUM;
    chan_cfg.dma_frame_num = DMA_FRAME_NUM;

    ESP_ERROR_CHECK(
        i2s_new_channel(&chan_cfg, NULL, &i2s_rx)
    );

    i2s_std_config_t i2s_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(ECO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_STEREO
        ),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_BCLK_GPIO,
            .ws = MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_DIN_GPIO,
            .invert_flags = {0},
        },
    };

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(i2s_rx, &i2s_cfg)
    );

    // Rede inicializada apos reservar os buffers de audio.
    ESP_ERROR_CHECK(eco_network_init());

    BaseType_t result = xTaskCreate(
        acquisition_task,
        "tinyclap_audio",
        CAPTURE_STACK,
        NULL,
        5,
        NULL
    );

    assert(result == pdPASS);

    ESP_LOGI(
        TAG,
        "Acquisition started | 44.1 kHz | left I2S slot"
    );
}
