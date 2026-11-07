/*
 * Edge Ecoacoustics: tinyCLAP end-to-end on ESP32-S3 N16R8.
 *
 * First deployment scheduling (deliberately non-continuous):
 * 6 s I2S capture at 44.1 kHz -> pause I2S -> 827x64 Log-Mel ->
 * PhiNet INT8 -> 41-prompt similarities -> Bio/Geo/Antro -> MQTT/TLS.
 * Capture resumes only after inference; no Ring Buffer overflow/lost blocks.
 * No legacy CNN is initialized or included in firmware.
 */
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "feature.h"
#include "inference.h"
#include "network.h"

#define MIC_BCLK_GPIO  GPIO_NUM_4
#define MIC_WS_GPIO    GPIO_NUM_5
#define MIC_DIN_GPIO   GPIO_NUM_6
#define DMA_FRAME_NUM  441
#define DMA_DESC_NUM   8
#define CAPTURE_STACK  8192
#define READ_TIMEOUT_MS 1000

static const char *TAG = "eco_audio";
static i2s_chan_handle_t i2s_rx = NULL;
static int32_t raw_buffer[ECO_BLOCK_SAMPLES * 2];
static uint32_t audio_errors = 0;
static uint32_t windows = 0;
static int16_t *pcm_buffer = NULL;
static eco_features_t *feature_buffer = NULL;

static void log_memory_status(const char *stage)
{
    ESP_LOGI(TAG, "Memory %s | internal=%u psram=%u largest_psram=%u",
             stage,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
}

static esp_err_t capture_segment(int16_t *pcm, double *rms, int64_t *end_us)
{
    esp_err_t err = i2s_channel_enable(i2s_rx);
    if (err != ESP_OK) return err;
    const int64_t start_us = esp_timer_get_time();
    double sum_squares = 0.0;
    size_t collected = 0;

    while (collected < ECO_SEGMENT_SAMPLES) {
        const size_t count = (ECO_SEGMENT_SAMPLES - collected > ECO_BLOCK_SAMPLES)
            ? ECO_BLOCK_SAMPLES : (ECO_SEGMENT_SAMPLES - collected);
        const size_t need_bytes = count * 2 * sizeof(int32_t);
        size_t received = 0;
        while (received < need_bytes) {
            size_t just_read = 0;
            err = i2s_channel_read(i2s_rx,
                                   (uint8_t *)raw_buffer + received,
                                   need_bytes - received,
                                   &just_read,
                                   pdMS_TO_TICKS(READ_TIMEOUT_MS));
            if (err != ESP_OK || just_read == 0) {
                if (err == ESP_OK) err = ESP_ERR_TIMEOUT;
                goto capture_exit;
            }
            received += just_read;
        }
        // Retain the same physical I2S LEFT slot used by the existing firmware.
        for (size_t i = 0; i < count; ++i) {
            const int16_t s = (int16_t)(raw_buffer[2 * i] >> 16);
            pcm[collected + i] = s;
            const double x = (double)s / 32768.0;
            sum_squares += x * x;
        }
        collected += count;
    }

capture_exit:
    *end_us = esp_timer_get_time();
    esp_err_t disable_err = i2s_channel_disable(i2s_rx);
    if (err == ESP_OK) err = disable_err;
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2S capture failed: %s", esp_err_to_name(err));
        return err;
    }
    *rms = sqrt(sum_squares / ECO_SEGMENT_SAMPLES);
    ESP_LOGI(TAG, "Capture #%" PRIu32 " | %.2f s | RMS=%.6f | samples=%u",
             windows + 1, (double)(*end_us - start_us) / 1e6,
             *rms, (unsigned)collected);
    return ESP_OK;
}

static void acquisition_task(void *arg)
{
    (void)arg;
    int16_t *pcm = pcm_buffer;
    eco_features_t *features = feature_buffer;
    log_memory_status("buffers_allocated");
    ESP_LOGW(TAG, "Scheduled capture: 6 s audio, then paused during inference");

    while (true) {
        double rms = 0.0;
        int64_t end_us = 0;
        esp_err_t err = capture_segment(pcm, &rms, &end_us);
        if (err != ESP_OK) {
            ++audio_errors;
            break;
        }

        int64_t start_dsp = esp_timer_get_time();
        err = eco_features_compute(pcm, ECO_SEGMENT_SAMPLES, features);
        const double dsp_ms = (double)(esp_timer_get_time() - start_dsp) / 1000.0;
        if (err != ESP_OK) {
            ++audio_errors;
            ESP_LOGE(TAG, "Log-Mel failed: %s", esp_err_to_name(err));
            break;
        }

        float indices[3] = {0.0f};
        int64_t inference_us = 0;
        err = eco_inference_run(&features->logmel[0][0], indices, &inference_us);
        if (err != ESP_OK) {
            ++audio_errors;
            ESP_LOGE(TAG, "tinyCLAP failed: %s", esp_err_to_name(err));
            break;
        }
        ++windows;
        ESP_LOGI(TAG, "Window %" PRIu32 " | Bio=%.4f Geo=%.4f Antro=%.4f | DSP=%.2f ms | PhiNet=%.2f ms",
                 windows, (double)indices[0], (double)indices[1],
                 (double)indices[2], dsp_ms, (double)inference_us / 1000.0);

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
        eco_network_publish(&telemetry);
        ESP_LOGI(TAG, "Cycle complete | next 6-s capture starts now | stack_free=%u",
                 (unsigned)uxTaskGetStackHighWaterMark(NULL));
        log_memory_status("window");
    }
    ESP_LOGE(TAG, "Acquisition stopped | errors=%" PRIu32, audio_errors);
    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "Edge Ecoacoustics - tinyCLAP INT8 + Bio/Geo/Antro + MQTT");
    // Allocate the large model arena first to avoid PSRAM fragmentation.
    log_memory_status("startup");
    ESP_ERROR_CHECK(eco_inference_init());
    ESP_ERROR_CHECK(eco_features_init());

    // Reserve remaining buffers before WiFi/TLS to avoid fragmentation.
    pcm_buffer = (int16_t *)heap_caps_aligned_alloc(
        16, ECO_SEGMENT_SAMPLES * sizeof(int16_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    feature_buffer = (eco_features_t *)heap_caps_aligned_alloc(
        16, sizeof(eco_features_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    assert(pcm_buffer != NULL && feature_buffer != NULL);

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = DMA_DESC_NUM;
    chan_cfg.dma_frame_num = DMA_FRAME_NUM;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &i2s_rx));

    i2s_std_config_t i2s_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(ECO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_BCLK_GPIO,
            .ws = MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_DIN_GPIO,
            .invert_flags = {0},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(i2s_rx, &i2s_cfg));
    ESP_ERROR_CHECK(eco_network_init());

    BaseType_t result = xTaskCreate(acquisition_task, "tinyclap_audio",
                                    CAPTURE_STACK, NULL, 5, NULL);
    assert(result == pdPASS);
    ESP_LOGI(TAG, "Scheduled acquisition started | 44.1 kHz | left slot | MQTT/TLS");
}
