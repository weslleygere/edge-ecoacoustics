
/*
 * Edge Ecoacoustics
 * ESP32-S3 / ESP-IDF 6.x
 *
 * I2S -> DMA -> Ring Buffer -> Log-Mel -> CNN float32
 * Continuous audio at 48 kHz; one inference every 6 seconds.
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
#include "freertos/ringbuf.h"
#include "freertos/task.h"

#include "feature.h"
#include "inference.h"

#define MIC_BCLK_GPIO GPIO_NUM_4
#define MIC_WS_GPIO   GPIO_NUM_5
#define MIC_DIN_GPIO  GPIO_NUM_6

#define DMA_FRAME_NUM 480
#define DMA_DESC_NUM 8
#define RING_BYTES (96 * 1024)
#define CAPTURE_STACK 6144
#define DSP_STACK 6144
#define READ_TIMEOUT_MS 1000
#define RING_TIMEOUT_MS 100

static const char *TAG = "eco_audio";

static RingbufHandle_t audio_ring = NULL;
static i2s_chan_handle_t i2s_rx = NULL;
static eco_features_t features;

static int32_t raw_buffer[ECO_BLOCK_SAMPLES * 2];
static int16_t mono_buffer[ECO_BLOCK_SAMPLES];

static volatile uint32_t audio_blocks = 0;
static volatile uint32_t audio_errors = 0;
static volatile uint32_t audio_overflows = 0;
static volatile bool capture_stopped = false;

static void log_memory_status(const char *stage)
{
    ESP_LOGI(TAG, "Memory %s | internal_free=%u | internal_largest=%u | psram_free=%u | psram_largest=%u",
             stage,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
}

static void capture_task(void *arg)
{
    (void)arg;

    esp_err_t err = i2s_channel_enable(i2s_rx);
    if (err != ESP_OK) {
        audio_errors++;
        capture_stopped = true;
        ESP_LOGE(TAG, "Cannot enable I2S: %s", esp_err_to_name(err));
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Continuous audio capture started");

    while (true) {
        size_t received = 0;

        while (received < sizeof(raw_buffer)) {
            size_t bytes_read = 0;
            err = i2s_channel_read(i2s_rx, (uint8_t *)raw_buffer + received,
                                   sizeof(raw_buffer) - received, &bytes_read,
                                   pdMS_TO_TICKS(READ_TIMEOUT_MS));

            if (err != ESP_OK || bytes_read == 0) {
                audio_errors++;
                ESP_LOGE(TAG, "I2S read failed: %s", esp_err_to_name(err));
                goto capture_exit;
            }

            received += bytes_read;
        }

        // Microphone data is in the left I2S slot.
        for (size_t i = 0; i < ECO_BLOCK_SAMPLES; i++) {
            mono_buffer[i] = (int16_t)(raw_buffer[2 * i] >> 16);
        }

        if (xRingbufferSend(audio_ring, mono_buffer, sizeof(mono_buffer),
                            pdMS_TO_TICKS(RING_TIMEOUT_MS)) != pdTRUE) {
            audio_overflows++;
            ESP_LOGE(TAG, "Ring Buffer overflow at block %" PRIu32, audio_blocks);
            goto capture_exit;
        }

        audio_blocks++;
    }

capture_exit:
    capture_stopped = true;

    err = i2s_channel_disable(i2s_rx);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Cannot disable I2S: %s", esp_err_to_name(err));
    }

    ESP_LOGE(TAG, "Capture stopped | blocks=%" PRIu32 " errors=%" PRIu32 " overflows=%" PRIu32,
             audio_blocks, audio_errors, audio_overflows);

    vTaskDelete(NULL);
}

static void dsp_task(void *arg)
{
    (void)arg;

    uint32_t windows = 0;
    uint32_t processed_blocks = 0;
    double sum_squares = 0.0;

    eco_features_reset(&features);
    int64_t window_start_us = esp_timer_get_time();

    ESP_LOGI(TAG, "DSP task started");

    while (true) {
        size_t bytes = 0;
        int16_t *audio = (int16_t *)xRingbufferReceive(audio_ring, &bytes, pdMS_TO_TICKS(100));

        if (audio == NULL) {
            if (capture_stopped) break;
            continue;
        }

        if (bytes != sizeof(mono_buffer)) {
            ESP_LOGE(TAG, "Unexpected audio block size: %u", (unsigned)bytes);
            vRingbufferReturnItem(audio_ring, audio);
            break;
        }

        for (size_t i = 0; i < ECO_BLOCK_SAMPLES; i++) {
            double sample = (double)audio[i] / 32768.0;
            sum_squares += sample * sample;
        }

        esp_err_t err = eco_features_push(&features, audio, ECO_BLOCK_SAMPLES);
        vRingbufferReturnItem(audio_ring, audio);

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Log-Mel processing failed: %s", esp_err_to_name(err));
            break;
        }

        processed_blocks++;

        if (eco_features_complete(&features)) {
            windows++;

            double rms = sqrt(sum_squares / ECO_SEGMENT_SAMPLES);
            double elapsed_s = (double)(esp_timer_get_time() - window_start_us) / 1e6;
            float indices[3] = {0};
            int64_t inference_us = 0;

            ESP_LOGI(TAG, "Window %" PRIu32 " | frames=%u | RMS=%.6f | elapsed=%.2fs",
                     windows, (unsigned)features.frames_generated, rms, elapsed_s);

            err = eco_inference_run(&features.logmel[0][0], indices, &inference_us);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "CNN inference failed: %s", esp_err_to_name(err));
                break;
            }

            ESP_LOGI(TAG, "CNN | Bio=%.4f | Geo=%.4f | Antro=%.4f | inference_ms=%.2f",
                     indices[0], indices[1], indices[2], (double)inference_us / 1000.0);

            ESP_LOGI(TAG, "Blocks | captured=%" PRIu32 " | processed=%" PRIu32
                     " | errors=%" PRIu32 " | overflows=%" PRIu32 " | stack_free=%u",
                     audio_blocks, processed_blocks, audio_errors, audio_overflows,
                     (unsigned)uxTaskGetStackHighWaterMark(NULL));

            log_memory_status("window");

            eco_features_reset(&features);
            sum_squares = 0.0;
            window_start_us = esp_timer_get_time();
        }
    }

    ESP_LOGE(TAG, "DSP task stopped");
    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "Edge Ecoacoustics - Continuous CNN float32");
    ESP_ERROR_CHECK(eco_features_init());

    audio_ring = xRingbufferCreate(RING_BYTES, RINGBUF_TYPE_NOSPLIT);
    assert(audio_ring != NULL);

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

    log_memory_status("before_cnn");
    ESP_ERROR_CHECK(eco_inference_init());
    log_memory_status("after_cnn");

    BaseType_t result = xTaskCreate(dsp_task, "dsp_task", DSP_STACK, NULL, 5, NULL);
    assert(result == pdPASS);

    result = xTaskCreate(capture_task, "audio_capture", CAPTURE_STACK, NULL, 7, NULL);
    assert(result == pdPASS);

    ESP_LOGI(TAG, "Continuous capture | 48 kHz | left slot | CNN float32 every 6 seconds");
}
