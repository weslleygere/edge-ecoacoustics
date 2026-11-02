
/*
 * Edge Ecoacoustics
 *
 * ESP32-S3 / ESP-IDF 6.x
 *
 * Stage 2:
 * - Deterministic Log-Mel validation
 * - Continuous I2S acquisition
 * - Streaming ESP-DSP features
 * - DC and RMS diagnostics
 *
 * Audio: 48 kHz, PCM16, 20-ms blocks.
 * Features: 299 frames x 40 Mel bands / 6 seconds.
 */

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

/* Actual pins must match the microphone wiring. */
#define MIC_BCLK_GPIO GPIO_NUM_4
#define MIC_WS_GPIO   GPIO_NUM_5
#define MIC_DIN_GPIO  GPIO_NUM_6

#define DMA_FRAME_NUM 480
#define DMA_DESC_NUM  8

#define RING_BYTES    (96 * 1024)
#define CAPTURE_STACK 6144
#define DSP_STACK     6144

#define READ_TIMEOUT_MS 1000
#define RING_TIMEOUT_MS 100

#define RAW_WORDS (ECO_BLOCK_SAMPLES * 2)
#define BLOCKS_PER_WINDOW \
    (ECO_SEGMENT_SAMPLES / ECO_BLOCK_SAMPLES)

#define DIAGNOSTIC_INTERVAL_BLOCKS 300

/* Set to 0 after completing the validation. */
#define ENABLE_DSP_SELFTEST 0

/* The microphone was confirmed on the LEFT I2S slot. */
#define AUDIO_SLOT_INDEX 0

static const char *TAG = "eco_audio";

static RingbufHandle_t audio_ring = NULL;
static i2s_chan_handle_t i2s_rx = NULL;

static int32_t raw_buffer[RAW_WORDS];
static int16_t mono_buffer[ECO_BLOCK_SAMPLES];

static eco_features_t feature_context;

/* Reused for synthetic validation before I2S starts. */
static int16_t test_block[ECO_BLOCK_SAMPLES];

static volatile uint32_t audio_blocks = 0;
static volatile uint32_t audio_errors = 0;
static volatile uint32_t audio_overflows = 0;
static volatile bool capture_stopped = false;

/*
 * ------------------------------------------------------------
 * SYNTHETIC TEST SIGNAL
 * ------------------------------------------------------------
 *
 * Periodic triangular ramp:
 *
 * sample[n] = ((n % 48) * 500) - 12000
 *
 * This generates a deterministic PCM16 signal.
 * Its period is 48 samples (1 ms at 48 kHz).
 *
 * We use integer arithmetic on both ESP32 and Python
 * to guarantee identical input samples.
 */
static void generate_test_block(
    int16_t *buffer,
    uint32_t block_index
)
{
    uint32_t first_sample =
        block_index * ECO_BLOCK_SAMPLES;

    for (size_t i = 0; i < ECO_BLOCK_SAMPLES; ++i) {
        uint32_t n = first_sample + i;

        buffer[i] = (int16_t)(
            ((n % 48U) * 500U) - 12000
        );
    }
}

/*
 * Print 40 coefficients in groups of 10.
 * This avoids long serial log lines.
 *
 * The Python validation script parses this format.
 */
static void print_test_frame(
    const eco_features_t *ctx,
    int frame_index
)
{
    for (int start = 0; start < ECO_MEL_BANDS; start += 10) {
        char values[180];
        size_t used = 0;

        values[0] = '\0';

        for (int j = 0; j < 10; ++j) {
            int m = start + j;

            int written = snprintf(
                values + used,
                sizeof(values) - used,
                j == 0 ? "%.6f" : ",%.6f",
                ctx->logmel[frame_index][m]
            );

            if (written < 0 ||
                (size_t)written >= sizeof(values) - used) {
                ESP_LOGE(TAG, "DSP_TEST output truncated");
                return;
            }

            used += (size_t)written;
        }

        ESP_LOGI(
            TAG,
            "DSP_TEST frame=%d start=%d values=%s",
            frame_index,
            start,
            values
        );
    }
}

/*
 * ------------------------------------------------------------
 * LOG-MEL SELF-TEST
 * ------------------------------------------------------------
 *
 * Runs before the microphone is enabled.
 *
 * Generates exactly 288000 synthetic PCM16 samples
 * and feeds them in 300 blocks of 960 samples.
 *
 * This exercises the same streaming features.c used
 * by the actual audio processing task.
 *
 * Frames 0, 149 and 298 are exported for comparison
 * against model/features.py.
 */
static esp_err_t run_dsp_selftest(void)
{
    ESP_LOGI(TAG, "========== DSP SELFTEST START ==========");

    eco_features_reset(&feature_context);

    int64_t start_us = esp_timer_get_time();

    for (uint32_t block = 0;
         block < BLOCKS_PER_WINDOW;
         ++block) {

        generate_test_block(test_block, block);

        esp_err_t err = eco_features_push(
            &feature_context,
            test_block,
            ECO_BLOCK_SAMPLES
        );

        if (err != ESP_OK) {
            ESP_LOGE(
                TAG,
                "DSP_TEST FAIL: feature extraction: %s",
                esp_err_to_name(err)
            );

            return err;
        }

        /*
         * Yield occasionally during the synthetic test.
         */
        if ((block % 25U) == 24U) {
            vTaskDelay(1);
        }
    }

    if (!eco_features_complete(&feature_context)) {
        ESP_LOGE(
            TAG,
            "DSP_TEST FAIL: samples=%u frames=%u",
            (unsigned)feature_context.samples_received,
            (unsigned)feature_context.frames_generated
        );

        return ESP_FAIL;
    }

    const int frames_to_check[] = {
        0,
        149,
        298
    };

    for (size_t i = 0;
         i < sizeof(frames_to_check) /
             sizeof(frames_to_check[0]);
         ++i) {

        int frame = frames_to_check[i];

        for (int m = 0; m < ECO_MEL_BANDS; ++m) {
            if (!isfinite(
                feature_context.logmel[frame][m]
            )) {
                ESP_LOGE(
                    TAG,
                    "DSP_TEST FAIL: non-finite value"
                );

                return ESP_FAIL;
            }
        }

        print_test_frame(
            &feature_context,
            frame
        );
    }

    double elapsed_ms =
        (double)(esp_timer_get_time() - start_us) / 1000.0;

    ESP_LOGI(
        TAG,
        "DSP_TEST frames=%u samples=%u elapsed_ms=%.2f",
        (unsigned)feature_context.frames_generated,
        (unsigned)feature_context.samples_received,
        elapsed_ms
    );

    ESP_LOGI(
        TAG,
        "DSP_TEST generated successfully; "
        "compare with Python before declaring PASS"
    );

    ESP_LOGI(TAG, "========== DSP SELFTEST END ==========");

    /*
     * Important: don't mix synthetic and microphone data.
     */
    eco_features_reset(&feature_context);

    return ESP_OK;
}

/*
 * ------------------------------------------------------------
 * OPTIONAL I2S CHANNEL DIAGNOSTICS
 * ------------------------------------------------------------
 */
static void log_i2s_diagnostics(const int32_t *raw)
{
    uint32_t left_nonzero = 0;
    uint32_t right_pcm_nonzero = 0;

    int16_t left_min = INT16_MAX;
    int16_t left_max = INT16_MIN;

    for (size_t i = 0; i < ECO_BLOCK_SAMPLES; ++i) {
        int16_t left = (int16_t)(raw[2 * i] >> 16);
        int16_t right = (int16_t)(raw[2 * i + 1] >> 16);

        if (left != 0) {
            left_nonzero++;
        }

        if (right != 0) {
            right_pcm_nonzero++;
        }

        if (left < left_min) {
            left_min = left;
        }

        if (left > left_max) {
            left_max = left;
        }
    }

    ESP_LOGI(
        TAG,
        "I2S LEFT pcm_nonzero=%" PRIu32
        "/%d min=%d max=%d | RIGHT pcm_nonzero=%" PRIu32,
        left_nonzero,
        ECO_BLOCK_SAMPLES,
        (int)left_min,
        (int)left_max,
        right_pcm_nonzero
    );
}

/*
 * ------------------------------------------------------------
 * CONTINUOUS I2S CAPTURE
 * ------------------------------------------------------------
 */
static void capture_task(void *arg)
{
    (void)arg;

    esp_err_t err = i2s_channel_enable(i2s_rx);

    if (err != ESP_OK) {
        audio_errors++;
        capture_stopped = true;

        ESP_LOGE(
            TAG,
            "Cannot enable I2S: %s",
            esp_err_to_name(err)
        );

        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Continuous audio capture started");

    while (true) {
        size_t received = 0;

        /*
         * Accumulate partial DMA reads until 20 ms of
         * stereo samples have been received.
         */
        while (received < sizeof(raw_buffer)) {
            size_t bytes_read = 0;

            err = i2s_channel_read(
                i2s_rx,
                (uint8_t *)raw_buffer + received,
                sizeof(raw_buffer) - received,
                &bytes_read,
                pdMS_TO_TICKS(READ_TIMEOUT_MS)
            );

            if (err != ESP_OK || bytes_read == 0) {
                audio_errors++;

                ESP_LOGE(
                    TAG,
                    "I2S read failure: %s",
                    esp_err_to_name(err)
                );

                goto capture_exit;
            }

            received += bytes_read;
        }

        /*
         * The microphone has been identified in LEFT.
         * Convert its upper 16 bits into provisional PCM16.
         */
        for (size_t i = 0; i < ECO_BLOCK_SAMPLES; ++i) {
            mono_buffer[i] = (int16_t)(
                raw_buffer[2 * i + AUDIO_SLOT_INDEX] >> 16
            );
        }

        if ((audio_blocks %
             DIAGNOSTIC_INTERVAL_BLOCKS) == 0) {
            log_i2s_diagnostics(raw_buffer);
        }

        BaseType_t ok = xRingbufferSend(
            audio_ring,
            mono_buffer,
            sizeof(mono_buffer),
            pdMS_TO_TICKS(RING_TIMEOUT_MS)
        );

        if (ok != pdTRUE) {
            audio_overflows++;

            ESP_LOGE(
                TAG,
                "Ring Buffer overflow: block=%" PRIu32,
                audio_blocks
            );

            goto capture_exit;
        }

        audio_blocks++;
    }

capture_exit:

    capture_stopped = true;

    err = i2s_channel_disable(i2s_rx);

    if (err != ESP_OK) {
        ESP_LOGE(
            TAG,
            "I2S disable failed: %s",
            esp_err_to_name(err)
        );
    }

    ESP_LOGE(
        TAG,
        "Capture stopped | blocks=%" PRIu32
        " errors=%" PRIu32
        " overflows=%" PRIu32,
        audio_blocks,
        audio_errors,
        audio_overflows
    );

    vTaskDelete(NULL);
}

/*
 * ------------------------------------------------------------
 * STREAMING DSP TASK
 * ------------------------------------------------------------
 */
static void dsp_task(void *arg)
{
    (void)arg;

    uint32_t windows = 0;
    uint32_t blocks_processed = 0;

    /*
     * Diagnostic statistics. These do not modify
     * the audio passed to eco_features_push().
     */
    double sum = 0.0;
    double sum_squares = 0.0;
    uint32_t sample_count = 0;

    eco_features_reset(&feature_context);

    int64_t window_start_us = esp_timer_get_time();

    ESP_LOGI(TAG, "DSP task started");

    while (true) {
        size_t bytes = 0;

        int16_t *audio = (int16_t *)xRingbufferReceive(
            audio_ring,
            &bytes,
            pdMS_TO_TICKS(100)
        );

        if (!audio) {
            if (capture_stopped) {
                ESP_LOGE(TAG, "DSP stopped: capture unavailable");
                break;
            }

            continue;
        }

        if (bytes != ECO_BLOCK_SAMPLES * sizeof(int16_t)) {
            ESP_LOGE(
                TAG,
                "Unexpected audio block size: %u",
                (unsigned)bytes
            );

            vRingbufferReturnItem(audio_ring, audio);
            break;
        }

        for (size_t i = 0; i < ECO_BLOCK_SAMPLES; ++i) {
            double x = (double)audio[i] / 32768.0;

            sum += x;
            sum_squares += x * x;
        }

        sample_count += ECO_BLOCK_SAMPLES;

        /*
         * Preserve the original samples.
         * Do not remove DC or alter the feature pipeline.
         */
        esp_err_t err = eco_features_push(
            &feature_context,
            audio,
            ECO_BLOCK_SAMPLES
        );

        vRingbufferReturnItem(audio_ring, audio);

        if (err != ESP_OK) {
            ESP_LOGE(
                TAG,
                "Log-Mel processing failed: %s",
                esp_err_to_name(err)
            );
            break;
        }

        blocks_processed++;

        if (eco_features_complete(&feature_context)) {
            windows++;

            double mean = sum / sample_count;
            double mean_square = sum_squares / sample_count;

            double rms = sqrt(mean_square);

            double variance = mean_square - mean * mean;

            if (variance < 0.0) {
                variance = 0.0;
            }

            double rms_without_dc = sqrt(variance);

            float first_mel =
                feature_context.logmel[0][0];

            float last_mel =
                feature_context.logmel
                    [ECO_FRAMES_PER_SEGMENT - 1]
                    [ECO_MEL_BANDS - 1];

            double elapsed_s =
                (double)(esp_timer_get_time() -
                         window_start_us) / 1e6;

            ESP_LOGI(
                TAG,
                "Window %" PRIu32
                " | frames=%u"
                " | RMS=%.6f"
                " | mean=%.6f"
                " | RMS_no_DC=%.6f",
                windows,
                (unsigned)feature_context.frames_generated,
                rms,
                mean,
                rms_without_dc
            );

            ESP_LOGI(
                TAG,
                "mel_first=%.4f"
                " mel_last=%.4f"
                " elapsed=%.2fs",
                first_mel,
                last_mel,
                elapsed_s
            );

            ESP_LOGI(
                TAG,
                "Blocks=%" PRIu32
                " processed=%" PRIu32
                " errors=%" PRIu32
                " overflows=%" PRIu32
                " heap=%u stack_free=%u",
                audio_blocks,
                blocks_processed,
                audio_errors,
                audio_overflows,
                (unsigned)esp_get_free_heap_size(),
                (unsigned)uxTaskGetStackHighWaterMark(NULL)
            );

            if (!isfinite(first_mel) ||
                !isfinite(last_mel)) {
                ESP_LOGE(TAG, "Non-finite Log-Mel value");
            }

            /*
             * Future:
             * - Normalize Log-Mel
             * - Quantize INT8
             * - TFLite Micro inference
             * - MQTT queue
             */
            eco_features_reset(&feature_context);

            sum = 0.0;
            sum_squares = 0.0;
            sample_count = 0;

            window_start_us = esp_timer_get_time();
        }
    }

    ESP_LOGE(TAG, "DSP task ended");
    vTaskDelete(NULL);
}

/*
 * ------------------------------------------------------------
 * APPLICATION ENTRY POINT
 * ------------------------------------------------------------
 */
void app_main(void)
{
    ESP_LOGI(TAG, "Edge Ecoacoustics - DSP validation");

    ESP_ERROR_CHECK(eco_features_init());

#if ENABLE_DSP_SELFTEST
    ESP_ERROR_CHECK(run_dsp_selftest());
#endif

    audio_ring = xRingbufferCreate(
        RING_BYTES,
        RINGBUF_TYPE_NOSPLIT
    );

    assert(audio_ring != NULL);

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

    i2s_std_config_t cfg = {
        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(ECO_SAMPLE_RATE),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
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
        i2s_channel_init_std_mode(i2s_rx, &cfg)
    );

    BaseType_t result = xTaskCreate(
        dsp_task,
        "dsp_task",
        DSP_STACK,
        NULL,
        5,
        NULL
    );

    assert(result == pdPASS);

    result = xTaskCreate(
        capture_task,
        "audio_capture",
        CAPTURE_STACK,
        NULL,
        7,
        NULL
    );

    assert(result == pdPASS);

    ESP_LOGI(
        TAG,
        "Continuous capture enabled | 48 kHz"
        " | left slot | 6-second windows"
    );
}
