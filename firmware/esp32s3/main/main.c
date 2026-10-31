
/*
 * Edge Ecoacoustics
 *
 * Stage 1: I2S DMA -> FreeRTOS Ring Buffer
 *
 * ESP32-S3 / ESP-IDF 6.x
 * 48 kHz, right channel, 20 ms blocks, 60 s capture.
 *
 * No DSP, neural inference, network or deep sleep yet.
 */

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

/* Change these to your actual microphone wiring. */
#define MIC_BCLK_GPIO GPIO_NUM_4
#define MIC_WS_GPIO   GPIO_NUM_5
#define MIC_DIN_GPIO  GPIO_NUM_6

#define SAMPLE_RATE       48000
#define BLOCK_MS          20
#define BLOCK_SAMPLES     960
#define TOTAL_BLOCKS      3000

#define I2S_SLOTS         2
#define I2S_SAMPLE_BYTES  4

#define DMA_FRAME_NUM     480
#define DMA_DESC_NUM      8

#define RING_BYTES        (96 * 1024)
#define CAPTURE_STACK     6144
#define PROCESS_STACK     4096

#define READ_TIMEOUT_MS   1000
#define RING_TIMEOUT_MS   100

#define RAW_BYTES (BLOCK_SAMPLES * I2S_SLOTS * I2S_SAMPLE_BYTES)

static const char *TAG = "eco_audio";

static RingbufHandle_t ring = NULL;
static i2s_chan_handle_t rx = NULL;

/*
 * Static buffers: do not consume the capture task stack.
 *
 * Raw:   960 stereo frames * 2 channels * 4 bytes = 7680 B
 * Right: 960 mono samples * 2 bytes            = 1920 B
 */
static int32_t raw_buffer[BLOCK_SAMPLES * I2S_SLOTS];
static int16_t right_buffer[BLOCK_SAMPLES];

static volatile bool acquisition_done = false;
static volatile uint32_t overflow_blocks = 0;
static volatile uint32_t read_errors = 0;

static void capture_task(void *arg)
{
    (void)arg;

    esp_err_t err = i2s_channel_enable(rx);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Cannot enable I2S: %s",
                 esp_err_to_name(err));
        read_errors++;
        acquisition_done = true;
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Acquisition started");

    for (uint32_t block = 0; block < TOTAL_BLOCKS; block++) {

        /*
         * A single I2S read is not guaranteed to fill the
         * requested buffer. Accumulate until exactly 20 ms.
         */
        size_t total_bytes = 0;

        while (total_bytes < sizeof(raw_buffer)) {

            size_t bytes_read = 0;

            err = i2s_channel_read(
                rx,
                (uint8_t *)raw_buffer + total_bytes,
                sizeof(raw_buffer) - total_bytes,
                &bytes_read,
                pdMS_TO_TICKS(READ_TIMEOUT_MS)
            );

            if (err != ESP_OK || bytes_read == 0) {
                ESP_LOGE(
                    TAG,
                    "I2S read error: block=%" PRIu32
                    ", bytes=%u/%u, err=%s",
                    block,
                    (unsigned)total_bytes,
                    (unsigned)sizeof(raw_buffer),
                    esp_err_to_name(err)
                );

                read_errors++;
                goto capture_finished;
            }

            total_bytes += bytes_read;
        }

        /*
         * Philips stereo I2S:
         * [left_0, right_0, left_1, right_1, ...]
         *
         * Provisional PCM16 representation:
         * retain the upper 16 bits of the right 32-bit slot.
         *
         * The actual microphone's alignment/gain must
         * still be verified before running the CNN.
         */
        for (size_t i = 0; i < BLOCK_SAMPLES; i++) {
            right_buffer[i] =
                (int16_t)(raw_buffer[2 * i + 1] >> 16);
        }

        /*
         * Preserve one complete 20-ms block per ring item.
         * Never silently discard audio.
         */
        BaseType_t sent = xRingbufferSend(
            ring,
            right_buffer,
            sizeof(right_buffer),
            pdMS_TO_TICKS(RING_TIMEOUT_MS)
        );

        if (sent != pdTRUE) {
            overflow_blocks++;

            ESP_LOGE(
                TAG,
                "Ring Buffer overflow at block %" PRIu32,
                block
            );

            break;
        }
    }

capture_finished:

    err = i2s_channel_disable(rx);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Cannot disable I2S: %s",
                 esp_err_to_name(err));
    }

    acquisition_done = true;

    ESP_LOGI(
        TAG,
        "Acquisition ended. Overflows=%" PRIu32
        ", read_errors=%" PRIu32,
        overflow_blocks,
        read_errors
    );

    vTaskDelete(NULL);
}

static void processing_task(void *arg)
{
    (void)arg;

    uint32_t received_blocks = 0;
    uint64_t received_samples = 0;
    double sum_squares = 0.0;

    int64_t start_us = esp_timer_get_time();

    while (received_blocks < TOTAL_BLOCKS) {

        size_t bytes = 0;

        int16_t *item = (int16_t *)xRingbufferReceive(
            ring,
            &bytes,
            pdMS_TO_TICKS(100)
        );

        if (item == NULL) {
            if (acquisition_done) {
                break;
            }
            continue;
        }

        if (bytes != sizeof(right_buffer)) {
            ESP_LOGE(
                TAG,
                "Invalid ring item size: %u bytes",
                (unsigned)bytes
            );

            vRingbufferReturnItem(ring, item);
            break;
        }

        for (size_t i = 0; i < BLOCK_SAMPLES; i++) {
            double value = (double)item[i] / 32768.0;
            sum_squares += value * value;
        }

        received_samples += BLOCK_SAMPLES;
        received_blocks++;

        /*
         * Return the item promptly so the producer can
         * reuse ring buffer memory.
         */
        vRingbufferReturnItem(ring, item);

        if (received_blocks % 300 == 0) {
            ESP_LOGI(
                TAG,
                "Processed %.1f seconds (%" PRIu32 " blocks)",
                (double)received_samples / SAMPLE_RATE,
                received_blocks
            );
        }
    }

    double rms = 0.0;

    if (received_samples > 0) {
        rms = sqrt(
            sum_squares / (double)received_samples
        );
    }

    double elapsed_s =
        (double)(esp_timer_get_time() - start_us) / 1e6;

    ESP_LOGI(TAG, "==============================");
    ESP_LOGI(TAG, "ACQUISITION REPORT");
    ESP_LOGI(
        TAG,
        "Blocks: %" PRIu32 " / %d",
        received_blocks,
        TOTAL_BLOCKS
    );
    ESP_LOGI(
        TAG,
        "Samples: %" PRIu64,
        received_samples
    );
    ESP_LOGI(
        TAG,
        "Read errors: %" PRIu32,
        read_errors
    );
    ESP_LOGI(
        TAG,
        "Ring overflows: %" PRIu32,
        overflow_blocks
    );
    ESP_LOGI(TAG, "RMS: %.6f", rms);
    ESP_LOGI(TAG, "Elapsed: %.2f s", elapsed_s);

    if (received_blocks == TOTAL_BLOCKS &&
        read_errors == 0 &&
        overflow_blocks == 0) {

        ESP_LOGI(TAG, "RESULT: PASS");
    } else {
        ESP_LOGE(TAG, "RESULT: FAIL");
    }

    ESP_LOGI(TAG, "==============================");

    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "Edge Ecoacoustics - I2S acquisition");
    ESP_LOGI(TAG, "Sample rate: %d Hz", SAMPLE_RATE);
    ESP_LOGI(TAG, "Block: %d samples / %d ms",
             BLOCK_SAMPLES, BLOCK_MS);
    ESP_LOGI(TAG, "Capture: %d blocks / 60 seconds",
             TOTAL_BLOCKS);

    /*
     * NO_SPLIT preserves each 20-ms PCM block as one
     * complete item for the consumer task.
     */
    ring = xRingbufferCreate(
        RING_BYTES,
        RINGBUF_TYPE_NOSPLIT
    );

    assert(ring != NULL);

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_AUTO,
            I2S_ROLE_MASTER
        );

    chan_cfg.dma_desc_num = DMA_DESC_NUM;
    chan_cfg.dma_frame_num = DMA_FRAME_NUM;

    ESP_ERROR_CHECK(
        i2s_new_channel(&chan_cfg, NULL, &rx)
    );

    i2s_std_config_t i2s_cfg = {
        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

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
        i2s_channel_init_std_mode(rx, &i2s_cfg)
    );

    /*
     * Start consumer before producer.
     */
    BaseType_t result = xTaskCreate(
        processing_task,
        "audio_processing",
        PROCESS_STACK,
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

    ESP_LOGI(TAG, "Tasks started");
}
