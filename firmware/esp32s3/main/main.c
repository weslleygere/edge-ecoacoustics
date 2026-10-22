// Edge Ecoacoustics, stage 1: 48-kHz I2S DMA -> FreeRTOS ring buffer.
// ESP-IDF 5.x; select actual microphone pins and confirm its bit alignment.
// Deliberately NO neural inference, DSP, radio, or sleep in this stage.
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

#define MIC_BCLK_GPIO GPIO_NUM_4  // CHANGE to actual wiring
#define MIC_WS_GPIO   GPIO_NUM_5  // CHANGE to actual wiring
#define MIC_DIN_GPIO  GPIO_NUM_6  // CHANGE to actual wiring
#define SAMPLE_RATE 48000
#define BLOCK_MS 20
#define BLOCK_SAMPLES (SAMPLE_RATE * BLOCK_MS / 1000)
#define TOTAL_BLOCKS (60000 / BLOCK_MS)
#define RING_BYTES (96 * 1024)

static const char *TAG = "eco_audio";
static RingbufHandle_t ring;
static i2s_chan_handle_t rx;
static volatile bool acquisition_done = false;
static volatile uint32_t overflow_blocks = 0;
static volatile uint32_t read_errors = 0;

// The frame contains stereo 32-bit slots, [left, right]. The right slot is
// extracted as a signed 32-bit value. The 16 highest bits are kept for
// provisional PCM16, pending microphone-specific gain / format verification.
static void capture_task(void *arg)
{
    int32_t raw[BLOCK_SAMPLES * 2];
    int16_t right[BLOCK_SAMPLES];
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
    for (uint32_t block = 0; block < TOTAL_BLOCKS; ++block) {
        size_t got = 0;
        esp_err_t err = i2s_channel_read(rx, raw, sizeof(raw), &got, pdMS_TO_TICKS(1000));
        if (err != ESP_OK || got != sizeof(raw)) {
            ++read_errors;
            ESP_LOGE(TAG, "I2S read failed at block %" PRIu32 ": %s, got %u bytes",
                     block, esp_err_to_name(err), (unsigned)got);
            break; // do not silently shift audio/label time alignment
        }
        for (size_t i = 0; i < BLOCK_SAMPLES; ++i) {
            right[i] = (int16_t)(raw[2 * i + 1] >> 16);
        }
        if (xRingbufferSend(ring, right, sizeof(right), 0) != pdTRUE) {
            ++overflow_blocks;
            ESP_LOGE(TAG, "Ring buffer overflow at block %" PRIu32, block);
            break; // lost PCM is unacceptable
        }
    }
    ESP_ERROR_CHECK(i2s_channel_disable(rx));
    acquisition_done = true;
    vTaskDelete(NULL);
}

static void processing_task(void *arg)
{
    uint32_t received_blocks = 0;
    uint64_t received_samples = 0;
    double sum_squares = 0;
    int64_t start_us = esp_timer_get_time();
    while (!acquisition_done || received_blocks < TOTAL_BLOCKS) {
        size_t bytes = 0;
        int16_t *item = xRingbufferReceive(ring, &bytes, pdMS_TO_TICKS(100));
        if (!item) {
            if (acquisition_done) break;
            continue;
        }
        if (bytes != BLOCK_SAMPLES * sizeof(int16_t)) {
            ESP_LOGE(TAG, "Unexpected ring item length: %u", (unsigned)bytes);
        } else {
            for (size_t i = 0; i < BLOCK_SAMPLES; ++i) {
                float sample = (float)item[i] / 32768.0f;
                sum_squares += (double)sample * sample;
            }
            received_samples += BLOCK_SAMPLES;
            ++received_blocks;
            if (received_blocks % 300 == 0) {
                ESP_LOGI(TAG, "Processed %.1f seconds", (double)received_samples / SAMPLE_RATE);
            }
            // Next stage: Hann -> 2048-FFT -> 40 Mel bands -> 299 frames / 6s.
        }
        vRingbufferReturnItem(ring, item);
    }
    double rms = received_samples ? sqrt(sum_squares / (double)received_samples) : 0.0;
    ESP_LOGI(TAG, "Capture finished: blocks=%" PRIu32 "/%d samples=%" PRIu64
             " overflows=%" PRIu32 " read_errors=%" PRIu32 " RMS=%.6f elapsed=%.1fs",
             received_blocks, TOTAL_BLOCKS, received_samples, overflow_blocks,
             read_errors, rms, (double)(esp_timer_get_time() - start_us) / 1e6);
    vTaskDelete(NULL);
}

void app_main(void)
{
    // NO_SPLIT preserves 20-ms block boundaries; producer and consumer run
    // concurrently, independent of driver-owned DMA ring buffers.
    ring = xRingbufferCreate(RING_BYTES, RINGBUF_TYPE_NOSPLIT);
    assert(ring != NULL);
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = BLOCK_SAMPLES;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx));
    i2s_std_config_t cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_BCLK_GPIO,
            .ws = MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_DIN_GPIO,
            .invert_flags = {0},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &cfg));
    ESP_LOGI(TAG, "I2S RX right channel: %d Hz, 20-ms blocks, 60-s recording", SAMPLE_RATE);
    // Both tasks pinned separately only if needed after actual profiling.
    BaseType_t p = xTaskCreate(processing_task, "dsp_input", 6144, NULL, 5, NULL);
    assert(p == pdPASS);
    p = xTaskCreate(capture_task, "i2s_capture", 4096, NULL, 7, NULL);
    assert(p == pdPASS);
}
