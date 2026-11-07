/*
 * tinyCLAP frontend matched to model/tinyclap_indices.py:
 * 44.1 kHz, 6 s PCM, Hann(periodic), FFT=1024, hop=320,
 * center=True, reflect padding, power=abs(FFT)^2,
 * librosa Slaney Mel (50..14000 Hz, 64 bands), 10*log10(max(x,1e-10)).
 * The Slaney filter matrix is generated offline by build_tinyclap_assets.py.
 */
#include "feature.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "esp_dsp.h"
#include "esp_log.h"

// Validation-only: set 0 to remove the synthetic Log-Mel self-test.
#define ECO_LOGMEL_SELFTEST 1
#if ECO_LOGMEL_SELFTEST
#include "esp_heap_caps.h"
#include "logmel_validation_reference.h"
static void run_logmel_selftest(void);
#endif

#define FFT_BINS (ECO_FFT_LENGTH / 2 + 1)
#define LOG_FLOOR 1e-10f
#define PI_F 3.14159265358979323846f

static const char *TAG = "eco_features";

extern const uint8_t mel_start[] asm("_binary_tinyclap_mel_filters_bin_start");
extern const uint8_t mel_end[]   asm("_binary_tinyclap_mel_filters_bin_end");

static const float *mel_weights = NULL;   // [64, 513], stored in Flash
static float hann[ECO_FFT_LENGTH];
static __attribute__((aligned(16))) float fft[2 * ECO_FFT_LENGTH];
static float power[FFT_BINS];
static bool initialized = false;

static int reflect_index(int index)
{
    // The only out-of-bounds distances are <= 512 for this configuration.
    if (index < 0) index = -index;
    if (index >= ECO_SEGMENT_SAMPLES) {
        index = 2 * ECO_SEGMENT_SAMPLES - 2 - index;
    }
    return index;
}

esp_err_t eco_features_init(void)
{
    const size_t expected = sizeof(float) * ECO_MEL_BANDS * FFT_BINS;
    const size_t actual = (size_t)(mel_end - mel_start);
    if (actual != expected) {
        ESP_LOGE(TAG, "Invalid mel matrix size: %u, expected %u",
                 (unsigned)actual, (unsigned)expected);
        return ESP_ERR_INVALID_SIZE;
    }
    if (((uintptr_t)mel_start % _Alignof(float)) != 0) {
        ESP_LOGE(TAG, "Mel matrix is unaligned");
        return ESP_ERR_INVALID_STATE;
    }
    mel_weights = (const float *)mel_start;
    for (size_t i = 0; i < ECO_FFT_LENGTH; ++i) {
        hann[i] = 0.5f - 0.5f * cosf(2.0f * PI_F * (float)i /
                                     (float)ECO_FFT_LENGTH);
    }
    esp_err_t err = dsps_fft2r_init_fc32(NULL, ECO_FFT_LENGTH);
    if (err == ESP_OK) {
        initialized = true;
        ESP_LOGI(TAG, "Frontend ready | %d Hz | 827x64 | Slaney mel | dB",
                 ECO_SAMPLE_RATE);
#if ECO_LOGMEL_SELFTEST
        run_logmel_selftest();
#endif
    }
    return err;
}

esp_err_t eco_features_compute(const int16_t *pcm, size_t samples,
                               eco_features_t *output)
{
    if (!initialized || pcm == NULL || output == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    if (samples != ECO_SEGMENT_SAMPLES) return ESP_ERR_INVALID_SIZE;

    for (size_t frame = 0; frame < ECO_FRAMES_PER_SEGMENT; ++frame) {
        const int center = (int)(frame * ECO_HOP_LENGTH);
        for (int i = 0; i < ECO_FFT_LENGTH; ++i) {
            const int pos = reflect_index(center + i - ECO_FFT_LENGTH / 2);
            fft[2 * i] = ((float)pcm[pos] / 32768.0f) * hann[i];
            fft[2 * i + 1] = 0.0f;
        }
        esp_err_t err = dsps_fft2r_fc32(fft, ECO_FFT_LENGTH);
        if (err != ESP_OK) return err;
        err = dsps_bit_rev_fc32(fft, ECO_FFT_LENGTH);
        if (err != ESP_OK) return err;

        for (size_t k = 0; k < FFT_BINS; ++k) {
            const float re = fft[2 * k];
            const float im = fft[2 * k + 1];
            // IMPORTANT: torchlibrosa does NOT divide spectrum by FFT size.
            power[k] = re * re + im * im;
        }
        for (size_t m = 0; m < ECO_MEL_BANDS; ++m) {
            const float *weights = mel_weights + m * FFT_BINS;
            float energy = 0.0f;
            for (size_t k = 0; k < FFT_BINS; ++k) {
                energy += power[k] * weights[k];
            }
            output->logmel[frame][m] = 10.0f * log10f(fmaxf(energy, LOG_FLOOR));
        }
    }
    return ESP_OK;
}

#if ECO_LOGMEL_SELFTEST
/* Validate all 52,928 Log-Mel values against the desktop NumPy reference.
 * Runs once at boot, before normal capture buffers are allocated.
 * Failures are logged without disrupting the regular device operation.
 */
static void run_logmel_selftest(void)
{
    int16_t *pcm = (int16_t *)heap_caps_malloc(
        ECO_SEGMENT_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    eco_features_t *features = (eco_features_t *)heap_caps_malloc(
        sizeof(eco_features_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!pcm || !features) {
        ESP_LOGE(TAG, "VALIDATION LOGMEL | insufficient PSRAM");
        heap_caps_free(features);
        heap_caps_free(pcm);
        return;
    }

    uint32_t state = 0x12345678u;
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < ECO_SEGMENT_SAMPLES; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const uint16_t bits = (uint16_t)(state >> 16);
        pcm[i] = (int16_t)bits;
        hash = (hash ^ (uint8_t)bits) * 16777619u;
        hash = (hash ^ (uint8_t)(bits >> 8)) * 16777619u;
    }
    ESP_LOGI(TAG, "VALIDATION PCM FNV1A=%08x | first8=%d,%d,%d,%d,%d,%d,%d,%d",
             (unsigned)hash, (int)pcm[0], (int)pcm[1], (int)pcm[2], (int)pcm[3],
             (int)pcm[4], (int)pcm[5], (int)pcm[6], (int)pcm[7]);

    ESP_LOGI(TAG, "VALIDATION LOGMEL start | elements=%u", (unsigned)LOGMEL_VALIDATION_ELEMENTS);
    const esp_err_t err = eco_features_compute(pcm, ECO_SEGMENT_SAMPLES, features);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "VALIDATION LOGMEL compute failed: %s", esp_err_to_name(err));
    } else {
        const float *got = &features->logmel[0][0];
        double abs_sum = 0.0, sq_sum = 0.0;
        double x_sum = 0.0, y_sum = 0.0, xx_sum = 0.0, yy_sum = 0.0, xy_sum = 0.0;
        double max_abs = 0.0;
        size_t small_01 = 0, small_1 = 0;
        for (size_t i = 0; i < LOGMEL_VALIDATION_ELEMENTS; ++i) {
            float expected;
            const uint32_t bits = kLogmelValidationReference[i];
            memcpy(&expected, &bits, sizeof(expected));
            const double x = got[i];
            const double y = expected;
            const double a = fabs(x - y);
            abs_sum += a;
            sq_sum += (x - y) * (x - y);
            if (a > max_abs) max_abs = a;
            if (a <= 0.01) ++small_01;
            if (a <= 0.1) ++small_1;
            x_sum += x; y_sum += y;
            xx_sum += x*x; yy_sum += y*y; xy_sum += x*y;
        }
        const double n = (double)LOGMEL_VALIDATION_ELEMENTS;
        const double den = (n*xx_sum - x_sum*x_sum) * (n*yy_sum - y_sum*y_sum);
        const double corr = den > 0.0 ? (n*xy_sum - x_sum*y_sum)/sqrt(den) : NAN;
        ESP_LOGI(TAG, "VALIDATION LOGMEL | MAE_dB=%.6f | RMSE_dB=%.6f | MAX_dB=%.6f | corr=%.8f",
                 abs_sum/n, sqrt(sq_sum/n), max_abs, corr);
        ESP_LOGI(TAG, "VALIDATION LOGMEL | abs_le_0.01=%u/%u | abs_le_0.1=%u/%u",
                 (unsigned)small_01, (unsigned)LOGMEL_VALIDATION_ELEMENTS,
                 (unsigned)small_1, (unsigned)LOGMEL_VALIDATION_ELEMENTS);
        float ref_first, ref_last;
        const uint32_t first_bits = kLogmelValidationReference[0];
        const uint32_t last_bits = kLogmelValidationReference[LOGMEL_VALIDATION_ELEMENTS-1];
        memcpy(&ref_first, &first_bits, sizeof(ref_first));
        memcpy(&ref_last, &last_bits, sizeof(ref_last));
        ESP_LOGI(TAG, "VALIDATION LOGMEL samples | got[0]=%.5f ref[0]=%.5f | got[last]=%.5f ref[last]=%.5f",
                 (double)got[0], (double)ref_first,
                 (double)got[LOGMEL_VALIDATION_ELEMENTS-1], (double)ref_last);
    }
    heap_caps_free(features);
    heap_caps_free(pcm);
}
#endif
