
/*
 * Edge Ecoacoustics
 *
 * Streaming Log-Mel feature extraction.
 *
 * ESP32-S3 / ESP-IDF 6.x / ESP-DSP
 *
 * Reference: model/features.py
 *
 * 48 kHz mono PCM16
 * 6 seconds per segment
 * 1200-sample periodic Hann
 * 960-sample hop
 * 2048-point complex FFT with zero imaginary input
 * Power spectrum = abs(FFT)^2 / 2048
 * 40 HTK Mel filters, 60 Hz to 20000 Hz
 * Natural logarithm, floor 1e-10
 * 299 x 40 Log-Mel output
 */

#include "feature.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "esp_dsp.h"

#define ECO_PI 3.14159265358979323846f
#define MEL_FMIN 60.0f
#define MEL_FMAX 20000.0f
#define LOG_FLOOR 1e-10f

#define FFT_BINS (ECO_FFT_LENGTH / 2 + 1)

/*
 * Precomputed analysis window and Mel filter boundaries.
 */
static float hann[ECO_FRAME_LENGTH];
static float mel_edges[ECO_MEL_BANDS + 2];

/*
 * Complex FFT buffer:
 *
 * [Re(0), Im(0), Re(1), Im(1), ...]
 *
 * Even though the original signal is real, the complex FFT
 * receives zero imaginary components.
 */
static __attribute__((aligned(16)))
float fft_buffer[2 * ECO_FFT_LENGTH];

/*
 * One-sided power spectrum, including DC and Nyquist.
 */
static float power_spectrum[FFT_BINS];

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f *
        (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/*
 * Initialize FFT tables and feature-extraction constants.
 * Call once before starting acquisition.
 */
esp_err_t eco_features_init(void)
{
    /*
     * Periodic Hann:
     *
     * w[n] = 0.5 - 0.5*cos(2*pi*n/1200)
     *
     * Note: denominator is 1200, not 1199.
     */
    for (int i = 0; i < ECO_FRAME_LENGTH; ++i) {
        hann[i] =
            0.5f -
            0.5f * cosf(
                2.0f * ECO_PI * (float)i /
                (float)ECO_FRAME_LENGTH
            );
    }

    /*
     * Equally spaced HTK Mel points converted to Hz.
     *
     * 40 triangular filters require 42 boundary points.
     */
    float mel_min = hz_to_mel(MEL_FMIN);
    float mel_max = hz_to_mel(MEL_FMAX);

    for (int i = 0; i < ECO_MEL_BANDS + 2; ++i) {
        float mel =
            mel_min +
            (mel_max - mel_min) *
            (float)i / (float)(ECO_MEL_BANDS + 1);

        mel_edges[i] = mel_to_hz(mel);
    }

    return dsps_fft2r_init_fc32(
        NULL,
        ECO_FFT_LENGTH
    );
}

/*
 * Reset the current six-second segment.
 */
void eco_features_reset(eco_features_t *ctx)
{
    if (ctx == NULL) {
        return;
    }

    ctx->buffered = 0;
    ctx->samples_received = 0;
    ctx->frames_generated = 0;
}

/*
 * Process one 1200-sample analysis frame.
 */
static esp_err_t process_frame(
    const float *samples,
    float *output
)
{
    /*
     * Clear real and imaginary FFT components.
     */
    memset(
        fft_buffer,
        0,
        sizeof(fft_buffer)
    );

    /*
     * Apply Hann window to 1200 real samples.
     * The remaining 848 samples are right-zero-padded.
     */
    for (int i = 0; i < ECO_FRAME_LENGTH; ++i) {
        fft_buffer[2 * i] =
            samples[i] * hann[i];

        fft_buffer[2 * i + 1] = 0.0f;
    }

    /*
     * Perform 2048-point COMPLEX FFT.
     *
     * Input imaginary components are already zero.
     */
    esp_err_t err = dsps_fft2r_fc32(
        fft_buffer,
        ECO_FFT_LENGTH
    );

    if (err != ESP_OK) {
        return err;
    }

    /*
     * Reorder FFT output into natural frequency order.
     */
    err = dsps_bit_rev_fc32(
        fft_buffer,
        ECO_FFT_LENGTH
    );

    if (err != ESP_OK) {
        return err;
    }

    /*
     * IMPORTANT FIX:
     *
     * Do NOT call dsps_cplx2reC_fc32() here.
     *
     * Our input is one real signal stored as a complex
     * sequence with imaginary components set to zero.
     *
     * The ordinary complex FFT already gives the
     * frequency bins needed to match numpy.fft.rfft().
     *
     * Calling dsps_cplx2reC_fc32() would incorrectly
     * change their amplitudes.
     */

    /*
     * One-sided power spectrum:
     *
     * P[k] = (Re[k]^2 + Im[k]^2) / 2048
     *
     * Bins 0 through 1024, inclusive.
     *
     * No additional factor of 2.
     */
    for (int k = 0; k < FFT_BINS; ++k) {
        float re = fft_buffer[2 * k];
        float im = fft_buffer[2 * k + 1];

        power_spectrum[k] =
            (re * re + im * im) /
            (float)ECO_FFT_LENGTH;
    }

    /*
     * Apply unit-height HTK triangular Mel filters.
     *
     * No Slaney area normalization.
     */
    for (int m = 0; m < ECO_MEL_BANDS; ++m) {
        float left = mel_edges[m];
        float center = mel_edges[m + 1];
        float right = mel_edges[m + 2];

        float energy = 0.0f;

        for (int k = 0; k < FFT_BINS; ++k) {
            float hz =
                (float)k *
                (float)ECO_SAMPLE_RATE /
                (float)ECO_FFT_LENGTH;

            if (hz <= left || hz >= right) {
                continue;
            }

            float weight;

            if (hz <= center) {
                weight =
                    (hz - left) /
                    (center - left);
            } else {
                weight =
                    (right - hz) /
                    (right - center);
            }

            energy +=
                power_spectrum[k] * weight;
        }

        /*
         * Natural logarithm with numerical floor.
         */
        if (energy < LOG_FLOOR) {
            energy = LOG_FLOOR;
        }

        output[m] = logf(energy);
    }

    return ESP_OK;
}

/*
 * Push successive PCM16 blocks into the streaming extractor.
 *
 * The function produces a Log-Mel frame whenever 1200
 * samples are available.
 *
 * Consecutive frames advance by 960 samples.
 */
esp_err_t eco_features_push(
    eco_features_t *ctx,
    const int16_t *samples,
    size_t count
)
{
    if (ctx == NULL || samples == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (count >
        ECO_SEGMENT_SAMPLES - ctx->samples_received) {
        return ESP_ERR_INVALID_SIZE;
    }

    size_t offset = 0;

    while (offset < count) {
        size_t capacity =
            ECO_FRAME_LENGTH - ctx->buffered;

        size_t take = count - offset;

        if (take > capacity) {
            take = capacity;
        }

        /*
         * Convert PCM16 to float in approximately [-1, 1).
         *
         * No DC removal or normalization is performed.
         */
        for (size_t i = 0; i < take; ++i) {
            ctx->pcm[ctx->buffered + i] =
                (float)samples[offset + i] /
                32768.0f;
        }

        ctx->buffered += take;
        ctx->samples_received += take;
        offset += take;

        /*
         * Once 1200 samples are available,
         * generate one 40-band Log-Mel frame.
         */
        if (ctx->buffered == ECO_FRAME_LENGTH) {
            if (ctx->frames_generated >=
                ECO_FRAMES_PER_SEGMENT) {
                return ESP_ERR_INVALID_SIZE;
            }

            esp_err_t err = process_frame(
                ctx->pcm,
                ctx->logmel[ctx->frames_generated]
            );

            if (err != ESP_OK) {
                return err;
            }

            ctx->frames_generated++;

            /*
             * Preserve the 240-sample overlap:
             *
             * 1200 - 960 = 240
             */
            memmove(
                ctx->pcm,
                ctx->pcm + ECO_HOP_LENGTH,
                (ECO_FRAME_LENGTH - ECO_HOP_LENGTH) *
                    sizeof(float)
            );

            ctx->buffered =
                ECO_FRAME_LENGTH - ECO_HOP_LENGTH;
        }
    }

    return ESP_OK;
}

/*
 * A complete 6-second segment contains 288000 samples
 * and exactly 299 analysis frames.
 */
bool eco_features_complete(const eco_features_t *ctx)
{
    return ctx != NULL &&
        ctx->samples_received == ECO_SEGMENT_SAMPLES &&
        ctx->frames_generated == ECO_FRAMES_PER_SEGMENT;
}
