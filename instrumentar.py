from pathlib import Path
import argparse

p = argparse.ArgumentParser(description='Instrumenta main.c do edge-ecoacoustics para benchmark')
p.add_argument('--repo', type=Path, default=Path('.'))
args = p.parse_args()
source = args.repo / 'firmware/esp32s3/main/main.c'
if not source.is_file():
    raise SystemExit(f'Arquivo nao encontrado: {source}')
s = source.read_text(encoding='utf-8')
if 'BENCH,seq=' in s:
    raise SystemExit('main.c ja instrumentado. Nenhuma alteracao realizada.')

changes = [
('static eco_features_t *feature_buffer = NULL;', '''static eco_features_t *feature_buffer = NULL;

/* BENCH: memoria interna e PSRAM (bytes livres e minimo desde o boot). */
static void bench_memory(const char *phase)
{
    const uint32_t internal = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const uint32_t psram = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    ESP_LOGI(TAG,
        "BENCH_MEM,phase=%s,internal_free=%u,internal_min=%u,psram_free=%u,psram_min=%u",
        phase,
        (unsigned)heap_caps_get_free_size(internal),
        (unsigned)heap_caps_get_minimum_free_size(internal),
        (unsigned)heap_caps_get_free_size(psram),
        (unsigned)heap_caps_get_minimum_free_size(psram));
}
'''),
('    while (true) {\n        double rms = 0.0;', '''    int64_t last_cycle_start = 0;
    while (true) {
        const int64_t cycle_start = esp_timer_get_time();
        const double interval_ms = last_cycle_start == 0 ? -1.0 :
            (double)(cycle_start - last_cycle_start) / 1000.0;
        last_cycle_start = cycle_start;
        double rms = 0.0;'''),
('        esp_err_t err = capture_segment(pcm, &rms, &end_us);', '''        const int64_t capture_start = esp_timer_get_time();
        esp_err_t err = capture_segment(pcm, &rms, &end_us);
        const double capture_ms =
            (double)(esp_timer_get_time() - capture_start) / 1000.0;'''),
('        err = eco_inference_run_with_top(', '''        const int64_t run_start = esp_timer_get_time();
        err = eco_inference_run_with_top('''),
('        ++windows;\n\n        ESP_LOGI(', '''        const double run_ms =
            (double)(esp_timer_get_time() - run_start) / 1000.0;
        const double process_ms =
            (double)(esp_timer_get_time() - start_dsp) / 1000.0;
        const double cycle_ms =
            (double)(esp_timer_get_time() - cycle_start) / 1000.0;
        const double invoke_ms = (double)inference_us / 1000.0;
        /* run_extra inclui quantizacao, indices e overhead da funcao. */
        const double run_extra_ms = run_ms - invoke_ms;
        ++windows;

        if (windows <= 30) {
            ESP_LOGI(TAG,
                "BENCH,seq=%" PRIu32
                ",capture_ms=%.3f,dsp_ms=%.3f,invoke_ms=%.3f"
                ",run_extra_ms=%.3f,process_ms=%.3f,cycle_ms=%.3f,interval_ms=%.3f",
                windows, capture_ms, dsp_ms, invoke_ms,
                run_extra_ms, process_ms, cycle_ms, interval_ms);
            if (windows == 1 || windows == 30) {
                bench_memory(windows == 1 ? "window_1" : "window_30");
            }
        }

        ESP_LOGI('''),
('    ESP_LOGI(TAG, "Edge Ecoacoustics - tinyCLAP INT8");', '''    ESP_LOGI(TAG, "Edge Ecoacoustics - tinyCLAP INT8");
    bench_memory("boot");'''),
('    ESP_ERROR_CHECK(eco_features_init());', '''    ESP_ERROR_CHECK(eco_features_init());
    bench_memory("model_ready");'''),
('    assert(pcm_buffer != NULL && feature_buffer != NULL);', '''    assert(pcm_buffer != NULL && feature_buffer != NULL);
    ESP_LOGI(TAG, "BENCH_BUFFERS,pcm_bytes=%u,logmel_bytes=%u",
        (unsigned)(ECO_SEGMENT_SAMPLES * sizeof(int16_t)),
        (unsigned)sizeof(eco_features_t));
    bench_memory("buffers_ready");'''),
('    ESP_ERROR_CHECK(eco_network_init());', '''    ESP_ERROR_CHECK(eco_network_init());
    bench_memory("network_started");'''),
]
for old, new in changes:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f'Anchor falhou ({n} correspondencias): {old[:70]}')
    s = s.replace(old, new, 1)
backup = source.with_name('main.c.pre-benchmark.bak')
if backup.exists():
    raise SystemExit(f'Backup existente: {backup}. Nenhuma alteracao realizada.')
backup.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
source.write_text(s, encoding='utf-8')
print(f'Instrumentado: {source}')
print(f'Backup: {backup}')
print('Somente main.c foi alterado; sem alteracoes na logica MQTT ou nos indices.')
