# Benchmark ESP32-S3 — Edge Ecoacoustics

Janelas analisadas: **30**.

## Estatísticas temporais

| Etapa | Média (ms) | Desvio-padrão (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Captura I2S | 6001.25 | 0.07 | 6001.21 | 6001.62 |
| Log-Mel (DSP) | 7995.82 | 16.43 | 7979.20 | 8065.52 |
| Inferência PhiNet | 7058.53 | 12.47 | 7041.87 | 7089.11 |
| Outros (inferência) | 75.33 | 0.85 | 74.46 | 78.47 |
| Processamento total | 15129.71 | 20.13 | 15101.32 | 15196.27 |
| Ciclo medido | 21130.99 | 20.12 | 21102.58 | 21197.53 |
| Intervalo entre capturas | 21145.01 | 21.54 | 21116.38 | 21221.57 |

## Armazenamento

| Recurso | KiB | Origem |
|---|---:|---|
| Modelo TFLite | 4590.81 | Arquivo TFLite |
| Firmware (.bin) | 6001.14 | Imagem de aplicação |
| Tensor Arena reservada | 6144.0 | Configuração do código |
| Tensor Arena usada | n/d | Log de inicialização |
| Buffer PCM | n/d | BENCH_BUFFERS |
| Buffer Log-Mel | n/d | BENCH_BUFFERS |

## Limitações

- Memória foi medida em poucas fases; não inferir tendências contínuas.

Os índices Bio, Geo e Antro não são probabilidades calibradas.

O tempo de `invoke_ms` mede somente a execução do modelo.

`process_ms` já inclui DSP e inferência (não somar novamente).

Valores de memória mínima são acumulados desde o boot.

O arquivo .bin não representa sozinho toda a Flash ocupada.
