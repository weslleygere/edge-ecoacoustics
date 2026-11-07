"""Compare ESP32-S3 synthetic INT8 validation with the desktop TFLite runtime.

Run from repository root:
    python model/validate_tinyclap_backend.py

The ESP firmware prints VALIDATION FNV1A, indices and top prompt scores.
This test validates the *inference + prompt score* stages, not the DSP.
"""
from pathlib import Path
import struct
import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'firmware/esp32s3/model/tinyclap_phinet_int8.tflite'
ASSETS = ROOT / 'firmware/esp32s3/model/tinyclap_text_assets.bin'
PROMPTS = ROOT / 'model/prompts.txt'
GROUPS = ('Bio', 'Geo', 'Antro')
BITS = (1, 2, 4)


def load_text_assets():
    content = ASSETS.read_bytes()
    magic, n, dim, scale = struct.unpack_from('<4sIIf', content, 0)
    offset = (16 + n + 3) & ~3
    if magic != b'ECO1' or (n, dim) != (41, 1024):
        raise ValueError('Unexpected asset header')
    if len(content) != offset + n * dim * 4:
        raise ValueError('Unexpected asset size')
    masks = np.frombuffer(content, dtype=np.uint8, count=n, offset=16).copy()
    embeddings = np.frombuffer(content, dtype='<f4', count=n*dim, offset=offset).reshape(n, dim)
    return masks, embeddings, np.float32(scale)


def fnv1a(data):
    h = 2166136261
    for b in data.view(np.uint8).ravel():
        h = ((h ^ int(b)) * 16777619) & 0xFFFFFFFF
    return h


def main():
    masks, texts, scale = load_text_assets()
    intr = Interpreter(model_content=MODEL.read_bytes(), num_threads=1)
    intr.allocate_tensors()
    ind = intr.get_input_details()[0]
    out = intr.get_output_details()[0]
    assert tuple(ind['shape']) == (1, 827, 64) and ind['dtype'] == np.int8
    assert tuple(out['shape']) == (1, 1024) and out['dtype'] == np.int8

    # Identical direct INT8 input to the ESP32-S3 selftest.
    i = np.arange(827 * 64, dtype=np.int64)
    q = (-100 + ((i * 37 + 13) % 160)).astype(np.int8).reshape(1, 827, 64)
    intr.set_tensor(ind['index'], q)
    intr.invoke()
    raw = intr.get_tensor(out['index']).reshape(-1)
    out_scale, out_zero = out['quantization']
    audio = (raw.astype(np.float32) - np.float32(out_zero)) * np.float32(out_scale)
    norm = np.float32(np.sqrt(np.sum(audio.astype(np.float64)**2)))
    inv = np.float32(1.0) / norm

    print(f'VALIDATION FNV1A={fnv1a(raw):08x} | first8=' + ','.join(str(int(v)) for v in raw[:8]))
    print(f'Embedding norm: {float(norm):.5f} | output_scale={out_scale} | zero={out_zero}')

    ranked = [[], [], []]
    for p in range(41):
        dot = float(np.sum(audio.astype(np.float64) * texts[p].astype(np.float64)))
        cosine = np.float32(np.float32(dot) * inv)
        sim = np.float32(np.float32(np.float32(cosine * scale) - np.float32(5.0)) / np.float32(15.0))
        sim = max(0.0, min(1.0, float(sim)))
        for g, bit in enumerate(BITS):
            if masks[p] & bit:
                ranked[g].append((p, sim))

    indices = []
    for g, name in enumerate(GROUPS):
        top = sorted(ranked[g], key=lambda x: -x[1])[:3]  # stable ties by prompt ID
        value = .75 * top[0][1] + .25 * top[1][1]
        indices.append(value)
        print('VALIDATION TOP ' + name + ' | ' + ' | '.join(f'{p}={s:.5f}' for p, s in top))
    print('VALIDATION indices | Bio=%.6f Geo=%.6f Antro=%.6f' % tuple(indices))
    if PROMPTS.exists():
        prompts = [s.strip() for s in PROMPTS.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        if len(prompts) == 41:
            for g, name in enumerate(GROUPS):
                top = sorted(ranked[g], key=lambda x: -x[1])[:3]
                print(f'{name} prompts:')
                for p, s in top:
                    print(f'  [{p:02d}] {s:.5f}  {prompts[p]}')


if __name__ == '__main__':
    main()
