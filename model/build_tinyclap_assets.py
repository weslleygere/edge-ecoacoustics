"""Build immutable ESP32-S3 tinyCLAP assets from the EXACT Python index pipeline.

Run at edge-ecoacoustics repository root:
    python model/build_tinyclap_assets.py

Needs model/prompts.txt, model/text_embeddings.npy, and CLAP 2022 checkpoint
model/checkpoints/CLAP_weights_2022.pth. Alternatively, supply --logit-scale
with a value measured from the SAME CLAP checkpoint.

Output:
    firmware/esp32s3/model/tinyclap_text_assets.bin
    firmware/esp32s3/model/tinyclap_mel_filters.bin

Reference: model/tinyclap_indices.py: normalize audio/text embeddings,
exp(teacher logit_scale), clip((similarity-5)/15), top2 weighted .75/.25.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--logit-scale", type=float, default=None,
                        help="Only use the exact exp(logit_scale) from the CLAP 2022 checkpoint")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    fw_model = model_dir.parent / "firmware" / "esp32s3" / "model"
    fw_model.mkdir(parents=True, exist_ok=True)

    import librosa
    import torch
    from unidecode import unidecode

    original = [s.strip() for s in
                (model_dir / "prompts.txt").read_text(encoding="utf-8-sig").splitlines()
                if s.strip()]
    if len(original) != 41:
        raise RuntimeError(f"Expected 41 prompts, got {len(original)}")
    effective = [unidecode(s.lower().split(" - ")[-1]) for s in original]

    manifest = model_dir / "text_embeddings.txt"
    if not manifest.exists():
        raise RuntimeError(
            f"{manifest} missing. Run model/tinyclap_indices.py once to regenerate "
            "and validate the prompt-aligned CLAP 2022 embeddings."
        )
    descriptions = [s.strip() for s in
                    manifest.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if descriptions != effective:
        raise RuntimeError("Prompt manifest differs from text_embeddings.npy; regenerate cache")

    text = np.load(model_dir / "text_embeddings.npy", allow_pickle=False)
    if text.shape != (41, 1024) or not np.isfinite(text).all():
        raise RuntimeError(f"Invalid text embeddings: {text.shape}")
    text = text.astype(np.float32)
    norms = np.linalg.norm(text, axis=1, keepdims=True)
    if np.any(norms <= 1e-10):
        raise RuntimeError("Zero text embedding")
    text = (text / norms).astype("<f4")

    masks = bytearray()
    for s in original:
        s = s.lower()
        mask = 0
        if "biophony" in s:
            mask |= 1
        if "geophony" in s:
            mask |= 2
        if "anthropophony" in s or "technophony" in s:
            mask |= 4
        masks.append(mask)
    sizes = [sum(bool(x & (1 << b)) for x in masks) for b in range(3)]
    if min(sizes) < 2:
        raise RuntimeError(f"At least 2 prompts per category required: {sizes}")

    if args.logit_scale is None:
        ckpt = model_dir / "checkpoints" / "CLAP_weights_2022.pth"
        if not ckpt.exists():
            raise RuntimeError(
                f"Missing {ckpt}; pass --logit-scale with the EXACT value printed "
                "by tinyclap_indices.py (no guessed default)."
            )
        teacher = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        logit_scale = float(teacher["model"]["logit_scale"].exp())
        del teacher
    else:
        logit_scale = args.logit_scale
    if not np.isfinite(logit_scale) or logit_scale <= 0:
        raise RuntimeError("Invalid CLAP logit_scale")

    header = struct.pack("<4sIIf", b"ECO1", 41, 1024, logit_scale)
    content = bytearray(header)
    content.extend(masks)
    while len(content) % 4:
        content.append(0)
    content.extend(text.tobytes(order="C"))
    out_text = fw_model / "tinyclap_text_assets.bin"
    out_text.write_bytes(content)

    # torchlibrosa LogmelFilterBank uses librosa's default Slaney mel scale,
    # including 'slaney' filter-area normalization, not HTK/triangles of height 1.
    mel = librosa.filters.mel(sr=44100, n_fft=1024, n_mels=64,
                              fmin=50, fmax=14000,
                              htk=False, norm="slaney").astype("<f4")
    if mel.shape != (64, 513) or not np.isfinite(mel).all():
        raise RuntimeError("Invalid mel matrix")
    out_mel = fw_model / "tinyclap_mel_filters.bin"
    out_mel.write_bytes(mel.tobytes(order="C"))

    print("Generated", out_text, out_text.stat().st_size, "bytes")
    print("Generated", out_mel, out_mel.stat().st_size, "bytes")
    print(f"Prompts Bio/Geo/Antro: {sizes}; exp(teacher logit_scale)={logit_scale:.8f}")
    print("Log-Mel: 44.1k / 1024 FFT / 320 hop / 64 Slaney Mel / 50..14k Hz")


if __name__ == "__main__":
    main()
