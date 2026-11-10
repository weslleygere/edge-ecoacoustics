
"""Prepara os artefatos tinyCLAP para o ESP32-S3.

Uso:
    python model/package_firmware.py
    python model/package_firmware.py --install

Sem --install, apenas compara os artefatos.
"""

import argparse
import hashlib
import os
import struct
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
FW = ROOT.parent / "firmware" / "esp32s3"
MODEL_NAME = "tinyclap_phinet_int8.tflite"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def make_assets():
    source = ROOT / "generated"
    tflite_path = source / MODEL_NAME
    prompts_path = ROOT / "text" / "prompts.txt"
    embeddings_path = source / "text_embeddings.npy"
    checkpoint_path = ROOT / "checkpoints" / "CLAP_weights_2022.pth"

    for path in (
        tflite_path,
        prompts_path,
        embeddings_path,
        checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Prompts
    prompts = [
        line.strip()
        for line in prompts_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    num_prompts = len(prompts)

    if not 3 <= num_prompts <= 256:
        raise ValueError(
            f"Invalid number of prompts: {num_prompts} "
            "(expected 3 to 256)"
        )

    labels = [
        prompt.split(" - ")[-1].strip()
        for prompt in prompts
    ]

    if any(not label for label in labels):
        raise ValueError("Empty prompt description")

    # Categorias
    mapping = {
        "biophony": 1,
        "geophony": 2,
        "anthropophony": 4,
        "technophony": 4,
        "none": 0,
    }

    masks = bytearray()

    for prompt in prompts:
        category = prompt.split(" - ", 1)[0].strip().lower()

        if category not in mapping:
            raise ValueError(
                f"Unknown category: {category}"
            )

        masks.append(mapping[category])

    category_counts = [
        sum(bool(mask & bit) for mask in masks)
        for bit in (1, 2, 4)
    ]

    if min(category_counts) < 3:
        raise ValueError(
            "At least 3 prompts are required for "
            f"each category (Bio/Geo/Antro): {category_counts}"
        )

    # Embeddings textuais
    text = np.load(
        embeddings_path,
        allow_pickle=False
    )

    if (
        text.shape != (num_prompts, 1024)
        or text.dtype != np.float32
        or not np.isfinite(text).all()
    ):
        raise ValueError(
            f"Invalid text embeddings: "
            f"shape={text.shape}, dtype={text.dtype}; "
            f"expected ({num_prompts}, 1024) float32"
        )

    norms = np.linalg.norm(
        text,
        axis=1,
        keepdims=True
    )

    if np.any(norms <= 1e-10):
        raise ValueError("Zero-norm text embedding")

    text = (text / norms).astype("<f4")

    # Logit scale do CLAP 2022
    weights = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    logit_scale = float(
        weights["model"]["logit_scale"].exp()
    )

    del weights

    if not np.isfinite(logit_scale) or logit_scale <= 0:
        raise ValueError("Invalid logit scale")

    # Binario de embeddings
    text_bin = bytearray(
        struct.pack(
            "<4sIIf",
            b"ECO1",
            num_prompts,
            1024,
            logit_scale
        )
    )

    text_bin.extend(masks)

    while len(text_bin) % 4:
        text_bin.append(0)

    text_bin.extend(
        text.tobytes(order="C")
    )

    expected_size = (
        ((16 + num_prompts + 3) // 4) * 4
        + num_prompts * 1024 * 4
    )

    if len(text_bin) != expected_size:
        raise ValueError(
            "Unexpected text asset size"
        )

    # Filtros Mel
    mel = librosa.filters.mel(
        sr=44100,
        n_fft=1024,
        n_mels=64,
        fmin=50,
        fmax=14000,
        htk=False,
        norm="slaney"
    ).astype("<f4")

    if (
        mel.shape != (64, 513)
        or not np.isfinite(mel).all()
    ):
        raise ValueError(
            "Invalid Mel filter bank"
        )

    mel_bin = mel.tobytes(order="C")

    # Validar interface TFLite INT8
    from ai_edge_litert.interpreter import Interpreter

    model = tflite_path.read_bytes()
    interpreter = Interpreter(model_content=model)

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    if (
        tuple(inp["shape"]) != (1, 827, 64)
        or tuple(out["shape"]) != (1, 1024)
        or inp["dtype"] != np.int8
        or out["dtype"] != np.int8
        or inp["quantization"][0] <= 0
        or out["quantization"][0] <= 0
    ):
        raise ValueError(
            "TFLite interface incompatible with ESP32-S3 firmware"
        )

    print(
        f"Prompts: {num_prompts} | "
        f"Bio/Geo/Antro: {category_counts} | "
        f"exp(logit_scale)={logit_scale:.8f}"
    )

    return {
        FW / "model" / MODEL_NAME: model,
        FW / "model" / "tinyclap_text_assets.bin": bytes(text_bin),
        FW / "model" / "tinyclap_mel_filters.bin": mel_bin,
    }


def publish(files, install):
    for destination, contents in files.items():
        current = (
            destination.read_bytes()
            if destination.is_file()
            else None
        )

        status = (
            "IDENTICAL" if current == contents
            else "DIFFERENT" if current is not None
            else "MISSING"
        )

        print(
            f"{status:9} "
            f"{destination.relative_to(ROOT.parent)} "
            f"({len(contents)} bytes) "
            f"SHA256={sha256(contents)}"
        )

    if not install:
        print(
            "\nPreview only. No files modified. "
            "Use --install to publish."
        )
        return

    # Substituicao atomica por arquivo
    for destination, contents in files.items():
        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if (
            destination.is_file()
            and destination.read_bytes() == contents
        ):
            continue

        fd, temporary = tempfile.mkstemp(
            prefix=".tinyclap-",
            dir=destination.parent
        )

        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(contents)

            os.replace(temporary, destination)

        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    print("\nFirmware artifacts published.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--install",
        action="store_true",
        help="Copy validated artifacts to the firmware"
    )

    args = parser.parse_args()

    publish(
        make_assets(),
        args.install
    )


if __name__ == "__main__":
    main()
