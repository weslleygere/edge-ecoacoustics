
"""Prepara dados reais para quantizacao INT8 do tinyCLAP."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "tinyclap_calibration.npy"
METADATA = ROOT / "tinyclap_calibration.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    wavs = sorted(
        p for p in args.audio_dir.iterdir()
        if p.suffix.lower() == ".wav"
    )

    if not wavs:
        raise FileNotFoundError("Nenhum WAV encontrado.")
    if args.count < 1:
        raise ValueError("count deve ser positivo.")

    indices = np.linspace(
        0, len(wavs) - 1,
        min(args.count, len(wavs)),
        dtype=int,
    )

    stft = Spectrogram(
        n_fft=1024,
        hop_length=320,
        win_length=1024,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    )

    mel = LogmelFilterBank(
        sr=44100,
        n_fft=1024,
        n_mels=64,
        fmin=50,
        fmax=14000,
        ref=1.0,
        amin=1e-10,
        top_db=None,
        freeze_parameters=True,
    )

    samples = []

    with torch.inference_mode():
        for n, index in enumerate(indices):
            path = wavs[index]
            segment_index = n % 10

            with sf.SoundFile(path) as wav:
                if (
                    wav.samplerate != 48000
                    or wav.channels != 2
                    or wav.frames < 60 * 48000
                ):
                    raise ValueError(f"WAV incompatível: {path}")

                wav.seek(segment_index * 6 * 48000)
                audio = wav.read(
                    6 * 48000,
                    dtype="float32",
                    always_2d=True,
                )[:, 1]

            audio = torch.from_numpy(audio.copy())
            audio = torchaudio.functional.resample(
                audio, 48000, 44100
            )

            if audio.numel() != 264600:
                raise ValueError("Comprimento inesperado.")

            features = mel(stft(audio.unsqueeze(0)))
            features = features.squeeze().numpy()

            if features.shape != (827, 64):
                raise ValueError(
                    f"Log-Mel inesperado: {features.shape}"
                )

            # Ordem utilizada pelo TFLite convertido.
            samples.append(features.T.copy())

            print(f"[{n + 1}/{len(indices)}] {path.name}")

    data = np.stack(samples).astype(np.float32)

    if not np.isfinite(data).all():
        raise ValueError("Dados de calibracao invalidos.")

    # O onnx2tf espera dados normalizados em [0, 1].
    # Registramos os parametros para reconstruir
    # os valores Log-Mel originais na conversao.
    lower = float(data.min())
    upper = float(data.max())
    scale = upper - lower

    if scale <= 0:
        raise ValueError("Amplitude de calibracao invalida.")

    normalized = (data - lower) / scale

    np.save(OUTPUT, normalized.astype(np.float32))

    metadata = {
        "shape": list(normalized.shape),
        "input_name": "logmel",
        "minimum": lower,
        "maximum": upper,
        "onnx2tf_mean": -lower / scale,
        "onnx2tf_std": 1.0 / scale,
    }

    METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nCalibracao preparada!")
    print("Shape:", normalized.shape)
    print("Minimo Log-Mel:", lower)
    print("Maximo Log-Mel:", upper)
    print("Arquivo:", OUTPUT)
    print("Parametros:", METADATA)


if __name__ == "__main__":
    main()
