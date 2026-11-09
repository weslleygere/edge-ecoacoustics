"""Reproduz o encoder de áudio PhiNet em TFLite INT8.

Na raiz do repositório:
    python model/build_audio.py --audio-dir "CAMINHO_DOS_WAVS"

Para testar primeiro com a calibração já validada:
    python model/build_audio.py --calibration-dir model/calibration

Nenhum arquivo do firmware é modificado.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAME = "phinet_alpha_1.50_beta_0.75_t0_6_N_7"
CHECKPOINT = ROOT / "checkpoints" / f"{NAME}.ckpt"
OUTPUT = ROOT / "generated" / "tinyclap_phinet_int8.tflite"
SOURCE_TFLITE = "tinyclap_phinet_full_integer_quant.tflite"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(path):
    import torch
    from torch import nn
    from modules import AudioEncoder
    import onnx
    import onnxruntime as ort

    class AudioEmbedding(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, logmel):
            embedding, _, _ = self.encoder(logmel)
            return embedding

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    encoder = AudioEncoder(NAME, 2048, 1024, 527)
    encoder.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    model = AudioEmbedding(encoder).eval()

    torch.manual_seed(42)
    sample = torch.randn(1, 827, 64)
    with torch.inference_mode():
        expected = model(sample).numpy()
        torch.onnx.export(
            model, (sample,), str(path),
            input_names=["logmel"], output_names=["audio_embedding"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )

    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["audio_embedding"], {"logmel": sample.numpy()})[0]
    difference = float(np.max(np.abs(expected - actual)))
    print(f"PyTorch x ONNX: diferença máxima {difference:.8g}")
    if not np.isfinite(difference) or difference > 2e-4:
        raise RuntimeError("ONNX não reproduziu a saída do PyTorch")


def wav_calibration(audio_dir, count):
    import soundfile as sf
    import torch
    import torchaudio
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank

    wavs = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() == ".wav")
    if not wavs:
        raise FileNotFoundError(f"Nenhum WAV encontrado em {audio_dir}")
    if count < 1:
        raise ValueError("--count precisa ser positivo")

    indices = np.linspace(0, len(wavs) - 1, min(count, len(wavs)), dtype=int)
    stft = Spectrogram(n_fft=1024, hop_length=320, win_length=1024,
                       window="hann", center=True, pad_mode="reflect",
                       freeze_parameters=True)
    mel = LogmelFilterBank(sr=44100, n_fft=1024, n_mels=64,
                           fmin=50, fmax=14000, ref=1.0, amin=1e-10,
                           top_db=None, freeze_parameters=True)
    features = []
    manifest = []
    with torch.inference_mode():
        for n, index in enumerate(indices):
            path = wavs[index]
            segment = n % 10
            with sf.SoundFile(path) as wav:
                if (wav.samplerate, wav.channels) != (48000, 2) or wav.frames < 60 * 48000:
                    raise ValueError(f"WAV incompatível (48 kHz, estéreo, 60 s): {path}")
                wav.seek(segment * 6 * 48000)
                audio = wav.read(6 * 48000, dtype="float32", always_2d=True)[:, 1]
            audio = torchaudio.functional.resample(torch.from_numpy(audio.copy()), 48000, 44100)
            if audio.numel() != 264600:
                raise RuntimeError(f"Duração inesperada: {path}")
            item = mel(stft(audio.unsqueeze(0))).squeeze().numpy()
            if item.shape != (827, 64):
                raise RuntimeError(f"Log-Mel inesperado: {item.shape}")
            features.append(item)
            manifest.append({"wav": path.name, "segment_6s": segment, "sha256": sha256(path)})
            print(f"Calibração {n + 1}/{len(indices)}: {path.name} / segmento {segment}")
    return np.stack(features).astype(np.float32), manifest


def existing_calibration(directory):
    data = np.load(directory / "tinyclap_calibration.npy", allow_pickle=False)
    metadata = json.loads((directory / "tinyclap_calibration.json").read_text(encoding="utf-8"))
    # A calibração original foi armazenada transposta para o onnx2tf antigo.
    if data.ndim != 3 or data.shape[1:] != (64, 827):
        raise ValueError(f"Calibração existente com dimensões inesperadas: {data.shape}")
    mean = float(metadata["onnx2tf_mean"])
    std = float(metadata["onnx2tf_std"])
    return data.transpose(0, 2, 1).copy(), mean, std


def prepare_calibration(work, audio_dir, count, calibration_dir):
    if calibration_dir:
        normalized, mean, std = existing_calibration(calibration_dir)
        manifest = None
    else:
        original, manifest = wav_calibration(audio_dir, count)
        lower, upper = float(original.min()), float(original.max())
        scale = upper - lower
        if scale <= 0:
            raise ValueError("Amplitude da calibração inválida")
        normalized = (original - lower) / scale
        mean, std = -lower / scale, 1.0 / scale

    normalized = np.ascontiguousarray(normalized, dtype=np.float32)
    if normalized.ndim != 3 or normalized.shape[1:] != (827, 64):
        raise ValueError(f"Formato incorreto da calibração: {normalized.shape}")
    if not np.isfinite(normalized).all() or std <= 0:
        raise ValueError("Dados ou parâmetros de calibração inválidos")
    path = work / "calibration.npy"
    np.save(path, normalized)
    return path, mean, std, manifest


def validate_tflite(path):
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(path))
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if (tuple(inp["shape"]) != (1, 827, 64) or
        tuple(out["shape"]) != (1, 1024) or
        inp["dtype"] != np.int8 or out["dtype"] != np.int8):
        raise RuntimeError(f"Tensor TFLite incompatível: {inp['shape']}/{inp['dtype']} -> {out['shape']}/{out['dtype']}")
    for info in (inp, out):
        if info["quantization"][0] <= 0:
            raise RuntimeError("Tensor sem parâmetros de quantização válidos")
    allowed = {"ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
               "LOGISTIC", "MEAN", "MUL", "PAD", "RESHAPE", "RSQRT",
               "SQUARED_DIFFERENCE", "SUB"}
    found = {op["op_name"] for op in interpreter._get_ops_details()}
    if not found <= allowed:
        raise RuntimeError(f"Operadores inesperados no TFLite: {sorted(found - allowed)}")
    interpreter.allocate_tensors()
    interpreter.set_tensor(inp["index"], np.zeros((1, 827, 64), dtype=np.int8))
    interpreter.invoke()
    if interpreter.get_tensor(out["index"]).shape != (1, 1024):
        raise RuntimeError("Inferência de teste produziu formato inválido")
    print("TFLite validado: INT8, dimensões, operadores e inferência OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio-dir", type=Path, help="Pasta dos WAVs originais para calibração")
    group.add_argument("--calibration-dir", type=Path, help="Reusar calibração .npy/.json anterior")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--work-dir", type=Path, default=Path("C:/esp") if sys.platform == "win32" else None)
    parser.add_argument("--reference", type=Path, help="TFLite funcional para comparar SHA-256")
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    cli = shutil.which("onnx2tf")
    if not cli:
        raise RuntimeError("onnx2tf não encontrado no ambiente virtual")

    with tempfile.TemporaryDirectory(prefix="tinyclap-", dir=args.work_dir) as tmp:
        work = Path(tmp)
        onnx_file = work / "tinyclap_phinet.onnx"
        export_onnx(onnx_file)
        calibration, mean, std, manifest = prepare_calibration(
            work, args.audio_dir, args.count, args.calibration_dir)
        target = work / "converted"
        command = [cli, "-i", str(onnx_file), "-o", str(target),
                   "-tb", "tf_converter", "-kat", "logmel",
                   "-oiqt", "-iqd", "int8", "-oqd", "int8",
                   "-cind", "logmel", str(calibration), f"[{mean}]", f"[{std}]"]
        print("Convertendo ONNX → TFLite INT8...")
        subprocess.run(command, check=True)
        candidate = target / SOURCE_TFLITE
        if not candidate.is_file():
            raise FileNotFoundError(f"TFLite não gerado: {candidate}")
        validate_tflite(candidate)
        digest = sha256(candidate)
        print(f"SHA-256 do candidato: {digest}")
        if args.reference:
            reference_digest = sha256(args.reference)
            print(f"SHA-256 da referência: {reference_digest}")
            if digest != reference_digest:
                raise RuntimeError("Modelo gerado difere da referência; saída NÃO publicada")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, OUTPUT)
        print(f"Modelo gerado: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
        if manifest is not None:
            report = {"checkpoint_sha256": sha256(CHECKPOINT),
                      "model_sha256": digest,
                      "calibration": manifest,
                      "onnx2tf_mean": mean, "onnx2tf_std": std}
            (OUTPUT.parent / "audio_build.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
