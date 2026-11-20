"""Avalia a fidelidade do tinyCLAP FP32 vs TFLite INT8 usando os mesmos áudios.

Instale o script na raiz do repositório como model/evaluate_quantization.py.
Executar da raiz:
  python model/evaluate_quantization.py --audio-dir "CAMINHO_DOS_WAVS" --count 10
  python model/evaluate_quantization.py --audio-dir "CAMINHO_DOS_WAVS" --count 500

O script NÃO altera o checkpoint, TFLite nem o firmware. A execução INT8 é
realizada no computador, com o interpretador TFLite/LiteRT, não no ESP32-S3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL_NAME = "phinet_alpha_1.50_beta_0.75_t0_6_N_7"
CHECKPOINT = ROOT / "checkpoints" / f"{MODEL_NAME}.ckpt"
TFLITE = ROOT / "generated" / "tinyclap_phinet_int8.tflite"
TEXT_ASSETS = REPO / "firmware" / "esp32s3" / "model" / "tinyclap_text_assets.bin"
BUILD_META = ROOT / "generated" / "audio_build.json"

SR_INPUT = 48000
SR_MODEL = 44100
DURATION_SECONDS = 6
SAMPLES_INPUT = SR_INPUT * DURATION_SECONDS
LOGMEL_SHAPE = (827, 64)
EMBEDDING_SIZE = 1024
GROUPS = {"bio": 0x01, "geo": 0x02, "antro": 0x04}


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_text_assets(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"Arquivo de embeddings muito pequeno: {path}")
    magic, n, dim, logit_scale = struct.unpack_from("<4sIIf", raw)
    if magic != b"ECO1" or dim != EMBEDDING_SIZE or n < 3:
        raise ValueError("Cabeçalho do ativo de texto incompatível")
    offset = (16 + n + 3) & ~3
    expected = offset + n * dim * 4
    if len(raw) != expected:
        raise ValueError(f"Tamanho incorreto dos ativos: {len(raw)} != {expected}")
    masks = np.frombuffer(raw, dtype=np.uint8, count=n, offset=16).copy()
    text = np.frombuffer(raw, dtype="<f4", count=n * dim, offset=offset).reshape(n, dim).copy()
    if not np.all(np.isfinite(text)) or not np.isfinite(logit_scale):
        raise ValueError("Ativo de texto contém valores inválidos")
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    return masks, text, float(logit_scale)


def choose_examples(audio_dir: Path, count: int, seed: int, include_calibration: bool):
    import soundfile as sf

    wav_files = sorted(p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not wav_files:
        raise FileNotFoundError(f"Nenhum WAV em {audio_dir}")

    calibration_names = set()
    if not include_calibration and BUILD_META.exists():
        try:
            metadata = json.loads(BUILD_META.read_text(encoding="utf-8"))
            calibration_names = {str(item["wav"]).lower() for item in metadata.get("calibration", [])}
        except (OSError, ValueError, KeyError, TypeError):
            print("AVISO: Não foi possível ler o manifesto de calibração; nenhum áudio excluído.")
    rng = np.random.default_rng(seed)
    valid = []
    skipped_calib = 0
    invalid = 0
    for path in wav_files:
        if path.name.lower() in calibration_names:
            skipped_calib += 1
            continue
        try:
            info = sf.info(str(path))
            n_segments = int(info.frames // SAMPLES_INPUT)
            if info.samplerate != SR_INPUT or info.channels < 2 or n_segments < 1:
                invalid += 1
                continue
            indexes = np.arange(n_segments)
            rng.shuffle(indexes)
            valid.append((path, indexes.tolist()))
        except (RuntimeError, OSError):
            invalid += 1
    if not valid:
        raise ValueError("Nenhum WAV compatível: esperado 48 kHz, estéreo, pelo menos 6 s")

    # Primeiro seleciona áudios de gravações diferentes; só depois repete gravações.
    order = rng.permutation(len(valid))
    selected = []
    depth = 0
    while len(selected) < count:
        added = 0
        for i in order:
            path, segments = valid[int(i)]
            if depth < len(segments):
                selected.append((path, int(segments[depth])))
                added += 1
                if len(selected) >= count:
                    break
        if added == 0:
            break
        depth += 1
    print(f"WAVs encontrados: {len(wav_files)} | aptos: {len(valid)} | "
          f"excluídos pela calibração (nome): {skipped_calib} | incompatíveis: {invalid}")
    print(f"Segmentos selecionados: {len(selected)} (solicitados: {count})")
    if not selected:
        raise ValueError("Nenhum segmento disponível")
    return selected


def load_frontend():
    import torch
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram

    stft = Spectrogram(
        n_fft=1024, hop_length=320, win_length=1024,
        window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
    ).eval()
    mel = LogmelFilterBank(
        sr=SR_MODEL, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
        ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
    ).eval()
    return stft, mel


def features_for_segment(path: Path, segment: int, frontend) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio

    with sf.SoundFile(str(path)) as sound:
        if sound.samplerate != SR_INPUT or sound.channels < 2:
            raise ValueError(f"WAV inválido: {path}")
        sound.seek(segment * SAMPLES_INPUT)
        samples = sound.read(SAMPLES_INPUT, dtype="float32", always_2d=True)
    if len(samples) != SAMPLES_INPUT:
        raise ValueError(f"Segmento incompleto: {path}, índice {segment}")
    # Reproduz o canal de calibração do script build_audio.py (coluna direita).
    right = torch.from_numpy(samples[:, 1].copy())
    resampled = torchaudio.functional.resample(right, SR_INPUT, SR_MODEL)
    if resampled.numel() != SR_MODEL * DURATION_SECONDS:
        raise ValueError("Quantidade inesperada de amostras após resampling")
    with torch.inference_mode():
        stft, mel = frontend
        features = mel(stft(resampled.unsqueeze(0))).squeeze().numpy()
    if features.shape != LOGMEL_SHAPE or not np.all(np.isfinite(features)):
        raise ValueError(f"Log-Mel inválido: {features.shape}")
    return np.ascontiguousarray(features.astype(np.float32))


def prepare_float_model(checkpoint: Path, threads: int):
    import torch
    from modules import AudioEncoder

    if threads > 0:
        torch.set_num_threads(threads)
    model = AudioEncoder(MODEL_NAME, 2048, EMBEDDING_SIZE, 527)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def prepare_int8_model(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_content=path.read_bytes())
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    if tuple(input_info["shape"]) != (1, *LOGMEL_SHAPE) or input_info["dtype"] != np.int8:
        raise ValueError("Entrada do TFLite não é INT8 [1,827,64]")
    if tuple(output_info["shape"]) != (1, EMBEDDING_SIZE) or output_info["dtype"] != np.int8:
        raise ValueError("Saída do TFLite não é INT8 [1,1024]")
    if input_info["quantization"][0] <= 0 or output_info["quantization"][0] <= 0:
        raise ValueError("Parâmetros de quantização inválidos")
    return interpreter, input_info, output_info


def evaluate_int8(interpreter, input_info, output_info, logmel: np.ndarray):
    scale, zero = input_info["quantization"]
    unbounded = np.rint(logmel.astype(np.float64) / scale + zero)
    clipped = np.count_nonzero((unbounded < -128) | (unbounded > 127))
    q = np.clip(unbounded, -128, 127).astype(np.int8)[None]
    interpreter.set_tensor(input_info["index"], q)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
    out_scale, out_zero = output_info["quantization"]
    embedding = (raw - out_zero) * out_scale
    return embedding, float(clipped / logmel.size)


def compute_scores(embedding, masks, text, logit_scale):
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Embedding com norma zero ou inválida")
    cos = (text @ (embedding / norm)).astype(np.float64)
    scores = np.clip((cos * logit_scale - 5.0) / 15.0, 0.0, 1.0)
    indexes = {}
    top1 = {}
    for name, bit in GROUPS.items():
        selected = np.flatnonzero((masks & bit) != 0)
        if len(selected) < 2:
            raise ValueError(f"Faltam prompts na categoria {name}")
        # Índices: mesmo critério do firmware, incluindo empates por saturação do score.
        ranked_score = selected[np.argsort(-scores[selected], kind="stable")]
        indexes[name] = float(0.75 * scores[ranked_score[0]] + 0.25 * scores[ranked_score[1]])
        # Top-1 da similaridade bruta evita empate artificial após clipping.
        ranked_cos = selected[np.argsort(-cos[selected], kind="stable")]
        top1[name] = int(ranked_cos[0])
    dominant = max(GROUPS, key=indexes.get)
    return indexes, top1, dominant


def summarize(rows, key: str):
    a = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "p95": float(np.percentile(a, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True, help="Pasta com WAVs 48 kHz estéreo")
    parser.add_argument("--count", type=int, default=500, help="Quantidade de segmentos (padrão: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="Threads PyTorch; 0 usa padrão")
    parser.add_argument("--include-calibration", action="store_true", help="Não excluir gravações do manifesto de calibração")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--tflite", type=Path, default=TFLITE)
    parser.add_argument("--text-assets", type=Path, default=TEXT_ASSETS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "quantization_eval")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count deve ser >= 1")
    if not args.audio_dir.is_dir():
        parser.error(f"Pasta de áudio não encontrada: {args.audio_dir}")
    for path in (args.checkpoint, args.tflite, args.text_assets):
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")

    masks, text, logit_scale = read_text_assets(args.text_assets)
    print(f"Ativos de texto: {len(masks)} prompts; categorias "
          + ", ".join(f"{k}={int(np.count_nonzero((masks & bit) != 0))}" for k, bit in GROUPS.items()))
    examples = choose_examples(args.audio_dir, args.count, args.seed, args.include_calibration)
    frontend = load_frontend()
    float_model = prepare_float_model(args.checkpoint, args.threads)
    interpreter, input_info, output_info = prepare_int8_model(args.tflite)

    import torch
    rows = []
    for idx, (path, segment) in enumerate(examples, 1):
        features = features_for_segment(path, segment, frontend)
        with torch.inference_mode():
            fp32, _, _ = float_model(torch.from_numpy(features[None]))
            fp32 = fp32.detach().cpu().numpy()[0].astype(np.float32)
        int8, clip_fraction = evaluate_int8(interpreter, input_info, output_info, features)
        if not np.all(np.isfinite(fp32)) or not np.all(np.isfinite(int8)):
            raise ValueError(f"Embedding inválido em {path.name}, segmento {segment}")

        fp32_norm = float(np.linalg.norm(fp32))
        int8_norm = float(np.linalg.norm(int8))
        if min(fp32_norm, int8_norm) < 1e-12:
            raise ValueError("Embedding de norma zero")
        cosine = float(np.dot(fp32, int8) / (fp32_norm * int8_norm))
        difference = np.abs(fp32.astype(np.float64) - int8.astype(np.float64))
        f_idx, f_top, f_dom = compute_scores(fp32, masks, text, logit_scale)
        q_idx, q_top, q_dom = compute_scores(int8, masks, text, logit_scale)
        row = {
            "wav": str(path.relative_to(args.audio_dir)), "segment_index": segment,
            "start_second": segment * DURATION_SECONDS,
            "embedding_cosine": cosine, "embedding_mae": float(difference.mean()),
            "embedding_max_abs": float(difference.max()),
            "input_clipped_fraction": clip_fraction,
            "dominant_fp32": f_dom, "dominant_int8": q_dom,
            "dominant_agree": int(f_dom == q_dom),
        }
        for key in GROUPS:
            row[f"{key}_fp32"] = f_idx[key]
            row[f"{key}_int8"] = q_idx[key]
            row[f"{key}_abs_diff"] = abs(f_idx[key] - q_idx[key])
            row[f"{key}_top1_cosine_fp32"] = f_top[key]
            row[f"{key}_top1_cosine_int8"] = q_top[key]
            row[f"{key}_top1_agree"] = int(f_top[key] == q_top[key])
        rows.append(row)
        if idx == 1 or idx % 10 == 0 or idx == len(examples):
            print(f"[{idx}/{len(examples)}] {path.name} segmento={segment} | "
                  f"cos(FP32, INT8)={cosine:.6f} | clipping={clip_fraction:.2%}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "quantization_per_segment.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "description": "Fidelidade FP32 vs TFLite INT8 no computador, não acurácia supervisionada",
        "segment_count": len(rows), "unique_wav_count": len(set(row["wav"] for row in rows)),
        "seed": args.seed, "input_sample_rate": SR_INPUT, "model_sample_rate": SR_MODEL,
        "channel": "right (índice 1)", "duration_seconds": DURATION_SECONDS,
        "logmel_shape": list(LOGMEL_SHAPE), "text_prompt_count": int(len(masks)),
        "embedding_cosine": summarize(rows, "embedding_cosine"),
        "embedding_mae": summarize(rows, "embedding_mae"),
        "input_clipped_fraction": summarize(rows, "input_clipped_fraction"),
        "dominant_agreement": summarize(rows, "dominant_agree")["mean"],
        "per_category": {
            key: {
                "index_absolute_error": summarize(rows, f"{key}_abs_diff"),
                "top1_raw_cosine_agreement": summarize(rows, f"{key}_top1_agree")["mean"],
            }
            for key in GROUPS
        },
        "model_files": {
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
            "tflite": str(args.tflite), "tflite_sha256": file_sha256(args.tflite),
            "text_assets": str(args.text_assets), "text_assets_sha256": file_sha256(args.text_assets),
        },
    }
    json_path = args.output_dir / "quantization_summary.json"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== RESUMO ===")
    print(f"Segmentos: {len(rows)} | WAVs únicos: {metrics['unique_wav_count']}")
    print(f"Cosseno médio embeddings: {metrics['embedding_cosine']['mean']:.6f}")
    print(f"MAE médio embeddings: {metrics['embedding_mae']['mean']:.6f}")
    print(f"Concordância categoria dominante: {metrics['dominant_agreement']:.2%}")
    print(f"Fraçao média de clipping na entrada: {metrics['input_clipped_fraction']['mean']:.2%}")
    for key in GROUPS:
        v = metrics["per_category"][key]
        print(f"{key}: MAE índice = {v['index_absolute_error']['mean']:.6f}; "
              f"Top-1 = {v['top1_raw_cosine_agreement']:.2%}")
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    if metrics["input_clipped_fraction"]["mean"] > 0.01:
        print("AVISO: clipping de entrada >1%; conferir compatibilidade do Log-Mel com a calibração.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise
