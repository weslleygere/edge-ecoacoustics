
"""Exporta o encoder de audio tinyCLAP (PhiNet) para ONNX.

Executar na raiz do repositorio edge-ecoacoustics:

    python model/export_tinyclap_onnx.py

O script:
1. Carrega o checkpoint original do PhiNet.
2. Executa uma inferencia de referencia no PyTorch.
3. Exporta o encoder para ONNX.
4. Verifica o modelo ONNX.
5. Compara numericamente as saidas PyTorch e ONNX.

O arquivo ONNX ainda nao e executavel diretamente no ESP32-S3.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# CONFIGURACAO
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "phinet_alpha_1.50_beta_0.75_t0_6_N_7"

CHECKPOINT = ROOT / f"{MODEL_NAME}.ckpt"

OUTPUT = ROOT / "tinyclap_phinet.onnx"

REPORT = ROOT / "tinyclap_phinet.json"

SAMPLE_RATE = 44100
DURATION = 6
HOP_LENGTH = 320

N_MELS = 64
EMBED_DIM = 1024

# STFT com center=True, conforme torchlibrosa
N_FRAMES = SAMPLE_RATE * DURATION // HOP_LENGTH + 1

OPSET_VERSION = 17


# ============================================================
# MODELO PARA EXPORTACAO
# ============================================================

class ProjectedEmbedding(nn.Module):
    """Retorna apenas o embedding de audio normalizado."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, logmel):

        embedding, _, _ = self.encoder(logmel)

        return embedding


# ============================================================
# FUNCAO PRINCIPAL
# ============================================================

def main():

    print("\n=== tinyCLAP -> ONNX ===\n")

    # --------------------------------------------------------
    # 1. Verificar arquivos
    # --------------------------------------------------------

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint nao encontrado: {CHECKPOINT}"
        )

    if not (ROOT / "modules.py").is_file():
        raise FileNotFoundError(
            f"modules.py nao encontrado em: {ROOT}"
        )

    sys.path.insert(0, str(ROOT))

    from modules import AudioEncoder

    # --------------------------------------------------------
    # 2. Construir arquitetura PhiNet
    # --------------------------------------------------------

    print("Construindo arquitetura PhiNet...")

    encoder = AudioEncoder(
        MODEL_NAME,
        2048,
        EMBED_DIM,
        527
    )

    # --------------------------------------------------------
    # 3. Carregar checkpoint
    # --------------------------------------------------------

    print("Carregando checkpoint...")

    weights = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    encoder.load_state_dict(
        weights,
        strict=True
    )

    encoder.eval()

    model = ProjectedEmbedding(encoder)
    model.eval()

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Checkpoint carregado corretamente.")
    print(f"Parametros: {parameters:,}")

    # --------------------------------------------------------
    # 4. Criar entrada de teste
    # --------------------------------------------------------

    print("\nCriando entrada de teste...")

    torch.manual_seed(42)

    x = torch.randn(
        1,
        N_FRAMES,
        N_MELS,
        dtype=torch.float32
    )

    print("Formato da entrada:", tuple(x.shape))

    # --------------------------------------------------------
    # 5. Inferencia PyTorch
    # --------------------------------------------------------

    print("\nExecutando inferencia PyTorch...")

    with torch.inference_mode():
        reference = model(x).detach().cpu().numpy()

    if reference.shape != (1, EMBED_DIM):
        raise RuntimeError(
            f"Dimensao inesperada: {reference.shape}"
        )

    if not np.isfinite(reference).all():
        raise RuntimeError(
            "Embedding contem valores invalidos."
        )

    print("Formato da saida:", reference.shape)

    print(
        "Norma do embedding:",
        np.linalg.norm(reference[0])
    )

    # --------------------------------------------------------
    # 6. Exportar para ONNX
    # --------------------------------------------------------

    print("\nExportando modelo para ONNX...")

    torch.onnx.export(
        model,
        (x,),
        str(OUTPUT),
        input_names=["logmel"],
        output_names=["audio_embedding"],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False
    )

    print("Modelo exportado:", OUTPUT)

    # --------------------------------------------------------
    # 7. Validar estrutura ONNX
    # --------------------------------------------------------

    import onnx

    onnx_model = onnx.load(str(OUTPUT))

    onnx.checker.check_model(onnx_model)

    print("Estrutura ONNX valida.")

    # --------------------------------------------------------
    # 8. Comparar PyTorch e ONNX Runtime
    # --------------------------------------------------------

    print("\nComparando PyTorch e ONNX Runtime...")

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(OUTPUT),
        providers=["CPUExecutionProvider"]
    )

    actual = session.run(
        ["audio_embedding"],
        {"logmel": x.numpy()}
    )[0]

    if actual.shape != reference.shape:
        raise RuntimeError(
            "Formato de saida ONNX diferente do PyTorch."
        )

    if not np.isfinite(actual).all():
        raise RuntimeError(
            "ONNX produziu valores invalidos."
        )

    max_abs_diff = float(
        np.max(np.abs(actual - reference))
    )

    mean_abs_diff = float(
        np.mean(np.abs(actual - reference))
    )

    print(f"Diferenca maxima: {max_abs_diff:.8f}")
    print(f"Diferenca media:  {mean_abs_diff:.8f}")

    tolerance = 1e-4

    if max_abs_diff > tolerance:
        raise RuntimeError(
            "ERRO: ONNX nao reproduziu "
            "as saidas PyTorch dentro da tolerancia."
        )

    print("Equivalencia numerica: OK")

    # --------------------------------------------------------
    # 9. Salvar relatorio
    # --------------------------------------------------------

    checkpoint_hash = hashlib.sha256(
        CHECKPOINT.read_bytes()
    ).hexdigest()

    report = {
        "model": MODEL_NAME,
        "parameters": parameters,
        "checkpoint_sha256": checkpoint_hash,
        "input_shape": [1, N_FRAMES, N_MELS],
        "output_shape": [1, EMBED_DIM],
        "precision": "float32",
        "onnx_opset": OPSET_VERSION,
        "onnx_bytes": OUTPUT.stat().st_size,
        "max_abs_difference": max_abs_diff,
        "mean_abs_difference": mean_abs_diff,
        "tolerance": tolerance,
        "parity_passed": True
    }

    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\n=== EXPORTACAO CONCLUIDA ===")
    print("Modelo:", OUTPUT)
    print("Relatorio:", REPORT)

    print(
        "Tamanho ONNX: "
        f"{OUTPUT.stat().st_size / (1024**2):.2f} MB"
    )


if __name__ == "__main__":
    main()
