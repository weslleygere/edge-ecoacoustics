"""Gera embeddings textuais CLAP 2022 a partir de model/text/prompts.txt.

Uso: python model/generate_text.py [--reference model/exports/text_embeddings.npy]
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
from unidecode import unidecode

from modules import Projection

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "CLAP_weights_2022.pth"
PROMPTS = ROOT / "text" / "prompts.txt"
OUTPUT = ROOT / "generated" / "text_embeddings.npy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="NPY anterior para comparação numérica")
    args = parser.parse_args()

    lines = [
        line.strip()
        for line in PROMPTS.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    prompts = [
        unidecode(line.lower().split(" - ")[-1])
        for line in lines
    ]

    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("Arquivo de prompts vazio ou inválido")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    text_state = {
        key.removeprefix("caption_encoder."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("caption_encoder.")
    }
    del checkpoint

    bert_state = {
        key.removeprefix("base."): value
        for key, value in text_state.items()
        if key.startswith("base.")
    }
    projection_state = {
        key.removeprefix("projection."): value
        for key, value in text_state.items()
        if key.startswith("projection.")
    }
    bert_state.pop("embeddings.position_ids", None)

    bert = BertModel(BertConfig())
    projection = Projection(768, 1024)
    bert.load_state_dict(bert_state, strict=True)
    projection.load_state_dict(projection_state, strict=True)
    bert.eval()
    projection.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(
        prompts, max_length=100, padding="max_length",
        truncation=True, return_tensors="pt",
    )

    results = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 16):
            batch = {key: value[start:start + 16] for key, value in tokens.items()}
            cls = bert(**batch).last_hidden_state[:, 0, :]
            results.append(F.normalize(projection(cls), dim=-1).cpu())

    embeddings = torch.cat(results).numpy().astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1, atol=1e-5):
        raise RuntimeError("Embeddings não normalizados")

    if args.reference:
        reference = np.load(args.reference, allow_pickle=False)
        if reference.shape != embeddings.shape:
            raise RuntimeError(f"Dimensões diferentes da referência: {reference.shape}")
        difference = float(np.max(np.abs(reference - embeddings)))
        print(f"Diferença máxima da referência: {difference:.10g}")
        if not np.allclose(reference, embeddings, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Embeddings não reproduziram a referência")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, embeddings)
    print(f"Embeddings: {embeddings.shape} | dtype={embeddings.dtype}")
    print(f"Arquivo: {OUTPUT}")
    print(f"SHA-256: {hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
