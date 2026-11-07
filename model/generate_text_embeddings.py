
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
from unidecode import unidecode

from modules import Projection


ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "text" / "prompts.txt"
CHECKPOINT = ROOT / "checkpoints" / "CLAP_weights_2022.pth"
OUTPUT = ROOT / "exports" / "text_embeddings.npy"


def main():
    # 1. Ler os prompts originais
    lines = [
        line.strip()
        for line in PROMPTS.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    prompts = [
        unidecode(line.lower().split(" - ")[-1])
        for line in lines
    ]

    print("Quantidade de prompts:", len(prompts))

    # 2. Carregar os pesos textuais do CLAP 2022
    print("Carregando checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    state = checkpoint["model"]

    text_state = {
        key.removeprefix("caption_encoder."): value
        for key, value in state.items()
        if key.startswith("caption_encoder.")
    }

    print("Tensores textuais:", len(text_state))

    # 3. Reconstruir o encoder de texto
    bert = BertModel(BertConfig())
    projection = Projection(768, 1024)

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
    bert.load_state_dict(bert_state, strict=True)
    projection.load_state_dict(projection_state, strict=True)

    bert.eval()
    projection.eval()

    print("Pesos textuais carregados corretamente.")

    # 4. Tokenizar os prompts
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    tokens = tokenizer(
        prompts,
        max_length=100,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 5. Gerar os embeddings
    results = []

    with torch.inference_mode():
        for start in range(0, len(prompts), 16):
            batch = {
                key: value[start:start + 16]
                for key, value in tokens.items()
            }

            hidden = bert(**batch).last_hidden_state
            cls = hidden[:, 0, :]

            embedding = projection(cls)
            embedding = F.normalize(embedding, dim=-1)

            results.append(embedding.cpu())

    embeddings = torch.cat(results).numpy()

    assert embeddings.shape == (len(prompts), 1024)
    assert np.isfinite(embeddings).all()

    # 6. Salvar os resultados
    np.save(OUTPUT, embeddings)

    print("\n=== RESULTADOS ===")
    print("Prompts:", len(prompts))
    print("Dimensoes:", embeddings.shape)
    print("Norma do primeiro embedding:",
          np.linalg.norm(embeddings[0]))
    print("Valores finitos:", np.isfinite(embeddings).all())
    print("Arquivo:", OUTPUT)


if __name__ == "__main__":
    main()
