#!/usr/bin/env python3
"""Visualizações exploratórias dos CEI (bio, geo, antro) por segmento de 6 s.

Uso (a partir da raiz do repositório):
    python dataset/visualize.py \
      --csv data/cei_6s.csv \
      --sample data/recordings.csv

Requer apenas: pandas, numpy, matplotlib.
O CSV de amostragem é opcional: sem ele, histogramas, dispersões e resumo
continuam disponíveis; comparações por local/horário são ignoradas.
"""

import argparse
import inspect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # funciona em servidor sem interface gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INDICES = ("bio", "geo", "antro")
NOMES = {"bio": "Biofonia", "geo": "Geofonia", "antro": "Antropofonia (+ tecnofonia)"}
PERIODOS = ("Madrugada (0–5h)", "Manhã (6–11h)", "Tarde (12–17h)", "Noite (18–23h)")


def preparar_dados(arquivo_cei: Path, arquivo_amostra: Path | None) -> pd.DataFrame:
    if not arquivo_cei.is_file():
        raise FileNotFoundError(f"CSV dos CEI não encontrado: {arquivo_cei}")

    df = pd.read_csv(arquivo_cei)
    ausentes = [col for col in INDICES if col not in df.columns]
    if ausentes:
        raise ValueError(f"Colunas de CEI ausentes: {ausentes}. Encontradas: {list(df.columns)}")

    for coluna in INDICES:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    qtd_inicial = len(df)
    df = df.dropna(subset=list(INDICES)).copy()
    if len(df) < qtd_inicial:
        print(f"AVISO: removidas {qtd_inicial-len(df)} linhas com CEI ausente ou inválido.")
    if df.empty:
        raise ValueError("Nenhuma linha válida no CSV.")

    fora = ((df[list(INDICES)] < 0) | (df[list(INDICES)] > 1)).sum().sum()
    if fora:
        print(f"AVISO: {fora} valores CEI fora do intervalo [0,1]. Confira a extração.")

    # Metadados são opcionais e não são necessários para histogramas e dispersões.
    if arquivo_amostra is None or not arquivo_amostra.is_file():
        print("Metadados de amostragem não encontrados; gráficos por local e horário serão ignorados.")
        return df

    if "audio_id" not in df.columns:
        print("CSV de CEI sem audio_id; não é possível relacionar com a amostra.")
        return df

    meta = pd.read_csv(arquivo_amostra)
    if "audio_id" not in meta.columns:
        print("CSV de amostra sem audio_id; gráficos por local/horário serão ignorados.")
        return df

    colunas = ["audio_id", "site_name", "hour_of_day", "recording_datetime_local"]
    colunas = [coluna for coluna in colunas if coluna in meta.columns and (coluna == "audio_id" or coluna not in df.columns)]
    meta = meta[colunas].copy()
    df["audio_id"] = df["audio_id"].astype("string")
    meta["audio_id"] = meta["audio_id"].astype("string")
    if meta["audio_id"].duplicated().any():
        raise ValueError("A amostra contém audio_id duplicado; corrija antes de juntar os dados.")

    df = df.merge(meta, on="audio_id", how="left", validate="many_to_one", indicator=True)
    sem_meta = int((df["_merge"] == "left_only").sum())
    if sem_meta:
        print(f"AVISO: {sem_meta} segmentos sem correspondência na amostra.")
    df = df.drop(columns="_merge")
    return df


def salvar_grafico(fig, arquivo: Path, dpi: int) -> None:
    fig.savefig(arquivo, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {arquivo}")


def histogramas(df: pd.DataFrame, saida: Path, dpi: int) -> None:
    for indice in INDICES:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[indice], bins=35, range=(0, 1))
        ax.set(title=f"Distribuição — {NOMES[indice]}", xlabel="CEI", ylabel="Número de janelas de 6 s", xlim=(0, 1))
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        salvar_grafico(fig, saida / f"hist_{indice}.png", dpi)


def dispersoes(df: pd.DataFrame, saida: Path, dpi: int, max_pontos: int) -> None:
    # Mesmo conjunto de pontos em todos os gráficos; amostragem só para visualização.
    viz = df.sample(n=min(len(df), max_pontos), random_state=42) if len(df) > max_pontos else df
    for x, y in (("bio", "geo"), ("bio", "antro"), ("geo", "antro")):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(viz[x], viz[y], s=9, alpha=0.25, linewidths=0, rasterized=True)
        ax.set(title=f"{NOMES[x]} × {NOMES[y]}", xlabel=f"CEI {x}", ylabel=f"CEI {y}",
               xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        salvar_grafico(fig, saida / f"scatter_{x}_{y}.png", dpi)


def boxplot_compativel(ax, valores, nomes, **kwargs):
    # matplotlib 3.8 usa "labels"; versões recentes usam "tick_labels".
    chave = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
    return ax.boxplot(valores, **{chave: nomes}, **kwargs)


def classificar_periodo(hora: float) -> str | None:
    if pd.isna(hora) or not (0 <= hora <= 23):
        return None
    return PERIODOS[int(hora) // 6]


def boxplots_horario(df: pd.DataFrame, saida: Path, dpi: int) -> None:
    if "hour_of_day" not in df.columns:
        print("Coluna hour_of_day indisponível; ignorando gráficos por horário.")
        return
    dados = df.copy()
    dados["periodo"] = pd.to_numeric(dados["hour_of_day"], errors="coerce").map(classificar_periodo)
    grupos = [p for p in PERIODOS if (dados["periodo"] == p).any()]
    if not grupos:
        print("Não há horários válidos para gráficos por período.")
        return

    for indice in INDICES:
        valores = [dados.loc[dados["periodo"] == p, indice].to_numpy() for p in grupos]
        fig, ax = plt.subplots(figsize=(9, 5))
        boxplot_compativel(ax, valores, grupos, showfliers=False)
        ax.set(title=f"{NOMES[indice]} por período do dia", xlabel="Período de gravação", ylabel="CEI", ylim=(0, 1))
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        salvar_grafico(fig, saida / f"box_periodo_{indice}.png", dpi)

    resumo = dados.dropna(subset=["periodo"]).groupby("periodo", observed=True)[list(INDICES)].agg(["count", "mean", "median"])
    resumo.to_csv(saida / "resumo_por_periodo.csv")


def boxplots_locais(df: pd.DataFrame, saida: Path, dpi: int, max_locais: int) -> None:
    if "site_name" not in df.columns:
        print("Coluna site_name indisponível; ignorando gráficos por local.")
        return
    dados = df.dropna(subset=["site_name"]).copy()
    if dados.empty:
        print("Não há locais válidos no CSV.")
        return

    # Escolher locais pelo número de gravações distintas, não pelas janelas.
    pesos = (dados.groupby("site_name")["audio_id"].nunique() if "audio_id" in dados.columns
             else dados.groupby("site_name").size())
    locais = pesos.sort_values(ascending=False).head(max_locais).index.tolist()
    dados_top = dados[dados["site_name"].isin(locais)]
    for indice in INDICES:
        valores = [dados_top.loc[dados_top["site_name"] == site, indice].to_numpy() for site in locais]
        fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(locais) + 2)))
        boxplot_compativel(ax, valores[::-1], locais[::-1], vert=False, showfliers=False)
        ax.set(title=f"{NOMES[indice]} por local (top {len(locais)})", xlabel="CEI", xlim=(0, 1))
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        salvar_grafico(fig, saida / f"box_local_{indice}.png", dpi)

    resumo = dados.groupby("site_name")[list(INDICES)].agg(["count", "mean", "median"])
    resumo.to_csv(saida / "resumo_por_local.csv")


def resumo_geral(df: pd.DataFrame, saida: Path) -> None:
    # Uma linha por índice; sem interpretar os valores como porcentagens.
    descr = df[list(INDICES)].describe(percentiles=[.05, .25, .5, .75, .95]).T
    descr = descr.rename(columns={"count": "n_janelas", "mean": "media", "std": "desvio_padrao",
                                 "min": "minimo", "5%": "p05", "25%": "p25", "50%": "mediana",
                                 "75%": "p75", "95%": "p95", "max": "maximo"})
    descr.to_csv(saida / "resumo_geral.csv", float_format="%.6f")
    print("\nResumo geral:\n", descr.round(3).to_string())
    if "audio_id" in df.columns:
        print(f"\nGravações distintas: {df['audio_id'].nunique()} | Janelas: {len(df)}")
    else:
        print(f"\nJanelas: {len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gráficos exploratórios dos CEI por janela de 6 segundos")
    parser.add_argument("--csv", default="data/cei_6s.csv", type=Path,
                        help="CSV gerado por generate_labels.py")
    parser.add_argument("--sample", default=Path("data/recordings.csv"), type=Path,
                        help="CSV original de amostragem, para comparar horários e locais")
    parser.add_argument("--out", default=Path("results/figures"), type=Path)
    parser.add_argument("--top-sites", default=12, type=int, help="Máximo de locais por boxplot")
    parser.add_argument("--max-scatter", default=20000, type=int, help="Limite visual de pontos por dispersão")
    parser.add_argument("--dpi", default=180, type=int)
    args = parser.parse_args()
    if args.top_sites < 1 or args.max_scatter < 1 or args.dpi < 50:
        parser.error("--top-sites e --max-scatter devem ser positivos, e --dpi >= 50")

    args.out.mkdir(parents=True, exist_ok=True)
    df = preparar_dados(args.csv, args.sample)
    print(f"Gerando visualizações para {len(df)} janelas em {args.out}/")
    resumo_geral(df, args.out)
    histogramas(df, args.out, args.dpi)
    dispersoes(df, args.out, args.dpi, args.max_scatter)
    boxplots_horario(df, args.out, args.dpi)
    boxplots_locais(df, args.out, args.dpi, args.top_sites)
    print("\nConcluído. Abra os PNGs e os CSVs da pasta de saída.")


if __name__ == "__main__":
    main()
