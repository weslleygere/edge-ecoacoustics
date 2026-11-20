
#!/usr/bin/env python3
"""
Benchmark ESP32-S3 — Edge Ecoacoustics.

Gera:
- 5 gráficos em PNG e SVG
- 3 tabelas em PNG e SVG
- 5 arquivos CSV
- 1 resumo em Markdown

Uso:
    python analisar.py eco_bench_log.txt --output resultados_benchmark

Dependência:
    python -m pip install matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as stats
from pathlib import Path


CAMPOS = (
    "capture_ms",
    "dsp_ms",
    "invoke_ms",
    "run_extra_ms",
    "process_ms",
    "cycle_ms",
    "interval_ms",
)

NOMES = {
    "capture_ms": "Captura I2S",
    "dsp_ms": "Log-Mel (DSP)",
    "invoke_ms": "Inferência PhiNet",
    "run_extra_ms": "Outros (inferência)",
    "process_ms": "Processamento total",
    "cycle_ms": "Ciclo medido",
    "interval_ms": "Intervalo entre capturas",
}

MEM = (
    "phase",
    "internal_free",
    "internal_min",
    "psram_free",
    "psram_min",
)

PADRAO = re.compile(r"\b(BENCH_MEM|BENCH_BUFFERS|BENCH),")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# ============================================================
# 1. LEITURA E TRATAMENTO DOS DADOS
# ============================================================

def numerico(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def csv_salvar(path, colunas, itens):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=colunas,
            extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(itens)


def ler_log(path):
    eventos = []

    linhas = path.read_text(
        encoding="utf-8-sig",
        errors="replace"
    ).splitlines()

    for n, s in enumerate(linhas, 1):
        s = ANSI.sub("", s)
        m = PADRAO.search(s)

        if m:
            atributos = {}

            for trecho in s[m.end():].split(","):
                if "=" in trecho:
                    k, v = trecho.split("=", 1)
                    atributos[k.strip()] = v.strip()

            eventos.append(
                (n, m.group(1), atributos, s)
            )

        elif (
            "tinyCLAP INT8 ready" in s
            and "arena_used=" in s
        ):
            eventos.append(
                (n, "MODEL_INFO", {}, s)
            )

    return eventos


def selecionar_execucao(eventos):
    """
    Se houver reinicializações, seleciona a sequência
    com mais janelas, sem misturar execuções diferentes.
    """

    grupos = []
    grupo = []
    ultimo = 0

    for n, tipo, dados, _ in eventos:
        if tipo != "BENCH":
            continue

        try:
            seq = int(dados.get("seq", ""))
        except ValueError:
            continue

        if seq < 1:
            continue

        if grupo and seq < ultimo:
            grupos.append(grupo)
            grupo = []

        grupo.append((n, seq, dados))
        ultimo = seq

    if grupo:
        grupos.append(grupo)

    if not grupos:
        raise ValueError(
            "Nenhum registro BENCH válido encontrado."
        )

    i = max(
        range(len(grupos)),
        key=lambda j: (
            len(set(x[1] for x in grupos[j])),
            j
        )
    )

    escolhido = grupos[i]

    primeiro = escolhido[0][0]
    ultimo_linha = escolhido[-1][0]

    unicos = {
        seq: dados
        for _, seq, dados in escolhido
    }

    janelas = []

    for seq, dados in sorted(unicos.items()):
        r = {"seq": seq}

        for k in CAMPOS:
            v = numerico(dados.get(k))
            r[k] = (
                v if v is not None and v >= 0
                else None
            )

        janelas.append(r)

    # Identificar o início da execução.
    boots = [
        n for n, t, d, _ in eventos
        if t == "BENCH_MEM"
        and d.get("phase") == "boot"
        and n <= primeiro
    ]

    inicio = (
        boots[-1]
        if boots
        else (
            0 if i == 0
            else grupos[i - 1][-1][0] + 1
        )
    )

    # Incluir BENCH_MEM que aparece depois de BENCH seq=30.
    proximos = [
        n for n, t, d, _ in eventos
        if t == "BENCH_MEM"
        and d.get("phase") == "boot"
        and n > ultimo_linha
    ]

    if i + 1 < len(grupos):
        proximos.append(grupos[i + 1][0][0])

    fim = min(proximos) if proximos else math.inf

    memoria = []
    buffers = {}
    modelo = {}

    for n, tipo, d, s in eventos:
        if not (inicio <= n < fim):
            continue

        if tipo == "BENCH_MEM":
            r = {
                "phase": d.get("phase", "?")
            }

            for k in MEM[1:]:
                r[k] = numerico(d.get(k))

            memoria.append(r)

        elif tipo == "BENCH_BUFFERS":
            buffers = {
                k: numerico(d.get(k))
                for k in (
                    "pcm_bytes",
                    "logmel_bytes"
                )
            }

        elif tipo == "MODEL_INFO":
            for k in ("model", "arena_used"):
                m = re.search(
                    rf"\b{k}=(\d+)",
                    s
                )

                if m:
                    modelo[k] = int(m.group(1))

    return (
        janelas,
        memoria,
        buffers,
        modelo,
        len(grupos)
    )


# ============================================================
# 2. ESTATÍSTICAS
# ============================================================

def percentil(valores, pct):
    v = sorted(valores)
    p = (len(v) - 1) * pct

    a = int(p)
    b = min(a + 1, len(v) - 1)

    return v[a] + (v[b] - v[a]) * (p - a)


def resumo(janelas):
    itens = []

    for k in CAMPOS:
        v = [
            r[k]
            for r in janelas
            if r[k] is not None
        ]

        if not v:
            continue

        itens.append({
            "metrica": k,
            "descricao": NOMES[k],
            "n": len(v),
            "media_ms": round(stats.mean(v), 3),
            "desvio_ms": (
                round(stats.stdev(v), 3)
                if len(v) > 1 else 0
            ),
            "min_ms": round(min(v), 3),
            "mediana_ms": round(
                stats.median(v), 3
            ),
            "p95_ms": round(
                percentil(v, .95), 3
            ),
            "max_ms": round(max(v), 3),
        })

    return itens


# ============================================================
# 3. MODELO, FIRMWARE E MEMÓRIA
# ============================================================

def arquivos_armazenamento(args, buffers, modelo):

    def localizar(path, candidatos):
        if path:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Arquivo não encontrado: {path}"
                )
            return path

        return next(
            (x for x in candidatos if x.is_file()),
            None
        )

    modelo_path = localizar(
        args.model,
        [
            Path(
                "firmware/esp32s3/model/"
                "tinyclap_phinet_int8.tflite"
            )
        ]
    )

    raiz = (
        args.build_dir
        or Path("C:/esp/edge-ecoacoustics-build/build")
    )

    fw_candidatos = [
        x for x in raiz.glob("*.bin")
        if x.name not in (
            "bootloader.bin",
            "partition-table.bin"
        )
    ]

    fw_path = localizar(
        args.firmware,
        fw_candidatos
        if len(fw_candidatos) == 1
        else []
    )

    def linha(nome, b, fonte):
        return {
            "item": nome,
            "bytes": (
                round(b) if b is not None
                else ""
            ),
            "KiB": (
                round(b / 1024, 2)
                if b is not None else ""
            ),
            "MiB": (
                round(b / 1024**2, 3)
                if b is not None else ""
            ),
            "origem": fonte,
        }

    return [
        linha(
            "Modelo TFLite",
            modelo_path.stat().st_size
            if modelo_path else modelo.get("model"),
            "Arquivo TFLite"
            if modelo_path else "Log do dispositivo"
        ),
        linha(
            "Firmware (.bin)",
            fw_path.stat().st_size
            if fw_path else None,
            "Imagem de aplicação"
            if fw_path else "Não encontrado"
        ),
        linha(
            "Tensor Arena reservada",
            6 * 1024**2,
            "Configuração do código"
        ),
        linha(
            "Tensor Arena usada",
            modelo.get("arena_used"),
            "Log de inicialização"
        ),
        linha(
            "Buffer PCM",
            buffers.get("pcm_bytes"),
            "BENCH_BUFFERS"
        ),
        linha(
            "Buffer Log-Mel",
            buffers.get("logmel_bytes"),
            "BENCH_BUFFERS"
        ),
    ]


# ============================================================
# 4. GRÁFICOS E TABELAS
# ============================================================

def construir_figuras(
    out,
    janelas,
    sumario,
    memoria,
    storage,
    dpi
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

    except ImportError:
        print(
            "Matplotlib ausente. Instale com:\n"
            "python -m pip install matplotlib"
        )
        return 0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Paleta para apresentação e relatório.
    AZUL = "#246EAD"
    TURQUESA = "#009688"
    LARANJA = "#E58A2B"
    ROXO = "#7656A3"
    TEXTO = "#182532"
    CINZA = "#657483"
    BORDA = "#DFE5EB"

    mapa = {
        r["metrica"]: r
        for r in sumario
    }

    nfig = 0

    def media(k):
        return (
            mapa[k]["media_ms"]
            if k in mapa
            else None
        )

    def fmt(v, casas=2):
        if v is None or v == "":
            return "n/d"

        return (
            f"{v:,.{casas}f}"
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

    def pagina(titulo, subtitulo):
        fig = plt.figure(
            figsize=(13.33, 7.5),
            facecolor="white"
        )

        fig.text(
            .065, .947,
            "EDGE ECOACOUSTICS  /  ESP32-S3",
            size=10,
            weight="bold",
            color=AZUL
        )

        fig.text(
            .065, .87,
            titulo,
            size=21,
            weight="bold",
            color=TEXTO
        )

        fig.text(
            .065, .817,
            subtitulo,
            size=11,
            color=CINZA
        )

        fig.text(
            .94, .045,
            f"{len(janelas)} janelas analisadas",
            ha="right",
            size=9,
            color=CINZA
        )

        return fig

    def salvar(fig, nome):
        nonlocal nfig

        fig.savefig(
            out / f"{nome}.png",
            dpi=dpi,
            bbox_inches="tight"
        )

        fig.savefig(
            out / f"{nome}.svg",
            bbox_inches="tight"
        )

        plt.close(fig)
        nfig += 1

    cap, dsp, inv, proc = (
        media(k)
        for k in (
            "capture_ms",
            "dsp_ms",
            "invoke_ms",
            "process_ms"
        )
    )

    intervalo = (
        media("interval_ms")
        or media("cycle_ms")
    )

    # ========================================================
    # GRÁFICO 1 — VISÃO GERAL
    # ========================================================

    if all(
        x is not None
        for x in (
            cap, dsp, inv, proc, intervalo
        )
    ):
        fig = pagina(
            "Panorama do desempenho",
            "Cada janela contém 6 segundos de áudio; "
            "processamento ocorre com a captura pausada."
        )

        kpis = [
            ("CAPTURA", cap / 1000),
            ("LOG-MEL", dsp / 1000),
            ("PHINET", inv / 1000),
            ("INTERVALO", intervalo / 1000)
        ]

        for i, (rotulo, valor) in enumerate(kpis):
            x = .065 + .226 * i

            fig.add_artist(
                FancyBboxPatch(
                    (x, .59),
                    .203,
                    .155,
                    transform=fig.transFigure,
                    boxstyle=(
                        "round,pad=0.008,"
                        "rounding_size=.012"
                    ),
                    facecolor="#F3F7FB",
                    edgecolor=BORDA
                )
            )

            fig.text(
                x + .015, .700,
                rotulo,
                size=10,
                weight="bold",
                color=CINZA
            )

            fig.text(
                x + .015, .625,
                f"{fmt(valor)} s",
                size=24,
                weight="bold",
                color=TEXTO
            )

        ax = fig.add_axes([
            .065, .35, .87, .145
        ])

        partes = [
            ("Captura", cap, AZUL),
            ("Log-Mel", dsp, LARANJA),
            ("PhiNet", inv, TURQUESA),
            (
                "Outros",
                max(
                    0,
                    intervalo - cap - dsp - inv
                ),
                ROXO
            )
        ]

        pos = 0

        for rotulo, valor, cor in partes:
            ax.barh(
                0,
                valor / 1000,
                left=pos / 1000,
                color=cor,
                height=.60
            )

            if valor > 1500:
                ax.text(
                    (pos + valor / 2) / 1000,
                    0,
                    f"{rotulo}\n"
                    f"{fmt(valor / 1000, 1)} s",
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold"
                )

            pos += valor

        ax.set_xlim(
            0,
            max(intervalo, pos) / 1000 * 1.005
        )

        ax.set_yticks([])

        ax.set_xlabel(
            "Tempo entre inícios de capturas (s)"
        )

        ax.spines[
            ["left", "bottom"]
        ].set_visible(False)

        fig.text(
            .065, .235,
            "Cobertura de áudio estimada: "
            f"{fmt(cap / intervalo * 100, 1)}%",
            size=18,
            weight="bold",
            color=TEXTO
        )

        fig.text(
            .065, .17,
            "Outros = quantização, pós-processamento "
            "e diferenças entre ciclos "
            "(resíduo aproximado).",
            color=CINZA,
            size=10
        )

        salvar(fig, "01_visao_geral")

    # ========================================================
    # GRÁFICO 2 — GARGALO COMPUTACIONAL
    # ========================================================

    chaves = [
        k for k in (
            "capture_ms",
            "dsp_ms",
            "invoke_ms"
        )
        if k in mapa
    ]

    if chaves:
        fig = pagina(
            "Onde está o gargalo?",
            "Comparação de tempos médios por etapa; "
            "barras de erro = desvio-padrão."
        )

        ax = fig.add_axes([
            .26, .26, .60, .45
        ])

        cores = [
            AZUL,
            LARANJA,
            TURQUESA
        ]

        means = [
            media(k) / 1000
            for k in chaves
        ]

        for i, (k, v) in enumerate(
            zip(chaves, means)
        ):
            ax.barh(
                i,
                v,
                height=.52,
                color=cores[i],
                xerr=(
                    mapa[k]["desvio_ms"] / 1000
                ),
                capsize=4
            )

            ax.text(
                v + .13,
                i,
                f"{fmt(v)} s",
                va="center",
                weight="bold",
                size=13
            )

        ax.set_yticks(
            range(len(chaves)),
            [NOMES[k] for k in chaves]
        )

        ax.invert_yaxis()

        ax.set_xlim(
            0,
            max(means) * 1.21
        )

        ax.set_xlabel(
            "Duração média (s)"
        )

        ax.grid(
            axis="x",
            color=BORDA
        )

        ax.set_axisbelow(True)

        if dsp is not None and inv is not None:
            fig.text(
                .07, .145,
                "O Log-Mel leva "
                f"{fmt(dsp / inv)}× "
                "o tempo da inferência PhiNet.",
                size=16,
                color=TEXTO,
                weight="bold"
            )

        salvar(
            fig,
            "02_latencia_por_etapa"
        )

    # ========================================================
    # GRÁFICO 3 — ESTABILIDADE DAS JANELAS
    # ========================================================

    chaves = [
        k for k in (
            "dsp_ms",
            "invoke_ms"
        )
        if k in mapa
    ]

    if chaves:
        fig = pagina(
            "Estabilidade nas 30 janelas",
            "Variação (ms) em relação "
            "à média de cada operação."
        )

        ax = fig.add_axes([
            .10, .22, .80, .52
        ])

        for k, cor in zip(
            chaves,
            (LARANJA, TURQUESA)
        ):
            dados = [
                (
                    r["seq"],
                    r[k] - media(k)
                )
                for r in janelas
                if r[k] is not None
            ]

            ax.plot(
                [p[0] for p in dados],
                [p[1] for p in dados],
                marker="o",
                markersize=4,
                linewidth=1.9,
                color=cor,
                label=NOMES[k]
            )

        ax.axhline(
            0,
            color=CINZA,
            linestyle="--",
            linewidth=1
        )

        ax.set_xlabel(
            "Janela de áudio"
        )

        ax.set_ylabel(
            "Diferença para a média (ms)"
        )

        ax.grid(
            axis="y",
            color=BORDA
        )

        ax.legend(
            frameon=False
        )

        salvar(
            fig,
            "03_estabilidade"
        )

    # ========================================================
    # GRÁFICO 4 — DISTRIBUIÇÃO DOS TEMPOS
    # ========================================================

    chaves = [
        k for k in (
            "dsp_ms",
            "invoke_ms"
        )
        if sum(
            r[k] is not None
            for r in janelas
        ) >= 3
    ]

    if chaves:
        fig = pagina(
            "Distribuição dos tempos",
            "Boxplot com dispersão e mediana; "
            "escalas independentes."
        )

        for i, (k, cor) in enumerate(
            zip(
                chaves,
                (LARANJA, TURQUESA)
            )
        ):
            ax = fig.add_axes([
                .09 + i * .47,
                .22,
                .37,
                .49
            ])

            vals = [
                r[k]
                for r in janelas
                if r[k] is not None
            ]

            ax.boxplot(
                vals,
                patch_artist=True,
                widths=.25,
                boxprops={
                    "facecolor": cor,
                    "alpha": .3,
                    "edgecolor": cor
                },
                medianprops={
                    "color": TEXTO,
                    "linewidth": 2
                }
            )

            # Dispersão determinística dos pontos.
            posicoes = [
                1 + .08 * math.sin(j * 1.81)
                for j in range(len(vals))
            ]

            ax.scatter(
                posicoes,
                vals,
                s=19,
                alpha=.55,
                color=cor
            )

            ax.axhline(
                stats.mean(vals),
                color=cor,
                linewidth=1.5,
                linestyle="--"
            )

            ax.set_title(
                NOMES[k],
                weight="bold",
                pad=13
            )

            ax.set_xticks([])

            ax.set_ylabel(
                "Tempo (ms)"
            )

            ax.grid(
                axis="y",
                color=BORDA
            )

            margem = max(
                10,
                (max(vals) - min(vals)) * .25
            )

            ax.set_ylim(
                min(vals) - margem,
                max(vals) + margem
            )

            ax.text(
                .5, -.13,
                f"Média: {fmt(stats.mean(vals), 1)} ms"
                "  |  "
                f"DP: {fmt(stats.stdev(vals), 1)} ms",
                transform=ax.transAxes,
                ha="center",
                color=CINZA,
                size=10
            )

        salvar(
            fig,
            "04_distribuicao"
        )

    # ========================================================
    # GRÁFICO 5 — RESUMO DE MEMÓRIA
    # ========================================================

    if memoria:
        fig = pagina(
            "Memória disponível",
            "Snapshots registrados pelo firmware "
            "(não é uma curva contínua)."
        )

        sel = []

        for fase in (
            "window_1",
            "window_30"
        ):
            item = next(
                (
                    m for m in reversed(memoria)
                    if m["phase"] == fase
                ),
                None
            )

            if item:
                sel.append(item)

        if not sel:
            sel = memoria[:2]

        campos_mem = [
            (
                .61,
                "RAM interna",
                "internal_free",
                "internal_min",
                1024,
                "KiB"
            ),
            (
                .36,
                "PSRAM",
                "psram_free",
                "psram_min",
                1024**2,
                "MiB"
            )
        ]

        for (
            y,
            nome,
            chave,
            min_chave,
            div,
            unidade
        ) in campos_mem:

            fig.text(
                .08,
                y + .075,
                nome.upper(),
                weight="bold",
                color=AZUL,
                size=12
            )

            for i, r in enumerate(sel):
                v = r.get(chave)

                fase = r["phase"].replace(
                    "window_",
                    "Janela "
                )

                fig.text(
                    .09 + i * .28,
                    y,
                    fase,
                    color=CINZA,
                    size=11
                )

                valor = (
                    fmt(v / div, 2) + " " + unidade
                    if v is not None
                    else "n/d"
                )

                fig.text(
                    .09 + i * .28,
                    y - .065,
                    valor,
                    color=TEXTO,
                    weight="bold",
                    size=22
                )

            minimo = sel[-1].get(
                min_chave
            )

            fig.text(
                .69,
                y,
                "Mínimo desde a inicialização",
                color=CINZA,
                size=11
            )

            minimo_texto = (
                fmt(minimo / div, 2)
                + " "
                + unidade
                if minimo is not None
                else "n/d"
            )

            fig.text(
                .69,
                y - .065,
                minimo_texto,
                color=TEXTO,
                weight="bold",
                size=20
            )

        fig.text(
            .08, .115,
            "Apenas duas leituras não comprovam "
            "ausência de vazamentos de memória.",
            color=CINZA,
            size=11
        )

        salvar(
            fig,
            "05_memoria_resumo"
        )

    # ========================================================
    # TABELAS EM IMAGEM
    # ========================================================

    def tabela(
        nome,
        titulo,
        subtitulo,
        colunas,
        dados,
        widths
    ):
        if not dados:
            return

        fig = pagina(
            titulo,
            subtitulo
        )

        ax = fig.add_axes([
            .065, .15, .87, .61
        ])

        ax.axis("off")

        t = ax.table(
            cellText=dados,
            colLabels=colunas,
            colWidths=widths,
            loc="center",
            cellLoc="left",
            colLoc="left",
            bbox=[0, .04, 1, .90]
        )

        t.auto_set_font_size(False)
        t.set_fontsize(10.7)

        for (i, j), c in t.get_celld().items():
            c.set_edgecolor(BORDA)
            c.set_linewidth(.5)

            c.set_facecolor(
                "#E8F1F8"
                if i == 0
                else (
                    "#F7F9FB"
                    if i % 2 == 0
                    else "white"
                )
            )

            if i == 0:
                c.get_text().set_weight(
                    "bold"
                )

        salvar(fig, nome)

    # Tabela 1 — Desempenho.
    tabela(
        "tabela_01_desempenho",
        "Tabela 1 — desempenho temporal",
        "Tempos em milissegundos (ms)",
        [
            "Etapa",
            "n",
            "Média",
            "DP",
            "Mín.",
            "Máx."
        ],
        [
            [
                r["descricao"],
                str(r["n"]),
                fmt(r["media_ms"], 1),
                fmt(r["desvio_ms"], 1),
                fmt(r["min_ms"], 1),
                fmt(r["max_ms"], 1)
            ]
            for r in sumario
        ],
        [.32, .07, .16, .13, .16, .16]
    )

    # Tabela 2 — Memória.
    if memoria:
        tabela(
            "tabela_02_memoria",
            "Tabela 2 — memória",
            "Valores livres e mínimos "
            "desde a inicialização",
            [
                "Fase",
                "RAM livre\nKiB",
                "RAM mín.\nKiB",
                "PSRAM livre\nMiB",
                "PSRAM mín.\nMiB"
            ],
            [
                [
                    r["phase"],
                    fmt(
                        r["internal_free"] / 1024, 2
                    )
                    if r["internal_free"] is not None
                    else "n/d",
                    fmt(
                        r["internal_min"] / 1024, 2
                    )
                    if r["internal_min"] is not None
                    else "n/d",
                    fmt(
                        r["psram_free"] / 1024**2, 3
                    )
                    if r["psram_free"] is not None
                    else "n/d",
                    fmt(
                        r["psram_min"] / 1024**2, 3
                    )
                    if r["psram_min"] is not None
                    else "n/d"
                ]
                for r in memoria
            ],
            [.22, .18, .18, .21, .21]
        )

    # Tabela 3 — Armazenamento.
    tabela(
        "tabela_03_armazenamento",
        "Tabela 3 — memória e armazenamento",
        "n/d = valor não disponível "
        "no log ou no build",
        [
            "Recurso",
            "Tamanho (KiB)",
            "Origem"
        ],
        [
            [
                r["item"],
                fmt(r["KiB"], 2),
                r["origem"]
            ]
            for r in storage
        ],
        [.33, .23, .44]
    )

    return nfig


# ============================================================
# 5. EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description=(
            "Analisa BENCH/BENCH_MEM e cria "
            "evidências de benchmark."
        )
    )

    p.add_argument(
        "log",
        type=Path
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("resultados_benchmark")
    )

    p.add_argument(
        "--build-dir",
        type=Path
    )

    p.add_argument(
        "--model",
        type=Path
    )

    p.add_argument(
        "--firmware",
        type=Path
    )

    p.add_argument(
        "--esperadas",
        type=int,
        default=30
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=220
    )

    p.add_argument(
        "--sem-graficos",
        action="store_true"
    )

    a = p.parse_args()

    if not a.log.is_file():
        p.error(
            f"Arquivo não encontrado: {a.log}"
        )

    if a.dpi < 72 or a.esperadas < 1:
        p.error(
            "--dpi >= 72 e --esperadas >= 1 "
            "são necessários."
        )

    try:
        (
            janelas,
            memoria,
            buffers,
            modelo,
            execucoes
        ) = selecionar_execucao(
            ler_log(a.log)
        )

        storage = arquivos_armazenamento(
            a,
            buffers,
            modelo
        )

    except (
        ValueError,
        FileNotFoundError
    ) as erro:
        p.error(str(erro))

    sumario = resumo(janelas)

    a.output.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salvar resultados em CSV.
    csv_salvar(
        a.output / "janelas.csv",
        ["seq", *CAMPOS],
        janelas
    )

    csv_salvar(
        a.output / "resumo.csv",
        [
            "metrica",
            "descricao",
            "n",
            "media_ms",
            "desvio_ms",
            "min_ms",
            "mediana_ms",
            "p95_ms",
            "max_ms"
        ],
        sumario
    )

    csv_salvar(
        a.output / "memoria.csv",
        list(MEM),
        memoria
    )

    csv_salvar(
        a.output / "buffers.csv",
        [
            "pcm_bytes",
            "logmel_bytes"
        ],
        [buffers] if buffers else []
    )

    csv_salvar(
        a.output / "armazenamento.csv",
        [
            "item",
            "bytes",
            "KiB",
            "MiB",
            "origem"
        ],
        storage
    )

    # Avisos sobre a qualidade da coleta.
    avisos = []

    if len(janelas) < a.esperadas:
        avisos.append(
            f"Foram encontradas {len(janelas)} "
            f"de {a.esperadas} janelas esperadas."
        )

    if execucoes > 1:
        avisos.append(
            f"Log contém {execucoes} sequências; "
            "foi selecionada a maior."
        )

    if len(memoria) < 3:
        avisos.append(
            "Memória foi medida em poucas fases; "
            "não inferir tendências contínuas."
        )

    # Gerar resumo em Markdown.
    resumo_md = [
        "# Benchmark ESP32-S3 — Edge Ecoacoustics",
        "",
        f"Janelas analisadas: **{len(janelas)}**.",
        "",
        "## Estatísticas temporais",
        "",
        "| Etapa | Média (ms) | Desvio-padrão (ms) "
        "| Min (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|"
    ]

    for r in sumario:
        resumo_md.append(
            f"| {r['descricao']} "
            f"| {r['media_ms']:.2f} "
            f"| {r['desvio_ms']:.2f} "
            f"| {r['min_ms']:.2f} "
            f"| {r['max_ms']:.2f} |"
        )

    resumo_md += [
        "",
        "## Armazenamento",
        "",
        "| Recurso | KiB | Origem |",
        "|---|---:|---|"
    ]

    for r in storage:
        tamanho = (
            r["KiB"]
            if r["KiB"] != ""
            else "n/d"
        )

        resumo_md.append(
            f"| {r['item']} "
            f"| {tamanho} "
            f"| {r['origem']} |"
        )

    if avisos:
        resumo_md += [
            "",
            "## Limitações",
            ""
        ] + [
            "- " + v for v in avisos
        ]

    resumo_md += [
        "",
        "Os índices Bio, Geo e Antro não são "
        "probabilidades calibradas.",
        "",
        "O tempo de `invoke_ms` mede somente "
        "a execução do modelo.",
        "",
        "`process_ms` já inclui DSP e inferência "
        "(não somar novamente).",
        "",
        "Valores de memória mínima são acumulados "
        "desde o boot.",
        "",
        "O arquivo .bin não representa sozinho "
        "toda a Flash ocupada."
    ]

    (
        a.output / "relatorio_resumo.md"
    ).write_text(
        "\n".join(resumo_md) + "\n",
        encoding="utf-8"
    )

    # Gerar as figuras.
    figuras = (
        0
        if a.sem_graficos
        else construir_figuras(
            a.output,
            janelas,
            sumario,
            memoria,
            storage,
            a.dpi
        )
    )

    print(
        f"\n{len(janelas)} janelas analisadas. "
        f"Resultados: {a.output.resolve()}"
    )

    for r in sumario:
        print(
            f"{r['descricao']:<28} "
            f"{r['media_ms']/1000:>7.3f} s "
            f"(DP {r['desvio_ms']:.2f} ms)"
        )

    print(
        "5 CSVs, 1 resumo Markdown e "
        f"{figuras} figuras/tabelas em PNG+SVG."
    )

    for aviso in avisos:
        print("Aviso:", aviso)


if __name__ == "__main__":
    main()
