"""
=============================================================
  ETAPA 1 — INGESTÃO E PRÉ-PROCESSAMENTO
  Lê os 20 GB da ANAC com Polars (muito mais rápido que Pandas)
  e salva em Parquet particionado por ano/mês para uso nos modelos
=============================================================
"""

import polars as pl                          # substitui Pandas para arquivos grandes
import polars.selectors as cs                # seletores de colunas por tipo
from pathlib import Path                     # manipulação de caminhos cross-platform
import logging                               # registro de progresso no terminal

# ── configuração básica de log ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)            # logger com o nome deste arquivo

# ── caminhos do projeto ─────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")                 # pasta onde estão os CSVs da ANAC
PARQUET   = Path("data/processed")          # pasta de saída em formato Parquet
PARQUET.mkdir(parents=True, exist_ok=True)   # cria a pasta se não existir

# ── colunas que realmente vamos usar (descarta as redundantes) ──────────────
COLUNAS_UTEIS = [
    # identificação do voo
    "id_basica", "nr_voo", "id_empresa",
    "sg_empresa_icao", "nm_empresa",

    # dimensão temporal (referência)
    "dt_referencia", "nr_ano_referencia", "nr_mes_referencia",
    "nr_semana_referencia", "nm_dia_semana_referencia",
    "nr_dia_referencia",

    # partida e chegada reais
    "dt_partida_real", "hr_partida_real",
    "dt_chegada_real",  "hr_chegada_real",
    "nr_mes_partida_real", "nm_dia_semana_partida_real",
    "nr_hora_partida_real",                  # derivada — criamos abaixo

    # aeroporto origem
    "sg_icao_origem", "sg_iata_origem",
    "nm_municipio_origem", "sg_uf_origem",
    "nm_regiao_origem", "nm_continente_origem",

    # aeroporto destino
    "sg_icao_destino", "sg_iata_destino",
    "nm_municipio_destino", "sg_uf_destino",
    "nm_regiao_destino", "nm_continente_destino",

    # tipo de voo
    "ds_tipo_linha", "ds_natureza_tipo_linha",

    # capacidade e ocupação
    "nr_assentos_ofertados", "nr_passag_pagos", "nr_passag_gratis",

    # bagagem
    "kg_bagagem_livre", "kg_bagagem_excesso",

    # carga e correio
    "kg_carga_paga", "kg_carga_gratis", "kg_correio",

    # operação
    "nr_decolagem", "nr_horas_voadas", "lt_combustivel",
    "km_distancia", "kg_peso",

    # métricas derivadas da ANAC
    "nr_ask", "nr_rpk",                      # capacidade e receita por km
]

# ── tipos explícitos evitam inferência errada pelo Polars ──────────────────
SCHEMA_OVERRIDE = {
    "nr_voo":               pl.Utf8,
    "hr_partida_real":      pl.Utf8,
    "hr_chegada_real":      pl.Utf8,
    "dt_referencia":        pl.Utf8,
    "dt_partida_real":      pl.Utf8,
    "dt_chegada_real":      pl.Utf8,
    "nr_assentos_ofertados":pl.Float32,      # ← era Int32
    "nr_passag_pagos":      pl.Float32,      # ← era Int32
    "nr_passag_gratis":     pl.Float32,      # ← era Int16
    "nr_decolagem":         pl.Float32,      # ← era Int16
    "lt_combustivel":       pl.Float32,
    "km_distancia":         pl.Float32,
    "kg_peso":              pl.Float32,
    "kg_bagagem_livre":     pl.Float32,
    "kg_bagagem_excesso":   pl.Float32,
    "kg_carga_paga":        pl.Float32,
    "nr_ask":               pl.Float64,
    "nr_rpk":               pl.Float64,
}


def ler_csv_anac(caminho: Path) -> pl.LazyFrame:
    """
    Lê UM arquivo CSV da ANAC como LazyFrame.
    LazyFrame = Polars não carrega nada ainda, só planeja a operação.
    O dado só sobe para RAM quando chamamos .collect().
    """
    log.info(f"Lendo: {caminho.name}")

    return (
        pl.scan_csv(                         # scan = leitura lazy (não carrega ainda)
            caminho,
            separator=",",                   # ANAC usa ponto-e-vírgula
            encoding="utf8-lossy",           # ignora bytes inválidos sem travar
            infer_schema_length=0,           # lê tudo como string primeiro
            schema_overrides=SCHEMA_OVERRIDE,# aplica tipos que definimos acima
            null_values=["", "NA", "N/A"],   # strings que representam nulo
            truncate_ragged_lines=True,      # linhas com colunas extras são ignoradas
        )
        .select(                             # mantém só as colunas que precisamos
            [c for c in COLUNAS_UTEIS
             if c != "nr_hora_partida_real"] # esta coluna vamos derivar depois
        )
    )


def limpar_e_transformar(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Transforma os dados brutos:
    - Converte datas/horas para tipos nativos
    - Cria features derivadas essenciais
    - Remove linhas inválidas
    """
    return (
        lf

        # ── parse de datas ──────────────────────────────────────────────────
        .with_columns([
            pl.col("dt_referencia").str.strptime(
                pl.Date, "%Y-%m-%d", strict=False   # strict=False → nulo se falhar
            ).alias("dt_referencia"),

            pl.col("dt_partida_real").str.strptime(
                pl.Date, "%Y-%m-%d", strict=False
            ).alias("dt_partida_real"),

            pl.col("dt_chegada_real").str.strptime(
                pl.Date, "%Y-%m-%d", strict=False
            ).alias("dt_chegada_real"),
        ])

        # ── extrai hora da partida real (0–23) ─────────────────────────────
        .with_columns([
            pl.col("hr_partida_real")
              .str.slice(0, 2)               # pega "HH" de "HH:MM:SS"
              .cast(pl.Int8, strict=False)   # converte para inteiro
              .alias("nr_hora_partida_real"),
        ])

        # ── features de taxa de ocupação ────────────────────────────────────
        .with_columns([
            # taxa_ocupacao: quanto do avião foi preenchido (0.0 a 1.0)
            (pl.col("nr_passag_pagos") / pl.col("nr_assentos_ofertados").cast(pl.Float32))
              .clip(0.0, 1.0)               # garante que fica entre 0 e 1
              .alias("taxa_ocupacao"),

            # assentos_vazios: quantos lugares sobraram no voo
            (pl.col("nr_assentos_ofertados") - pl.col("nr_passag_pagos"))
              .clip(0, None)                 # sem valores negativos
              .alias("assentos_vazios"),

            # flag_bagagem_excesso: 1 se houve cobrança de excesso, 0 se não
            (pl.col("kg_bagagem_excesso") > 0)
              .cast(pl.Int8)
              .alias("flag_bagagem_excesso"),

            # rota_od: chave única de par origem-destino (ex: "GRU→CGH")
            (pl.col("sg_icao_origem") + "→" + pl.col("sg_icao_destino"))
              .alias("rota_od"),

            # flag_internacional: 1 se voo sai do Brasil
            (pl.col("nm_continente_destino") != "América do Sul")
              .cast(pl.Int8)
              .alias("flag_internacional"),

            # load_factor_km: passageiros × km / assentos × km (eficiência real)
            (pl.col("nr_rpk") / pl.col("nr_ask").clip(1.0, None))
              .alias("load_factor_km"),
        ])

        # ── remove linhas sem dados operacionais essenciais ─────────────────
        .filter(
            (pl.col("nr_assentos_ofertados").is_not_null()) &  # precisa ter capacidade
            (pl.col("nr_assentos_ofertados") > 0)            & # avião com 0 assentos = erro
            (pl.col("dt_partida_real").is_not_null())         & # precisa de data de voo
            (pl.col("nr_decolagem") > 0)                       # voo precisa ter decolado
        )
    )


def salvar_parquet(lf: pl.LazyFrame, ano: int, mes: int) -> Path:
    """
    Executa o plano lazy e salva um arquivo Parquet por (ano, mês).
    Parquet: formato colunar comprimido — leitura ~10x mais rápida que CSV.
    """
    saida = PARQUET / f"anac_{ano}_{mes:02d}.parquet"  # ex: anac_2023_06.parquet

    if saida.exists():                       # pula se já foi processado antes
        log.info(f"  Já existe: {saida.name} — pulando")
        return saida

    log.info(f"  Salvando {saida.name}...")
    df = lf.collect(engine="streaming")          # streaming=True: processa em chunks de RAM

    df.write_parquet(
        saida,
        compression="zstd",                 # zstd: melhor compressão que snappy
        compression_level=3,                # nível 3: bom equilíbrio velocidade/tamanho
        statistics=True,                    # salva estatísticas para leitura mais rápida
    )

    log.info(f"  Salvo: {df.shape[0]:,} linhas, {saida.stat().st_size/1e6:.1f} MB")
    return saida


def processar_todos_csvs():
    """
    Ponto de entrada principal.
    Itera sobre todos os CSVs da pasta raw, processa e salva.
    """
    csvs = sorted(RAW_DIR.glob("*.csv"))     # lista todos os CSVs em ordem

    if not csvs:
        log.error(f"Nenhum CSV encontrado em {RAW_DIR}")
        return

    log.info(f"Encontrados {len(csvs)} arquivo(s) CSV")

    for csv in csvs:
        # tenta extrair ano e mês do nome do arquivo (ex: "basica2023-06.csv")
        partes = csv.stem.replace("basica", "").split("-")
        try:
            ano, mes = int(partes[0]), int(partes[1])
        except (IndexError, ValueError):
            ano, mes = 0, 0                  # se não conseguir, usa 0/0 como fallback

        lf = ler_csv_anac(csv)               # cria o plano lazy de leitura
        lf = limpar_e_transformar(lf)        # adiciona transformações ao plano
        salvar_parquet(lf, ano, mes)         # executa e persiste

    log.info("✓ Ingestão concluída! Arquivos em data/processed/")


def carregar_tudo() -> pl.LazyFrame:
    """
    Utilitário: carrega TODOS os Parquets processados como um único LazyFrame.
    Use nos scripts de treinamento dos modelos.
    Exemplo de uso:
        from 01_ingestao import carregar_tudo
        df = carregar_tudo().filter(pl.col("nr_ano_referencia") >= 2019).collect()
    """
    parquets = sorted(PARQUET.glob("*.parquet"))

    if not parquets:
        raise FileNotFoundError("Nenhum Parquet encontrado. Rode processar_todos_csvs() primeiro.")

    return pl.scan_parquet(parquets)         # scan lê todos de uma vez de forma lazy


if __name__ == "__main__":
    processar_todos_csvs()                   # executa quando rodado diretamente
