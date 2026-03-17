"""
=============================================================
  ETAPA 2 — ENGENHARIA DE FEATURES
  Transforma os dados limpos em tabelas de features prontas
  para cada um dos 4 modelos preditivos.
  Roda UMA VEZ e salva os resultados — modelos apenas leem.
=============================================================
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

PROCESSED = Path("data/processed")          # parquets da etapa 1
FEATURES  = Path("data/features")           # saída desta etapa
FEATURES.mkdir(parents=True, exist_ok=True)

# ── feriados nacionais do Brasil (enriquecimento externo) ───────────────────
FERIADOS_BR = {                             # conjunto de datas — busca em O(1)
    "2019-01-01","2019-04-19","2019-04-21","2019-05-01","2019-06-20",
    "2019-09-07","2019-10-12","2019-11-02","2019-11-15","2019-12-25",
    "2020-01-01","2020-02-24","2020-02-25","2020-04-10","2020-04-21",
    "2020-05-01","2020-06-11","2020-09-07","2020-10-12","2020-11-02",
    "2020-11-15","2020-12-25",
    "2021-01-01","2021-02-15","2021-02-16","2021-04-02","2021-04-21",
    "2021-05-01","2021-06-03","2021-09-07","2021-10-12","2021-11-02",
    "2021-11-15","2021-12-25",
    "2022-01-01","2022-02-28","2022-03-01","2022-04-15","2022-04-21",
    "2022-05-01","2022-06-16","2022-09-07","2022-10-12","2022-11-02",
    "2022-11-15","2022-12-25",
    "2023-01-01","2023-02-20","2023-02-21","2023-04-07","2023-04-21",
    "2023-05-01","2023-06-08","2023-09-07","2023-10-12","2023-11-02",
    "2023-11-15","2023-12-25",
}


def carregar_dados_base() -> pl.LazyFrame:
    """Carrega todos os Parquets processados como LazyFrame único."""
    parquets = sorted(PROCESSED.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError("Rode 01_ingestao.py antes desta etapa.")
    log.info(f"Carregando {len(parquets)} parquet(s)...")
    return pl.scan_parquet(parquets)         # leitura lazy — sem uso de RAM ainda


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SET 1 — LOTAÇÃO DE AEROPORTO (modelo LSTM)
#  Granularidade: aeroporto × data × hora → total de movimentos
# ══════════════════════════════════════════════════════════════════════════════

def features_aeroporto() -> None:
    """
    Agrega voos por aeroporto + data + hora.
    Cria lags temporais (7d, 14d, 30d) para alimentar o LSTM.
    """
    log.info("Construindo features de lotação de aeroporto...")

    lf = carregar_dados_base()

    # agrega movimentos por aeroporto, data e hora de partida
    agg = (
        lf
        .group_by(["sg_icao_origem", "dt_partida_real", "nr_hora_partida_real"])
        .agg([
            pl.col("nr_decolagem").sum().alias("total_decolagens"),         # voos que partiram
            pl.col("nr_passag_pagos").sum().alias("total_passageiros"),     # pax total
            pl.col("nr_assentos_ofertados").sum().alias("total_assentos"),  # capacidade total
            pl.col("nr_voo").n_unique().alias("voos_distintos"),            # companhias diferentes
            pl.col("taxa_ocupacao").mean().alias("ocupacao_media"),         # % médio de lotação
        ])
        .sort(["sg_icao_origem", "dt_partida_real", "nr_hora_partida_real"])
        .collect()                           # coleta em RAM — dataset agregado é pequeno
    )

    # adiciona flag de feriado nacional
    agg = agg.with_columns([
        pl.col("dt_partida_real")
          .cast(pl.Utf8)                     # converte data para string para comparar
          .is_in(list(FERIADOS_BR))
          .cast(pl.Int8)
          .alias("flag_feriado"),

        # dia da semana como número (0=segunda … 6=domingo)
        pl.col("dt_partida_real")
          .dt.weekday()
          .alias("dia_semana_num"),

        # semana do ano (1–52) — captura sazonalidade anual
        pl.col("dt_partida_real")
          .dt.week()
          .alias("semana_ano"),
    ])

    # ── lags temporais por aeroporto ──────────────────────────────────────
    # O LSTM precisa de histórico → calculamos médias de 7, 14 e 30 dias atrás
    # Fazemos no Pandas por janelas — mais simples que fazer rolling no Polars
    import pandas as pd                      # só aqui para a janela móvel

    df_pd = agg.to_pandas()                  # conversão pontual para janelas
    df_pd = df_pd.sort_values(["sg_icao_origem", "dt_partida_real", "nr_hora_partida_real"])

    for aeroporto, grupo in df_pd.groupby("sg_icao_origem"):
        for janela, nome in [(7, "lag7d"), (14, "lag14d"), (30, "lag30d")]:
            # média de decolagens nos N dias anteriores para aquele aeroporto
            df_pd.loc[grupo.index, f"decol_{nome}"] = (
                grupo["total_decolagens"]
                  .rolling(window=janela * 24,  # janela em horas (max frequência = hora)
                           min_periods=1)
                  .mean()
                  .shift(janela * 24)          # garante que não vaza o futuro
                  .values
            )

    agg = pl.from_pandas(df_pd)              # volta para Polars

    saida = FEATURES / "feat_aeroporto.parquet"
    agg.write_parquet(saida, compression="zstd")
    log.info(f"  ✓ {agg.shape[0]:,} linhas salvas em {saida.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SET 2 — ASSENTOS DISPONÍVEIS (modelo Transformer tabular)
#  Granularidade: um voo individual → quantos assentos vão sobrar
# ══════════════════════════════════════════════════════════════════════════════

def features_assentos() -> None:
    """
    Cada linha = um voo.
    Target: taxa_ocupacao (regressão) ou bucket de lotação (classificação).
    """
    log.info("Construindo features de assentos...")

    df = (
        carregar_dados_base()
        .select([
            # identificação
            "sg_icao_origem", "sg_icao_destino", "rota_od",
            "sg_empresa_icao", "nm_empresa",

            # temporais
            "nr_ano_referencia", "nr_mes_referencia",
            "nr_semana_referencia", "nm_dia_semana_referencia",
            "nr_hora_partida_real",

            # operacionais
            "nr_assentos_ofertados", "km_distancia",
            "ds_tipo_linha", "ds_natureza_tipo_linha",
            "flag_internacional",

            # targets
            "taxa_ocupacao", "assentos_vazios",
            "nr_passag_pagos",
        ])
        .collect()
    )

    # ── cria categoria ordinal de lotação (para classificação) ─────────────
    df = df.with_columns([
        pl.when(pl.col("taxa_ocupacao") < 0.50).then(pl.lit(0))  # avião vazio
          .when(pl.col("taxa_ocupacao") < 0.75).then(pl.lit(1))  # parcialmente cheio
          .when(pl.col("taxa_ocupacao") < 0.90).then(pl.lit(2))  # quase cheio
          .otherwise(pl.lit(3))                                   # lotado
          .alias("bucket_ocupacao"),

        # distância em bins (curta <500km, média <1500km, longa >1500km)
        pl.when(pl.col("km_distancia") < 500).then(pl.lit("curta"))
          .when(pl.col("km_distancia") < 1500).then(pl.lit("media"))
          .otherwise(pl.lit("longa"))
          .alias("faixa_distancia"),
    ])

    # ── histórico médio de ocupação por rota (media histórica como feature) ─
    media_rota = (
        df.group_by("rota_od")
          .agg(pl.col("taxa_ocupacao").mean().alias("ocupacao_media_historica_rota"))
    )                                        # contexto histórico de demanda da rota

    media_empresa_mes = (
        df.group_by(["sg_empresa_icao", "nr_mes_referencia"])
          .agg(pl.col("taxa_ocupacao").mean().alias("ocupacao_media_empresa_mes"))
    )                                        # sazonalidade por empresa

    # join das médias como features adicionais
    df = (
        df
        .join(media_rota, on="rota_od", how="left")
        .join(media_empresa_mes, on=["sg_empresa_icao", "nr_mes_referencia"], how="left")
    )

    saida = FEATURES / "feat_assentos.parquet"
    df.write_parquet(saida, compression="zstd")
    log.info(f"  ✓ {df.shape[0]:,} linhas salvas em {saida.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SET 3 — PRECIFICAÇÃO DINÂMICA (modelo MLP + Embeddings)
#  Granularidade: rota × mês × dia_semana × empresa → índice de pressão
# ══════════════════════════════════════════════════════════════════════════════

def features_precificacao() -> None:
    """
    Agrega dados por rota-empresa-período.
    O índice de pressão de preço é derivado de demanda vs capacidade.
    """
    log.info("Construindo features de precificação...")

    df = (
        carregar_dados_base()
        .group_by([
            "rota_od", "sg_empresa_icao",
            "nr_mes_referencia", "nm_dia_semana_referencia",
            "nr_hora_partida_real",
        ])
        .agg([
            pl.col("taxa_ocupacao").mean().alias("ocupacao_media"),
            pl.col("nr_passag_pagos").sum().alias("total_pax"),
            pl.col("nr_assentos_ofertados").sum().alias("total_assentos"),
            pl.col("lt_combustivel").mean().alias("combustivel_medio"),        # custo operacional
            pl.col("km_distancia").first().alias("km_distancia"),              # distância é fixa por rota
            pl.col("load_factor_km").mean().alias("load_factor_medio"),        # eficiência por km
            pl.col("nr_decolagem").sum().alias("frequencia_voos"),             # quantos voos nesse slot
        ])
        .collect()
    )

    # ── índice de pressão de preço (target do modelo) ─────────────────────
    # Fórmula heurística: quanto maior a ocupação e menor a capacidade,
    # maior a "pressão" para cobrar mais caro
    df = df.with_columns([
        # pressao_preco: normalizado 0–1 (1 = cobrar mais, 0 = dar desconto)
        (
            pl.col("ocupacao_media") * 0.6 +              # peso maior para ocupação
            (pl.col("total_pax") / pl.col("total_assentos").cast(pl.Float32)) * 0.3 +
            (pl.col("combustivel_medio") / 10000.0).clip(0.0, 0.1) * 0.1   # custo normalizado
        )
        .clip(0.0, 1.0)
        .alias("pressao_preco"),

        # receita por assento disponível (proxy de preço médio praticado)
        (pl.col("total_pax") / pl.col("total_assentos").cast(pl.Float32))
          .alias("receita_por_assento"),
    ])

    saida = FEATURES / "feat_precificacao.parquet"
    df.write_parquet(saida, compression="zstd")
    log.info(f"  ✓ {df.shape[0]:,} linhas salvas em {saida.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SET 4 — COBRANÇA DE BAGAGEM (modelo GBT + DNN)
#  Granularidade: um voo → cobrar despacho de mala?
# ══════════════════════════════════════════════════════════════════════════════

def features_bagagem() -> None:
    """
    Features focadas em bagagem por voo.
    Target binário: flag_bagagem_excesso (1 = passageiro vai despachar mala extra).
    """
    log.info("Construindo features de bagagem...")

    df = (
        carregar_dados_base()
        .select([
            "rota_od", "sg_empresa_icao",
            "sg_icao_origem", "sg_icao_destino",
            "nm_regiao_origem", "nm_regiao_destino",
            "nm_continente_destino",
            "flag_internacional",
            "nr_mes_referencia", "nm_dia_semana_referencia",
            "nr_hora_partida_real",
            "km_distancia",              # ← sem faixa_distancia aqui
            "ds_tipo_linha",
            "nr_passag_pagos",
            "flag_bagagem_excesso",
            "kg_bagagem_excesso",
            "kg_bagagem_livre",
        ])
        .collect()
    )
        # cria faixa_distancia aqui, depois do collect
    df = df.with_columns([
        pl.when(pl.col("km_distancia") < 500).then(pl.lit("curta"))
          .when(pl.col("km_distancia") < 1500).then(pl.lit("media"))
          .otherwise(pl.lit("longa"))
          .alias("faixa_distancia"),
    ])

    # ── média histórica de excesso por rota (memória de comportamento passado) ─
    media_excesso_rota = (
        df.group_by("rota_od")
          .agg([
              pl.col("flag_bagagem_excesso").mean().alias("taxa_excesso_historica_rota"),
              pl.col("kg_bagagem_excesso").mean().alias("kg_excesso_medio_rota"),
          ])
    )

    media_excesso_empresa = (
        df.group_by("sg_empresa_icao")
          .agg(
              pl.col("flag_bagagem_excesso").mean().alias("taxa_excesso_historica_empresa")
          )
    )

    df = (
        df
        .join(media_excesso_rota, on="rota_od", how="left")
        .join(media_excesso_empresa, on="sg_empresa_icao", how="left")
    )

    saida = FEATURES / "feat_bagagem.parquet"
    df.write_parquet(saida, compression="zstd")
    log.info(f"  ✓ {df.shape[0]:,} linhas salvas em {saida.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    features_aeroporto()
    features_assentos()
    features_precificacao()
    features_bagagem()
    log.info("✓ Todas as feature tables geradas em data/features/")
