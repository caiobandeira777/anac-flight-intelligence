"""
=============================================================
  MONITORAMENTO DE DRIFT
  Compara as predições históricas com os dados reais da ANAC
  e avisa quando o modelo começa a ficar desatualizado.
  
  Rodar: python 07_monitorar_drift.py
  Agendar: executar mensalmente após novos dados da ANAC
=============================================================
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models")
DRIFT_DIR    = Path("data/drift")
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

# limiares de alerta — se o erro ultrapassar, emite aviso
LIMIARES = {
    "mae_ocupacao":    0.12,   # erro médio absoluto de taxa de ocupação > 12% = drift
    "auc_bagagem":     0.80,   # AUC de bagagem abaixo de 80% = drift
    "mae_pressao":     0.15,   # erro de pressão de preço > 15% = drift
}


def avaliar_modelo_assentos() -> dict:
    """
    Compara a taxa_ocupacao real (nos dados) com o que o modelo preveria.
    Usa os dados de 2025 como 'janela de monitoramento' — dados que o modelo
    nunca viu durante o treino (treino foi até 2023).
    """
    log.info("Avaliando modelo de assentos (FT-Transformer)...")

    import torch
    from sklearn.preprocessing import StandardScaler

    df = pl.read_parquet(FEATURES_DIR / "feat_assentos.parquet")

    # janela de monitoramento: dados mais recentes
    df_monitor = df.filter(pl.col("nr_ano_referencia").cast(pl.Int32) >= 2025)

    if len(df_monitor) == 0:
        log.warning("  Sem dados de 2025 para monitorar assentos.")
        return {"status": "sem_dados", "mae": None}

    # amostra para avaliar (máx 50k linhas para ser rápido)
    df_monitor = df_monitor.sample(n=min(50_000, len(df_monitor)), seed=42)

    # carrega o modelo
    ckpt = torch.load(MODELS_DIR / "modelo_assentos.pt", map_location="cpu", weights_only=False)

    from torch import nn

    class FeatureTokenizer(nn.Module):
        def __init__(self, vocab_sizes, n_num, embed_dim):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim, padding_idx=0) for v in vocab_sizes])
            self.num_proj = nn.Linear(1, embed_dim)
            self.n_num = n_num
        def forward(self, x_cat, x_num):
            tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            for i in range(self.n_num): tokens.append(self.num_proj(x_num[:, i].unsqueeze(1)))
            return torch.stack(tokens, dim=1)

    class FTTransformer(nn.Module):
        def __init__(self, vocab_sizes, n_num, cfg):
            super().__init__()
            self.tokenizer = FeatureTokenizer(vocab_sizes, n_num, cfg["embed_dim"])
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg["embed_dim"]))
            enc = nn.TransformerEncoderLayer(d_model=cfg["embed_dim"], nhead=cfg["n_heads"], dim_feedforward=cfg["ffn_dim"], dropout=cfg["dropout"], batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(enc, num_layers=cfg["n_layers"])
            self.cabeca_reg = nn.Sequential(nn.Linear(cfg["embed_dim"], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            self.cabeca_clf = nn.Sequential(nn.Linear(cfg["embed_dim"], 64), nn.ReLU(), nn.Linear(64, 4))
        def forward(self, x_cat, x_num):
            tokens = self.tokenizer(x_cat, x_num)
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            out = self.transformer(torch.cat([cls, tokens], dim=1))
            cls_out = out[:, 0, :]
            return self.cabeca_reg(cls_out), self.cabeca_clf(cls_out)

    modelo = FTTransformer(ckpt["vocab_sizes"], ckpt["n_numericas"], ckpt["config"])
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()

    # encoda features
    from sklearn.preprocessing import LabelEncoder
    cats = ckpt["categoricas"]
    encoders = {k: type('E', (), {'classes_': v, 'transform': lambda self, x: [list(self.classes_).index(i) if i in self.classes_ else 0 for i in x]})() for k, v in ckpt["encoders"].items()}

    cat_arrays = []
    for col in cats:
        vals = df_monitor[col].fill_null("__desconhecido__").to_numpy().astype(str)
        lookup = {v: i for i, v in enumerate(ckpt["encoders"][col])}
        cat_arrays.append(np.array([lookup.get(v, 0) for v in vals], dtype=np.int64))

    x_cat = torch.tensor(np.array(cat_arrays, dtype=np.int64).T, dtype=torch.long)

    sc = StandardScaler()
    sc.mean_  = np.array(ckpt["scaler_mean"])
    sc.scale_ = np.array(ckpt["scaler_scale"])
    num_np = df_monitor.select(ckpt["numericas"]).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
    x_num  = torch.tensor(sc.transform(num_np), dtype=torch.float32)
    y_real = df_monitor["taxa_ocupacao"].fill_null(0.0).to_numpy()

    # prediz em batches
    preds = []
    batch = 1024
    for i in range(0, len(x_cat), batch):
        with torch.no_grad():
            pred, _ = modelo(x_cat[i:i+batch], x_num[i:i+batch])
            preds.append(pred.numpy().flatten())
    y_pred = np.concatenate(preds)

    mae    = float(np.mean(np.abs(y_real - y_pred)))
    status = "🔴 DRIFT DETECTADO" if mae > LIMIARES["mae_ocupacao"] else "🟢 OK"

    log.info(f"  MAE ocupação (2025): {mae:.4f} | limiar: {LIMIARES['mae_ocupacao']} | {status}")
    return {"status": status, "mae": round(mae, 4), "n_amostras": len(df_monitor), "limiar": LIMIARES["mae_ocupacao"]}


def avaliar_modelo_bagagem() -> dict:
    """Avalia o LightGBM de bagagem nos dados mais recentes."""
    log.info("Avaliando modelo de bagagem (LightGBM)...")

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    df = pl.read_parquet(FEATURES_DIR / "feat_bagagem.parquet")
    df_monitor = df.filter(pl.col("nr_mes_referencia").cast(pl.Int32).is_in([11, 12]))  # últimos meses

    CAT_BAGAGEM = ["rota_od","sg_empresa_icao","nm_regiao_origem","nm_regiao_destino",
                   "nm_continente_destino","faixa_distancia","ds_tipo_linha","nm_dia_semana_referencia"]

    for col in CAT_BAGAGEM:
        df_monitor = df_monitor.with_columns(pl.col(col).fill_null("desconhecido"))
    df_monitor = df_monitor.with_columns([
        pl.col("taxa_excesso_historica_rota").fill_null(0.0).fill_nan(0.0),
        pl.col("kg_excesso_medio_rota").fill_null(0.0).fill_nan(0.0),
        pl.col("taxa_excesso_historica_empresa").fill_null(0.0).fill_nan(0.0),
        pl.col("nr_passag_pagos").fill_null(0.0),
        pl.col("km_distancia").fill_null(0.0),
        pl.col("flag_bagagem_excesso").fill_null(0),
    ])

    FEATURES = ["rota_od","sg_empresa_icao","nm_regiao_origem","nm_regiao_destino",
                "nm_continente_destino","flag_internacional","nr_mes_referencia",
                "nm_dia_semana_referencia","nr_hora_partida_real","km_distancia",
                "faixa_distancia","ds_tipo_linha","nr_passag_pagos",
                "taxa_excesso_historica_rota","kg_excesso_medio_rota","taxa_excesso_historica_empresa"]

    X = df_monitor.select(FEATURES).to_pandas()
    for col in CAT_BAGAGEM:
        X[col] = X[col].astype("category")
    X["nr_mes_referencia"] = X["nr_mes_referencia"].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
    y = df_monitor["flag_bagagem_excesso"].to_numpy().astype(int)

    if len(y) < 100 or y.sum() < 10:
        log.warning("  Dados insuficientes para avaliar bagagem.")
        return {"status": "sem_dados", "auc": None}

    modelo = lgb.Booster(model_file=str(MODELS_DIR / "lgb_bagagem.txt"))
    prob   = modelo.predict(X)
    auc    = roc_auc_score(y, prob)
    status = "🔴 DRIFT DETECTADO" if auc < LIMIARES["auc_bagagem"] else "🟢 OK"

    log.info(f"  AUC bagagem (monitor): {auc:.4f} | limiar: {LIMIARES['auc_bagagem']} | {status}")
    return {"status": status, "auc": round(auc, 4), "n_amostras": len(df_monitor), "limiar": LIMIARES["auc_bagagem"]}


def gerar_relatorio(resultados: dict) -> None:
    """Salva um relatório JSON e imprime um resumo no terminal."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida = DRIFT_DIR / f"drift_report_{ts}.json"

    # determina status geral
    algum_drift = any(
        "DRIFT" in str(r.get("status", ""))
        for r in resultados.values()
    )

    relatorio = {
        "timestamp":     datetime.now().isoformat(),
        "status_geral":  "🔴 RETREINAMENTO RECOMENDADO" if algum_drift else "🟢 MODELOS SAUDÁVEIS",
        "modelos":       resultados,
        "acao":          "Retreine os modelos com dados mais recentes." if algum_drift else "Nenhuma ação necessária.",
    }

    with open(saida, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    # imprime resumo
    print("\n" + "═" * 55)
    print(f"  RELATÓRIO DE DRIFT — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("═" * 55)
    print(f"  STATUS GERAL: {relatorio['status_geral']}")
    print()
    for nome, res in resultados.items():
        print(f"  {nome.upper()}: {res.get('status', 'N/A')}")
        for k, v in res.items():
            if k not in ("status",) and v is not None:
                print(f"    {k}: {v}")
    print()
    print(f"  Relatório salvo em: {saida}")
    print(f"  AÇÃO: {relatorio['acao']}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    resultados = {}
    resultados["assentos"] = avaliar_modelo_assentos()
    resultados["bagagem"]  = avaliar_modelo_bagagem()
    gerar_relatorio(resultados)
