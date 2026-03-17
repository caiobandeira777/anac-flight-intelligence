"""
=============================================================
  ETAPA 5 — MODELOS DE PRECIFICAÇÃO E BAGAGEM
  5a) Precificação dinâmica: MLP com Embeddings categóricos
      Target: pressao_preco (0.0–1.0)
  5b) Cobrança de bagagem: LightGBM + camada DNN (stacking)
      Target: flag_bagagem_excesso (0 ou 1)
=============================================================
"""

import torch
import torch.nn as nn
import numpy as np
import polars as pl
import lightgbm as lgb
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  5a — MODELO DE PRECIFICAÇÃO DINÂMICA
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PRECO = {
    "embed_dim":  32,
    "hidden":    [256, 128, 64],
    "dropout":    0.2,
    "batch_size": 1024,
    "epochs":     40,
    "lr":         5e-4,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}

CATEGORICAS_PRECO = [
    "rota_od",
    "sg_empresa_icao",
    "nm_dia_semana_referencia",
]

NUMERICAS_PRECO = [
    "nr_mes_referencia",
    "nr_hora_partida_real",
    "km_distancia",
    "ocupacao_media",
    "combustivel_medio",
    "load_factor_medio",
    "frequencia_voos",
]


class PrecoDataset(Dataset):
    def __init__(self, df, encoders, scaler, fit=False):
        cat_arrays = []
        for col in CATEGORICAS_PRECO:
            vals = df[col].fill_null("__desconhecido__").to_numpy().astype(str)
            if fit:
                encoders[col] = LabelEncoder()
                encoders[col].fit(np.append(vals, "__desconhecido__"))
            lookup = {v: i for i, v in enumerate(encoders[col].classes_)}
            encoded = np.array([lookup.get(v, 0) for v in vals], dtype=np.int64)
            cat_arrays.append(encoded)

        self.X_cat = torch.tensor(np.array(cat_arrays, dtype=np.int64).T, dtype=torch.long)
        num_np = df.select(NUMERICAS_PRECO).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
        if fit:
            scaler.fit(num_np)
        self.X_num = torch.tensor(scaler.transform(num_np), dtype=torch.float32)
        self.y = torch.tensor(
            df["pressao_preco"].fill_null(0.5).fill_nan(0.5).to_numpy().astype(np.float32),
            dtype=torch.float32,
        ).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class MLPPreco(nn.Module):
    def __init__(self, vocab_sizes, n_num):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, CONFIG_PRECO["embed_dim"], padding_idx=0)
            for vocab in vocab_sizes
        ])
        dim_entrada = len(CATEGORICAS_PRECO) * CONFIG_PRECO["embed_dim"] + n_num
        camadas = []
        dim_atual = dim_entrada
        for dim_saida in CONFIG_PRECO["hidden"]:
            camadas.extend([
                nn.Linear(dim_atual, dim_saida),
                nn.LayerNorm(dim_saida),
                nn.GELU(),
                nn.Dropout(CONFIG_PRECO["dropout"]),
            ])
            dim_atual = dim_saida
        camadas.append(nn.Linear(dim_atual, 1))
        camadas.append(nn.Sigmoid())
        self.rede = nn.Sequential(*camadas)

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embeds + [x_num], dim=1)
        return self.rede(x)


def treinar_precificacao():
    log.info("Treinando modelo de precificação dinâmica...")
    df = pl.read_parquet(FEATURES_DIR / "feat_precificacao.parquet")

    df_treino = df.filter(pl.col("nr_mes_referencia").cast(pl.Int32).is_in(list(range(1, 11))))
    df_val    = df.filter(pl.col("nr_mes_referencia").cast(pl.Int32).is_in([11, 12]))
    log.info(f"Treino: {len(df_treino):,} | Validação: {len(df_val):,}")

    encoders = {}
    scaler   = StandardScaler()
    ds_treino = PrecoDataset(df_treino, encoders, scaler, fit=True)
    ds_val    = PrecoDataset(df_val,    encoders, scaler)

    dl_treino = DataLoader(ds_treino, batch_size=CONFIG_PRECO["batch_size"], shuffle=True,  num_workers=0)
    dl_val    = DataLoader(ds_val,    batch_size=CONFIG_PRECO["batch_size"] * 2, num_workers=0)

    device = CONFIG_PRECO["device"]
    log.info(f"Treinando em: {device}")

    vocab_sizes = [len(encoders[c].classes_) + 1 for c in CATEGORICAS_PRECO]
    modelo  = MLPPreco(vocab_sizes, len(NUMERICAS_PRECO)).to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=CONFIG_PRECO["lr"])
    criterio   = nn.MSELoss()
    melhor     = float("inf")

    for epoch in range(1, CONFIG_PRECO["epochs"] + 1):
        modelo.train()
        loss_treino = 0.0
        for x, n, y in dl_treino:
            x, n, y = x.to(device), n.to(device), y.to(device)
            otimizador.zero_grad()
            loss = criterio(modelo(x, n), y)
            loss.backward()
            otimizador.step()
            loss_treino += loss.item()
        loss_treino /= len(dl_treino)

        modelo.eval()
        loss_val = 0.0
        with torch.no_grad():
            for x, n, y in dl_val:
                loss_val += criterio(modelo(x.to(device), n.to(device)), y.to(device)).item()
        loss_val /= max(len(dl_val), 1)

        if epoch % 5 == 0:
            log.info(f"Epoch {epoch:02d}/{CONFIG_PRECO['epochs']} | Treino: {loss_treino:.4f} | Val: {loss_val:.4f}")

        if loss_val < melhor:
            melhor = loss_val
            torch.save({
                "model_state":  modelo.state_dict(),
                "config":       CONFIG_PRECO,
                "vocab_sizes":  vocab_sizes,
                "encoders":     {k: v.classes_.tolist() for k, v in encoders.items()},
                "scaler_mean":  scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }, MODELS_DIR / "modelo_precificacao.pt")

    log.info("✓ Modelo de precificação salvo!")


# ══════════════════════════════════════════════════════════════════════════════
#  5b — MODELO DE BAGAGEM (stacking: LightGBM + DNN)
# ══════════════════════════════════════════════════════════════════════════════

FEATURES_BAGAGEM = [
    "rota_od", "sg_empresa_icao",
    "nm_regiao_origem", "nm_regiao_destino",
    "nm_continente_destino", "flag_internacional",
    "nr_mes_referencia", "nm_dia_semana_referencia",
    "nr_hora_partida_real", "km_distancia",
    "faixa_distancia", "ds_tipo_linha",
    "nr_passag_pagos",
    "taxa_excesso_historica_rota",
    "kg_excesso_medio_rota",
    "taxa_excesso_historica_empresa",
]

CAT_BAGAGEM = [
    "rota_od", "sg_empresa_icao",
    "nm_regiao_origem", "nm_regiao_destino",
    "nm_continente_destino", "faixa_distancia",
    "ds_tipo_linha", "nm_dia_semana_referencia",
]


def treinar_bagagem():
    log.info("Treinando modelo de bagagem...")
    df = pl.read_parquet(FEATURES_DIR / "feat_bagagem.parquet")

    for col in CAT_BAGAGEM:
        df = df.with_columns(pl.col(col).fill_null("desconhecido"))

    df = df.with_columns([
        pl.col("taxa_excesso_historica_rota").fill_null(0.0).fill_nan(0.0),
        pl.col("kg_excesso_medio_rota").fill_null(0.0).fill_nan(0.0),
        pl.col("taxa_excesso_historica_empresa").fill_null(0.0).fill_nan(0.0),
        pl.col("nr_passag_pagos").fill_null(0.0),
        pl.col("km_distancia").fill_null(0.0),
        pl.col("flag_bagagem_excesso").fill_null(0),
    ])

    log.info("  Treinando LightGBM (etapa 1 do stacking)...")
    X = df.select(FEATURES_BAGAGEM).to_pandas()
    for col in CAT_BAGAGEM:
        X[col] = X[col].astype("category")
    X["nr_mes_referencia"] = X["nr_mes_referencia"].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
    y = df["flag_bagagem_excesso"].to_numpy().astype(int)

    n_treino = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:n_treino], X.iloc[n_treino:]
    y_tr, y_val = y[:n_treino],      y[n_treino:]

    lgb_treino = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_BAGAGEM)
    lgb_val    = lgb.Dataset(X_val, y_val, reference=lgb_treino)

    modelo_lgb = lgb.train(
        {
            "objective": "binary", "metric": "auc",
            "n_estimators": 500, "learning_rate": 0.05,
            "num_leaves": 63, "min_child_samples": 50,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "verbose": -1, "n_jobs": -1,
        },
        lgb_treino,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    prob_lgb_tr  = modelo_lgb.predict(X_tr)
    prob_lgb_val = modelo_lgb.predict(X_val)
    log.info(f"  LightGBM AUC: {roc_auc_score(y_val, prob_lgb_val):.4f}")
    modelo_lgb.save_model(str(MODELS_DIR / "lgb_bagagem.txt"))

    log.info("  Treinando DNN de refinamento (etapa 2 do stacking)...")
    num_cols_dnn = [
        "flag_internacional", "nr_mes_referencia",
        "nr_hora_partida_real", "km_distancia",
        "nr_passag_pagos",
        "taxa_excesso_historica_rota",
        "kg_excesso_medio_rota",
        "taxa_excesso_historica_empresa",
    ]

    scaler_dnn = StandardScaler()
    X_tr_num = np.column_stack([
        df.select(num_cols_dnn).to_numpy()[:n_treino].astype(np.float32),
        prob_lgb_tr.reshape(-1, 1).astype(np.float32),
    ])
    X_val_num = np.column_stack([
        df.select(num_cols_dnn).to_numpy()[n_treino:].astype(np.float32),
        prob_lgb_val.reshape(-1, 1).astype(np.float32),
    ])
    scaler_dnn.fit(X_tr_num)
    X_tr_norm  = scaler_dnn.transform(X_tr_num)
    X_val_norm = scaler_dnn.transform(X_val_num)

    # DNN roda na CPU em mini-batches — evita estourar RAM e VRAM
    dnn = nn.Sequential(
        nn.Linear(X_tr_norm.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 1), nn.Sigmoid(),
    )

    opt      = torch.optim.Adam(dnn.parameters(), lr=1e-3)
    criterio = nn.BCELoss()
    batch_size_dnn = 4096  # processa 4096 linhas por vez — cabe na RAM

    for epoch in range(50):
        dnn.train()
        idx = np.random.permutation(len(X_tr_norm))  # embaralha a cada época
        for i in range(0, len(idx), batch_size_dnn):
            batch_idx = idx[i:i + batch_size_dnn]
            xb = torch.tensor(X_tr_norm[batch_idx], dtype=torch.float32)
            yb = torch.tensor(y_tr[batch_idx],      dtype=torch.float32).unsqueeze(1)
            opt.zero_grad()
            criterio(dnn(xb), yb).backward()
            opt.step()

        if epoch % 10 == 0:
            dnn.eval()
            with torch.no_grad():
                # valida em batches para não estourar RAM
                probs = []
                for i in range(0, len(X_val_norm), batch_size_dnn):
                    xb = torch.tensor(X_val_norm[i:i + batch_size_dnn], dtype=torch.float32)
                    probs.append(dnn(xb).numpy().flatten())
                auc_dnn = roc_auc_score(y_val, np.concatenate(probs))
                log.info(f"  DNN epoch {epoch:02d} | val AUC: {auc_dnn:.4f}")

    torch.save({
        "model_state":  dnn.state_dict(),
        "input_dim":    X_tr_norm.shape[1],
        "scaler_mean":  scaler_dnn.mean_.tolist(),
        "scaler_scale": scaler_dnn.scale_.tolist(),
        "num_cols":     num_cols_dnn,
    }, MODELS_DIR / "dnn_bagagem.pt")

    log.info("✓ Modelos de bagagem (LightGBM + DNN) salvos!")


if __name__ == "__main__":
    treinar_precificacao()
    treinar_bagagem()