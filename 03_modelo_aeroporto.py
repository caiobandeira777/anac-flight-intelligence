"""
=============================================================
  ETAPA 3 — MODELO DE LOTAÇÃO DE AEROPORTO
  Arquitetura: LSTM bidirecional + camada de atenção temporal
=============================================================
"""

import gc
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CONFIG = {
    "janela_historica": 48,      # 2 dias de contexto
    "horizonte":        12,      # prever próximas 12h
    "hidden_size":      64,
    "num_layers":       2,
    "dropout":          0.2,
    "batch_size":       32,
    "epochs":           20,
    "lr":               1e-3,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
}

FEATURES_SERIE = [
    "total_decolagens",
    "total_passageiros",
    "total_assentos",
    "ocupacao_media",
    "voos_distintos",
    "dia_semana_num",
    "nr_hora_partida_real",
    "semana_ano",
    "flag_feriado",
]


class AeroportoDataset(Dataset):
    def __init__(self, df: pl.DataFrame, scaler: StandardScaler):
        self.janela    = CONFIG["janela_historica"]
        self.horizonte = CONFIG["horizonte"]
        self.series    = []

        for icao in df["sg_icao_origem"].unique().to_list():
            serie = (
                df.filter(pl.col("sg_icao_origem") == icao)
                  .sort(["dt_partida_real", "nr_hora_partida_real"])
                  .select(FEATURES_SERIE)
                  .to_numpy().astype(np.float32)
            )
            if len(serie) >= self.janela + self.horizonte:
                self.series.append(scaler.transform(serie))

        # guarda só os índices — não pré-computa as janelas em RAM
        self.indices = []
        for s_idx, serie in enumerate(self.series):
            for i in range(len(serie) - self.janela - self.horizonte):
                self.indices.append((s_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s_idx, i = self.indices[idx]
        serie = self.series[s_idx]
        X = torch.tensor(serie[i : i + self.janela], dtype=torch.float32)
        y = torch.tensor(serie[i + self.janela : i + self.janela + self.horizonte, :2], dtype=torch.float32)
        return X, y


class AtencaoTemporal(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.atencao = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores = self.atencao(lstm_out)
        pesos  = torch.softmax(scores, dim=1)
        return (lstm_out * pesos).sum(dim=1)


class LSTMAeroporto(nn.Module):
    def __init__(self, n_features: int, horizonte: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=CONFIG["hidden_size"],
            num_layers=CONFIG["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=CONFIG["dropout"] if CONFIG["num_layers"] > 1 else 0.0,
        )
        self.atencao = AtencaoTemporal(CONFIG["hidden_size"])
        self.head = nn.Sequential(
            nn.Linear(CONFIG["hidden_size"] * 2, 128),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(128, horizonte * 2),
        )
        self.horizonte = horizonte

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        ctx = self.atencao(lstm_out)
        out = self.head(ctx)
        return out.view(-1, self.horizonte, 2)


def treinar():
    log.info("Carregando features de aeroporto...")
    df = pl.read_parquet(FEATURES_DIR / "feat_aeroporto.parquet")

    # remove linha com ano claramente errado
    df = df.filter(pl.col("dt_partida_real").dt.year() <= 2025)

    # usa só os 20 aeroportos com mais voos para economizar RAM
    top_aeroportos = (
        df.group_by("sg_icao_origem")
          .agg(pl.len().alias("n"))
          .sort("n", descending=True)
          .head(20)["sg_icao_origem"]
          .to_list()
    )
    df = df.filter(pl.col("sg_icao_origem").is_in(top_aeroportos))
    log.info(f"Aeroportos selecionados: {top_aeroportos}")

    # split temporal — nunca aleatório em séries temporais!
    df_treino = df.filter(pl.col("dt_partida_real").dt.year() <= 2023)
    df_val    = df.filter(pl.col("dt_partida_real").dt.year() >= 2024)

    gc.collect()  # limpa RAM antes de criar os datasets

    # preenche nulos com 0 antes de normalizar
    df_treino = df_treino.with_columns([
        pl.col(c).fill_null(0).fill_nan(0) for c in FEATURES_SERIE
    ])
    df_val = df_val.with_columns([
        pl.col(c).fill_null(0).fill_nan(0) for c in FEATURES_SERIE
    ])

    # normaliza apenas com dados de treino
    scaler = StandardScaler()
    scaler.fit(df_treino.select(FEATURES_SERIE).to_numpy().astype(np.float32))

    ds_treino = AeroportoDataset(df_treino, scaler)
    ds_val    = AeroportoDataset(df_val,    scaler)

    dl_treino = DataLoader(
        ds_treino,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,    # 0 no Windows — evita erro de multiprocessing
        pin_memory=False,
    )
    dl_val = DataLoader(ds_val, batch_size=CONFIG["batch_size"] * 2, num_workers=0)

    device = CONFIG["device"]
    log.info(f"Treinando em: {device} | {len(ds_treino):,} amostras de treino")

    modelo = LSTMAeroporto(
        n_features=len(FEATURES_SERIE),
        horizonte=CONFIG["horizonte"],
    ).to(device)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=CONFIG["epochs"], eta_min=1e-5)
    criterio   = nn.HuberLoss(delta=1.0)

    melhor_val_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        modelo.train()
        perda_treino = 0.0
        for X, y in dl_treino:
            X, y = X.to(device), y.to(device)
            otimizador.zero_grad()
            loss = criterio(modelo(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perda_treino += loss.item()

        modelo.eval()
        perda_val = 0.0
        with torch.no_grad():
            for X, y in dl_val:
                X, y = X.to(device), y.to(device)
                perda_val += criterio(modelo(X), y).item()

        perda_treino /= len(dl_treino)
        perda_val    /= max(len(dl_val), 1)
        scheduler.step()

        log.info(f"Epoch {epoch:02d}/{CONFIG['epochs']} | Treino: {perda_treino:.4f} | Val: {perda_val:.4f}")

        if perda_val < melhor_val_loss:
            melhor_val_loss = perda_val
            torch.save({
                "model_state": modelo.state_dict(),
                "config":      CONFIG,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale":scaler.scale_.tolist(),
                "features":    FEATURES_SERIE,
            }, MODELS_DIR / "modelo_aeroporto.pt")
            log.info(f"  ★ Melhor modelo salvo (val={perda_val:.4f})")

    log.info("✓ Treinamento concluído!")


if __name__ == "__main__":
    treinar()