"""
=============================================================
  ETAPA 4 — MODELO DE ASSENTOS DISPONÍVEIS
  Arquitetura: FT-Transformer (Feature Tokenizer + Transformer)
  Target: taxa_ocupacao (0.0–1.0) e bucket_ocupacao (0–3)
=============================================================
"""
 
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import logging
 
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
 
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
 
CONFIG = {
    "embed_dim":    64,
    "n_heads":      8,
    "n_layers":     3,
    "ffn_dim":      256,
    "dropout":      0.1,
    "batch_size":   512,
    "epochs":       30,
    "lr":           3e-4,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}
 
CATEGORICAS = [
    "sg_icao_origem",
    "sg_icao_destino",
    "rota_od",
    "sg_empresa_icao",
    "nm_dia_semana_referencia",
    "ds_tipo_linha",
    "faixa_distancia",
]
 
NUMERICAS = [
    "nr_mes_referencia",
    "nr_semana_referencia",
    "nr_hora_partida_real",
    "nr_assentos_ofertados",
    "km_distancia",
    "flag_internacional",
    "ocupacao_media_historica_rota",
    "ocupacao_media_empresa_mes",
]
 
TARGET_REG = "taxa_ocupacao"
TARGET_CLF = "bucket_ocupacao"
 
 
class VooDataset(Dataset):
    def __init__(self, df: pl.DataFrame, encoders: dict, scaler: StandardScaler, fit: bool = False):
 
        # encoding vetorizado — processa todo o array de uma vez, sem loop por linha
        cat_arrays = []
        for col in CATEGORICAS:
            vals = df[col].fill_null("__desconhecido__").to_numpy().astype(str)
            if fit:
                encoders[col] = LabelEncoder()
                encoders[col].fit(np.append(vals, "__desconhecido__"))
            # monta lookup dict para transformação rápida
            lookup = {v: i for i, v in enumerate(encoders[col].classes_)}
            encoded = np.array([lookup.get(v, 0) for v in vals], dtype=np.int64)
            cat_arrays.append(encoded)
 
        self.X_cat = torch.tensor(
            np.array(cat_arrays, dtype=np.int64).T,
            dtype=torch.long,
        )
 
        # normaliza numéricas — preenche nulos com 0 antes
        num_np = df.select(NUMERICAS).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
        if fit:
            scaler.fit(num_np)
        self.X_num = torch.tensor(scaler.transform(num_np), dtype=torch.float32)
 
        # targets
        self.y_reg = torch.tensor(
            df[TARGET_REG].fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32),
            dtype=torch.float32,
        ).unsqueeze(1)
 
        self.y_clf = torch.tensor(
            df[TARGET_CLF].fill_null(0).to_numpy().astype(np.int64),
            dtype=torch.long,
        )
 
    def __len__(self):
        return len(self.y_reg)
 
    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y_reg[idx], self.y_clf[idx]
 
 
class FeatureTokenizer(nn.Module):
    def __init__(self, vocab_sizes: list, n_numericas: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, embed_dim, padding_idx=0)
            for vocab in vocab_sizes
        ])
        self.num_proj = nn.Linear(1, embed_dim)
        self.n_numericas = n_numericas
 
    def forward(self, x_cat, x_num):
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        for i in range(self.n_numericas):
            tokens.append(self.num_proj(x_num[:, i].unsqueeze(1)))
        return torch.stack(tokens, dim=1)
 
 
class FTTransformer(nn.Module):
    def __init__(self, vocab_sizes: list, n_numericas: int):
        super().__init__()
        self.tokenizer = FeatureTokenizer(vocab_sizes, n_numericas, CONFIG["embed_dim"])
        self.cls_token = nn.Parameter(torch.randn(1, 1, CONFIG["embed_dim"]))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG["embed_dim"],
            nhead=CONFIG["n_heads"],
            dim_feedforward=CONFIG["ffn_dim"],
            dropout=CONFIG["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG["n_layers"])
        self.cabeca_reg = nn.Sequential(
            nn.Linear(CONFIG["embed_dim"], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.cabeca_clf = nn.Sequential(
            nn.Linear(CONFIG["embed_dim"], 64), nn.ReLU(), nn.Linear(64, 4)
        )
 
    def forward(self, x_cat, x_num):
        tokens = self.tokenizer(x_cat, x_num)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.transformer(seq)
        cls_out = out[:, 0, :]
        return self.cabeca_reg(cls_out), self.cabeca_clf(cls_out)
 
 
def treinar():
    log.info("Carregando features de assentos...")
    df = pl.read_parquet(FEATURES_DIR / "feat_assentos.parquet")
 
    # remove anos inválidos
    df = df.filter(pl.col("nr_ano_referencia").cast(pl.Int32) <= 2025)
 
    # usa 20% dos dados para caber na RAM sem travar
    df = df.sample(fraction=0.2, seed=42)
    log.info(f"Usando {len(df):,} linhas (20% do total)")
 
    # split temporal
    df_treino = df.filter(pl.col("nr_ano_referencia").cast(pl.Int32) <= 2023)
    df_val    = df.filter(pl.col("nr_ano_referencia").cast(pl.Int32) >= 2024)
    log.info(f"Treino: {len(df_treino):,} | Validação: {len(df_val):,}")
 
    encoders = {}
    scaler   = StandardScaler()
 
    log.info("Codificando features (pode demorar alguns minutos)...")
    ds_treino = VooDataset(df_treino, encoders, scaler, fit=True)
    ds_val    = VooDataset(df_val,    encoders, scaler, fit=False)
    log.info("Features prontas, iniciando treino...")
 
    dl_treino = DataLoader(
        ds_treino,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,       # 0 no Windows — evita crash de multiprocessing
        pin_memory=False,
    )
    dl_val = DataLoader(ds_val, batch_size=CONFIG["batch_size"] * 2, num_workers=0)
 
    device = CONFIG["device"]
    log.info(f"Treinando em: {device}")
 
    vocab_sizes = [len(encoders[c].classes_) + 1 for c in CATEGORICAS]
    modelo = FTTransformer(vocab_sizes=vocab_sizes, n_numericas=len(NUMERICAS)).to(device)
 
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        otimizador,
        max_lr=CONFIG["lr"],
        epochs=CONFIG["epochs"],
        steps_per_epoch=len(dl_treino),
    )
    criterio_reg = nn.MSELoss()
    criterio_clf = nn.CrossEntropyLoss()
    melhor_val_loss = float("inf")
 
    for epoch in range(1, CONFIG["epochs"] + 1):
        modelo.train()
        total_loss = 0.0
        for x_cat, x_num, y_reg, y_clf in dl_treino:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            y_reg, y_clf = y_reg.to(device), y_clf.to(device)
            otimizador.zero_grad()
            pred_reg, pred_clf = modelo(x_cat, x_num)
            loss = 0.7 * criterio_reg(pred_reg, y_reg) + 0.3 * criterio_clf(pred_clf, y_clf)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            scheduler.step()
            total_loss += loss.item()
 
        modelo.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, y_reg, y_clf in dl_val:
                x_cat, x_num = x_cat.to(device), x_num.to(device)
                y_reg, y_clf = y_reg.to(device), y_clf.to(device)
                pred_reg, pred_clf = modelo(x_cat, x_num)
                val_loss += (
                    0.7 * criterio_reg(pred_reg, y_reg) +
                    0.3 * criterio_clf(pred_clf, y_clf)
                ).item()
 
        total_loss /= len(dl_treino)
        val_loss   /= max(len(dl_val), 1)
        log.info(f"Epoch {epoch:02d}/{CONFIG['epochs']} | Treino: {total_loss:.4f} | Val: {val_loss:.4f}")
 
        if val_loss < melhor_val_loss:
            melhor_val_loss = val_loss
            torch.save({
                "model_state":  modelo.state_dict(),
                "config":       CONFIG,
                "vocab_sizes":  vocab_sizes,
                "n_numericas":  len(NUMERICAS),
                "encoders":     {k: v.classes_.tolist() for k, v in encoders.items()},
                "scaler_mean":  scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "categoricas":  CATEGORICAS,
                "numericas":    NUMERICAS,
            }, MODELS_DIR / "modelo_assentos.pt")
            log.info(f"  ★ Melhor modelo salvo (val={val_loss:.4f})")
 
    log.info("✓ Treinamento de assentos concluído!")
 
 
if __name__ == "__main__":
    treinar()
 