# ✈️ ANAC Flight Intelligence

> **4 deep learning models trained on 21 million real Brazilian aviation records to predict airport congestion, seat availability, dynamic pricing pressure, and baggage fee recommendations — served via a real-time REST API with an interactive dashboard.**

---

## 🧠 Models & Architecture

This project trains **4 independent deep learning models** on the [ANAC Brazilian Civil Aviation dataset](https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/dados-estatisticos), covering domestic and international flights from 2000 to 2025.

| Model | Architecture | Target | Result |
|-------|-------------|--------|--------|
| Airport Congestion | Bidirectional LSTM + Temporal Attention | Probability of high movement | Val loss: **0.1319** |
| Seat Availability | FT-Transformer (Feature Tokenizer) | Occupancy rate + bucket class | Val loss: **0.3816** |
| Dynamic Pricing | MLP + Categorical Embeddings | Demand pressure index (0–1) | Val loss: **0.0005** |
| Baggage Surcharge | LightGBM + DNN Stacking | Excess baggage probability | AUC: **0.8659** |

### Architecture Overview

```
21M rows ANAC CSV
        ↓
  Polars ETL (Parquet)
        ↓
  Feature Engineering (4 feature tables)
        ↓
┌─────────────┬──────────────┬─────────────────┬──────────────────┐
│ LSTM        │ FT-          │ MLP +           │ LightGBM +       │
│ Bidir +     │ Transformer  │ Embeddings      │ DNN Stacking     │
│ Attention   │ (Tabular)    │                 │                  │
└─────────────┴──────────────┴─────────────────┴──────────────────┘
        ↓
  FastAPI — single endpoint → 4 predictions in parallel
        ↓
  Interactive Dashboard (HTML + JS)
```

---

## 📊 Dataset

- **Source:** [ANAC — Agência Nacional de Aviação Civil](https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/dados-estatisticos)
- **Size:** ~20 GB raw CSV, 21,115,242 flight records
- **Period:** 2000–2025
- **Key columns:** route OD, airline, date/time, seats offered, passengers carried, baggage weight, fuel, distance, cargo

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/caiobandeira/anac-flight-intelligence.git
cd anac-flight-intelligence

# Download ANAC data → data/raw/
# Train models (see Training section below)

docker compose up --build
```

- **Dashboard** → http://localhost:3000
- **API docs** → http://localhost:8000/docs

### Option 2 — Local

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn 06_api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔧 Training Pipeline

```bash
python 01_ingestao.py              # Ingest 20GB CSV → Parquet (Polars streaming)
python 02_feature_engineering.py   # Generate 4 feature tables
python 03_modelo_aeroporto.py      # Train Bidirectional LSTM
python 04_modelo_assentos.py       # Train FT-Transformer
python 05_modelos_preco_bagagem.py # Train MLP + LightGBM/DNN
uvicorn 06_api:app --host 0.0.0.0 --port 8000
```

---

## 🔌 API Reference

### `POST /prever`

```json
{
  "aeroporto_origem": "SBGR",
  "aeroporto_destino": "SBPA",
  "empresa": "TAM",
  "data_voo": "2024-06-14",
  "hora_partida": 18,
  "mes": 6,
  "dia_semana": "Sexta-feira",
  "assentos": 180,
  "distancia_km": 1100,
  "continente_destino": "América do Sul",
  "semana_ano": 24
}
```

**Response:**

```json
{
  "prob_aeroporto_cheio": 0.65,
  "lotacao_aeroporto": "alta",
  "taxa_ocupacao_voo": 0.805,
  "assentos_vazios_est": 35,
  "bucket_ocupacao": "quase cheio",
  "pressao_preco": 0.682,
  "recomendacao_preco": "premium",
  "prob_excesso_bagagem": 0.13,
  "recomendar_cobranca": false,
  "resumo": "🟡 AIRPORT (65% congestion)... 🟡 FLIGHT (80% occupancy)..."
}
```

Additional endpoints:
- `GET /health` — API status
- `GET /historico?origem=SBGR&destino=SBPA` — Monthly occupancy history
- `GET /cache/stats` — Cache usage
- `DELETE /cache` — Clear cache

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data ingestion | Polars (streaming, 20GB) |
| Deep learning | PyTorch, FT-Transformer, LSTM |
| Gradient boosting | LightGBM |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Dashboard | Vanilla HTML/CSS/JS |
| Drift monitoring | Custom script |

---

## 📁 Project Structure

```
anac-flight-intelligence/
├── 01_ingestao.py              # CSV → Parquet pipeline (Polars)
├── 02_feature_engineering.py   # 4 feature tables
├── 03_modelo_aeroporto.py      # Bidirectional LSTM + Attention
├── 04_modelo_assentos.py       # FT-Transformer (tabular)
├── 05_modelos_preco_bagagem.py # MLP pricing + LightGBM/DNN baggage
├── 06_api.py                   # FastAPI with cache + CORS + /historico
├── 07_monitorar_drift.py       # Monthly model health check
├── dashboard.html              # Interactive frontend
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 📈 Model Results

### Airport Congestion (LSTM)
- 3,011,327 training samples · 20 top Brazilian airports
- Best validation loss: **0.1319** at epoch 14/20

### Seat Availability (FT-Transformer)
- 4,223,048 samples · dual output (regression + classification)
- Best validation loss: **0.3816** at epoch 25/30

### Dynamic Pricing (MLP + Embeddings)
- 2,311,939 training samples
- Best validation loss: **0.0005** at epoch 40/40

### Baggage Surcharge (LightGBM + DNN Stacking)
- 21,115,242 samples · 80/20 temporal split
- LightGBM AUC: **0.8659** · DNN refinement AUC: **0.8634**

---

## 👤 Author

**Caio de Brito Bandeira**
Built with Python, PyTorch, and 21 million rows of real aviation data.

---

## 📄 License

MIT License
