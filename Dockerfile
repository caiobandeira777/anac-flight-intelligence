# ============================================================
#  Dockerfile — ANAC Prediction API
#  Build: docker build -t anac-api .
#  Run:   docker-compose up
# ============================================================

# imagem base com Python 3.11 slim — menor que a full
FROM python:3.11-slim

# evita prompts interativos durante o build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1   
ENV PYTHONUNBUFFERED=1          

WORKDIR /app

# instala dependências do sistema necessárias para LightGBM e compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \       
    && rm -rf /var/lib/apt/lists/*

# copia e instala dependências Python primeiro
# (camada separada = rebuild mais rápido se só o código mudar)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# instala PyTorch CPU (menor que a versão CUDA — ideal para produção em servidor)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# copia o código da aplicação
COPY 06_api.py .

# copia os modelos treinados
COPY models/ ./models/

# copia as features para o endpoint /historico
COPY data/features/ ./data/features/

# porta que a API vai expor
EXPOSE 8000

# comando de inicialização — sem reload em produção
CMD ["uvicorn", "06_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
