FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    MODEL_ID=openai/clip-vit-base-patch32

WORKDIR /app

COPY backend/app/requirements.txt /app/backend/app/requirements.txt
COPY backend/app/requirements-dev.txt /app/backend/app/requirements-dev.txt


RUN apt-get update && \
    apt-get install -y --no-install-recommends gawk && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    awk '!/^torch==/' /app/backend/app/requirements.txt > /tmp/requirements-no-torch.txt && \
    pip install -r /tmp/requirements-no-torch.txt && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install -r /app/backend/app/requirements-dev.txt


COPY backend /app/backend
COPY data /app/data

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
