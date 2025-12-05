# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.2-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends     python3-pip python3-dev git curl build-essential &&     rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --upgrade pip &&     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 &&     pip install triton==3.0.0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_bench_phase2_vs_phase3.py"]
