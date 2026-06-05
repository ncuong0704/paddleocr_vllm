FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --index-strategy unsafe-best-match

RUN pip install \
    openai>=1.0.0 \
    runpod>=0.8.0 \
    requests>=2.31.0 \
    Pillow>=10.0.0 \
    transformers>=4.35.0

COPY handler.py /workspace/handler.py

EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/v1/models || exit 1

CMD bash -c "\
    echo 'Starting vLLM server...' && \
    vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --max-num-batched-tokens 16384 \
        --no-enable-prefix-caching \
        --mm-processor-cache-gb 0 \
        --port 8000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.9 &\
    \
    sleep 45 && \
    echo 'Starting RunPod handler...' && \
    python3 /workspace/handler.py"
