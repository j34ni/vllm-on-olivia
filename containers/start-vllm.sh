#!/bin/bash
MODEL=/mnt/ns1000k/models/Qwen2-7B-Instruct

# Attendre que le modèle soit disponible
if [ ! -d "$MODEL" ]; then
    echo "Model not found at $MODEL, skipping vLLM startup"
    exit 0
fi

export NO_PROXY=localhost,127.0.0.1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name qwen2-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.87 \
    --tensor-parallel-size 2 \
    > /mnt/ns1000k/vllm.log 2>&1 &

echo "vLLM started with PID $!"
echo $! > /mnt/ns1000k/vllm.pid
