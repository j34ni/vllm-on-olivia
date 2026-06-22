#!/bin/bash
MODEL=$PROJECT/models/Qwen2-7B-Instruct
LOGDIR=$PROJECT/vllm-on-olivia/logs

mkdir -p "$LOGDIR"

if [ ! -d "$MODEL" ]; then
    echo "Model not found at $MODEL, aborting"
    exit 1
fi

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name qwen2-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.87 \
    > "$LOGDIR/vllm.log" 2>&1 &

echo "vLLM started with PID $!"
echo $! > "$LOGDIR/vllm.pid"
