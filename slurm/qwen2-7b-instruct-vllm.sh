#!/bin/bash
#SBATCH --account=nn9997k
#SBATCH --job-name=qwen2-7b-1gpu
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=96G
#SBATCH --time=24:00:00

SIF=/cluster/projects/nn9997k/jeani/vllm-on-olivia/containers/vllm-aarch64.sif
MODEL=/cluster/projects/nn9997k/jeani/llm-inference-on-olivia/models/Qwen2-7B-Instruct
CACHE=/cluster/projects/nn9997k/jeani/vllm-on-olivia/cache

export APPTAINER_TMPDIR=${CACHE}/apptainer/tmp
export APPTAINER_CACHEDIR=${CACHE}/apptainer/cache
mkdir -p ${APPTAINER_TMPDIR} ${APPTAINER_CACHEDIR}

export no_proxy=localhost,127.0.0.1

apptainer exec --nv \
    --env XDG_CACHE_HOME=${CACHE} \
    --env TORCHINDUCTOR_CACHE_DIR=${CACHE}/torchinductor \
    --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    --env no_proxy=localhost,127.0.0.1 \
    --env NCCL_NET=Socket \
    --env NCCL_IB_DISABLE=1 \
    --bind /cluster/projects/nn9997k:/cluster/projects/nn9997k \
    ${SIF} \
    python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL} \
        --served-model-name qwen2-7b \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.87
