#!/bin/bash
#SBATCH --account=nn9997k
#SBATCH --partition=accel
#SBATCH --job-name=build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --gpus-per-node=0
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

apptainer build --fakeroot vllm-aarch64.sif vllm-aarch64.def
