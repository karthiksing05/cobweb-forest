#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --exclude=hal,tars,t1000,vicki
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch -J name scripts/cpu.sbatch <python args...>   (run from the repo root)
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
exec /coc/flash10/ksingara3/miniconda3/envs/cobweb-llm/bin/python "$@"
