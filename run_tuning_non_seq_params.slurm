#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=64GB
#SBATCH --job-name=tune_ns
#SBATCH --error=./tune_ns.%J.err
#SBATCH --output=./tune_ns.%J.out

cd /work/yesselmanlab/ewhiting/ss_confidence_scorer
activate .venv/bin/activate
bash run_tuning_swan_non_seq_params
