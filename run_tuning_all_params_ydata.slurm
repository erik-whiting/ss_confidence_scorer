#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=64GB
#SBATCH --job-name=ydtune_all
#SBATCH --error=./ydtune_all.%J.err
#SBATCH --output=./ydtune_all.%J.out

cd /work/yesselmanlab/ewhiting/ss_confidence_scorer
activate .venv/bin/activate
bash run_tuning_swan_with_ydata
