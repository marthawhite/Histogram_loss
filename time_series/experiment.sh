#!/bin/bash
#SBATCH --job-name=Analysis
#SBATCH --output=%x-%A-%a.out
#SBATCH --array=0-2
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

MODELS=(Linear Transformer LSTM)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY_FILE=Histogram_loss/model_analysis.py
BASE_DIR=~/scratch

module load python/3.10 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python $BASE_DIR/$PY_FILE $MODEL
