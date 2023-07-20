#!/bin/bash
#SBATCH --job-name=ETTm2-HL
#SBATCH --output=%x-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

MODEL=HL
PY_FILE=Histogram_loss/Time_Series/time_series_models.py
BASE_DIR=~/scratch
DATA_FILE=ETTm2.csv

module load python/3.10 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python $BASE_DIR/$PY_FILE $MODEL
