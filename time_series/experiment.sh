#!/bin/bash
#SBATCH --job-name=Transformer-100
#SBATCH --output=%x-%j.out
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

PY_FILE=Histogram_loss/main.py
BASE_DIR=~/scratch

module load python/3.10 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python $BASE_DIR/$PY_FILE
