#!/bin/bash
#SBATCH --job-name=Reg-Aligned
#SBATCH --output=%x-%j-%a.out
#SBATCH --array=1-8
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

DATA=megaage_asian.tgz
HYPERS=reg_aligned_hypers${SLURM_ARRAY_TASK_ID}.tgz
TUNER=keras_tuner-1.3.5-py3-none-any.whl
PY_FILE=Histogram_loss/regression_tuner.py

module load python/3.10 scipy-stack cuda cudnn 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install --upgrade $TUNER

mkdir $SLURM_TMPDIR/data
tar -xzf $DATA -C $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/hypers

python $PY_FILE $SLURM_TMPDIR $SLURM_ARRAY_TASK_ID

tar -czf $HYPERS -C $SLURM_TMPDIR hypers
