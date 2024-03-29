#!/bin/bash
#SBATCH --job-name=fgnet
#SBATCH --output=%x-%j.out
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

DATA=data/fgnet.tar
HYPERS=fgnet_hypers.tgz
TUNER=keras_tuner-1.3.5-py3-none-any.whl
PY_FILE=Histogram_loss/fgnet.py
BASE_DIR=~/scratch

module load python/3.10 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r $BASE_DIR/requirements.txt
pip install --upgrade $BASE_DIR/$TUNER

mkdir $SLURM_TMPDIR/data
tar -xzf $BASE_DIR/$DATA -C $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/hypers

python $BASE_DIR/$PY_FILE $SLURM_TMPDIR

tar -czf $HYPERS -C $SLURM_TMPDIR hypers
