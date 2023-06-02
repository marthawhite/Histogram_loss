#!/bin/bash
#SBATCH --job-name=UTKFace-hypers
#SBATCH --output=%x-%j.out
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

DATA=UTKFace.tar.gz
HYPERS=hypers.tar
TUNER=keras_tuner-1.3.5-py3-none-any.whl

module load python/3.10 scipy-stack cuda cudnn 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install --upgrade $TUNER

mkdir $SLURM_TMPDIR/data
tar xzf $DATA -C $SLURM_TMPDIR/data
tar xf $HYPERS -C $SLURM_TMPDIR

python Histogram_loss/tuner.py $SLURM_TMPDIR

tar cf $HYPERS -C $SLURM_TMPDIR hypers