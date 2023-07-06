#!/bin/bash
#SBATCH --job-name=atari
#SBATCH --output=%x-%j.out
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

GAME=PongNoFrameskip-v4
PY_FILE=atari_main.py
BASE_DIR=~/scratch/Histogram_loss
RETURNS_FILE=returns/$GAME.npy
OUT_DIR=$GAME

POLICY_DIR=$SLURM_TMPDIR/data/policies
DATA=$BASE_DIR/data/policies.zip
ACTION_FILE=$POLICY_DIR/$GAME.txt


module load python/3.10 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r atari_requirements.txt
pip install --upgrade AutoROM-0.6.1-py3-none-any.whl AutoROM.accept-rom-license-0.6.1.tar.gz ../keras_tuner-1.3.5-py3-none-any.whl

mkdir $SLURM_TMPDIR/data
unzip $DATA -d $SLURM_TMPDIR/data

mkdir $OUT_DIR
cd $OUT_DIR
python $BASE_DIR/$PY_FILE $ACTION_FILE $RETURNS_FILE
