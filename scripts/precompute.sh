#!/bin/bash
#SBATCH --job-name=precompute
#SBATCH --output=%x-%j.out
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

PY_FILE=Histogram_loss/atari_prediction/precompute.py
BASE_DIR=~/scratch
POLICY_DIR=$SLURM_TMPDIR/data/policies
DATA=$BASE_DIR/data/policies.zip

module load python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r precompute_requirements.txt
pip install AutoROM-0.6.1-py3-none-any.whl AutoROM.accept-rom-license-0.6.1.tar.gz

mkdir $SLURM_TMPDIR/data
unzip $DATA -d $SLURM_TMPDIR/data
ls $POLICY_DIR | parallel python $BASE_DIR/$PY_FILE $POLICY_DIR {} returns
