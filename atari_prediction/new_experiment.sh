#!/bin/bash
#SBATCH --job-name=Atari
#SBATCH --output=%x-%A-%a.out
#SBATCH --array=0-57
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

readarray -t GAMES < games.txt
NAME=${GAMES[$SLURM_ARRAY_TASK_ID]}
GAME=${NAME}NoFrameskip-v4
PY_FILE=Histogram_loss/atari_main.py
BASE_DIR=~/scratch
RETURNS_FILE=$BASE_DIR/atari/returns/$GAME.npy
OUT_DIR=$NAME

POLICY_DIR=$SLURM_TMPDIR/data/policies
DATA=$BASE_DIR/data/policies.zip
ACTION_FILE=$POLICY_DIR/$GAME.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.10 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r atari_requirements.txt
pip install AutoROM-0.6.1-py3-none-any.whl AutoROM.accept-rom-license-0.6.1.tar.gz ../keras_tuner-1.3.5-py3-none-any.whl

mkdir $SLURM_TMPDIR/data
unzip $DATA -d $SLURM_TMPDIR/data

cd $SLURM_TMPDIR
mkdir $OUT_DIR
cd $OUT_DIR
python $BASE_DIR/$PY_FILE $ACTION_FILE $RETURNS_FILE

cd $SLURM_TMPDIR
tar -czf $NAME.tgz $OUT_DIR
mv $NAME.tgz $BASE_DIR/atari/.
