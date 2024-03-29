#!/bin/bash
#SBATCH --job-name=atari
#SBATCH --output=%x-%A-%a.out
#SBATCH --array=0-26%10
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

GAMES=(Gopher Hero JamesBond JourneyEscape Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman Phoenix Pitfall Pooyan PrivateEye RoadRunner Robotank Seaquest Solaris StarGunner Tennis TimePilot Tutankham UpNDown Venture VideoPinball WizardOfWor YarsRevenge Zaxxon)
GAME_NAME=${GAMES[$SLURM_ARRAY_TASK_ID]}

GAME=${GAME_NAME}NoFrameskip-v4
PY_FILE=Histogram_loss/atari_main.py
BASE_DIR=~/scratch
RETURNS_FILE=../returns/$GAME.npy
OUT_DIR=$GAME_NAME

POLICY_DIR=$SLURM_TMPDIR/data/policies
DATA=$BASE_DIR/data/policies.zip
ACTION_FILE=$POLICY_DIR/$GAME.txt


module load python/3.10 scipy-stack cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r atari_requirements.txt
pip install AutoROM-0.6.1-py3-none-any.whl AutoROM.accept-rom-license-0.6.1.tar.gz

mkdir $SLURM_TMPDIR/data
unzip $DATA -d $SLURM_TMPDIR/data

mkdir $OUT_DIR
cd $OUT_DIR
python $BASE_DIR/$PY_FILE $ACTION_FILE $RETURNS_FILE
