NAME=$1
GAME=${NAME}NoFrameskip-v4
PY_FILE=Histogram_loss/atari_main.py
BASE_DIR=~/scratch
RETURNS_FILE=$BASE_DIR/atari/returns/$GAME.npy
OUT_DIR=$NAME

POLICY_DIR=$SLURM_TMPDIR/data/policies
DATA=$BASE_DIR/data/policies.zip
ACTION_FILE=$POLICY_DIR/$GAME.txt

cd $SLURM_TMPDIR
mkdir $OUT_DIR
cd $OUT_DIR
python $BASE_DIR/$PY_FILE $ACTION_FILE $RETURNS_FILE

cd $SLURM_TMPDIR
tar -czf $NAME.tgz $OUT_DIR
mv $NAME.tgz $BASE_DIR/atari/.
