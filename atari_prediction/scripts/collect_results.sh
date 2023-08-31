TAR_FILE=all_results.tar
DIR_NAME=rerun_results
OUT_TAR=rerun_full.tgz
BASE_DIR=~/scratch/atari

tar -cf $TAR_FILE *.tgz
mkdir $SLURM_TMPDIR/$DIR_NAME
cp $TAR_FILE $SLURM_TMPDIR/$DIR_NAME/.
cd $SLURM_TMPDIR/$DIR_NAME
tar -xf $TAR_FILE

for f in *.tgz
do
    tar -xzf "$f"
done

rm *.tgz
cd $SLURM_TMPDIR
tar -czf $OUT_TAR $DIR_NAME
cp $OUT_TAR $BASE_DIR/.
