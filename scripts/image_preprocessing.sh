#!/bin/bash
#SBATCH --job-name=UTKFace-Preproc
#SBATCH --output=%x-%j.out
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=20000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

DATA=UTKFace.tar.gz
PY_FILE=Histogram_loss/mtcnn_test.py
N_CPUS=5

module load python/3.10 scipy-stack 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r preproc_requirements.txt
pip install --no-index --no-deps mtcnn

mkdir $SLURM_TMPDIR/data
tar xzf $DATA -C $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/data/UTKFace/aligned
for i in $(seq 1 $N_CPUS)
do
    python $PY_FILE $SLURM_TMPDIR $N_CPUS $i &
done
wait

tar czf $DATA -C $SLURM_TMPDIR/data UTKFace