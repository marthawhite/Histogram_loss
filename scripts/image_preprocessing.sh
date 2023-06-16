#!/bin/bash
#SBATCH --job-name=MegaAge-Preproc
#SBATCH --output=%x-%j.out
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=28000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

DATA=megaage_asian.tar
PY_FILE=Histogram_loss/mtcnn_test.py
N_CPUS=7

module load python/3.10 scipy-stack 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r preproc_requirements.txt
pip install --no-index --no-deps mtcnn

mv mtcnn.py $SLURM_TMPDIR/env/lib/site-packages/mtcnn/mtcnn.py

mkdir $SLURM_TMPDIR/data
tar xf $DATA -C $SLURM_TMPDIR/data
for i in $(seq 1 $N_CPUS)
do
    python $PY_FILE $SLURM_TMPDIR $N_CPUS $i &
done
wait

tar cf $DATA -C $SLURM_TMPDIR/data megaage_asian