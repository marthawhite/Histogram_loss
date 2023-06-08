#!/bin/bash
#SBATCH --job-name=MegaAge-dist-reg
#SBATCH --output=%x-%j.out
#SBATCH --time=0-9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

DATA=megaage_asian.tar
HYPERS=hypers.tar
TUNER=keras_tuner-1.3.5-py3-none-any.whl
N_WORKERS=2

module load python/3.10 scipy-stack cuda cudnn 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install --upgrade $TUNER

mkdir $SLURM_TMPDIR/data
tar xf $DATA -C $SLURM_TMPDIR/data
tar xf $HYPERS -C $SLURM_TMPDIR

PY_FILE=Histogram_loss/regression_tuner.py

srun --ntasks=1 ./Histogram_loss/chief.sh $PY_FILE $SLURM_TMPDIR &
for i in $(seq 1 $N_WORKERS)
do 
    srun --ntasks=1 ./Histogram_loss/worker.sh $PY_FILE $SLURM_TMPDIR $i &
wait

tar cf hypers.tar -C $SLURM_TMPDIR hypers
