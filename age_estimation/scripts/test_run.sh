#!/bin/bash
#SBATCH --job-name=MegaAge-test
#SBATCH --output=%x-%j.out
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --mail-user=kluedema@ualberta.ca
#SBATCH --mail-type=ALL

module load python/3.10 cuda cudnn 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

mkdir $SLURM_TMPDIR/data
tar xf fgnet.tar -C $SLURM_TMPDIR/data

python Histogram_loss/main.py $SLURM_TMPDIR/data
