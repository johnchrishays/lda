#!/bin/bash
#
#SBATCH --job-name=lda
#
#SBATCH --ntasks=1
#SBATCH -p sched_mit_sloan_batch_r8
#SBATCH -o "slurm-%a.out"
#SBATCH -e "slurm-%a.out"
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-type=ERROR,END
#SBATCH --mail-user=jhays@mit.edu
#SBATCH --array=0-49


export JHAYS="/home/jhays"
python3 $JHAYS/lda-1/model_training.py -i $SLURM_ARRAY_TASK_ID

