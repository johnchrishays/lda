#!/bin/bash
#
#SBATCH --job-name=lda
#
#SBATCH --ntasks=1
#SBATCH -p sched_mit_sloan_interactive_r8
#SBATCH -o "slurm-%a.out"
#SBATCH -e "slurm-%a.out0V"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=100000
#SBATCH --mail-type=ERROR,END
#SBATCH --mail-user=jhays@mit.edu
#SBATCH --array=0-50


export JHAYS="/home/jhays"


# mamba init
conda activate lda
python3 $JHAYS/lda-1/model_training.py -i $SLURM_ARRAY_TASK_ID

