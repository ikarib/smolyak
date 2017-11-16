#!/bin/bash
#SBATCH --job-name=smolyak
#SBATCH --output=smolyak.out
#SBATCH --error=smolyak.err
#SBATCH --nodelist=n28
#SBATCH --time=10
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
##SBATCH --mail-type=END
##SBATCH --licenses=matlab@10.1.65.64:1
##SBATCH --gres=gpu:4

#module load matlab/90
matlab -r multi_country

