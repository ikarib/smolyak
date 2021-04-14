#!/bin/bash
#
#SBATCH --job-name=smolyak
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=smolyak.out
#SBATCH --error=smolyak.err
##SBATCH --nodelist=n28
##SBATCH --time=60
#SBATCH --mail-type=END
#SBATCH --mem=100G

module load matlab
matlab -r multi_country
