#!/bin/bash
#SBATCH --partition gpu4_short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 16G
#SBATCH --gres gpu:1
#SBATCH --time 0-04:00:00

#SBATCH --job-name runner
#SBATCH --output runner_%J.log 

source ~/.bashrc
conda init
cd /gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/victor/cvfm4rehab
conda activate cvfm4rehab
python -m tools.grounding_dino.inferencer
python -m data.visualization.examples
