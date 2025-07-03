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
cd "$(pwd)"
conda activate cvfm4rehab

python -m data.sam2_prompting.sam2_tracking
