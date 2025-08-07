#!/bin/bash
#SBATCH --partition cpu_short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 16G
#SBATCH --time 0-04:00:00
#SBATCH --job-name runner
#SBATCH --output runner_%J.log 

source ~/.bashrc
conda init
cd /gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/victor/cvfm4rehab
conda activate cvfm4rehab
python -m data.ia.batch_convert_FM_videos
# python -m tools.grounding_dino.inferencer
# python -m data.visualization.examples
