#!/bin/bash
#SBATCH --partition gpu4_short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 16G
#SBATCH --gres gpu:1
#SBATCH --time 0-04:00:00

#SBATCH --job-name get_sam2_inputs_auto
#SBATCH --output get_sam2_inputs_auto_%J.log 

source ~/.bashrc
conda init
cd "$(pwd)"
conda activate sam2
python -m data.pipeline.backend.core.auto_collect \
    --device cuda \
    --dataset_kwargs 'filter_for_testset=True;patients=all;activity=all;reps=all'
