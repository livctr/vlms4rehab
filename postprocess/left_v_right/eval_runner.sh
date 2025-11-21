#!/bin/bash
#SBATCH --job-name=left_v_right_eval
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=18:00:00
#SBATCH --output=results/output_%j.txt  # Save output in a file named with job ID
#SBATCH --error=results/output_%j.txt    # Save error logs


export HF_HOME="***REMOVED***"
export HF_TOKEN="***REMOVED***"

deactivate_all_conda_envs() {
  while [[ -n "$CONDA_DEFAULT_ENV" ]]; do
    echo "Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
  done
}

source ~/.bashrc
conda init
cd "$(pwd)"
conda activate cvfm4rehab

python -m postprocess.left_v_right.eval
