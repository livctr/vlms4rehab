#!/bin/bash
#SBATCH --job-name=visualize_video_labels
#SBATCH --partition=cpu_short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output=results/output_%j.txt  # Save output in a file named with job ID
#SBATCH --mail-type=BEGIN,END,FAIL         # Email on job end (optional)
#SBATCH --mail-user=Victor.Li@nyulangone.org  # Your email (optional)


source ~/.bashrc  # my bashrc loads the following modules, uncomment if not
# module purge
# module load json-c/0.17
# module load slurm/current
# module load standard-tools/1.0
# module load default-environment
# module load gcc/11.2.0
# module load ffmpeg/3.4.2

cd ~/gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/victor/cvfm4rehab
conda init
conda activate cvfm4rehab

python -m data.verification.show_video_label_pair

echo "Job complete"
