#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --array=0-4
#SBATCH --output=array_test_%A_%a.out
#SBATCH --error=array_test_%A_%a.error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48G 

# Your job commands go here
cd /home/wagnew/github/gen-audio-ethics/LyricWhiz/code
conda activate gen-audio-ethics2

python run_whisper.py --input_dir$1 --output_dir$2 --n_shard=5 --shard_rank=$SLURM_ARRAY_TASK_ID