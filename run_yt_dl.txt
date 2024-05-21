#!/bin/bash
#SBATCH --job-name=yt_dl
#SBATCH --partition=ckpt
#SBATCH --account=jamiemmt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<>

source ~/.bashrc
conda activate gen-audio-ethics2
cd /gscratch/scrubbed/wagnew2/github/gen-audio-ethics
echo "-----------"
echo $1
python transcription_analysis.py --save_loc=/gscratch/scrubbed/wagnew2/audioset/audio/unbal_train/ --youtube_ids=/gscratch/scrubbed/wagnew2/unbalanced_train_segments.csv --num_processes=32 --info_transcribe=0


