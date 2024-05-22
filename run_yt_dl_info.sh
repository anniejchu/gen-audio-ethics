#!/bin/bash
#SBATCH --job-name=yt_dl
#SBATCH --partition=ckpt
#SBATCH --account=jamiemmt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<>

source ~/.bashrc
conda activate gen-audio-ethics2
cd /gscratch/scrubbed/wagnew2/github/gen-audio-ethics
echo "-----------"
curl https://ipinfo.io/ip
echo $1
python transcription_analysis.py --save_loc=/gscratch/scrubbed/wagnew2/audioset_info --youtube_ids=/gscratch/scrubbed/wagnew2/unbalanced_train_segments.csv --num_processes=2 --info_transcribe=1


