import os
import random
import shutil

data_paths=["/data/user_data/wagnew/wav-caps/WavCaps/Zip_files/AudioSet_SL/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac",
            "/data/user_data/wagnew/wav-caps/WavCaps/Zip_files/BBC_Sound_Effects/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/BBC_Sound_Effects_flac",
            "/data/user_data/wagnew/wav-caps/WavCaps/Zip_files/FreeSound/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/FreeSound_flac",
            "/data/user_data/wagnew/wav-caps/WavCaps/Zip_files/SoundBible/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/SoundBible_flac"]
save_folder="/data/user_data/wagnew/wav-caps/WavCaps/sample"

for data_path in data_paths:
    files=os.listdir(data_path)
    random.shuffle(files)
    for i in range(500):
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(save_folder, files[i]))
        
        