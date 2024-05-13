import os
import compress_pickle as pickle
from tqdm import tqdm



audio_info_file_dir="/media/willie/1caf5422-4135-4f2c-9619-c44041b51146/audio_data/audioset/audioset_downloads/"

for audio_info_file in tqdm(os.listdir(audio_info_file_dir)):
    if audio_info_file[-4:]=='.lz4':
        audio_info=pickle.load(os.path.join(audio_info_file_dir, audio_info_file))
        if isinstance(audio_info, dict):
            title=audio_info['yt_info']['title']
            transcription=audio_info['transcript']