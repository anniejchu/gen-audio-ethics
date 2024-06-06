import os
import compress_pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

audio_info_file_dir="/gscratch/scrubbed/wagnew2/audioset/audio/unbal_train/"

audio_infos={}
num_empty=0
for audio_info_file in tqdm(os.listdir(audio_info_file_dir)):
    if audio_info_file[-4:]=='.lz4':
        
        try:
            audio_info=pickle.load(os.path.join(audio_info_file_dir, audio_info_file))
        except EOFError:
            print(f'EOFError {audio_info_file}')
            
        
        # if isinstance(audio_info, dict):
        #     audio_infos[audio_info_file[:-4]]=audio_info['yt_info']['uploader']

        if 'music_info' in audio_info and not audio_info['music_info'] is None and len(audio_info['music_info'])>0:
            audio_infos[audio_info_file[:-4]]=audio_info
            print("found music")
        else:
            num_empty+=1
            #transcription=audio_info['transcript']
    # if len(audio_infos)>10000:
    #     break
print('num_empty', num_empty)   
pickle.dump(audio_infos, open("/gscratch/scrubbed/wagnew2/audioset_info_collated.lz4", 'wb'))
            
# Date

# years=[]
# for audio_info in audio_infos:
#     years.append(int(audio_info['yt_info']['upload_date'][0:4]))
# plt.hist(years)
# plt.xlabel('Year')
# plt.ylabel('Number Audio Files')
# plt.title('Year Audio Files Uploaded')
# plt.show()

# Language

# languages=[]
# for audio_info in audio_infos:
#     languages.append(audio_info['langauge'])
# plt.hist(languages, bins=range(len(set(languages))), align="left")
# plt.xlabel('Language')
# plt.ylabel('Number Audio Files')
# plt.title('Audio Language')
# plt.show()

# Tags

# audio_tagss=[]
# for audio_info in audio_infos:
#     audio_tags=audio_info['audio_tags']
#     for audio_tag in audio_tags:
#         audio_tagss.append(audio_tag[0])
# plt.hist(audio_tagss, bins=range(len(set(audio_tagss))), align="left")
# plt.xlabel('Audio Tag')
# plt.ylabel('Number Audio Files')
# plt.title('Audio Tags')
# plt.show()


# Music IDed, Artist name, label name

creator_names=[]
for audio_info in audio_infos:
    if not audio_info['music_info'] is None and len(audio_info['music_info'])>0:
        u=0
    audio_tags=audio_info['audio_tags']
    for audio_tag in audio_tags:
        audio_tagss.append(audio_tag[0])
plt.hist(audio_tagss, bins=range(len(set(audio_tagss))), align="left")
plt.xlabel('Audio Tag')
plt.ylabel('Number Audio Files')
plt.title('Audio Tags')
plt.show()
