import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import re
import compress_pickle as pickle
from datetime import datetime

def parse_credits(credits_list):
    names=[]
    for credits_dict in credits_list:
        for artist in credits_dict['artists']:
            names.append(artist['name'])
    return names
        
audioset_meta_location="/gscratch/scrubbed/wagnew2/audioset_info_collated.lz4"
audioset_song_infos=pickle.load(audioset_meta_location)

audioset_meta_location="/gscratch/scrubbed/wagnew2/audioset_metadata_collated.lz4"
audioset_metas=pickle.load(audioset_meta_location)

num_print=5
printed=0
all_names=[]
file_ids=list(set(list(audioset_metas.keys())+list(audioset_song_infos.keys())))
over_2016=0
for fid in tqdm(file_ids):
    names=[]
    
    if fid in audioset_song_infos:
        release_date=datetime.strptime(audioset_song_infos[fid]['music_info']['top_hit_full_info']['song']['release_date'], '%Y-%m-%d').date()
        if release_date.year<=2016:
            
            
            names.append(audioset_song_infos[fid]['music_info']['top_hit_full_info']['song']['artist_names'])
            for k in audioset_song_infos[fid]['music_info']['top_hit_full_info']['song']:
                info_list=audioset_song_infos[fid]['music_info']['top_hit_full_info']['song'][k]
                if isinstance(info_list, list):
                    for info_dict in info_list:
                        if isinstance(info_dict, dict) and 'relationship_type' in info_dict and 'name' in info_dict:
                            names.append(info_dict['name'])
            
            names+=parse_credits(audioset_song_infos[fid]['music_info']['song_credits'])
            names+=parse_credits(audioset_song_infos[fid]['music_info']['xtra_credits'])
        else:
            over_2016+=1
            continue
    if fid in audioset_metas:
        names.append(audioset_metas[fid]['yt_info']['uploader'])
        
    names=[name.lower() for name in names]
    names=list(set(names))
    all_names+=names

print('num datasets over 2016', over_2016, 'total', len(audioset_song_infos))
pickle.dump(all_names, open("/gscratch/scrubbed/wagnew2/all_artist_names_audioset.lz4", 'wb'))
    
        # for k in audioset_song_infos[fid]['music_info']:
        #     print(k, audioset_song_infos[fid][k])
        #     print("---------------------------------")
        # printed+=1
        # if printed>num_print:
        #     exit()

# # load lakh
# lakh_copyholders="/Users/williamagnew/eclipse-workspace/gen-audio-ethics/lakh_copyright.txt"
# with open(lakh_copyholders) as file:
#     copyright_infos=[]
#     for line in file:
#         copyright_infos.append(re.sub(r"\d{4}", "", line.rstrip().lower()))
# copyright_counts = Counter(copyright_infos)
#
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
#
# frequencies = list(copyright_counts.values())
# names = list(copyright_counts.keys())
#
# names=([x for _, x in sorted(zip(frequencies, names))])[::-1]
# frequencies.sort()
# frequencies=frequencies[::-1]
#
# num_display=50
# x_coordinates = np.arange(len(copyright_counts))
# y_pos = np.arange(num_display)
#
# ax.barh(y_pos, frequencies[:num_display], align='center')
# ax.set_yticks(y_pos, labels=names[:num_display], fontsize=7)
# ax.set_xlabel('Number Name References')
# ax.set_title('Artist Names and Copyright Claims in Audited Datasets', y=1.0, x=-0.0)
#
# # ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
# # ax.xaxis.set_major_formatter(plt.FixedFormatter(names))
# plt.ylim([num_display-0.5, 0-0.5])
# plt.tight_layout()
# plt.show()