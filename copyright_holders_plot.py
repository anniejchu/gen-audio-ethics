import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import re
import compress_pickle as pickle

audioset_meta_location="/gscratch/scrubbed/wagnew2/audioset_info_collated.lz4"
audioset_metas=pickle.load(audioset_meta_location)
num_print=5
printed=0
for fid in audioset_metas:
    for k in audioset_meta[fid]:
        print(k, audioset_meta[fid][k])
        print("---------------------------------")
    printed+=1
    if printed>num_print:
        exit()

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