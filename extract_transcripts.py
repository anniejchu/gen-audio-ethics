import os
import compress_pickle as pickle
from tqdm import tqdm
import pretty_midi
import re
import codecs

# VCTK
# vctk_location="/Users/williamagnew/datasets/vctk/VCTK-Corpus-0.92/txt"
#
# all_text=[]
# for folder in tqdm(os.listdir(vctk_location)):
#     folder_path=os.path.join(vctk_location, folder)
#     if os.path.isdir(folder_path):
#         for text_file in os.listdir(folder_path):
#             text_file_path=os.path.join(folder_path, text_file)
#             with open(text_file_path, "r") as f:
#                 all_text.append(f.read())
# pickle.dump(all_text, open("/Users/williamagnew/datasets/vctk_all_transcripts.lz4", "wb"))

# Lakh

# def find_midi(folder_path, all_text):
#     for search_file in tqdm(os.listdir(folder_path)):
#         new_folder_path=os.path.join(folder_path, search_file)
#         if os.path.isdir(new_folder_path):
#             all_text=find_midi(new_folder_path, all_text)
#         elif search_file[-4:]==".mid":
#             try:
#                 pm = pretty_midi.PrettyMIDI(new_folder_path)
#                 if len(pm.lyrics)>0:
#                     text=""
#                     for lyric in pm.lyrics:
#                         text=text+lyric.text+" "
#                     all_text.append(text)
#             except Exception as e:
#                 print(e)
#     return all_text
#
# lakh_location="/media/willie/1caf5422-4135-4f2c-9619-c44041b51146/audio_data/lmd_full"
# all_text=find_midi(lakh_location, [])
# #pickle.dump(all_text, open("/gscratch/scrubbed/wagnew2/audio_transcripts/lakh_all_transcripts.lz4", "wb"))


# LibriVox

start_str="***"
end_str="***END OF THIS PROJECT GUTENBERG EBOOK"
librivox_location="/Users/williamagnew/datasets/LibriSpeech/books/ascii"
all_text=[]
for librivox_location in ["/Users/williamagnew/datasets/LibriSpeech/books/ascii", "/Users/williamagnew/datasets/LibriSpeech/books/utf-8"]:
    for folder in tqdm(os.listdir(librivox_location)):
        folder_path=os.path.join(librivox_location, folder)
        if os.path.isdir(folder_path):
            for text_file in os.listdir(folder_path):
                text_file_path=os.path.join(folder_path, text_file)
                with codecs.open(text_file_path, encoding='latin-1') as f:
                    print(text_file_path)
                    book_text=f.read()
                    oringinal_text_len=len(book_text)
                    try:
                        ind=book_text.find(start_str)
                        start_ind=ind+book_text[ind+len(start_str):].find(start_str)+2*len(start_str)
                        p=re.compile("\*\*\*\s*END OF")
                        m = p.search(book_text)
                        end_ind=m.start()
                        book_text=book_text[start_ind:end_ind]
                        u=book_text[-200:]
                    except:
                        print("parse 1 error!")
                        start_str="further information is included below.  We need your donations."
                        start_ind=book_text.find(start_str)+len(start_str)
                        end_ind=book_text.find("End of the Project Gutenberg Etext")
                        book_text=book_text[start_ind:end_ind]
                        u=book_text[-200:]
                        t=0
                    all_text.append(book_text)
                    if len(book_text)/oringinal_text_len<0.9:
                        u=0
                    print(f"Len book text {len(book_text)}")
pickle.dump(all_text, open("/Users/williamagnew/datasets/librivox_all_transcripts.lz4", "wb"))

