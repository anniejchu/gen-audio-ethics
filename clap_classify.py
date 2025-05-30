import numpy as np
import librosa
import torch
import laion_clap
import json
import random
from tqdm import tqdm

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class Clap_Classifier:
    
    def __init__(self):
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt() # download the default pretrained checkpoint.
        class_index_dict = {v: k for v, k in json.load(open('CLAP/class_labels/audioset_fsd50k_class_labels_indices.json')).items()}
        #self.ground_truth_idx = [class_index_dict[json.load(open(jf))['tag'][0]] for jf in json_files]
        self.classes=list(class_index_dict.keys())
        all_texts = ["This is a sound of " + t for t in self.classes]
        self.text_embed = self.model.get_text_embedding(all_texts)
        
    def classify(self, audio_files):
        chunk_size=10
        classes=[]
        for chunk_ind in tqdm(range(0, len(audio_files), chunk_size)):
            audio_embed = self.model.get_audio_embedding_from_filelist(x=[audio])
            similarities=torch.tensor(audio_embed) @ torch.tensor(self.text_embed).t()
            sort_similarities=torch.sort(similarities, descending=True).detach().cpu().numpy()
            ranking = torch.argsort(similarities, descending=True).detach().cpu().numpy()
            for file_ind in range(sort_similarities.shape[0]):
                classes.append([])
                for sim_ind in range(sort_similarities.shape[1]):
                    if sort_similarities[file_ind, sim_ind]<0.9:
                        break
                    classes[-1].append(self.classes[int(ranking[file_ind, sim_ind])])
                    
            # print(ranking)
            # top_class=self.classes[int(ranking[0,0])]
        print(top_class)

def get_audio_files(dir):
    audio_files=[]
    for file in os.listdir(dir):
        if os.path.isdir(file):
            audio_files+=get_audio_files(os.path.join(dir, file))
        elif file[-4:] in ['.mp3', '.wav', 'flac']:
            audio_files.append(os.path.join(dir, file))
    return audio_files
        

if __name__ == '__main__':
    #MCV
    #speaker
    
    #VCTK
    #speaker
    
    #Librivox
    #speaker
    
    #Lakh
    #genre labels
    
    #Audioset
    #labels
    
    #FMA
    #genre labels
    audio_files=get_audio_files('/data/user_data/wagnew/fma_large/')
    audio_files=random.shuffle(audio_files)
    
    #Jamendo
    #genre labels
    
    classifier=Clap_Classifier()
    classifier.classify('/data/user_data/wagnew/fma_large/000/000002.mp3')

# # Directly get audio embeddings from audio files
# audio_file = [
#     '/home/data/test_clap_short.wav',
#     '/home/data/test_clap_long.wav'
# ]
# audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)
#
# # Get audio embeddings from audio data
# audio_data, _ = librosa.load('/home/data/test_clap_short.wav', sr=48000) # sample rate should be 48000
# audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
# audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)
#
# # Directly get audio embeddings from audio files, but return torch tensor
# audio_file = [
#     '/home/data/test_clap_short.wav',
#     '/home/data/test_clap_long.wav'
# ]
# audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=True)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)
#
# # Get audio embeddings from audio data
# audio_data, _ = librosa.load('/home/data/test_clap_short.wav', sr=48000) # sample rate should be 48000
# audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
# audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
# audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)
#
# # Get text embedings from texts:
# text_data = ["I love the contrastive learning", "I love the pretrain model"] 
# text_embed = model.get_text_embedding(text_data)
# print(text_embed)
# print(text_embed.shape)
#
# # Get text embedings from texts, but return torch tensor:
# text_data = ["I love the contrastive learning", "I love the pretrain model"] 
# text_embed = model.get_text_embedding(text_data, use_tensor=True)
# print(text_embed)
# print(text_embed.shape)
