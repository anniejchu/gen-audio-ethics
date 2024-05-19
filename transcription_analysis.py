import os
import array
from tqdm import tqdm
from lyricsgenius import Genius #uses BeautifulSoup, should think about use
import pprint as j
from ShazamAPI import Shazam
from torch import tensor
from torchmetrics.audio import SignalNoiseRatio
from yt_dlp import YoutubeDL
from EfficientAT.inference import Audio_Tagger
import compress_pickle as pickle
import torchaudio
import torch
from wada_snr import wada_snr
from whisper_transcription import Whisper_Transcriber
import requests
from vtt2text import vtt2text
import csv
import random
import yt_dlp
import multiprocessing as mp
from optparse import OptionParser
import time
from pydub import AudioSegment
# import api_key #importing api_key.py file that holds my personal client access token
#whisper_transcriber=None#Whisper_Transcriber()
from multiprocessing import Process, Manager,Queue, Pipe
import queue
from contextlib import redirect_stdout
from pathlib import Path
import io
import tempfile
import codecs
from miniaudio import DecodeError, ffi, lib
import resampy
import soundfile as sf
from fs.memoryfs import MemoryFS
import time

def mp3_read_f32(data: bytes) -> array:
    '''Reads and decodes the whole mp3 audio data. Resulting sample format is 32 bits float.'''
    config = ffi.new('drmp3_config *')
    num_frames = ffi.new('drmp3_uint64 *')
    memory = lib.drmp3_open_memory_and_read_pcm_frames_f32(data, len(data), config, num_frames, ffi.NULL)
    if not memory:
        raise DecodeError('cannot load/decode data')
    try:
        samples = array.array('f')
        buffer = ffi.buffer(memory, num_frames[0] * config.channels * 4)
        samples.frombytes(buffer)
        return samples, config.sampleRate, config.channels
    finally:
        lib.drmp3_free(memory, ffi.NULL)
        ffi.release(num_frames)
        
def mult_analyze(analyzer,q,child_conn):
    # print('a')
    # analyzer,q,child_conn=args
    analyzer.mp_get_audio_info_youtube(q,child_conn)
    
class Transcription_Analyzer:
    
    def __init__(self):
        token = "rnoCATq04jbnSTIEsW9fvahh8Q--OBMbeUf76AFOKKpVVEIYTwOhtLOLPqrYBptW"
        self.genius = Genius(token, remove_section_headers = True, timeout=30)
        #self.audio_tagger=Audio_Tagger()
        
        self.audio_features={}
        with open('class_labels_indices.csv', newline='') as f:
            reader = csv.reader(f)
            feats = list(reader)
        
        for ind in range(1, len(feats)):
            self.audio_features[feats[ind][1]]=feats[ind][2]
            
        with open("music_codes.txt") as file:
            self.music_codes = [line.rstrip() for line in file]
        u=0

    def search_song_by_lyric(self, lyric_query, return_lyrics=False):
        """
        Search for songs by a given lyric query and return the hits.
    
        Args:
        - lyric_query (str): The lyric query to search for.
        - return_lyrics (bool): Whether to include full lyrics in the results. Defaults to False. If True, takes a little longer
    
        Returns:
        - hits (list): Returns songs that match the query (sorted by Popularity) / List of dictionaries containing information about each hit (song).
        """
        request = self.genius.search_lyrics(lyric_query, per_page=2)
        hits = []
        for hit in request['sections'][0]['hits']:
            lyric_url = hit['result']['url']
            hit_info = {
                'genius_id': hit['result']['id'],
                'song': hit['result']['title'],
                'artists': hit['result']['artist_names'],
                'full_song_title': hit['result']['full_title'],
                'lyric_match': hit['highlights'][0]['value'],
                'full_lyrics_url': lyric_url,
            }
            if return_lyrics:
                hit_info['full_lyrics'] = genius.lyrics(song_url=lyric_url)
            hits.append(hit_info)
        return hits
    
    def extract_attribution_info(self, performance_list):
        performances_dict = {}
    
        for item in performance_list:
            label = item['label']
            artists = item['artists']
            artists_info = []
    
            for artist in artists:
                artist_info = (artist['name'], artist['id'], artist['url']) #dict of lists
                # artist_info = {'name': artist['name'], 'id': artist['id'], 'url': artist['url']} #dict of dicts
                artists_info.append(artist_info)
    
            performances_dict[label] = artists_info
    
        return performances_dict
    
    # # test shazam api
    audio_path="/media/willie/1caf5422-4135-4f2c-9619-c44041b51146/audio_data/MusicBench/datashare/data"
    # for fname in tqdm(os.listdir(audio_path)):
    #     if fname[-4:]==".wav":
    #         mp3_file_content_to_recognize = open(os.path.join(audio_path, fname), 'rb').read()
    #         shazam = Shazam(mp3_file_content_to_recognize)
    #         recognize_generator = shazam.recognizeSong()
    #         for result in recognize_generator:
    #             if len(result[1]['matches'])>0:
    #                 print(fname, result) # current offset & shazam response to recognize requests
    
    def get_audio_info_spotify(self, spotify_link):
        u=0
    
    def mp_get_audio_info_youtube(self, q, child_conn):
        while not q.empty():
            youtube_id, save_dir, start_time, stop_time, features=q.get()
            self.get_audio_info_youtube(youtube_id, save_dir, start_time, stop_time, features, child_conn)
        
    
    def get_audio_info_youtube(self, youtube_id, save_dir, start_time, stop_time, features, whisper_pipe):
        if not os.path.exists(os.path.join(save_dir, f'{youtube_id}_info.lz4')):
            ydl_opts = {
                'format': 'bestaudio/best',
                # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
                'postprocessors': [{  # Extract audio using ffmpeg
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'paths': {'home': save_dir},
            }
            
            # mem_fs = MemoryFS()
            # output = io.BytesIO()#tempfile.SpooledTemporaryFile(max_size=1e12)#
            # StreamWriter = codecs.getwriter('utf-8')  # here you pass the encoding
            # wrapper_file = StreamWriter(output)
            # with mem_fs.open('test.wav', 'rw') as f:
            s_time=time.time()
            with YoutubeDL(ydl_opts) as ydl:
                # error_code=ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                try:
                    error_code=ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                except yt_dlp.utils.DownloadError:
                    print(f'video unavailable! {youtube_id}')
                    pickle.dump('video unavailable!', open(os.path.join(save_dir, f'{youtube_id}_info.lz4'), 'wb'))
                    return
                    
                info = ydl.extract_info(f'https://www.youtube.com/watch?v={youtube_id}', download=False)
                # temp_data, temp_sr = sf.read(output.read(), channels=2, samplerate=44100,
                #        subtype='FLOAT', format='RAW')
                    #waveform=mp3_read_f32(output.read())
                    #waveform, sample_rate = torchaudio.load(f)
            for file_name in os.listdir(save_dir):
                if f'[{youtube_id}].wav' in file_name:
                    break
            # audio = AudioSegment.from_file(os.path.join(save_dir, file_name))
            # audio.export(os.path.join(save_dir, file_name[:-3]+'wav'), format='wav')
            waveform, sample_rate = torchaudio.load(os.path.join(save_dir, file_name[:-3]+'wav'))#
            e_time=time.time()
            print('time a', e_time-s_time)   
            # for file_name in os.listdir(save_dir):
            #     if f'[{youtube_id}].m4a' in file_name:
            #         break
            
            # audio = AudioSegment.from_file(os.path.join(save_dir, file_name))
            # audio.export(os.path.join(save_dir, file_name[:-3]+'wav'), format='wav')
            #waveform, sample_rate = torchaudio.load()#os.path.join(save_dir, file_name[:-3]+'wav'))
            # cut audio to 10s sample length
            
            s_time=time.time()
            waveform=waveform[:, start_time*sample_rate:stop_time*sample_rate]
            torchaudio.save(os.path.join(save_dir, file_name[:-3]+'wav'), waveform, sample_rate)
            wada_snr_measure=float('nan')
            if waveform.shape[1]>0:
                wada_snr_measure=wada_snr(waveform)
            e_time=time.time()
            print('time b', e_time-s_time)   
            # audio_tags=self.get_sound_tags(waveform.numpy())
            audio_tags=[]
            for feature in features:
                f=feature.replace('"', '')
                audio_tags.append((self.audio_features[feature.replace('"', '').strip()], 1.0))
            
            
                
            
            transcript=None
            # Get creator made English transcript
            # if 'subtitles' in info:
            #     subtitles_keys=list(info['subtitles'].keys())
            #     for subtitles_key in subtitles_keys:
            #         if subtitles_key[:3]=='en-':
            #             transcript_links=info['subtitles'][subtitles_key]
            #             for transcript_link in transcript_links:
            #                 if transcript_link['ext']=='vtt':
            #                     transcript=self.get_transcript_from_url(transcript_link['url'])
            # # If this fails get youtube transcript
            # if transcript is None:
            #     if 'automatic_captions' in info and 'en' in info['automatic_captions']:
            #         transcript_links=info['automatic_captions']['en']
            #         for transcript_link in transcript_links:
            #             if transcript_link['ext']=='vtt':
            #                 transcript=self.get_transcript_from_url(transcript_link['url'])
            # If this fails use whisper
            s_time=time.time()
            if transcript is None:
                result=self.whisper_transcribe(whisper_pipe, os.path.join(save_dir, file_name[:-3]+'wav'))#whisper_transcriber.transcribe(os.path.join(save_dir, file_name[:-3]+'wav'))#
                transcript=result['text']
                langauge=result['language']
            e_time=time.time()
            print('translate time', e_time-s_time)   
                
            # If audio has music, try to get audio and artist info
            music_info=None
            music_tags=[feat for feat in audio_tags if feat[0] in self.music_codes]
            #print('music_tags', music_tags)
            if len(music_tags)>0:
                print('d')
                music_info=self.get_audio_info_unk(os.path.join(save_dir, file_name[:-3]+'wav'), transcript)
            #print('music_info', music_info)
            #langauge=whisper_transcriber.get_language(os.path.join(save_dir, file_name))
                
            print('e')
            pickle.dump({'yt_info': info, 'wada_snr': wada_snr_measure, 'audio_tags': audio_tags, 'transcript': transcript, 'langauge': langauge, 'music_info': music_info}, open(os.path.join(save_dir, f'{youtube_id}_info.lz4'), 'wb'))
            print('f')
            os.remove(os.path.join(save_dir, file_name[:-3]+'wav'))
            print('g')
    
    def whisper_transcribe(self, whisper_pipe, text_path):
        whisper_pipe[0].put(text_path)
        transcription=whisper_pipe[1].get()
        return transcription
    
    def get_transcript_from_url(self, transcript_url):
        data = requests.get(transcript_url)
        text=vtt2text(data.text)
        return data
    
    def compute_snr(self, audio):
        snr = SignalNoiseRatio()
        return snr(audio)
    
    def get_sound_tags(self, audio):
        return self.audio_tagger.tag_audio(audio)
    
    def get_audio_info_unk(self, audio_file, transcript):
        #drop first word which might be clipped to increase search accuracy
        print('i')
        search_lyrics=transcript
        if len(transcript.split(" "))>1:
            search_lyrics=transcript[len(transcript.split(" ")[0]):]
        print('j')
        hits=self.search_song_by_lyric(search_lyrics, return_lyrics=False)
        # Getting the ID of the top hit (single song)
        print('k')
        top_hit_id=None
        # If match is exact and unique
        if ((len(hits)>0 and transcript.lower() in hits[0]['lyric_match'].lower().replace("\n", " ")) 
            and (len(hits)==1 or not transcript.lower() in hits[1]['lyric_match'].lower().replace("\n", " "))):
            top_hit_id = hits[0]['genius_id']
        else: #try shazam
            mp3_file_content_to_recognize = open(audio_file, 'rb').read()
            shazam = Shazam(mp3_file_content_to_recognize)
            recognize_generator = shazam.recognizeSong()
            for result in recognize_generator:
                if len(result[1]['matches'])==1:
                    title=result[1]['track']['title']
                    artist=result[1]['track']['subtitle']
                    hits=self.genius.search_song(title=title, artist=artist)
                    if not hits is None:
                        top_hit_id = hits.id
                break
        if not top_hit_id is None:    
            print('l')
            # d=hits[1]['lyric_match'].lower().replace("\n", " ")
            # a=transcript.lower() in hits[1]['lyric_match'].lower().replace("\n", " ")
            # b=not transcript.lower() in hits[1]['lyric_match'].lower().replace("\n", " ")
            # c=len(hits)==1
            
            # Fetching full information about the top hit song // Ref: https://docs.genius.com/#songs-h2
            top_hit_full_info = self.genius.song(song_id=top_hit_id)
            
            # Getting basic song info
            basic_song_info = top_hit_full_info['song']
            # print(basic_song_info)  # Entire basic song info
            # print(basic_song_info['title'])  # Title of the song
            # print(basic_song_info['artist_names'])  # Names of the artists
            # print(basic_song_info['release_date'])  # Release date of the song
            # print(basic_song_info['recording_location'])  # Recording location of the song
            
            # Accessing album information if available
            album_info = basic_song_info['album']
            print(album_info)  # Entire album info
            
            # *** Getting other important song credits! Includes copyright / label info ****
            song_credits = top_hit_full_info['song']['custom_performances']
            
            xtra_credits = top_hit_full_info['song']['custom_performances']
            # j.pprint(xtra_credits)
            # for item in xtra_credits:
            #     if item['label'] == 'Copyright ©':
            #         artists = item['artists']
            #         for artist in artists:
            #             print("Name:", artist['name'])
            #             print("URL:", artist['url'])
            
            for item in xtra_credits:
                label = item['label']
                artists = item['artists']
                print("Label:", label)
                for artist in artists:
                    print("\tName:", artist['name'])
                    print("\tID:", artist['id'])
                    print("\tURL:", artist['url'])
            
            xtra_credits = top_hit_full_info['song']['custom_performances']
            performances_info = self.extract_attribution_info(xtra_credits)
            print('found music info!')
            
            return {'top_hit_full_info': top_hit_full_info,
                    'basic_song_info': basic_song_info,
                    'album_info': album_info,
                    'song_credits': song_credits,
                    'xtra_credits': xtra_credits,
                    'performances_info': performances_info}
        else:
            return {}
        
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--save_loc", dest="save_loc", default='/Users/williamagnew/eclipse-workspace/gen-audio-ethics/')
    parser.add_option("--youtube_ids", dest="youtube_ids", default='/media/willie/1caf5422-4135-4f2c-9619-c44041b51146/audio_data/audioset/unbalanced_train_segments.csv')
    parser.add_option("--num_processes", type="int", dest="num_processes", default=24)
    
    (options, args) = parser.parse_args()
    print(options)
    
    whisper_transcriber=Whisper_Transcriber()
    analyzer=Transcription_Analyzer()
    #analyzer.get_audio_info_unk("/Users/williamagnew/eclipse-workspace/gen-audio-ethics/Flume - Never Be Like You feat. Kai [Ly7uj0JwgKg].m4a", "Oh, can't you see I made, I made a mistake Please just look me in my face Tell me everything's okay Cause I got this")
    # analyzer.get_audio_info_youtube("Ly7uj0JwgKg", options.save_loc, 120, 130, [], None)
    # exit()
    
    with open(options.youtube_ids, newline='') as f:
        reader = csv.reader(f)
        yt_ids = list(reader)
    yt_ids=yt_ids[3:]
    random.shuffle(yt_ids)
    # for yt_id in yt_ids:
    #     analyzer.get_audio_info_youtube(yt_id[0], options.save_loc, int(float(yt_id[1].strip())), int(float(yt_id[2].strip())), yt_id[3:], None)
    
    # Multithreaded
    all_parallel_runs=[]
    with mp.Pool(processes=options.num_processes, maxtasksperchild=1) as pool:
        manager=Manager()
        q = manager.Queue()
        for ind in tqdm(range(len(yt_ids))):
             q.put((yt_ids[ind][0], options.save_loc, int(float(yt_ids[ind][1].strip())), int(float(yt_ids[ind][2].strip())), yt_ids[ind][3:]))
        whisper_pipes=[]
        for ind in tqdm(range(options.num_processes)):
            parent_queue=manager.Queue()
            child_queue=manager.Queue()
            whisper_pipes.append((parent_queue, child_queue))
            # mult_analyze((analyzer, q,child_conn))
            run=pool.apply_async(mult_analyze, args=(analyzer, q,(parent_queue, child_queue)))
            all_parallel_runs.append(run)
        
        while not q.empty():
            print(q.qsize())
            time.sleep(1)
            
        all_done=False
        while not all_done:
            all_done=True
            for run in all_parallel_runs:
                if not run.ready():
                    all_done=False
            print(q.qsize())
            for queues in whisper_pipes:
                while not queues[0].empty():
                    audio_file=queues[0].get()
                    result=whisper_transcriber.transcribe(audio_file)
                    queues[1].put(result)
            time.sleep(0.1)
    print("finished!")
    # Single threaded
    # for ind in tqdm(range(len(yt_ids))):
    #     analyzer.get_audio_info_youtube(yt_ids[ind][0], options.save_loc, int(float(yt_ids[ind][1].strip())), int(float(yt_ids[ind][2].strip())), yt_ids[ind][3:])
        
           