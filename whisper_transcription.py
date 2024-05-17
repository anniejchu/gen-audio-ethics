import whisper
import os
from tqdm import tqdm
import pickle

class Whisper_Transcriber:
    
    def __init__(self):
        self.model = whisper.load_model("tiny")
        
    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result
        
    def get_language(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)