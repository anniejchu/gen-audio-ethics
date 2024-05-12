import argparse
import torch
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext

from EfficientAT.models.mn.model import get_model as get_mobilenet
from EfficientAT.models.dymn.model import get_model as get_dymn
from EfficientAT.models.ensemble import get_ensemble_model
from EfficientAT.models.preprocess import AugmentMelSTFT
from EfficientAT.helpers.utils import NAME_TO_WIDTH, labels

class Audio_Tagger:
    
    def __init__(self):
        model_name = 'dymn20_as'
        self.device = torch.device('cuda')
        self.sample_rate = 32000
        self.window_size = 800
        self.hop_size = 320
        self.n_mels = 128
        ensemble=[]
        strides=[2, 2, 2, 2]
        head_type="mlp"
    
        # load pre-trained model
        if len(ensemble) > 0:
            self.model = get_ensemble_model(ensemble)
        else:
            if model_name.startswith("dymn"):
                self.model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                      strides=strides)
            else:
                self.model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                      strides=strides, head_type=head_type)
        self.model.to(self.device)
        self.model.eval()
        
    def tag_audio(self, audio):
        # model to preprocess waveform into mel spectrograms
        mono_audio=(audio[0,:]+audio[1,:])/2.0
        mel = AugmentMelSTFT(n_mels=self.n_mels, sr=self.sample_rate, win_length=self.window_size, hopsize=self.hop_size)
        mel.to(self.device)
        mel.eval()
    
        waveform=mono_audio
        waveform = torch.from_numpy(waveform[None, :]).to(self.device)
    
        # our models are trained in half precision mode (torch.float16)
        # run on cuda with torch.float16 to get the best performance
        # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
        with torch.no_grad(), autocast(device_type=self.device.type):
            spec = mel(waveform)
            preds, features = self.model(spec.unsqueeze(0))
        preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()
    
        sorted_indexes = np.argsort(preds)[::-1]
        
        audio_events=[]
        for k in range(50):
            audio_events.append((labels[sorted_indexes[k]], preds[sorted_indexes[k]]))
        # print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
        #     preds[sorted_indexes[k]]))
    
        return audio_events

def audio_tagging(args):
    """
    Running Inference on an audio clip.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=args.strides)
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=args.strides, head_type=args.head_type)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(20):
        print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
            preds[sorted_indexes[k]]))
    print("********************************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn10_as')
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--audio_path', type=str, default="/media/willie/1caf5422-4135-4f2c-9619-c44041b51146/audio_data/OBGYN - SNL [KSg7a0yMCSg].m4a")

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])

    args = parser.parse_args()

    audio_tagging(args)
