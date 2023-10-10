import os
import io
import numpy as np
import json
import math
import torch
import torch.nn as nn
import collections
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
import subprocess
import torchaudio
from audio_models.beats.BEATs import BEATs, BEATsConfig
from audio_models.cavmae.models import CAVMAE, CAVMAEFT
from petrel_client.client import Client
client = Client('~/petreloss.conf', enable_mc=True)

def init_beats(model_dir):
    checkpoint = torch.load(model_dir, map_location="cpu")
    audio_cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(audio_cfg)
    model.load_state_dict(checkpoint['model'])
    return model

def init_cavmae(model_dir):
    checkpoint = torch.load(model_dir, map_location="cpu")
    new_state_dict = OrderedDict()
    for k in checkpoint:
        name = k.replace('module.', '')
        new_state_dict[name] = checkpoint.setdefault(k)
    # model = CAVMAE(modality_specific_depth=11)、
    model = CAVMAEFT(label_dim=527, modality_specific_depth=11)
    model.load_state_dict(new_state_dict, strict=True)
    return model

def load_audio(audio_path, sr):
    if audio_path.startswith('s3') or audio_path.startswith('p2'):
        audio_bytes = client.get(audio_path)
        buff = io.BytesIO(audio_bytes)
    else:
        buff = audio_path
    torchaudio.set_audio_backend('soundfile')   # for flac files
    audio, csr = torchaudio.load(buff)
    audio = torch.mean(audio, dim=0, keepdim=True)
    if csr != sr:
        trans = torchaudio.transforms.Resample(csr, sr)
        audio = trans(audio)
    return audio

def get_fbank(waveform):
    fbank_mean = -5.081
    fbank_std = 4.4849
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    # fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
    fbank = fbank.squeeze(0)
    fbank = (fbank - fbank_mean) / (fbank_std)
    pad_len = math.ceil(fbank.size(0)/1024)*1024-fbank.size(0)
    print(waveform.shape, fbank.shape, pad_len, waveform.shape[1]/16000)
    m = torch.nn.ZeroPad2d((0, 0, 0, pad_len))
    fbank = m(fbank)
    fbank = fbank.view(fbank.size(0)//1024, 1024, -1)
    return fbank, pad_len


def beats_feat_extract(audio_dir, split_ls, feat_dir, model_dir):
    model = init_beats(model_dir)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    model = model.cuda()

    for name in split_ls:
    # for name in os.listdir(split_ls):
        if os.path.exists(os.path.join(feat_dir, name)):
            # print(f"{name} exist!", flush=True)     
            continue
        try:
            # data = load_audio(os.path.join(audio_dir, name[:-4]+'.wav'), 16000)
            data = load_audio(os.path.join(audio_dir, name+'.wav'), 16000)
        except Exception as e:
            print(e, flush=True)
            # subprocess.run(['aws', 's3', '--endpoint-url=http://10.135.3.249:80', 'cp', 's3://perception/audios/'+name[:-4]+'.wav', './data/pt/tmp_audio/'])
            continue
        data = data.cuda()
        feats = model.extract_features(data)[0].squeeze()
        print(feats.shape, name)
        feats = feats.cpu().numpy()
        np.save(os.path.join(feat_dir, name), feats)


def cavmae_feat_extract(audio_dir, split_ls, feat_dir, model_dir):
    model = init_cavmae(model_dir)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    model = model.cuda()

    for name in os.listdir(split_ls):
        if os.path.exists(os.path.join(feat_dir, name)):
            # print(f"{name} exist!", flush=True)     
            continue
        try:
            data = load_audio(os.path.join(audio_dir, name[:-4]+'.wav'), 16000)
            data, pad_len = get_fbank(data)
        except Exception as e:
            print(e, flush=True)
            # subprocess.run(['aws', 's3', '--endpoint-url=http://10.135.3.249:80', 'cp', 's3://perception/audios/'+name[:-4]+'.wav', './data/pt/tmp_audio/'])
            continue
        data = data.cuda()
        feats = model.forward_feat(data, None, mode='a')
        feats = feats.view(-1, 768)
        feats = feats[:feats.size(0)-pad_len//2]
        print(feats.shape, name)
        feats = feats.cpu().numpy()
        np.save(os.path.join(feat_dir, name), feats)


if __name__ == "__main__":
    audio_dir = "/mnt/petrelfs/yujiashuo/dataset/pt/audios/"
    feat_dir = "./data/pt/sound_localisation_beats/missing_audio/"
    # split_ls = './data/pt/sound_localisation_valid_audio_features/'
    with open('missing_audio.txt', 'r') as f:
        split_ls = f.readlines()
    split_ls = [f.strip() for f in split_ls]
    print(len(split_ls))
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir, exist_ok=True)
    model_dir = "/mnt/petrelfs/yujiashuo/model/beats/BEATs_iter3+.pt"
    # model_dir = "/mnt/petrelfs/yujiashuo/model/cavmae/as_46.6.pth"
    # cavmae_feat_extract(audio_dir, split_ls, feat_dir, model_dir)
    beats_feat_extract(audio_dir, split_ls, feat_dir, model_dir)
