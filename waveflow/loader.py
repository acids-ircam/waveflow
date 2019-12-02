from torch.utils.data import Dataset
import numpy as np
import torch
import librosa as li

class SineGen(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def __getitem__(self, i):
        N = 8192
        f0 = 400*np.random.rand(1) + 100
        x = np.cos(2*np.pi*f0*np.linspace(0,.5,N))
        
        S = li.feature.melspectrogram(x, hop_length=128, n_fft=2048, win_length=2048, n_mels=80)[...,:N//128]
        S = np.log(S + 1e-3) / 10
        
        return torch.from_numpy(x).float(), torch.from_numpy(S)
    
    def __len__(self):
        return 1000