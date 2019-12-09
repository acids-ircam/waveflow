from waveflow import WaveFlow
import torch

c = torch.randn(1,80,64)
wf = WaveFlow()
wf.synthesize_fast(c)