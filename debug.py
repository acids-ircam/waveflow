from waveflow import WaveFlow
import torch

c = torch.randn(1,80,96000)
wf = WaveFlow().cuda()
with torch.no_grad():
    print(wf.synthesize_fast(c).shape)