from waveflow import WaveFlow, hp
import torch

cdt = torch.randn(1,hp.cdt_size, 2048)
flow = WaveFlow()

with torch.no_grad():
    y = flow.synthesize(cdt)

print(y.shape)

