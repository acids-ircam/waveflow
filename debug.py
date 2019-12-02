from waveflow import WaveFlow, hp
import torch

x = torch.randn(1, 2048)
cdt = torch.randn(1,hp.cdt_size, 2048)

print(x.shape, cdt.shape)

flow = WaveFlow(True)

print(flow.loss(x,cdt))