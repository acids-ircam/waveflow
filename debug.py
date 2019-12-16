from waveflow import WaveFlow
import torch
from time import time

torch.set_grad_enabled(False)

c = torch.randn(1,80,8192)
wf = WaveFlow() 
wf.remove_weight_norm()

st = time()
wf.synthesize_fast(c)

print(time()-st)
st = time()

wf.synthesize_fast(c)

print(time()-st)
