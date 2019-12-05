from waveflow import WaveFlow, hp, SineGen
import matplotlib.pyplot as plt
import torch
import os

import seaborn
seaborn.set()

singen = SineGen()
bs = 1
data = []

for i in range(bs):
    data.append(singen[i])

x = torch.stack([d[0] for d in data],0)
S = torch.stack([d[1] for d in data],0)

S = S.repeat_interleave(128, -1)

flow = WaveFlow(debug=False)
flow.load_state_dict(torch.load("step_60000.pth", map_location="cpu")[1])
flow = flow.cuda()
print("State loaded !")

with torch.no_grad():
    y_trained, zs = flow.synthesize(S, demo_pass=True)
    y_trained = y_trained.squeeze().cpu().numpy()

plt.figure(figsize=(10,5))
plt.plot(x.reshape(-1).squeeze().numpy())
plt.plot(y_trained.reshape(-1))
plt.show()

plt.ion()

fps = 20
N   = 100

y = y_trained.reshape(-1)[:N]

x = zs[0,:N]

plt.subplot(121)
plt.plot(x)
plt.title(f"source")
plt.subplot(122)
plt.plot(y)
plt.title("target")
plt.pause(2)

# os.makedirs("temp/", exist_ok=True)

for f in range(hp.n_flow):
    for i in range(fps):
        x = zs[f,:N]*(fps-i)/fps + zs[f+1,:N]*i/fps
        plt.clf()
        plt.subplot(121)
        plt.plot(x)
        plt.ylim([-3,3])
        plt.title(f"Flow {f} to {f+1}")
        plt.subplot(122)
        plt.plot(y)
        plt.ylim([-3,3])
        plt.title("target")
        # plt.savefig("temp/img_{:05d}.png".format(i+f*fps))
        plt.pause(1/60)

plt.pause(5)
