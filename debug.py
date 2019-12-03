from waveflow import WaveFlow, hp, SineGen
import torch
import matplotlib.pyplot as plt

# x,S = SineGen()[0]
# x = x.unsqueeze(0)
# S = S.unsqueeze(0)

# # plt.figure(figsize=(10,5))
# # plt.subplot(121)
# # plt.plot(x.squeeze())
# # plt.subplot(122)
# # plt.imshow(S.squeeze(), origin="lower", aspect="auto")
# # plt.show()

# S = S.repeat_interleave(128, -1)

# flow = WaveFlow(debug=False)
# flow.load_state_dict(torch.load("step_16000.pth", map_location="cpu")[1])

# print("State loaded !")

# with torch.no_grad():
#     y_trained = flow.synthesize(S).squeeze().numpy()

# plt.figure(figsize=(10,5))
# plt.plot(y_trained)
# plt.show()

wf = WaveFlow(True)
x = torch.arange(128).unsqueeze(0).float()
c = torch.randn(1,80,128)
print(wf.loss(x,c)[-1])