from waveflow import WaveFlow, SineGen, hp
import waveflow.central_training as ct

import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_step(model, opt_list, step, data_list):
    if step:
        # exit()
        pass

    opt = opt_list[0]
    opt.zero_grad()

    x,S = data_list
    S = S.repeat_interleave(128,-1)
    
    z,mean,logvar,loss = model.loss(x,S)
    loss.backward()
    

    opt.step()

    if step % trainer.image_every == 0:
        with torch.no_grad():
            y = model.synthesize(S).reshape(-1)
        writer.add_audio("input", x.reshape(-1).cpu(), step, 16000)
        writer.add_audio("reconstruction", y.cpu(), step, 16000)
        writer.add_histogram("mean", mean.reshape(-1), step)
        writer.add_histogram("logvar", logvar.reshape(-1), step)
        writer.add_histogram("z", z.reshape(-1), step)

    return {"loss":loss.item()}

trainer = ct.Trainer(**ct.args.__dict__)

debug = False

trainer.set_model(lambda : WaveFlow(debug))
trainer.setup_model()

trainer.add_optimizer(torch.optim.Adam(trainer.model.parameters()))
trainer.setup_optim()

trainer.set_dataset_loader(SineGen)
trainer.set_lr(np.linspace(2e-4, 2e-4, ct.args.step))

trainer.set_train_step(train_step)

writer = SummaryWriter(f"runs/{ct.args.name}/", flush_secs=10)

for i,losses in enumerate(trainer.train_loop()):
    for loss in losses:
        writer.add_scalar(loss, losses[loss], i)
