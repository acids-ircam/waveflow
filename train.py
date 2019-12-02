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
    
    loss = model.loss(x,S)
    loss.backward()
    

    opt.step()


    return {"loss":loss.item()}

trainer = ct.Trainer(**ct.args.__dict__)

trainer.set_model(lambda : WaveFlow(True))
trainer.setup_model()

trainer.add_optimizer(torch.optim.Adam(trainer.model.parameters()))
trainer.setup_optim()

trainer.set_dataset_loader(SineGen)
trainer.set_lr(np.linspace(1e-3, 1e-4, ct.args.step))

trainer.set_train_step(train_step)

writer = SummaryWriter(f"runs/{ct.args.name}/", flush_secs=10)

for i,losses in enumerate(trainer.train_loop()):
    for loss in losses:
        writer.add_scalar(loss, losses[loss], i)
