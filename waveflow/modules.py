import torch
import torch.nn as nn
from . import hparams as hp
import numpy as np
from tqdm import tqdm

class Debugger:
    def debug_msg(self, msg):
        if self.debug:
            print(msg)

class ResidualBlock(nn.Module, Debugger):
    def __init__(self, dilation, debug=False):
        super().__init__()
        self.debug = debug

        total_size = (hp.kernel_size - 1) * dilation + 1
        padding_h  = total_size - 1
        padding_w  = total_size // 2

        self.debug_msg([dilation, total_size, padding_h, padding_w])

        self.initial_conv = nn.Conv2d(hp.res_size, hp.hidden_size * 2,
                                      hp.kernel_size, dilation=dilation,
                                      padding=(padding_h, padding_w))

        self.cdtconv      = nn.Conv2d(hp.cdt_size, hp.hidden_size * 2, 1)
        
        self.resconv      = nn.Conv2d(hp.hidden_size, hp.res_size, 1)
        self.skipconv     = nn.Conv2d(hp.hidden_size, hp.skp_size, 1)

    def forward(self, x, c):
        res = x.clone()

        x = self.initial_conv(x)[:,:,:hp.h,:]
        c = self.cdtconv(c)

        xa,xb = torch.split(x, hp.hidden_size, 1)
        ca,cb = torch.split(c, hp.hidden_size, 1)

        x = torch.tanh(xa + ca) * torch.sigmoid(xb + cb)

        res = self.resconv(x) + res
        skp = self.skipconv(x)

        return res, skp

class ResidualStack(nn.Module, Debugger):
    def __init__(self, debug=False):
        super().__init__()

        self.first_conv = nn.Conv2d(hp.in_size, hp.res_size, 1)

        self.stack = nn.ModuleList([
            ResidualBlock(2**i, debug) for i in np.arange(hp.n_layer) % hp.cycle_size
        ])

        self.last_convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hp.skp_size, hp.skp_size, 1),
            nn.ReLU(),
            nn.Conv2d(hp.skp_size, hp.out_size, 1)
        )

        self.debug = debug
    
    def forward(self, x, c):
        self.debug_msg("first conv")
        res = self.first_conv(x)

        skp_list = []

        self.debug_msg(f"iterating over {len(self.stack)} resblock...")
        for i,resblock in enumerate(self.stack):
            self.debug_msg(f"Residual block {i}")
            res, skp = resblock(res, c)
            skp_list.append(skp)
        
        self.debug_msg("sum")
        x = sum(skp_list)

        self.debug_msg("last convs")
        return self.last_convs(x)



class WaveFlow(nn.Module, Debugger):
    def __init__(self, debug=False):
        super().__init__()
        self.flows = nn.ModuleList([
            ResidualStack(debug) for i in range(hp.n_flow)
        ])

        self.debug = debug
        self.receptive_field = (hp.kernel_size-1)*(sum([2**(i%hp.cycle_size) for i in range(hp.n_layer)])) + 1

        skipped = 0
        for p in self.parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                skipped += 1

        print(f"Skipped {skipped} parameters during initialisation")
        print(f"Built waveflow with squeezed height {hp.h} and receptive field {self.receptive_field}")

    def forward(self, x, c, squeezed=False):
        if not squeezed:
            self.debug_msg(f"Squeezing input")
            x = x.reshape(x.shape[0], 1, x.shape[-1] // hp.h, -1).transpose(2,3)
            c = c.reshape(c.shape[0], c.shape[1], c.shape[-1] // hp.h, -1).transpose(2,3)

        global_mean    = None
        global_logvar  = None

        self.debug_msg(f"Iterating over {len(self.flows)} flows")
        for i,flow in enumerate(self.flows):
            self.debug_msg(f"Passing through flow {i}")
            mean, logvar = torch.split(flow(x,c), 1, 1)
            
            if global_mean is not None and global_logvar is not None:
                global_mean    = global_mean * torch.exp(logvar) + mean
                global_logvar  = global_logvar + logvar
            
            else:
                global_mean   = mean
                global_logvar = logvar

            x = torch.exp(logvar) * x + mean

        return x, global_mean, global_logvar

    def loss(self, x, c):
        z, _, logvar = self.forward(x,c)
        loglikelihood = - torch.mean(z ** 2 + .5 * np.log(2*np.pi))  + torch.mean(logvar)
        return - loglikelihood

    def synthesize(self, c):
        device = next(self.parameters()).device

        c = c.reshape(c.shape[0], c.shape[1], c.shape[-1] // hp.h, -1).transpose(2,3).to(device)
        z = torch.randn(c.shape[0], 1, c.shape[2], c.shape[3]).to(device)
        x = torch.zeros_like(z).to(device)

        pad = torch.zeros(1,1,hp.h,1).to(device)

        c = torch.cat([pad.expand_as(c),c], 2)
        z = torch.cat([pad.expand_as(z),z], 2)
        x = torch.cat([pad.expand_as(x),x], 2)

        for step in tqdm(range(hp.h)):
            x_ = x[:,:,step:step+hp.h,:]
            c_ = c[:,:,step:step+hp.h,:]

            _, mean, logvar = self.forward(x_,c_, squeezed=True)

            x[:,:,hp.h + step,:] = (z[:,:,hp.h + step,:] - mean[:,:,-1,:]) * torch.exp(-logvar[:,:,-1,:])
        
        x = x[:,:,hp.h:,:].transpose(2,3).reshape(x.shape[0], -1)

        return x