import torch
import torch.nn as nn
from . import hparams as hp
from .fast_utils import CircularTensor
import numpy as np
from tqdm import tqdm

def full_flip(x):
    return torch.flip(x, (2,))

def half_flip(x):
    x1,x2 = torch.split(x, x.shape[2]//2, 2)
    x1 = torch.flip(x1,(2,))
    x2 = torch.flip(x2,(2,))
    return torch.cat([x1,x2], 2)

class Debugger:
    def debug_msg(self, msg):
        if self.debug:
            print(msg)

class ResidualBlock(nn.Module, Debugger):
    def __init__(self, dilation, debug=False):
        super().__init__()
        self.debug = debug

        total_size = (hp.kernel_size - 1) * dilation + 1
        self.padding_h  = total_size - 1 
        self.padding_w  = total_size // 2

        self.debug_msg([dilation, total_size, self.padding_h, self.padding_w])

        self.initial_conv = nn.Conv2d(hp.res_size, hp.hidden_size * 2,
                                      hp.kernel_size, dilation=dilation,
                                      padding=(self.padding_h, self.padding_w), bias=False)

        self.cdtconv      = nn.Conv2d(hp.cdt_size, hp.hidden_size * 2, 1, bias=False)
        
        self.resconv      = nn.Conv2d(hp.hidden_size, hp.res_size, 1, bias=False)
        self.skipconv     = nn.Conv2d(hp.hidden_size, hp.skp_size, 1, bias=False)

        self.apply_weight_norm()

    def forward(self, x, c):
        res = x.clone()

        x = self.initial_conv(x)[:,:,:-self.padding_h,:]
        c = self.cdtconv(c)

        xa,xb = torch.split(x, hp.hidden_size, 1)
        ca,cb = torch.split(c, hp.hidden_size, 1)

        x = torch.tanh(xa + ca) * torch.sigmoid(xb + cb)

        res = self.resconv(x) + res
        skp = self.skipconv(x)

        return res, skp
    
    def apply_weight_norm(self):
        self.initial_conv = nn.utils.weight_norm(self.initial_conv)
        self.cdtconv     = nn.utils.weight_norm(self.cdtconv)
        self.resconv      = nn.utils.weight_norm(self.resconv)
        self.skipconv     = nn.utils.weight_norm(self.skipconv)

class ResidualStack(nn.Module, Debugger):
    def __init__(self, debug=False):
        super().__init__()

        self.first_conv = nn.Conv2d(hp.in_size, hp.res_size, (2,1),
                                    padding=(2,0), bias=False)

        self.stack = nn.ModuleList([
            ResidualBlock(2**i, debug) for i in np.arange(hp.n_layer) % hp.cycle_size
        ])

        self.last_convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hp.skp_size, hp.skp_size, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hp.skp_size, hp.out_size, 1, bias=False)
        )

        self.debug = debug

        self.apply_weight_norm()
    
    def forward(self, x, c):
        self.debug_msg("first conv")
        res = self.first_conv(x)[:,:,:-3,:]
        
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
    
    def arTransform(self, z):
        """
        TO BE DONE: FAST auto regressive transformation of z
        """
        for step in range(hp.h):
            z_in = z[:,:,:step+1,:]
            c_in = c[:,:,:step+1,:]
            mean, logvar = torch.split(self.forward(z_in,c_in), 1, 1)
            z[:,:,step,:] = (z[:,:,step,:] - mean[:,:,-1,:]) * torch.exp(-logvar[:,:,-1,:])
        return z

    def apply_weight_norm(self):
        for i in [1,3]:
            self.last_convs[i] = nn.utils.weight_norm(self.last_convs[i])


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

            x = full_flip(x) if i < 4 else half_flip(x)
            c = full_flip(c) if i < 4 else half_flip(c)

        return x, global_mean, global_logvar

    def loss(self, x, c):
        z, mean, logvar = self.forward(x,c)
      
        self.debug_msg(f"z.shape={z.shape}\nmean.shape={mean.shape}\nlogvar.shape={logvar.shape}")

        loss = torch.mean(z ** 2 - logvar)
        
        return z, mean, logvar, loss

    def synthesize(self, c, temp=1.0, demo_pass=False):
        device = next(self.parameters()).device
        print(f"Synthesizing on device {device}")

        c = c.reshape(c.shape[0], c.shape[1], c.shape[-1] // hp.h, -1).transpose(2,3).to(device)
        z = torch.randn(c.shape[0], 1, c.shape[2], c.shape[3]).to(device)
    
        z = z * temp

        if demo_pass:
            zs = []
            zs.append(z.transpose(2,3).reshape(-1).cpu().numpy())

        for i,flow in enumerate(tqdm(self.flows[::-1], desc="Iterating overs flows")):
            z = full_flip(z) if i > 4 else half_flip(z)
            c = full_flip(c) if i > 4 else half_flip(c)
            
            for step in range(hp.h):
                z_in = z[:,:,:step+1,:]
                c_in = c[:,:,:step+1,:]

                mean, logvar = torch.split(flow(z_in,c_in), 1, 1)

                z[:,:,step,:] = (z[:,:,step,:] - mean[:,:,-1,:]) * torch.exp(-logvar[:,:,-1,:])
            
            if demo_pass:
                # demo = full_flip(z) if i > 4 else half_flip(z)
                zs.append(z.transpose(2,3).reshape(-1).cpu().numpy())

      
            
        z = z.transpose(2,3).reshape(z.shape[0], -1)

        if demo_pass:
            return z,np.asarray(zs)
        else:
            return z

    def synthesize_fast(self, c, temp=1.0):

        device = next(self.parameters()).device
        print(f"Synthesizing on device {device}")

        c = c.reshape(c.shape[0], c.shape[1], c.shape[-1] // hp.h, -1).transpose(2,3).to(device)
        z = torch.randn(c.shape[0], 1, c.shape[2], c.shape[3]).to(device)
    
        z = z * temp

        for f in self.flows:
            pass

        for i,flow in enumerate(tqdm(self.flows[::-1], desc="Iterating overs flows")):
            z = full_flip(z) if i > 4 else half_flip(z)
            c = full_flip(c) if i > 4 else half_flip(c)
            
            z = flow.arTransform(z_in,cin)
            # for step in range(hp.h):
            #     z_in = z[:,:,:step+1,:]
            #     c_in = c[:,:,:step+1,:]

            #     mean, logvar = torch.split(flow(z_in,c_in), 1, 1)

            #     z[:,:,step,:] = (z[:,:,step,:] - mean[:,:,-1,:]) * torch.exp(-logvar[:,:,-1,:])
                  
            
        z = z.transpose(2,3).reshape(z.shape[0], -1)

        return z