import torch

class CircularTensor(object):
    def __init__(self, tensor, dim):
        self.tensor  = tensor
        self.dim     = dim
        self.roll    = 0
        self.index   = torch.arange(self.tensor.shape[dim])
    
    def __getattr__(self, name):
        if name == "shape":
            return self.tensor.shape
    
    def __call__(self):
        idx       = (self.index + self.roll) % self.shape[self.dim]
        self.roll = (self.roll+1) % self.shape[self.dim]
        return self.tensor.index_select(self.dim, idx)

    def set_current(self, x):
        dim = [slice(None,None,None) for i in range(len(self.shape))]
        dim[self.dim] = self.roll - 1
        self.tensor[dim] = x


if __name__ == "__main__":
    import torch
    x = torch.zeros(3,3)
    c = CircularTensor(x,0)
    for i in range(3):
        c.set_current(torch.randn(3))
        print(c())