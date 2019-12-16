import torch

class CircularTensor(object):
    def __init__(self, tensor, dim, n_slice):
        self.tensor  = tensor
        self.dim     = dim
        self.n_slice = n_slice
        self.roll    = 0
        self.index   = torch.arange(self.n_slice)
    
    def __getattr__(self, name):
        if name == "shape":
            return self.tensor.shape
    
    def next_slice(self):
        self.roll = (self.roll+1) % self.shape[self.dim]
        idx       = (self.index + self.roll) % self.shape[self.dim]
        return self.tensor.index_select(self.dim, idx)

    def set_current(self, x):
        dim = [slice(None,None,None) for i in range(len(self.shape))]
        dim[self.dim] = (self.roll + self.n_slice) % self.shape[self.dim]
        self.tensor[dim] = x


if __name__ == "__main__":
    import torch
    x = torch.arange(9).reshape(3,3)
    c = CircularTensor(x,0,2)
    for i in range(10):
        print(c.next_slice())
        c.set_current(x[0,:])