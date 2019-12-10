class CircularTensor:
    """
    Define a Tensor mapped on a cylinder in order to speed up roll time during
    fast generation loop.
    """
    def __init__(self, tensor, dim):
        self.dim = dim
        self.roll = 0
        self.tensor = tensor
        self.mod = self.tensor.shape[self.dim]

    def __getitem__(self,index):
        index = list(index)
        index[self.dim] = (index[self.dim] + self.roll) % self.mod
        return self.tensor[tuple(index)]

    def __setitem__(self, index, value):
        index = list(index)
        index[self.dim] = (index[self.dim] + self.roll) % self.mod
        self.tensor[tuple(index)] = value

    def rollBy(self, amount):
        self.roll -= amount
        self.roll  = self.roll % self.mod

    def __str__(self):
        return str(self.tensor)

def getFastConv(conv):
    kernel = conv.kernel_size
    
    dilation = (1, conv.dilation[1])
    padding = (0, conv.padding[1])
    
    stride = conv.stride
    weight = conv.weight
    bias = conv.bias
    
    inchan, outchan = weight.shape[1], weight.shape[0]
    
    newconv = nn.Conv2d(inchan, outchan, kernel, dilation=dilation, stride=stride, padding=padding)
    newconv.weight = weight
    newconv.bias = bias
    
    return newconv