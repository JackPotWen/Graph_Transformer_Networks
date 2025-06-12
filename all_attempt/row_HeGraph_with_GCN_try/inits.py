import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def glorot(input_adj, num_nodes=None):
    """Initialize a parameter according to the Glorot initialization."""
    init_range = math.sqrt(6.0/(input_adj.size(0)+input_adj.size(1)))
    input_adj.data.uniform_(-init_range, init_range)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0) 