import torch
from torch import nn

class DyT(nn.Module):
    def __init__(self, num_features = 0, alpha_init_value=0.5):
        super().__init__()
        # assert num_features==0, 'Number of features cannot be 0'
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
