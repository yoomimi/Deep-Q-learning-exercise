import torch
import torch.nn as nn
import numpy as np

# Model
# MultiLayerPerceptron
MIN = torch.Tensor(np.array([0,0]))
MAX = torch.Tensor(np.array([4,9]))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(2, 200, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            )
        
        # output layer
        self.linear_out = nn.Linear(200,4)
        self.linear_out.weight = nn.Parameter(torch.zeros(4,200)) # weight initialization
        self.linear_out.bias = nn.Parameter(torch.zeros(4)) # bias initialization

    def forward(self, x):
        # forward pass
        x = (x - MIN)/(MAX-MIN) # input normalization
        x = self.fc_layers(x) # fully connected layers
        x = self.linear_out(x) # output layer
        return x
    
    