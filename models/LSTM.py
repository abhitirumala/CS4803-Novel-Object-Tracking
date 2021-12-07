import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
LSTM Network to forecast the next value of a multivariate time series of 4 variables
"""
class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x