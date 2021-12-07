import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Transformer Network to forecast the next value of a multivariate time series of 4 variables
"""

class Transformer(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        
        return x