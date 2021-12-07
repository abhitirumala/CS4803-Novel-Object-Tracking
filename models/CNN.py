import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNForecaster(nn.Module):
    
    def __init__(self, input_shape, output_shape, num_layers, dropout):
        super(CNNForecaster, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Linear(input_shape[0] * input_shape[1], input_shape[1] * 32)
        
        self.conv1 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv_layers = []
        
        for i in range(1, num_layers):
            self.conv_layers.append(nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1))
        
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
        self.fc = nn.Linear(32 * input_shape[1], output_shape[0] * output_shape[1])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        x = x.flatten(start_dim=1)
        embedded_x = self.embedding(x)
        embedded_x = embedded_x.view(embedded_x.size(0), self.input_shape[1], 32)
        embedded_x = embedded_x.permute(0, 2, 1)
        
        output = self.max_pool(F.relu(self.conv1(embedded_x)))
        for layer in self.conv_layers:
            output = self.max_pool(F.relu(layer(output)))
        
        output = output.flatten(start_dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

    def reshape_output(self, output):
        """
        Reshapes the output to the original shape
        """
        output = output.view(output.size(0), self.output_shape[0], self.output_shape[1])
        return output
    
if __name__ == "__main__":
    """
    Initialize the model
    
    create a model with the following parameters:
    input shape: (2, 8)
    output shape: (2, 8)
    num_layers: 4
    """
    model = CNNForecaster((2, 8), (2, 8), 4, 0.5)
    test_tensor = torch.randn(32, 2, 8)
    
    output = model(test_tensor)
    output = model.reshape_output(output)
    print(output.shape)
        
        
            