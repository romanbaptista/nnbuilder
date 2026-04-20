# IMPORTS
import torch
import torch.nn as nn

# Neural Network Builder

class NNBuilder(nn.Module):

    # Define initialisation
    def __init__(self):
        # Inherit initialisation from nn.Module
        super().__init__()
        # Set layers
        self.layers = nn.ModuleList()
    
    # REGISTRY

    ACTIVATIONS = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid
    }

    # UTILITY FUNCTIONS

    def get_activation_layer(self, name: str):
        return self.ACTIVATIONS[name]


    # LAYER FUNCTIONS

    def add_dense(self, in_features: int, out_features: int, activation: str = None):
        # Add linear layer
        self.layers.append(nn.Linear(in_features, out_features))
        # Add activation layer
        if activation:
            activation_layer = self.get_activation_layer(activation)
            self.layers.append(activation_layer())

        return self
    
    def add_dropout(self, p: float = 0.5):
        # Add dropout layer
        self.layers.append(nn.Dropout(p))
        
        return self

    # FORWARD FUNCTION

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x