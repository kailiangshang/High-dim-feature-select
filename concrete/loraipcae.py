import torch.nn as nn



class LoraIndirectlyConcreteAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, device):
        ...

    def forward(self, x):
        ...

    def loss(self, prediction, target):
        ...
    
    