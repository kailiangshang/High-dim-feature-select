import torch
import torch.nn as nn
import torch.nn.functional as F
from .cls import MLPClassifier


class IndirectConcreteLayer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=100, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(IndirectConcreteLayer, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_epochs = total_epochs
        self.temperature = torch.tensor(initial_temperature, device=device)
        self.psi = nn.Parameter(torch.randn(output_dim, embedding_dim))    # (k_select, emb_dim)
        self.W = nn.Parameter(torch.randn(input_dim, embedding_dim))     # (n_feat, emb_dim)
        self.b = nn.Parameter(torch.randn(input_dim))                    # (n_feat)
        self.device = device

    def forward(self, x, training=True):
        # Compute logits using the transformation of embeddings
        logits = self.get_logits()
        if training:
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            y = F.softmax((logits + gumbel_noise) / self.temperature, dim=0)
        else:
            y = F.one_hot(torch.argmax(logits, dim=0), self.input_dim).T.float()
        return torch.matmul(x, y)

    def update_temperature(self, epoch):
        b = epoch
        B = self.total_epochs
        T_0 = self.initial_temperature
        T_B = self.final_temperature
        new_temperature = T_0 * (T_B / T_0) ** (b / B)
        self.temperature = torch.tensor(new_temperature, device=self.device)

    def get_logits(self):
        logits = (torch.matmul(self.psi, self.W.T) + self.b).T # (n_feat, k_select)
        return logits

    def get_prob(self):
        return F.softmax(self.get_logits(), dim=0)

    def get_mean_max_prob(self):
        p = torch.mean(torch.max(self.get_prob(), dim=0)[0])
        return p.item()

    def get_hard_selection(self):
        return F.one_hot(torch.argmax(self.get_logits(), dim=0), self.input_dim).T.float()
        
    @property
    def train_parameters_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    


class IndirectConcreteClassifier(nn.Module):
    def __init__(self, input_dim, k_feature_select, hidden_dim_cls, output_dim, embedding_dim=100, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(IndirectConcreteClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_cls
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_epochs = total_epochs
        self.temperature = torch.tensor(initial_temperature, device=device)
        self.indirect_layer = IndirectConcreteLayer(input_dim, k_feature_select, embedding_dim, initial_temperature, final_temperature, total_epochs, device)
        self.mlp = MLPClassifier(k_feature_select, hidden_dim_cls, output_dim)
        
        self.device = device

    def forward(self, x, training=True):
        x = self.indirect_layer(x, training)
        x = self.mlp(x)
        return x
    
    @property
    def train_parameters_number(self):
        return self.indirect_layer.train_parameters_number + self.mlp.train_parameters_number
    
    def get_prob(self):
        return self.indirect_layer.get_prob()
    
    def get_mean_max_prob(self):
        return self.indirect_layer.get_mean_max_prob()

    def get_hard_selection(self):
        return self.indirect_layer.get_hard_selection()
    
    def update_temperature(self, epoch):
        self.indirect_layer.update_temperature(epoch)
    