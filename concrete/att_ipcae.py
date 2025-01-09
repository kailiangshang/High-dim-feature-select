import torch
import torch.nn as nn
from torch.nn import functional as F
from .cls import MLPClassifier


class IndirectConcreteLayerWithNNAttention(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=100, num_heads=8, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(IndirectConcreteLayerWithNNAttention, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_epochs = total_epochs
        self.temperature = torch.tensor(initial_temperature, device=device)
        self.psi = nn.Parameter(torch.randn(output_dim, embedding_dim, device=device))  # (k_select, emb_dim)
        self.W = nn.Parameter(torch.randn(input_dim, embedding_dim, device=device))     # (n_feat, emb_dim)
        self.b = nn.Parameter(torch.randn(input_dim, device=device))                    # (n_feat)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, device=device)
        self.device = device

    def forward(self, x, training=True):
        # Compute embeddings for each feature
        W = self.W.unsqueeze(0).expand(x.size(0), -1, -1)  # [batch_size, input_dim, embedding_dim]
        
        # Apply MultiheadAttention to compute attention weights
        att_output, _ = self.attention(W, W, W)  # [batch_size, input_dim, embedding_dim]

        # Adjust embeddings with attention output
        W_att = att_output + W  # Residual connection

        # Compute logits using the adjusted embeddings
        logits = self.get_logits(W_att)

        if training:
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            y = F.softmax((logits + gumbel_noise) / self.temperature, dim=0)
        else:
            y = F.one_hot(torch.argmax(logits, dim=0), self.input_dim).T.float()
        return torch.matmul(x, y)

    def get_logits(self, W_att):
        # Compute logits using the adjusted embeddings
        logits = torch.matmul(self.psi, W_att.transpose(1, 2)).transpose(1, 2)  # [batch_size, n_feat, k_select]
        return logits

    def update_temperature(self, epoch):
        b = epoch
        B = self.total_epochs
        T_0 = self.initial_temperature
        T_B = self.final_temperature
        print(f'b: {b}, B: {B}, T_0: {T_0}, T_B: {T_B}')
        new_temperature = T_0 * (T_B / T_0) ** (b / B)
        self.temperature = torch.tensor(new_temperature, device=self.device)

    def get_prob(self):
        logits = self.get_logits(self.W.unsqueeze(0))
        return F.softmax(logits, dim=1)

    def get_mean_max_prob(self):
        p = torch.mean(torch.max(self.get_prob(), dim=1)[0])
        return p.item()

    def get_hard_selection(self):
        logits = self.get_logits(self.W.unsqueeze(0))
        return F.one_hot(torch.argmax(logits, dim=1), self.input_dim).float()
    
    @property
    def train_parameters_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IndirectConcreteClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, k_feature_select, hidden_dim_cls, output_dim, embedding_dim=100, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(IndirectConcreteClassifierWithAttention, self).__init__()
        self.input_dim = input_dim

        self.atten_ipcae_layer = IndirectConcreteLayerWithNNAttention(input_dim, k_feature_select, embedding_dim, 1, initial_temperature, final_temperature, total_epochs, device)
        self.classifier = MLPClassifier(k_feature_select, hidden_dim_cls, output_dim)
        self.device = device

    def forward(self, x, training=True):
        x = self.atten_ipcae_layer(x, training)
        x = self.classifier(x, training)
        return x

    @property
    def train_parameters_number(self):
        return self.atten_ipcae_layer.train_parameters_number + self.classifier.train_parameters_number

    def get_prob(self):
        return self.atten_ipcae_layer.get_prob()

    def get_mean_max_prob(self):
        return self.atten_ipcae_layer.get_mean_max_prob()

    def get_hard_selection(self):
        return self.atten_ipcae_layer.get_hard_selection()

    def update_temperature(self, epoch):
        self.atten_ipcae_layer.update_temperature(epoch)
    
    def get_logits(self):
        return self.atten_ipcae_layer.get_logits()
    
    