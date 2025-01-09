import torch
import torch.nn as nn
import torch.nn.functional as F

from .cls import MLPClassifier


class ConcreteLayer(nn.Module):
    def __init__(self, input_dim, output_dim, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(ConcreteLayer, self).__init__()
        self.input_dim = input_dim
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature 
        self.total_epochs = total_epochs
        self.temperature = torch.tensor(initial_temperature, device=device)
        self.logit_scale = nn.Parameter(torch.randn(input_dim, output_dim)) # (n_feat, k_selection) 
        self.device = device

    def forward(self, x, training=True):
        if training:
            u = torch.rand_like(self.logit_scale) # (n_feat, k_select) 
            gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10) # (n_feat, k_select) 
            y = F.softmax((self.logit_scale + gumbel_noise) / self.temperature, dim=0) # (n_feat, k_select) 
        else:
            y = F.one_hot(torch.argmax(self.logit_scale, dim=0), self.input_dim).T.float() # (n_feat, k_select) 
        return torch.matmul(x, y)
            
    def update_temperature(self, epoch):
        b = epoch
        B = self.total_epochs
        T_0 = self.initial_temperature
        T_B = self.final_temperature
        new_temperature = T_0 * (T_B / T_0) ** (b / B)
        self.temperature = torch.tensor(new_temperature, device=self.device)

    def get_prob(self):
        return F.softmax(self.logit_scale, dim=0)

    def get_mean_max_prob(self):
        p = torch.mean(torch.max(self.get_prob(), dim=0)[0])
        return p.item()

    def get_hard_selection(self):
        return F.one_hot(torch.argmax(self.logit_scale, dim=0), self.input_dim).T.float()
    
    @property
    def train_parameters_number(self):
        # 获取所有训练时需要更新梯度的参数量
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConcreteClassifier(nn.Module):
    def __init__(self, input_dim, k_feature_select, hidden_dim_cls, output_dim, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(ConcreteClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_cls
        self.output_dim = output_dim
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_epochs = total_epochs
        self.temperature = torch.tensor(initial_temperature, device=device)
        self.concrete_layer = ConcreteLayer(input_dim, k_feature_select, initial_temperature, final_temperature, total_epochs, device)
        self.mlp = MLPClassifier(k_feature_select, hidden_dim_cls, output_dim)
        self.device = device

    def forward(self, x, training=True):
        x = self.concrete_layer(x, training)
        x = self.mlp(x)
        return x
    
    @property
    def train_parameters_number(self):
        # 获取所有训练时需要更新梯度的参数量
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_prob(self):
        return self.concrete_layer.get_prob()
    
    def get_mean_max_prob(self):
        return self.concrete_layer.get_mean_max_prob()

    def get_hard_selection(self):
        return self.concrete_layer.get_hard_selection()
    
    def update_temperature(self, epoch):
        self.concrete_layer.update_temperature(epoch)