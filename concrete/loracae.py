import torch
import torch.nn as nn
import torch.nn.functional as F
from .cls import MLPClassifier


class LoraConcreteLayer(nn.Module):
    def __init__(self, input_dim, k_feature_select, rank=4, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(LoraConcreteLayer, self).__init__()
        self.input_dim = input_dim
        self.k_select = k_feature_select
        self.rank = rank  # 低秩矩阵的秩
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature 
        self.total_epochs = total_epochs
        self.temperature = torch.tensor(initial_temperature, device=device)
        self.device = device
        
        # 初始化低秩矩阵 A 和 B
        self.A = nn.Parameter(torch.randn(input_dim, rank, device=device))  # (input_dim, rank)
        self.B = nn.Parameter(torch.randn(rank, k_feature_select, device=device))   # (rank, k_select)
        
    def forward(self, x, training=True):
        if training:
            # 使用 A * B 替代原始的 logit_scale
            logit_scale = torch.matmul(self.A, self.B)  # (input_dim, k_select)
            u = torch.rand_like(logit_scale, device=self.device)  # (input_dim, k_select)
            gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)  # (input_dim, k_select)
            y = F.softmax((logit_scale + gumbel_noise) / self.temperature, dim=0)  # (input_dim, k_select)
        else:
            # 推理时使用原始的 logit_scale
            logit_scale = torch.matmul(self.A, self.B)  # (input_dim, k_select)
            y = F.one_hot(torch.argmax(logit_scale, dim=0), self.input_dim).T.float()  # (input_dim, k_select)
        return torch.matmul(x, y)

    def update_temperature(self, epoch):
        b = epoch
        B = self.total_epochs
        T_0 = self.initial_temperature
        T_B = self.final_temperature
        new_temperature = T_0 * (T_B / T_0) ** (b / B)
        self.temperature = torch.tensor(new_temperature, device=self.device)

    def get_prob(self):
        logit_scale = torch.matmul(self.A, self.B)  # (input_dim, k_select)
        return F.softmax(logit_scale, dim=0)

    def get_mean_max_prob(self):
        p = torch.mean(torch.max(self.get_prob(), dim=0)[0])
        return p.item()

    def get_hard_selection(self):
        logit_scale = torch.matmul(self.A, self.B)  # (input_dim, k_select)
        return F.one_hot(torch.argmax(logit_scale, dim=0), self.input_dim).T.float()
    
    @property
    def train_parameters_number(self):
        # 获取所有训练时需要更新梯度的参数量
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoraConcreteClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_cls, output_dim, k_feature_select, rank=4, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None):
        super(LoraConcreteClassifier, self).__init__()
        self.device = device
        self.input_dim = input_dim
                
        # LoraConcreteLayer 选择重要的特征
        self.lora_layer = LoraConcreteLayer(input_dim, k_feature_select, rank, initial_temperature, final_temperature, total_epochs, device)
        self.cls_layer = MLPClassifier(k_feature_select, hidden_dim_cls, output_dim)

    def forward(self, x, training=True):
        # 通过 LoraConcreteLayer
        x = self.lora_layer(x, training=training)
        x = self.cls_layer(x, training=training)        
        return x

    def update_temperature(self, epoch):
        self.lora_layer.update_temperature(epoch)

    def get_prob(self):
        return self.lora_layer.get_prob()

    def get_mean_max_prob(self):
        return self.lora_layer.get_mean_max_prob()

    def get_hard_selection(self):
        return self.lora_layer.get_hard_selection()
    
    @property
    def train_parameters_number(self):
        # 获取所有训练时需要更新梯度的参数量
        return sum(p.numel() for p in self.parameters() if p.requires_grad)