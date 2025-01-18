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



class AdaptiveGrowthPruningLayer(IndirectConcreteLayer):
    def __init__(self, input_dim, output_dim, embedding_dim=100, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None, growth_threshold=0.95, prune_threshold=0.05):
        super().__init__(input_dim, output_dim, embedding_dim, initial_temperature, final_temperature, total_epochs, device)
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold

    def adapt(self, performance_metric, epoch):
        # Growth for k_select (output_dim)
        if performance_metric < self.growth_threshold and self.psi.size(0) < self.input_dim:
            new_output_dim = self.psi.size(0) + 1
            new_psi_cat = nn.Parameter(torch.randn(1, self.embedding_dim, requires_grad=True, device=self.device))
            new_psi = nn.Parameter(torch.cat((self.psi.data, new_psi_cat)))
            
            self.psi = new_psi
        
        # Pruning for k_select (output_dim)
        importance_scores = torch.max(self.get_prob(), dim=0)[0]        
        
        keep_mask = importance_scores > self.prune_threshold
        if torch.sum(keep_mask) < self.psi.size(0):  # Only prune if there are features to remove
            new_psi = nn.Parameter(self.psi[keep_mask])
            self.psi = new_psi

    def forward(self, x, training=True):
        logits = self.get_logits()
        logits = logits.to(self.device)
        if training:
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            y = F.softmax((logits + gumbel_noise) / self.temperature, dim=0)
        else:
            y = F.one_hot(torch.argmax(logits, dim=0), self.input_dim).T.float()
        return torch.matmul(x, y)


class IndirectConcreteClassifier(nn.Module):
    def __init__(self, input_dim, k_feature_select, hidden_dim_cls, output_dim, embedding_dim=100, initial_temperature=1.0, final_temperature=0.01, total_epochs=200, device=None, growth_threshold=0.95, prune_threshold=0.05):
        super().__init__()
        self.indirect_layer = AdaptiveGrowthPruningLayer(input_dim, k_feature_select, embedding_dim, initial_temperature, final_temperature, total_epochs, device, growth_threshold, prune_threshold)
        self.mlp = MLPClassifier(k_feature_select, hidden_dim_cls, output_dim)
        self.device = device

    def forward(self, x, training=True):
        x = self.indirect_layer(x, training)
        x = self.mlp(x)
        return x

    def adapt(self, performance_metric, epoch):
        # if epoch <= 30:
        #     pass
        # else:
        #     old_output_dim = self.indirect_layer.psi.size(0)
        #     self.indirect_layer.adapt(performance_metric, epoch)
        #     new_output_dim = self.indirect_layer.psi.size(0)

        #     if old_output_dim != new_output_dim:
        #         # 动态调整MLPClassifier的输入维度
        #         self._adapt_mlp(new_output_dim)
        old_output_dim = self.indirect_layer.psi.size(0)
        self.indirect_layer.adapt(performance_metric, epoch)
        new_output_dim = self.indirect_layer.psi.size(0)

        if old_output_dim != new_output_dim:
            # 动态调整MLPClassifier的输入维度
            self._adapt_mlp(new_output_dim)

        
    def _adapt_mlp(self, new_input_dim):
        old_input_dim = self.mlp.input_dim
        new_mlp = MLPClassifier(new_input_dim, self.mlp.hidden_dim, self.mlp.output_dim).to(self.device)
        
        # 如果新维度大于旧维度，则复制旧参数并随机初始化新增部分
        if new_input_dim > old_input_dim:
            with torch.no_grad():
                new_mlp.fc1.weight = nn.Parameter(
                    torch.cat([self.mlp.fc1.weight.data, torch.randn(self.mlp.hidden_dim, new_input_dim - old_input_dim, device=self.device)], dim=1)
                )
                # new_mlp.fc1.bias = nn.Parameter(
                #     torch.cat([self.mlp.fc1.bias.data, torch.randn(new_input_dim - old_input_dim, device=self.device)])
                #     )
                # print(new_mlp.fc1.weight.data.size())
                # print(new_mlp.fc1.bias.data.size())
        # 如果新维度小于旧维度，则基于重要性得分保留最重要的特征
        elif new_input_dim < old_input_dim:
            importance_scores = self.indirect_layer.get_prob().mean(dim=0)
            _, indices = torch.topk(importance_scores, new_input_dim)
            with torch.no_grad():
                new_mlp.fc1.weight = nn.Parameter(self.mlp.fc1.weight[indices].data)
                new_mlp.fc1.bias = nn.Parameter(self.mlp.fc1.bias[indices].data)
        else:
            # 维度不变时无需操作
            pass
         
        self.mlp = new_mlp
        
    @property
    def train_parameters_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def now_feature(self):
        return self.indirect_layer.psi.size(0)
    
    def update_temperature(self, epoch):
        self.indirect_layer.update_temperature(epoch)
        