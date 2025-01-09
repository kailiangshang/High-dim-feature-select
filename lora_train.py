from __future__ import annotations
from data.dataset_pancreas import MetaData, generate_train_test_loader
import torch
from concrete.cls import MLPClassifier
from concrete.cae import ConcreteClassifier
from concrete.ipcae import IndirectConcreteClassifier
from concrete.loracae import LoraConcreteClassifier

import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from loguru import logger
import numpy as np
import pandas as pd

class Trainer:
    
    def __init__(self, k_feature: int, input_dim: int, hidden_dim_cls: int, embedding_dim: int, meta_data: MetaData, device, epochs=200):
        logger.info(f'k_feature: {k_feature}, input_dim: {input_dim}, hidden_dim_cls: {hidden_dim_cls}, embedding_dim: {embedding_dim}, epochs: {epochs}')
        
        self.model_dict = {
            'org_cls': MLPClassifier(
                input_dim=input_dim, hidden_dim=hidden_dim_cls, output_dim=meta_data.cls_num
                ).to(device),
            'concrete_cls': ConcreteClassifier(
                input_dim=input_dim, hidden_dim_cls=hidden_dim_cls, output_dim=meta_data.cls_num, k_feature_select=k_feature, total_epochs=epochs
                ).to(device),
            'indirect_concrete_cls': IndirectConcreteClassifier(
                input_dim=input_dim, hidden_dim_cls=hidden_dim_cls, output_dim=meta_data.cls_num, k_feature_select=k_feature, total_epochs=epochs, embedding_dim=embedding_dim
                ).to(device),
            # 'lora_cae_cls': LoraConcreteClassifier(
            #     input_dim=input_dim, hidden_dim_cls=hidden_dim_cls, output_dim=meta_data.cls_num, k_feature_select=k_feature, total_epochs=epochs, rank=embedding_dim
            # ).to(device)
        }
        model_params = {
            model_name: model.train_parameters_number for model_name, model in self.model_dict.items()
        }
        
        logger.info(f'Model parameters: {model_params}')
        self.train_name = f'{meta_data.name}-k_{k_feature}-h_{hidden_dim_cls}-e_{embedding_dim}-ep_{epochs}-random_{meta_data.random_state}'
        self.train_loader, self.test_loader = meta_data.train_loader, meta_data.test_loader
        self.optimizers = {
            'org_cls': torch.optim.Adam(self.model_dict['org_cls'].parameters(), lr=1e-3),
            'concrete_cls': torch.optim.Adam(self.model_dict['concrete_cls'].parameters(), lr=1e-3),
            'indirect_concrete_cls': torch.optim.Adam(self.model_dict['indirect_concrete_cls'].parameters(), lr=1e-3),
            # 'lora_cae_cls': torch.optim.Adam(self.model_dict['lora_cae_cls'].parameters(), lr=1e-3)
        }
        self.device = device
        self.epochs = epochs
        self.result_dict = {
            name: {'train_loss': [], 'test_loss': [], 'test_acc': [], 'report': None} for name in self.model_dict.keys()
        }
    
    def _train_val_org_cls_epoch(self):
        self.model_dict['org_cls'].train()
        running_loss_train = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizers['org_cls'].zero_grad()
            y_pred = self.model_dict['org_cls'](batch.X)
            loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
            loss.backward()
            self.optimizers['org_cls'].step()
            running_loss_train += loss.item()
        avg_train_loss = running_loss_train / len(self.train_loader)
        
        self.model_dict['org_cls'].eval()
        running_loss_test = 0
        correct = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                y_pred = self.model_dict['org_cls'](batch.X, training=False)
                loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
                running_loss_test += loss.item() * batch.X.size(0)
                correct += (y_pred.argmax(1) == batch.obs['cell_type']).sum().item()
                
                # Collect predictions and labels for F1 score calculation
                all_preds.extend(y_pred.argmax(1).cpu().numpy())
                all_labels.extend(batch.obs['cell_type'].cpu().numpy())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        cls_report_frame = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True, zero_division='warn')).transpose()
        cls_report_frame.columns = [f'orgcls_{col}' for col in cls_report_frame.columns]
        avg_test_loss = running_loss_test / len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)

        return avg_train_loss, avg_test_loss, accuracy, cls_report_frame
        
    def _train_val_cae_cls_epoch(self, epoch):
        self.model_dict['concrete_cls'].train()
        self.model_dict['concrete_cls'].update_temperature(epoch)
        running_loss_train = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizers['concrete_cls'].zero_grad()
            y_pred = self.model_dict['concrete_cls'](batch.X)
            loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
            loss.backward()
            self.optimizers['concrete_cls'].step()
            running_loss_train += loss.item()
        avg_train_loss = running_loss_train / len(self.train_loader)

        self.model_dict['concrete_cls'].eval()
        running_loss_test = 0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                y_pred = self.model_dict['concrete_cls'](batch.X)
                loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
                running_loss_test += loss.item() * batch.X.size(0)
                correct += (y_pred.argmax(1) == batch.obs['cell_type']).sum().item()

                # Collect predictions and labels for F1 score calculation
                all_preds.extend(y_pred.argmax(1).cpu().numpy())
                all_labels.extend(batch.obs['cell_type'].cpu().numpy())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        cls_report_frame = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True, zero_division='warn')).transpose()
        cls_report_frame.columns = [f'caecls_{col}' for col in cls_report_frame.columns]
        avg_test_loss = running_loss_test / len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return avg_train_loss, avg_test_loss, accuracy, cls_report_frame

    def _train_val_ipcae_cls_epoch(self, epoch):
        self.model_dict['indirect_concrete_cls'].train()
        self.model_dict['indirect_concrete_cls'].update_temperature(epoch)
        running_loss_train = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizers['indirect_concrete_cls'].zero_grad()
            y_pred = self.model_dict['indirect_concrete_cls'](batch.X)
            loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
            loss.backward()
            self.optimizers['indirect_concrete_cls'].step()
            running_loss_train += loss.item()
        avg_train_loss = running_loss_train / len(self.train_loader)

        self.model_dict['indirect_concrete_cls'].eval()
        running_loss_test = 0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                y_pred = self.model_dict['indirect_concrete_cls'](batch.X)
                loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
                running_loss_test += loss.item() * batch.X.size(0)
                correct += (y_pred.argmax(1) == batch.obs['cell_type']).sum().item()
                all_preds.extend(y_pred.argmax(1).cpu().numpy())
                all_labels.extend(batch.obs['cell_type'].cpu().numpy())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        cls_report_frame = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True, zero_division='warn')).transpose()
        cls_report_frame.columns = [f'ipcaec_{col}' for col in cls_report_frame.columns]
        avg_test_loss = running_loss_test / len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return avg_train_loss, avg_test_loss, accuracy, cls_report_frame
    
    def _train_val_loracae_cls_epoch(self, epoch):
        self.model_dict['lora_cae_cls'].train()
        self.model_dict['lora_cae_cls'].update_temperature(epoch)
        running_loss_train = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizers['lora_cae_cls'].zero_grad()
            y_pred = self.model_dict['lora_cae_cls'](batch.X)
            loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
            loss.backward()
            self.optimizers['lora_cae_cls'].step()
            running_loss_train += loss.item()
        avg_train_loss = running_loss_train / len(self.train_loader)

        self.model_dict['lora_cae_cls'].eval()
        running_loss_test = 0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                y_pred = self.model_dict['lora_cae_cls'](batch.X)
                loss = F.cross_entropy(y_pred, batch.obs['cell_type'])
                running_loss_test += loss.item() * batch.X.size(0)
                correct += (y_pred.argmax(1) == batch.obs['cell_type']).sum().item()
                all_preds.extend(y_pred.argmax(1).cpu().numpy())
                all_labels.extend(batch.obs['cell_type'].cpu().numpy())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        cls_report_frame = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True, zero_division='warn')).transpose()
        cls_report_frame.columns = [f'lora_cae_cls_{col}' for col in cls_report_frame.columns]
        avg_test_loss = running_loss_test / len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return avg_train_loss, avg_test_loss, accuracy, cls_report_frame
    
    def train(self):
        
        os.makedirs(f'./result_cls_{self.train_name}', exist_ok=True)
        
        for epoch in range(self.epochs):
            result_org = self._train_val_org_cls_epoch()
            result_cae = self._train_val_cae_cls_epoch(epoch=epoch)
            result_ipcae = self._train_val_ipcae_cls_epoch(epoch=epoch)
            # result_loracae = self._train_val_loracae_cls_epoch(epoch=epoch)
            
            self.result_dict['org_cls']['train_loss'].append(result_org[0])
            self.result_dict['org_cls']['test_loss'].append(result_org[1])
            self.result_dict['org_cls']['test_acc'].append(result_org[2])
            self.result_dict['org_cls']['report'] = result_org[3]
            self.result_dict['concrete_cls']['train_loss'].append(result_cae[0])
            self.result_dict['concrete_cls']['test_loss'].append(result_cae[1])
            self.result_dict['concrete_cls']['test_acc'].append(result_cae[2])
            self.result_dict['concrete_cls']['report'] = result_cae[3]
            self.result_dict['indirect_concrete_cls']['train_loss'].append(result_ipcae[0])
            self.result_dict['indirect_concrete_cls']['test_loss'].append(result_ipcae[1])
            self.result_dict['indirect_concrete_cls']['test_acc'].append(result_ipcae[2])
            self.result_dict['indirect_concrete_cls']['report'] = result_ipcae[3]
            # self.result_dict['lora_cae_cls']['train_loss'].append(result_loracae[0])
            # self.result_dict['lora_cae_cls']['test_loss'].append(result_loracae[1])
            # self.result_dict['lora_cae_cls']['test_acc'].append(result_loracae[2])
            # self.result_dict['lora_cae_cls']['report'] = result_loracae[3]
            
            self.result_dict['org_cls']['report'].to_csv(f'./result_cls_{self.train_name}/org_cls_report.csv')
            self.result_dict['concrete_cls']['report'].to_csv(f'./result_cls_{self.train_name}/concrete_cls_report.csv')
            self.result_dict['indirect_concrete_cls']['report'].to_csv(f'./result_cls_{self.train_name}/indirect_concrete_cls_report.csv')
            # self.result_dict['lora_cae_cls']['report'].to_csv(f'./result_cls_{self.train_name}/lora_cae_cls_report.csv')
            
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Org Train Loss: {result_org[0]:.4f}, Org Test Loss: {result_org[1]:.4f}, Org Accuracy: {result_org[2]:.4f}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, CAE Train Loss: {result_cae[0]:.4f}, CAE Test Loss: {result_cae[1]:.4f}, CAE Accuracy: {result_cae[2]:.4f}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, iPCAE Train Loss: {result_ipcae[0]:.4f}, iPCAEC Test Loss: {result_ipcae[1]:.4f}, iPCAEC Accuracy: {result_ipcae[2]:.4f}")
            # logger.info(f"Epoch {epoch + 1}/{self.epochs}, LoraCAE Train Loss: {result_loracae[0]:.4f}, LoraCAE Test Loss: {result_loracae[1]:.4f}, LoraCAE Accuracy: {result_loracae[2]:.4f}")
            self.plot_result(epoch=epoch)
    
    def plot_result(self, epoch):
        epochs = range(1, epoch + 2)

        plt.figure(figsize=(15, 10))

        # Training Loss
        plt.subplot(3, 1, 1)
        for model_name in self.model_dict.keys():
            plt.plot(epochs, self.result_dict[model_name]['train_loss'], label=model_name)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Test Loss
        plt.subplot(3, 1, 2)
        for model_name in self.model_dict.keys():
            plt.plot(epochs, self.result_dict[model_name]['test_loss'], label=model_name)
        plt.title('Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Test Accuracy
        plt.subplot(3, 1, 3)
        for model_name in self.model_dict.keys():
            plt.plot(epochs, self.result_dict[model_name]['test_acc'], label=model_name)
        plt.title('Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        
        # Save the figure to a file
        result_dir = f'./result_cls_{self.train_name}'
        os.makedirs(result_dir, exist_ok=True)  # Ensure the directory exists
        plt.savefig(os.path.join(result_dir, 'training_history.png'))
        plt.close()


# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta_data = generate_train_test_loader(
        path="E:/桌面/DFS/test_with_mnist/high-dim-fs/data/pancreas.h5ad",
        batch_size=512,
        device=device,
        
        random_state=42
        )
    for k_select in [70, 80, 90, 100]:
        for embedding_dim in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            trainer = Trainer(
                k_feature=k_select,
                hidden_dim_cls=256,
                epochs=50,
                input_dim=58482,
                embedding_dim=embedding_dim,
                meta_data=meta_data,
                device=device)
            
            trainer.train()
    
    # train_loader, val_loader = generate_train_test_loader(
    #     path="E:/桌面/DFS/test_with_mnist/high-dim-fs/data/pancreas.h5ad",
    #     batch_size=512,
    #     device=device,
    #     random_state=42
    #     )
    # hidden_dim_cls = 256
    # k_feature_select = 10
    # # classifier = MLPClassifier(
    # #     input_dim=58482,
    # #     hidden_dim=256,
    # #     output_dim=15,
    # # ).to(device)
    # # classifier = ConcreteClassifier(
    # #     input_dim=58482,
    # #     k_feature_select=k_feature_select,
    # #     hidden_dim_cls=hidden_dim_cls,
    # #     output_dim=15,
    # # ).to(device)
    # classifier = IndirectConcreteClassifier(
    #     input_dim=58482,
    #     k_feature_select=k_feature_select,
    #     hidden_dim_cls=hidden_dim_cls,
    #     output_dim=15,
    # ).to(device)
    
    # print(f"Model parameters: {classifier.train_parameters_number}")
    # print(f"Model Architecture: {classifier}")
    
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    # # 构建保存文件夹名称
    # save_dir_name = f"results_ipcaecls_input{classifier.input_dim}_hidden{hidden_dim_cls}_output{k_feature_select}_kselectNone"
    # save_dir = os.path.join("results", save_dir_name)

    # trained_model = train(classifier, train_loader, val_loader, optimizer, device, epochs=200, save_dir=save_dir)