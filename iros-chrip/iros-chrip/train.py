import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import torch.nn.functional as F
import timm  # <--- 引入 timm

# ==========================================
# 1. 数据集定义 (保持不变)
# ==========================================
class UltrasoundDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        print(f"Loading data from {data_path}...")
        data = torch.load(data_path)
        self.images = data['images'] 
        self.labels = data['labels'] 
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ==========================================
# 2. 模型定义: PoseHRNet (使用 timm)
# ==========================================
class PoseHRNet(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        # 使用 timm 创建 HRNet-W32
        # in_chans=1: 自动将第一层卷积改为单通道，并适配预训练权重
        # num_classes=0: 移除原本的分类全连接层 (FC)
        # global_pool='avg': 添加全局平均池化层，使输出变为特征向量
        self.backbone = timm.create_model(
            'hrnet_w32', 
            pretrained=True, 
            in_chans=1, 
            num_classes=0, 
            global_pool='avg'
        )
        
        # 获取 Backbone 输出的特征维度
        # HRNet-W32 的输出特征维度通常是 2048 (Head融合后)
        self.num_features = self.backbone.num_features
        
        # 回归头 (Regression Head)
        # 输出: 8个位姿参数 + 3个不确定性权重 (s_trans, s_rot, s_curv)
        self.fc = nn.Linear(self.num_features, output_dim + 3)
        
    def forward(self, x):
        # 提取特征 [B, num_features]
        features = self.backbone(x)
        # 回归预测 [B, 11]
        out = self.fc(features)
        return out

# ==========================================
# 3. 模型定义: ResNet-50 (使用 timm 简化版)
# ==========================================
# 既然有了 timm，ResNet 也可以写得更简洁
class PoseResNet50(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50', 
            pretrained=True, 
            in_chans=1, 
            num_classes=0, 
            global_pool='avg'
        )
        self.fc = nn.Linear(self.backbone.num_features, output_dim + 3)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

# ==========================================
# 4. 自动加权损失函数 (保持不变)
# ==========================================
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        """
        pred: [B, 11] -> [Predictions(8) | LogVars(3)]
        target: [B, 8]
        Labels: 0:Z, 1:s_th, 2:c_th, 3:s_yaw, 4:s_pit, 5:K, 6:s_phi, 7:c_phi
        """
        val_pred = pred[:, :8]
        log_vars = pred[:, 8:] # s1, s2, s3
        
        # 1. Translation Loss (Z)
        l_trans = F.smooth_l1_loss(val_pred[:, 0], target[:, 0])
        
        # 2. Rotation Loss
        l_theta = (val_pred[:, 1] - target[:, 1])**2 + (val_pred[:, 2] - target[:, 2])**2
        l_yaw   = (val_pred[:, 3] - target[:, 3])**2
        l_pitch = (val_pred[:, 4] - target[:, 4])**2
        l_phi   = (val_pred[:, 6] - target[:, 6])**2 + (val_pred[:, 7] - target[:, 7])**2
        
        # 加大 Yaw 和 Pitch 的权重，因为它们对应复杂的几何畸变
        l_rot = l_theta.mean() + 2.0*l_yaw.mean() + 2.0*l_pitch.mean() + l_phi.mean()
        
        # 3. Curvature Loss
        l_curv = F.mse_loss(val_pred[:, 5], target[:, 5])
        
        # 4. Uncertainty Weighting
        s_trans = log_vars[:, 0].mean()
        s_rot   = log_vars[:, 1].mean()
        s_curv  = log_vars[:, 2].mean()
        
        loss = 0.5 * torch.exp(-s_trans) * l_trans + 0.5 * s_trans + \
               0.5 * torch.exp(-s_rot)   * l_rot   + 0.5 * s_rot + \
               0.5 * torch.exp(-s_curv)  * l_curv  + 0.5 * s_curv
               
        return loss, l_trans.item(), l_rot.item(), l_curv.item()

# ==========================================
# 5. 主训练流程
# ==========================================
def train_model(model_name="hrnet", epochs=50, batch_size=32, lr=1e-4):
    # 路径设置
    train_path = "dataset/train_data_6dof.pt"
    val_path = "dataset/val_data_6dof.pt"
    save_dir = f"checkpoints/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 数据准备
    if not os.path.exists(train_path):
        print(f"❌ 数据文件未找到: {train_path}. 请先运行 generate_dataset.py")
        return

    train_dataset = UltrasoundDataset(train_path)
    val_dataset = UltrasoundDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. 模型初始化 (现在使用基于 timm 的类)
    print(f"Initializing {model_name} with timm...")
    if model_name == "resnet50":
        model = PoseResNet50().to(device)
    elif model_name == "hrnet":
        model = PoseHRNet().to(device)
    else:
        raise ValueError("Unknown model name")
        
    criterion = MultiTaskUncertaintyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 学习率调度器: 当 val_loss 不下降时减小 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 3. 记录器
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'l_trans': [], 'l_rot': [], 'l_curv': []
    }
    
    print(f"Start training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        r_trans, r_rot, r_curv = 0.0, 0.0, 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss, lt, lr_val, lc = criterion(preds, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            r_trans += lt * imgs.size(0)
            r_rot += lr_val * imgs.size(0)
            r_curv += lc * imgs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        avg_lt = r_trans / len(train_dataset)
        avg_lr = r_rot / len(train_dataset)
        avg_lc = r_curv / len(train_dataset)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                loss, _, _, _ = criterion(preds, labels)
                val_loss += loss.item() * imgs.size(0)
        
        avg_val_loss = val_loss / len(val_dataset)
        scheduler.step(avg_val_loss)
        
        # --- Logging ---
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(avg_val_loss)
        history['l_trans'].append(avg_lt)
        history['l_rot'].append(avg_lr)
        history['l_curv'].append(avg_lc)
        
        # 实时打印
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} (Val: {avg_val_loss:.4f}) | "
              f"Trans: {avg_lt:.4f} Rot: {avg_lr:.4f} Curv: {avg_lc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存 CSV
        df = pd.DataFrame(history)
        df.to_csv(f"{model_name}_result.csv", index=False)
        
        # 保存最佳权重
        if epoch == 0 or avg_val_loss < min(history['val_loss'][:-1]):
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            
    # 4. 绘图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Total Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['l_trans'], label='Trans')
    plt.plot(history['l_curv'], label='Curv')
    plt.title('Component Losses')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['l_rot'], label='Rot')
    plt.title('Rotation Loss')
    
    plt.savefig(f"{model_name}_loss_curve.png")
    print(f"Done! Results saved to {model_name}_result.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认使用 hrnet
    parser.add_argument("--model", type=str, default="hrnet", choices=["resnet50", "hrnet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train_model(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size)