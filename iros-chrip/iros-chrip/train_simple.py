import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import torch.nn.functional as F
import timm

# 1. Dataset (通用)
class UltrasoundDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = torch.load(data_path)
        self.images = data['images']
        self.labels = data['labels']
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

# 2. Model (HRNet)
class PoseHRNet(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        # 尝试加载预训练，失败则不用
        try:
            self.backbone = timm.create_model('hrnet_w32', pretrained=True, in_chans=1, num_classes=0, global_pool='avg')
        except:
            self.backbone = timm.create_model('hrnet_w32', pretrained=False, in_chans=1, num_classes=0, global_pool='avg')
        self.fc = nn.Linear(self.backbone.num_features, output_dim + 3) # +3 for uncertainty
    def forward(self, x):
        return self.fc(self.backbone(x))

# 3. Loss (回归正常权重)
class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        val_pred = pred[:, :8]
        log_vars = pred[:, 8:]
        
        # Z轴损失
        l_trans = F.smooth_l1_loss(val_pred[:, 0], target[:, 0])
        
        # 旋转损失 (Theta 为主，Yaw/Pitch 应该是 0)
        l_theta = (val_pred[:, 1] - target[:, 1])**2 + (val_pred[:, 2] - target[:, 2])**2
        # Yaw/Pitch 虽然是0，但也算进去，确保网络输出0
        l_yaw   = (val_pred[:, 3] - target[:, 3])**2
        l_pitch = (val_pred[:, 4] - target[:, 4])**2
        
        # [修改] 移除之前的 x10 权重，因为现在没有复杂的几何畸变
        l_rot = l_theta.mean() + l_yaw.mean() + l_pitch.mean()
        
        # 曲率损失
        l_curv = F.mse_loss(val_pred[:, 5], target[:, 5])
        
        # 不确定性加权
        s_trans, s_rot, s_curv = log_vars[:, 0].mean(), log_vars[:, 1].mean(), log_vars[:, 2].mean()
        loss = 0.5 * torch.exp(-s_trans) * l_trans + 0.5 * s_trans + \
               0.5 * torch.exp(-s_rot)   * l_rot   + 0.5 * s_rot + \
               0.5 * torch.exp(-s_curv)  * l_curv  + 0.5 * s_curv
        
        # 单独返回 Theta Loss 供观察
        return loss, l_trans.item(), l_rot.item(), l_curv.item(), l_theta.mean().item()

# 4. 主训练
def train_simple(epochs=30, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 3-DoF 数据集
    train_path = "dataset/train_data_3dof.pt"
    val_path = "dataset/val_data_3dof.pt"
    
    train_loader = DataLoader(UltrasoundDataset(train_path), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(UltrasoundDataset(val_path), batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = PoseHRNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # 弱正则化即可
    criterion = SimpleLoss()
    
    print("🚀 开始 3-DoF 验证实验 (Z, Theta, Curvature)...")
    
    history = {'train_z': [], 'val_z': [], 'train_theta': [], 'val_theta': []}
    
    for epoch in range(epochs):
        model.train()
        rz, rth = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Resize
            imgs = F.interpolate(imgs, size=(96, 128), mode='bilinear', align_corners=False)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss, l_z, l_rot, l_k, l_th_val = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            rz += l_z * imgs.size(0)
            rth += l_th_val * imgs.size(0)
            
        train_z = rz / len(train_loader.dataset)
        train_th = rth / len(train_loader.dataset)
        
        # Validation
        model.eval()
        vz, vth = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                imgs = F.interpolate(imgs, size=(96, 128), mode='bilinear', align_corners=False)
                preds = model(imgs)
                _, l_z, _, _, l_th_val = criterion(preds, labels)
                vz += l_z * imgs.size(0)
                vth += l_th_val * imgs.size(0)
        
        val_z = vz / len(val_loader.dataset)
        val_th = vth / len(val_loader.dataset)
        
        history['train_z'].append(train_z)
        history['val_z'].append(val_z)
        history['train_theta'].append(train_th)
        history['val_theta'].append(val_th)
        
        print(f"Ep {epoch+1}: Z_loss={train_z:.4f}/{val_z:.4f} | Theta_loss={train_th:.4f}/{val_th:.4f}")
        
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_z'], label='Train Z')
    plt.plot(history['val_z'], label='Val Z')
    plt.title('Z-Axis Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_theta'], label='Train Theta')
    plt.plot(history['val_theta'], label='Val Theta')
    plt.title('Theta Loss')
    plt.legend()
    
    plt.savefig('simple_experiment_result.png')
    print("✅ 实验完成。请查看 simple_experiment_result.png")

if __name__ == "__main__":
    train_simple()