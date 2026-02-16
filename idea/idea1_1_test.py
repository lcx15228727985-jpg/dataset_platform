import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import os
import cv2
import numpy as np
import time
from PIL import Image  # <--- 关键修复：导入 PIL

# ==========================================
# 配置区域
# ==========================================
# 数据集绝对路径
DATA_ROOT = "/root/pinn_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
IMG_DIR = os.path.join(DATA_ROOT, "images")

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4


# ==========================================
# 1. 自定义数据集 (修复 NumPy -> PIL 问题)
# ==========================================
class PINNDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取图片路径
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])

        # OpenCV 读取
        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError(f"无法找到图片: {img_name}")

        # 1. 转为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 关键修复：将 NumPy 数组转换为 PIL Image 对象
        # PyTorch 的 transforms (如 Resize) 期望输入是 PIL Image
        image = Image.fromarray(image)

        # 读取物理标签
        kappa = self.data.iloc[idx]['kappa']
        phi = self.data.iloc[idx]['phi']
        u_pos = self.data.iloc[idx]['u_position']

        # 构建标签
        labels = torch.tensor([kappa, phi], dtype=torch.float32)
        physics_params = torch.tensor([u_pos], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels, physics_params


# ==========================================
# 2. PINN 模型结构 (ResNet-18 Regressor)
# ==========================================
class RobotPINN(nn.Module):
    def __init__(self):
        super(RobotPINN, self).__init__()
        # 加载预训练的 ResNet
        self.backbone = models.resnet18(pretrained=True)

        num_ftrs = self.backbone.fc.in_features

        # 修改全连接层为回归头
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出 [kappa, phi]
        )

    def forward(self, x):
        return self.backbone(x)


# ==========================================
# 3. 物理损失函数 (Physics-Informed Loss)
# ==========================================
class PhysicsLoss(nn.Module):
    def __init__(self, lambda_phy=0.1):
        super(PhysicsLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phy = lambda_phy

    def forward(self, preds, targets, u_pos):
        # --- A. 数据驱动损失 (Data Loss) ---
        pred_kappa = preds[:, 0]
        true_kappa = targets[:, 0]

        pred_phi = preds[:, 1]
        true_phi = targets[:, 1]

        # 给 kappa 更高的权重 (x1000)，因为它的数值很小 (0.001 ~ 0.02)
        # 而 phi 数值很大 (0 ~ 6.28)，不加权重会导致模型只学 phi 忽略 kappa
        loss_kappa = self.mse(pred_kappa, true_kappa) * 1000.0
        loss_phi = self.mse(pred_phi, true_phi)

        loss_data = loss_kappa + loss_phi

        # --- B. 物理约束损失 (Physics Constraint) ---
        # 约束: 曲率 kappa 应该 >= 0
        loss_negative_kappa = torch.mean(torch.relu(-pred_kappa))

        # 总损失
        total_loss = loss_data + self.lambda_phy * loss_negative_kappa

        return total_loss, loss_kappa.item(), loss_phi.item()


# ==========================================
# 4. 训练主程序
# ==========================================
def train_pinn():
    # 检查数据
    if not os.path.exists(CSV_PATH):
        print(f"错误: 找不到元数据文件 {CSV_PATH}")
        print("请先运行数据生成脚本！")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载数据集
    full_dataset = PINNDataset(CSV_PATH, IMG_DIR, transform=transform)

    # 划分训练/验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Windows/Linux 兼容性：如果报错，尝试把 num_workers 改为 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # 初始化模型
    model = RobotPINN().to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 损失函数
    criterion = PhysicsLoss(lambda_phy=1.0)

    # 学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # 训练循环
    best_loss = float('inf')

    print("🚀 开始 PINN 回归训练...")

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_kappa_loss = 0.0

            for imgs, targets, u_pos in dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                u_pos = u_pos.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    loss, l_k, l_p = criterion(outputs, targets, u_pos)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                running_kappa_loss += l_k * imgs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_kappa_loss = running_kappa_loss / len(dataloader.dataset)

            # 打印格式化：总Loss 和 曲率误差
            print(f'{phase:<5} Total Loss: {epoch_loss:.4f} | Kappa MSE (x1000): {epoch_kappa_loss:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_pinn_regressor.pth')
                    print("  --> 🌟 模型已保存 (New Best)")

    print(f"\n✅ 训练完成! Min Val Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_pinn()