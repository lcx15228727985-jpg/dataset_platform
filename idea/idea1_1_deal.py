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
from PIL import Image
import torch.nn.functional as F

# ==========================================
# 配置区域
# ==========================================
DATA_ROOT = "/root/pinn_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
IMG_DIR = os.path.join(DATA_ROOT, "images")

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4


# ==========================================
# 1. 数据集
# ==========================================
class PINNDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])

        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError(f"无法找到图片: {img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        kappa = self.data.iloc[idx]['kappa']
        phi = self.data.iloc[idx]['phi']
        u_pos = self.data.iloc[idx]['u_position']

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        labels = torch.tensor([kappa, cos_phi, sin_phi], dtype=torch.float32)
        physics_params = torch.tensor([u_pos], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels, physics_params


# ==========================================
# 2. 模型
# ==========================================
class RobotPINN(nn.Module):
    def __init__(self):
        super(RobotPINN, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [kappa, cos, sin]
        )

    def forward(self, x):
        return self.backbone(x)


# ==========================================
# 3. 改进的损失函数 (余弦损失 + 动态权重)
# ==========================================
class CurriculumPhysicsLoss(nn.Module):
    def __init__(self):
        super(CurriculumPhysicsLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets, u_pos, w_kappa, w_dir):
        # preds: [batch, 3] -> [kappa, cos, sin]

        # --- 1. 曲率损失 (MSE) ---
        loss_kappa = self.mse(preds[:, 0], targets[:, 0])

        # --- 2. 方向损失 (Cosine Embedding Loss) ---
        # 核心修改：使用余弦相似度，而不是 MSE
        pred_vec = preds[:, 1:]
        true_vec = targets[:, 1:]

        # Cosine Similarity: 1 表示方向相同，-1 表示相反
        # 我们希望 maximize similarity，所以 minimize (1 - sim)
        # F.cosine_similarity 会自动处理归一化，所以模型不需要输出单位向量
        cos_sim = F.cosine_similarity(pred_vec, true_vec, dim=1)
        loss_dir = torch.mean(1.0 - cos_sim)

        # --- 3. 几何约束 ---
        # 仍然希望模型输出的模长接近 1，避免梯度爆炸
        pred_norm = torch.norm(pred_vec, dim=1)
        loss_norm = torch.mean((pred_norm - 1.0) ** 2)

        # 动态加权
        # 注意：这里 Kappa 不乘 100 了，因为我们用课程学习控制它
        total_loss = (loss_kappa * w_kappa * 1000.0) + (loss_dir * w_dir) + (loss_norm * 0.1)

        return total_loss, loss_kappa.item(), loss_dir.item()


# ==========================================
# 4. 训练主程序 (带课程学习)
# ==========================================
def train_pinn_cosine():
    if not os.path.exists(CSV_PATH):
        print(f"错误: 找不到 {CSV_PATH}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on: {device}")
    print("⚡ 策略: 余弦损失 + 课程学习 (前5轮只练方向)")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    full_dataset = PINNDataset(CSV_PATH, IMG_DIR, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = RobotPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CurriculumPhysicsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # === 课程学习调度器 ===
        if epoch < 5:
            # 第一阶段：只看方向，完全忽略曲率
            # 强迫模型从图像中找角度特征
            w_kappa = 0.0
            w_dir = 10.0
            phase_msg = "Focus: DIRECTION"
        else:
            # 第二阶段：方向 + 曲率
            w_kappa = 1.0
            w_dir = 10.0
            phase_msg = "Focus: GLOBAL"

        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS} [{phase_msg}]')
        print('-' * 70)
        print(f"{'Phase':<6} | {'Total Loss':<10} | {'Kappa MSE':<12} | {'Dir CosLoss':<12}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_kappa = 0.0
            running_dir = 0.0

            for imgs, targets, u_pos in dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                u_pos = u_pos.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    loss, l_k, l_d = criterion(outputs, targets, u_pos, w_kappa, w_dir)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                running_kappa += l_k * imgs.size(0)  # 记录原始MSE
                running_dir += l_d * imgs.size(0)  # 记录原始CosLoss

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_kappa = running_kappa / len(dataloader.dataset)
            epoch_dir = running_dir / len(dataloader.dataset)

            # Dir CosLoss 说明:
            # 1.0 = 完全反向 (最差)
            # 0.5 = 垂直/瞎猜
            # 0.0 = 完全同向 (完美)
            print(f'{phase:<6} | {epoch_loss:.4f}     | {epoch_kappa:.6f}     | {epoch_dir:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_pinn_cosine.pth')
                    print("  --> 🌟 Best Model Saved")

    print(f"\n✅ 训练完成!")


if __name__ == "__main__":
    train_pinn_cosine()