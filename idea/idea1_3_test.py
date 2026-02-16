import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import os
import cv2
import numpy as np

# ==========================================
# 配置
# ==========================================
DATA_ROOT = "/root/asym_pinn_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
SEQ_LEN = 5


# ==========================================
# 1. 数据集
# ==========================================
class AsymPINNDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        npy_path = os.path.join(self.root_dir, row['filename'])

        stack_gray = np.load(npy_path)
        stack_tensor = torch.from_numpy(stack_gray).float() / 255.0
        stack_tensor = (stack_tensor - 0.5) / 0.5

        kappa = row['kappa']
        phi = row['phi']

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # 标签：回归 kappa 和 phi 的三角函数
        labels = torch.tensor([kappa, cos_phi, sin_phi], dtype=torch.float32)

        return stack_tensor, labels


# ==========================================
# 2. 网络 (ResNet-18)
# ==========================================
class StackedResNet(nn.Module):
    def __init__(self):
        super(StackedResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        # 修改第一层适应 5 帧堆叠
        old_conv = resnet.conv1
        self.new_conv = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.new_conv.weight[:] = torch.mean(old_conv.weight, 1, keepdim=True).repeat(1, 5, 1, 1)
        resnet.conv1 = self.new_conv

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # [kappa, cos, sin]
        )

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        output = self.fc(feat)
        return output


# ==========================================
# 3. 物理 Loss (加权)
# ==========================================
class WeightedPhysicsLoss(nn.Module):
    def __init__(self):
        super(WeightedPhysicsLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # 1. 曲率损失 (放大 100 倍，因为数值小)
        loss_kappa = self.mse(preds[:, 0], targets[:, 0]) * 100.0

        # 2. 方向损失 (Cos/Sin)
        # 这次因为有大圆点作为特征，网络能看见方向了，所以给正常权重
        loss_dir = self.mse(preds[:, 1:], targets[:, 1:]) * 10.0

        # 3. 几何约束 (模长=1)
        pred_norm = torch.norm(preds[:, 1:], dim=1)
        loss_norm = torch.mean((pred_norm - 1.0) ** 2)

        total = loss_kappa + loss_dir + loss_norm
        return total, loss_kappa.item(), loss_dir.item()


# ==========================================
# 4. 训练
# ==========================================
def train_asym_pinn():
    if not os.path.exists(CSV_PATH):
        print("❌ 数据集不存在")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training Asymmetric PINN on: {device}")

    dataset = AsymPINNDataset(CSV_PATH, DATA_ROOT)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = StackedResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = WeightedPhysicsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    best_loss = float('inf')

    print(f"{'Phase':<6} | {'Total':<8} | {'Kappa':<8} | {'Dir MSE':<8}")

    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            run_loss = 0.0
            rk = 0.0
            rd = 0.0

            for stack, labels in dataloader:
                stack = stack.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(stack)
                    loss, lk, ld = criterion(out, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                run_loss += loss.item() * stack.size(0)
                rk += lk * stack.size(0)
                rd += ld * stack.size(0)

            epoch_loss = run_loss / len(dataloader.dataset)
            ek = rk / len(dataloader.dataset)
            ed = rd / len(dataloader.dataset)

            if phase == 'val':
                print(f'{phase:<6} | {epoch_loss:.4f}   | {ek:.4f}   | {ed:.4f}')
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_asym_pinn.pth')
                    print("  --> 🌟 Best Model Saved")


if __name__ == "__main__":
    train_asym_pinn()