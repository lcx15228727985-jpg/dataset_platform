import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 配置
# ==========================================
DATA_ROOT = "/root/unrolled_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4


# ==========================================
# 1. 数据集
# ==========================================
class UnrolledDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])

        # 读取单通道灰度图
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        # 标签：末端坐标 (归一化 /100mm)
        label = torch.tensor([row['tx'] / 100.0, row['ty'] / 100.0, row['tz'] / 100.0], dtype=torch.float32)

        return img, label


# ==========================================
# 2. 网络 (输入通道=1)
# ==========================================
class MapResNet(nn.Module):
    def __init__(self):
        super(MapResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        # 修改第一层：1通道输入
        old_conv = resnet.conv1
        self.new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 取 RGB 均值初始化
        with torch.no_grad():
            self.new_conv.weight[:] = torch.mean(old_conv.weight, 1, keepdim=True)
        resnet.conv1 = self.new_conv

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # x, y, z
        )

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return self.fc(x)


# ==========================================
# 3. 训练
# ==========================================
def train_unrolled():
    if not os.path.exists(CSV_PATH):
        print("❌ 数据集不存在")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on Unrolled Maps: {device}")

    transform = transforms.Compose([
        transforms.Resize((200, 360)),  # 保持生成时的尺寸
        transforms.ToTensor(),
        # 不需要 ImageNet 的 normalization，因为我们是灰度图
    ])

    dataset = UnrolledDataset(CSV_PATH, DATA_ROOT, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MapResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    best_loss = float('inf')

    print(f"{'Phase':<6} | {'Loss':<10} | {'Error (mm)':<12}")

    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            run_loss = 0.0

            for img, label in dataloader:
                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(img)
                    loss = criterion(out, label)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                run_loss += loss.item() * img.size(0)

            epoch_loss = run_loss / len(dataloader.dataset)
            mm_error = (epoch_loss ** 0.5) * 100.0

            if phase == 'val':
                print(f'{phase:<6} | {epoch_loss:.6f}   | {mm_error:.2f} mm')
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_unrolled_model.pth')
                    print("  --> 🌟 Best Model Saved")


if __name__ == "__main__":
    train_unrolled()