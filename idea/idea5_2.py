import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import os
from PIL import Image
import numpy as np

# ==========================================
# 配置
# ==========================================
DATA_ROOT = "/root/cross_helix_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4


# ==========================================
# 数据集
# ==========================================
class HelixDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        img = Image.open(img_path).convert('L')
        if self.transform: img = self.transform(img)
        # 归一化标签 / 100mm
        label = torch.tensor([row['tx'] / 100.0, row['ty'] / 100.0, row['tz'] / 100.0], dtype=torch.float32)
        return img, label


# ==========================================
# 模型 (1通道 ResNet)
# ==========================================
class MapResNet(nn.Module):
    def __init__(self):
        super(MapResNet, self).__init__()
        resnet = models.resnet18(pretrained=False)  # 不用预训练，特征太简单了，从头学更快
        self.new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = self.new_conv
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, x):
        return self.fc(self.backbone(x).view(x.size(0), -1))


# ==========================================
# 训练
# ==========================================
def train():
    if not os.path.exists(CSV_PATH): return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on Cross-Helix Data: {device}")

    transform = transforms.Compose([
        transforms.Resize((200, 360)),
        transforms.ToTensor(),
    ])

    dataset = HelixDataset(CSV_PATH, DATA_ROOT, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MapResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    best_loss = float('inf')

    print(f"{'Phase':<6} | {'Loss':<10} | {'Error (mm)':<12}")

    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            run_loss = 0.0
            for img, label in (train_loader if phase == 'train' else val_loader):
                img = img.to(device);
                label = label.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(img)
                    loss = criterion(out, label)
                    if phase == 'train': loss.backward(); optimizer.step()
                run_loss += loss.item() * img.size(0)

            epoch_loss = run_loss / len(dataset) if phase == 'train' else run_loss / val_size
            mm_error = (epoch_loss ** 0.5) * 100.0

            if phase == 'val':
                print(f'{phase:<6} | {epoch_loss:.6f}   | {mm_error:.2f} mm')
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_cross_helix.pth')


if __name__ == "__main__":
    train()