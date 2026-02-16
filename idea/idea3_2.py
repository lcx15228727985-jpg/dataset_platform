import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# ==========================================
# 配置
# ==========================================
CSV_FILE = "wire_features.csv"
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3  # MLP 可以用大一点的学习率


# ==========================================
# 1. 纯数值数据集
# ==========================================
class WireFeatureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 输入特征: 6个角度 + 距离
        # 这里的角度是 -pi 到 pi。为了方便网络，我们可以计算差值
        # 差值 delta = b - a
        # 考虑周期性: (b - a + pi) % (2*pi) - pi

        a = np.array([row['a0'], row['a1'], row['a2']])
        b = np.array([row['b0'], row['b1'], row['b2']])

        # 计算角度差 (Unwrapped Phase Diff)
        diff = b - a
        # 修正跨越 +/- PI 的情况
        diff = np.arctan2(np.sin(diff), np.cos(diff))

        # 我们把原始角度和差值都喂进去，信息量最全
        # Input dim = 3 (A) + 3 (Diff) = 6
        features = np.concatenate([a, diff])

        input_tensor = torch.from_numpy(features).float()

        # 标签
        kappa = row['kappa']
        phi = row['phi']
        label = torch.tensor([kappa, np.cos(phi), np.sin(phi)], dtype=torch.float32)

        return input_tensor, label


# ==========================================
# 2. 极简 MLP 网络
# ==========================================
class WireMLP(nn.Module):
    def __init__(self):
        super(WireMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 3)  # [kappa, cos, sin]
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. 训练
# ==========================================
def train_wire_mlp():
    if not os.path.exists(CSV_FILE):
        print("❌ 请先运行 extract_wire_features.py")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training Wire MLP on: {device}")

    dataset = WireFeatureDataset(CSV_FILE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WireMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"{'Phase':<6} | {'Total':<8} | {'Kappa':<8} | {'Dir':<8}")

    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            run_loss = 0.0
            rk, rd = 0.0, 0.0

            for feats, labels in (train_loader if phase == 'train' else val_loader):
                feats = feats.to(device);
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(feats)

                    # 依然加权
                    lk = criterion(out[:, 0], labels[:, 0]) * 100.0
                    ld = criterion(out[:, 1:], labels[:, 1:]) * 1.0  # MLP学得快，权重可以正常点
                    loss = lk + ld

                    if phase == 'train':
                        loss.backward();
                        optimizer.step()

                run_loss += loss.item() * feats.size(0)
                rk += lk.item() * feats.size(0)
                rd += ld.item() * feats.size(0)

            epoch_loss = run_loss / len(dataset)
            ek = rk / len(dataset)
            ed = rd / len(dataset)

            if phase == 'val':
                print(f'{phase:<6} | {epoch_loss:.4f}   | {ek:.4f}   | {ed:.4f}')
                scheduler.step(epoch_loss)
                if epoch_loss < 0.1:  # 随便设个阈值
                    torch.save(model.state_dict(), 'best_wire_mlp.pth')


if __name__ == "__main__":
    train_wire_mlp()