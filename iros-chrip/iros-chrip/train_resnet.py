import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= 配置区域 =================
DATASET_DIR = "dataset_engraving_v1"  # 数据集路径
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30  # 稍微增加轮数，让自动权重有时间收敛
TEST_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 数据集定义 (带归一化) =================
class UltrasoundPoseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.labels_df = pd.read_csv(os.path.join(root_dir, "labels.csv"))
        self.transform = transform
        
        # 预计算位置统计信息 (用于归一化)
        # 仅针对位置 (pos_x, pos_y, pos_z) 进行归一化，四元数不需要
        pos_data = self.labels_df[['pos_x', 'pos_y', 'pos_z']].values
        self.pos_mean = torch.tensor(pos_data.mean(axis=0), dtype=torch.float32)
        self.pos_std = torch.tensor(pos_data.std(axis=0), dtype=torch.float32)
        
        print(f"📊 数据集统计: Pos Mean={self.pos_mean.numpy()}, Std={self.pos_std.numpy()}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # 读取图像
        img_name = self.labels_df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取灰度图
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # [1, H, W] 归一化
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        # 读取标签
        row = self.labels_df.iloc[idx]
        pos = np.array([row['pos_x'], row['pos_y'], row['pos_z']], dtype=np.float32)
        quat = np.array([row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']], dtype=np.float32)
        
        # 位置归一化 (Z-Score Normalization)
        # 这对于平衡 Loss 非常重要，让位置数值范围接近 0~1
        pos = (torch.from_numpy(pos) - self.pos_mean) / self.pos_std
        quat = torch.from_numpy(quat) # 四元数本身范围就是 -1~1，无需归一化
        
        # 合并为 7D 向量
        label = torch.cat([pos, quat])
        
        return image, label

# ================= 2. 模型定义 (方案 3: 强制归一化) =================
class PoseResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PoseResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # 修改第一层适应单通道
        original_first_layer = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)

        # 修改全连接层输出 7 维
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 7)

    def forward(self, x):
        x = self.backbone(x)
        
        # [方案 3 核心]：硬性约束 (Hard Constraint)
        # 将输出向量拆分为 位置(3) 和 姿态(4)
        pos = x[:, :3]
        quat_raw = x[:, 3:]
        
        # 强制四元数归一化，使其位于单位超球面上
        # 这样网络只需要学习方向，不需要学习模长，极大降低姿态回归难度
        quat_norm = torch.nn.functional.normalize(quat_raw, p=2, dim=1)
        
        return torch.cat([pos, quat_norm], dim=1)

# ================= 3. 损失函数 (方案 1: 自动加权) =================
class AutomaticWeightedLoss(nn.Module):
    """
    基于同方差不确定性 (Homoscedastic Uncertainty) 的多任务 Loss
    Loss = (1/2σ1^2)*L1 + log(σ1) + (1/2σ2^2)*L2 + log(σ2)
    """
    def __init__(self, num_tasks=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 初始化可学习的参数 log_var (初始化为 0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.l1_loss = nn.L1Loss() # 位置使用 L1 Loss (对异常值更鲁棒)

    def forward(self, pred, target):
        # 拆分
        pos_pred, quat_pred = pred[:, :3], pred[:, 3:]
        pos_target, quat_target = target[:, :3], target[:, 3:]
        
        # --- 任务 1: 位置 Loss (基于归一化后的坐标) ---
        loss_pos_raw = self.l1_loss(pos_pred, pos_target)
        
        # 动态加权公式
        # prec_pos = exp(-log_var) 相当于 1/σ^2
        precision_pos = torch.exp(-self.log_vars[0])
        weighted_loss_pos = 0.5 * precision_pos * loss_pos_raw + 0.5 * self.log_vars[0]
        
        # --- 任务 2: 姿态 Loss (Geodesic Distance) ---
        # Loss = 1 - |<q1, q2>|
        # 即使模型输出了归一化四元数，点积仍可能略大于1或略小于-1 (数值误差)，clamp一下
        dot_product = torch.sum(quat_pred * quat_target, dim=1)
        loss_quat_raw = 1.0 - torch.mean(torch.abs(dot_product))
        
        precision_quat = torch.exp(-self.log_vars[1])
        weighted_loss_quat = 0.5 * precision_quat * loss_quat_raw + 0.5 * self.log_vars[1]
        
        # 总 Loss
        total_loss = weighted_loss_pos + weighted_loss_quat
        
        return total_loss, loss_pos_raw, loss_quat_raw

# ================= 4. 训练与验证流程 =================
def train():
    print(f"🚀 启动训练 (方案3+方案1)...")
    print(f"⚙️  设备: {DEVICE}")
    
    # 1. 准备数据
    if not os.path.exists(os.path.join(DATASET_DIR, "labels.csv")):
        print("❌ 错误: 找不到数据集，请先运行 generate_dataset.py")
        return

    full_dataset = UltrasoundPoseDataset(DATASET_DIR)
    
    # 保存统计数据用于反归一化验证
    stats = {
        'mean': full_dataset.pos_mean.to(DEVICE),
        'std': full_dataset.pos_std.to(DEVICE)
    }
    
    test_size = int(len(full_dataset) * TEST_SPLIT)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 初始化模型与Loss
    model = PoseResNet().to(DEVICE)
    criterion = AutomaticWeightedLoss().to(DEVICE)
    
    # [关键] 将 Loss 的 log_vars 也加入优化器
    # 通常给 log_vars 一个稍大的学习率有助于快速找到平衡点
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters(), 'lr': 1e-3} 
    ], lr=LEARNING_RATE)
    
    # 3. 训练循环
    history = {'train_loss': [], 'val_pos_err': [], 'val_quat_err': []}
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # 打印当前的权重分配 (Debug用)
        w_pos = torch.exp(-criterion.log_vars[0]).item()
        w_quat = torch.exp(-criterion.log_vars[1]).item()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [W_pos:{w_pos:.2f}, W_quat:{w_quat:.2f}]")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            loss, l_pos, l_quat = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.2f}", 'L_P': f"{l_pos.item():.3f}", 'L_Q': f"{l_quat.item():.3f}"})
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_epoch_loss)
        
        # 验证 (计算真实的物理误差)
        val_pos_mm, val_quat_dist = evaluate(model, test_loader, stats)
        history['val_pos_err'].append(val_pos_mm)
        history['val_quat_err'].append(val_quat_dist)
        
        print(f"   Done. Train Loss: {avg_epoch_loss:.4f}")
        print(f"   >> Val Error: Position = {val_pos_mm:.2f} mm | Rotation = {val_quat_dist:.4f} (Geodesic)")

    # 4. 保存
    torch.save(model.state_dict(), "resnet18_auto_weighted.pth")
    print("✅ 模型已保存至 resnet18_auto_weighted.pth")
    
    # 绘图
    plot_history(history)

def evaluate(model, loader, stats):
    model.eval()
    total_pos_error = 0.0
    total_quat_error = 0.0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            
            # 反归一化位置，计算真实物理误差 (mm)
            pos_pred_real = outputs[:, :3] * stats['std'] + stats['mean']
            pos_gt_real = labels[:, :3] * stats['std'] + stats['mean']
            
            # 位置误差 (L2 Euclidean)
            batch_pos_err = torch.norm(pos_pred_real - pos_gt_real, dim=1).mean().item()
            total_pos_error += batch_pos_err
            
            # 姿态误差 (Geodesic distance: 1 - |q1.q2|)
            # outputs 已经在 forward 中强制归一化了，labels 也是归一化的
            dot = torch.sum(outputs[:, 3:] * labels[:, 3:], dim=1)
            batch_quat_err = (1.0 - torch.abs(dot)).mean().item()
            total_quat_error += batch_quat_err
            
    return total_pos_error / len(loader), total_quat_error / len(loader)

def plot_history(hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(hist['val_pos_err'], 'b-')
    ax1.set_title("Position Error (mm)")
    ax1.set_xlabel("Epoch")
    ax1.grid(True)
    
    ax2.plot(hist['val_quat_err'], 'r-')
    ax2.set_title("Rotation Error (1 - |q.q|)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    
    plt.savefig("training_metrics.png")
    print("📈 评估曲线已保存至 training_metrics.png")

# ================= 5. 推理演示 =================
def run_demo():
    print(f"\n🔮 推理演示...")
    if not os.path.exists("resnet18_auto_weighted.pth"): return
    
    # 需要重新加载 Dataset 获取统计数据用于反归一化
    ds = UltrasoundPoseDataset(DATASET_DIR)
    mean, std = ds.pos_mean.to(DEVICE), ds.pos_std.to(DEVICE)
    
    model = PoseResNet().to(DEVICE)
    model.load_state_dict(torch.load("resnet18_auto_weighted.pth", map_location=DEVICE))
    model.eval()
    
    # 读取一张图
    img_path = os.path.join(DATASET_DIR, "images", "000000.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_t = torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(img_t)[0]
    
    # 解析结果
    pos_norm = out[:3]
    quat = out[3:]
    
    # 反归一化位置
    pos_real = pos_norm * std + mean
    
    print(f"   原始输出 (Norm Pos): {pos_norm.cpu().numpy()}")
    print(f"   真实位置 (Real Pos): {pos_real.cpu().numpy()} mm")
    print(f"   预测姿态 (Quat): {quat.cpu().numpy()} (模长: {torch.norm(quat).item():.4f})")

if __name__ == "__main__":
    train()
    run_demo()