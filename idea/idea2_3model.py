import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 配置与模型定义 (需与训练一致)
# ==========================================
MODEL_PATH = 'best_unrolled_model.pth'
DATA_ROOT = "/root/unrolled_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MapResNet(nn.Module):
    def __init__(self):
        super(MapResNet, self).__init__()
        resnet = models.resnet18(pretrained=False)
        # 1通道输入
        self.new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
# 2. 几何工具箱 (PCC 逆解与正解)
# ==========================================
def pcc_forward(kappa, phi, length=100.0, num_points=50):
    s = np.linspace(0, length, num_points)
    if abs(kappa) < 1e-5:
        # 直线
        x = np.zeros_like(s)
        y = np.zeros_like(s)
        z = s
    else:
        # 圆弧
        r = 1.0 / kappa
        theta = s * kappa
        x_loc = r * (1 - np.cos(theta))
        z_loc = r * np.sin(theta)
        # 旋转 phi
        x = x_loc * np.cos(phi)
        y = x_loc * np.sin(phi)
        z = z_loc
    return x, y, z


def inverse_pcc_from_tip(tip_pos, length=100.0):
    """
    简易逆运动学：已知末端 (x,y,z)，反求 kappa, phi
    这里我们假设是常曲率弯曲
    """
    x, y, z = tip_pos

    # 1. 计算方向角 phi
    phi = np.arctan2(y, x)

    # 2. 计算弯曲平面内的末端坐标 (d, z)
    d = np.sqrt(x ** 2 + y ** 2)  # 水平偏距

    # 3. 计算曲率 kappa
    # 几何关系: d = (1-cos(theta))/k, z = sin(theta)/k, theta = k*L
    # 弦长 chord = sqrt(d^2 + z^2)
    # 半径 R = chord / (2 * sin(theta/2))
    # 这是一个超越方程，但对于小弯曲可以用几何近似：
    # 2*R*d = d^2 + z^2 (相交弦定理近似) -> R = (d^2+z^2)/(2d) -> k = 2d / (d^2+z^2)

    if abs(d) < 1e-4:
        kappa = 0.0
    else:
        kappa = 2 * d / (d ** 2 + z ** 2)

    return kappa, phi


# ==========================================
# 3. 演示主程序
# ==========================================
def run_demo():
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型未找到")
        return

    # 加载模型
    model = MapResNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 读取数据
    df = pd.read_csv(CSV_PATH)

    # 随机抽取一个样本
    idx = np.random.randint(0, len(df))
    row = df.iloc[idx]

    print(f"🔎 演示样本 #{idx}")
    print(f"   真值坐标: ({row['tx']:.2f}, {row['ty']:.2f}, {row['tz']:.2f})")

    # 预处理图片
    img_path = os.path.join(DATA_ROOT, row['filename'])
    pil_img = Image.open(img_path).convert('L')

    transform = transforms.Compose([
        transforms.Resize((200, 360)),
        transforms.ToTensor()
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # 预测
    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy()[0]

    # 反归一化 (x100)
    pred_tip = pred_norm * 100.0

    error = np.linalg.norm(np.array([row['tx'], row['ty'], row['tz']]) - pred_tip)
    print(f"   预测坐标: ({pred_tip[0]:.2f}, {pred_tip[1]:.2f}, {pred_tip[2]:.2f})")
    print(f"   末端误差: {error:.2f} mm")

    # === 3D 重建 ===
    # 1. 真实骨架
    gt_x, gt_y, gt_z = pcc_forward(row['kappa'], row['phi'])

    # 2. 预测骨架 (通过逆解反推形态)
    pred_k, pred_phi = inverse_pcc_from_tip(pred_tip)
    pred_x, pred_y, pred_z = pcc_forward(pred_k, pred_phi)

    # === 绘图 ===
    fig = plt.figure(figsize=(14, 6))

    # 左图：输入的展开图
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(pil_img, cmap='gray', aspect='auto')
    ax1.set_title("Input: Unrolled Map (Cylindrical Projection)")
    ax1.set_xlabel("Angle (Phi)")
    ax1.set_ylabel("Length (s)")

    # 右图：3D 重建对比
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # 画真值 (绿色实线)
    ax2.plot(gt_x, gt_y, gt_z, 'g-', linewidth=3, label='Ground Truth')
    # 画末端点
    ax2.scatter(row['tx'], row['ty'], row['tz'], c='g', s=50, marker='o')

    # 画预测 (红色虚线)
    ax2.plot(pred_x, pred_y, pred_z, 'r--', linewidth=2, label=f'AI Pred (Err={error:.1f}mm)')
    ax2.scatter(pred_tip[0], pred_tip[1], pred_tip[2], c='r', s=50, marker='x')

    # 设置坐标轴比例一致，不然看着会变形
    max_range = 100
    ax2.set_xlim([-50, 50])
    ax2.set_ylim([-50, 50])
    ax2.set_zlim([0, 100])

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (Depth)')
    ax2.set_title("3D Shape Reconstruction")
    ax2.legend()

    plt.tight_layout()
    save_path = 'final_demo.png'
    plt.savefig(save_path)
    print(f"✨ 3D 可视化已保存至: {save_path}")


if __name__ == "__main__":
    run_demo()