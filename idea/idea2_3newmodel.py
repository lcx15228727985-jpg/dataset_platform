import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 配置
# ==========================================
MODEL_PATH = 'best_unrolled_model.pth'
DATA_ROOT = "/root/unrolled_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 物理参数 (必须与生成数据时一致)
ROBOT_RADIUS = 4.0
TURNS = 3.0  # 缠绕圈数
LENGTH = 100.0  # 机器人长度


# ==========================================
# 2. 模型定义
# ==========================================
class MapResNet(nn.Module):
    def __init__(self):
        super(MapResNet, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = self.new_conv
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, x):
        return self.fc(self.backbone(x).view(x.size(0), -1))


# ==========================================
# 3. 高级几何工具：带螺旋线的弯曲变换
# ==========================================
def apply_pcc_bending(points, kappa, phi):
    """
    将一组直线状态下的点 (x,y,z)，根据 kappa, phi 弯曲成圆弧状
    points: (N, 3) 数组
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]  # z 是长度方向

    # 1. 如果几乎是直线
    if abs(kappa) < 1e-5:
        # 只需要绕 Z 轴旋转 phi (虽然直线旋转没意义，但为了坐标系统一)
        # 对于直线，x,y 保持不变，z 保持不变
        # 但我们需要把 x,y 旋转到 phi 方向
        x_new = x * np.cos(phi) - y * np.sin(phi)
        y_new = x * np.sin(phi) + y * np.cos(phi)
        z_new = z
        return np.stack([x_new, y_new, z_new], axis=1)

    # 2. 常曲率弯曲变换
    r_bend = 1.0 / kappa

    # 将点转换到弯曲平面坐标系 (Bend Plane)
    # 假设弯曲发生在 X-Z 平面 (phi=0时)
    # 在弯曲前，点的切片坐标是 (x, y)，深度是 z
    # 弯曲后，z 变成弧长 -> 角度 theta
    theta = z * kappa

    # 核心几何变换：
    # 新的 Z_local (垂直向上) = (R - x) * sin(theta)
    # 新的 X_local (水平向圆心) = R - (R - x) * cos(theta)
    # Y_local 保持不变 (垂直于弯曲平面)

    z_bent = (r_bend - x) * np.sin(theta)
    x_bent = r_bend - (r_bend - x) * np.cos(theta)
    y_bent = y

    # 3. 绕 Z 轴旋转 phi，将弯曲平面转到实际方向
    # [ X_final ]   [ cos(phi)  -sin(phi) ] [ x_bent ]
    # [ Y_final ] = [ sin(phi)   cos(phi) ] [ y_bent ]

    x_final = x_bent * np.cos(phi) - y_bent * np.sin(phi)
    y_final = x_bent * np.sin(phi) + y_bent * np.cos(phi)
    z_final = z_bent

    return np.stack([x_final, y_final, z_final], axis=1)


def generate_helix_points(radius, length, turns, phases):
    """生成直状态下的螺旋线点云"""
    points_all = []

    s_steps = np.linspace(0, length, 300)  # 采样密度

    for phase in phases:
        # 螺旋方程: x = r*cos(wt+phi), y = r*sin(wt+phi), z = t
        angle = (s_steps / length) * turns * 2 * np.pi + phase

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = s_steps

        pts = np.stack([x, y, z], axis=1)
        points_all.append(pts)

    return points_all  # 返回 List[Array]


def inverse_pcc_from_tip(tip_pos):
    x, y, z = tip_pos
    phi = np.arctan2(y, x)
    d = np.sqrt(x ** 2 + y ** 2)
    if abs(d) < 1e-4:
        kappa = 0.0
    else:
        kappa = 2 * d / (d ** 2 + z ** 2)
    return kappa, phi


# ==========================================
# 4. 主程序
# ==========================================
def run_demo_with_helix():
    if not os.path.exists(MODEL_PATH): return

    model = MapResNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    df = pd.read_csv(CSV_PATH)
    idx = np.random.randint(0, len(df))
    row = df.iloc[idx]

    print(f"🔎 样本 #{idx}")
    print(f"   真值坐标: ({row['tx']:.2f}, {row['ty']:.2f}, {row['tz']:.2f})")

    # 预测
    img_path = os.path.join(DATA_ROOT, row['filename'])
    pil_img = Image.open(img_path).convert('L')
    transform = transforms.Compose([transforms.Resize((200, 360)), transforms.ToTensor()])
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_tip = model(input_tensor).cpu().numpy()[0] * 100.0

    print(f"   预测坐标: ({pred_tip[0]:.2f}, {pred_tip[1]:.2f}, {pred_tip[2]:.2f})")
    print(f"   误差: {np.linalg.norm(np.array([row['tx'], row['ty'], row['tz']]) - pred_tip):.2f} mm")

    # === 重建 ===
    # 1. 骨架点 (Backbone)
    backbone_straight = np.stack([np.zeros(100), np.zeros(100), np.linspace(0, 100, 100)], axis=1)

    # 2. 螺旋线点 (Helix / Wire Harness)
    helix_straight_list = generate_helix_points(ROBOT_RADIUS, LENGTH, TURNS, [0, 2 * np.pi / 3, 4 * np.pi / 3])

    # 3. 弯曲变换 (Ground Truth)
    gt_backbone = apply_pcc_bending(backbone_straight, row['kappa'], row['phi'])
    gt_helices = [apply_pcc_bending(h, row['kappa'], row['phi']) for h in helix_straight_list]

    # 4. 弯曲变换 (Prediction)
    pred_k, pred_phi = inverse_pcc_from_tip(pred_tip)
    pred_backbone = apply_pcc_bending(backbone_straight, pred_k, pred_phi)
    pred_helices = [apply_pcc_bending(h, pred_k, pred_phi) for h in helix_straight_list]

    # === 绘图 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画真值 (绿色)
    ax.plot(gt_backbone[:, 0], gt_backbone[:, 1], gt_backbone[:, 2], 'g-', linewidth=4, alpha=0.5, label='GT Backbone')
    for h in gt_helices:
        ax.plot(h[:, 0], h[:, 1], h[:, 2], 'g-', linewidth=1, alpha=0.3)  # 细绿线表示真值线束

    # 画预测 (红色)
    ax.plot(pred_backbone[:, 0], pred_backbone[:, 1], pred_backbone[:, 2], 'r--', linewidth=3, label='Pred Backbone')
    for i, h in enumerate(pred_helices):
        label = 'Pred Wire Harness' if i == 0 else None
        ax.plot(h[:, 0], h[:, 1], h[:, 2], 'r-', linewidth=2)  # 鲜红线表示预测线束

    ax.scatter(row['tx'], row['ty'], row['tz'], c='g', s=100, marker='o', label='GT Tip')
    ax.scatter(pred_tip[0], pred_tip[1], pred_tip[2], c='r', s=100, marker='x', label='Pred Tip')

    # 视角与比例
    ax.set_xlim([-40, 40]);
    ax.set_ylim([-40, 40]);
    ax.set_zlim([0, 100])
    ax.set_xlabel('X');
    ax.set_ylabel('Y');
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(
        f"3D Reconstruction with Wire Harness\nError: {np.linalg.norm(np.array([row['tx'], row['ty'], row['tz']]) - pred_tip):.2f} mm")

    plt.savefig('helix_demo.png')
    print("✨ 带线束的 3D 图已保存: helix_demo.png")


if __name__ == "__main__":
    run_demo_with_helix()