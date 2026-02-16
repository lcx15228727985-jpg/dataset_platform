import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# ==========================================
# 配置
# ==========================================
CSV_FILE = "wire_features.csv"
MODEL_PATH = "best_wire_mlp.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROBOT_LENGTH = 100.0  # 机器人长度 100mm


# ==========================================
# 1. 模型定义 (需与训练一致)
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
# 2. 几何工具 (PCC)
# ==========================================
def pcc_forward(kappa, phi, length=ROBOT_LENGTH):
    """根据 kappa, phi 重建 3D 骨架"""
    s = np.linspace(0, length, 100)

    if abs(kappa) < 1e-5:
        # 直线
        x = np.zeros_like(s)
        y = np.zeros_like(s)
        z = s
    else:
        # 圆弧
        r = 1.0 / kappa
        theta = s * kappa

        # 弯曲平面内的坐标
        x_loc = r * (1 - np.cos(theta))
        z_loc = r * np.sin(theta)

        # 旋转 phi
        x = x_loc * np.cos(phi)
        y = x_loc * np.sin(phi)
        z = z_loc

    return np.stack([x, y, z], axis=1)


# ==========================================
# 3. 评估主程序
# ==========================================
def run_wire_demo():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CSV_FILE):
        print("❌ 找不到模型或数据文件")
        return

    # 加载模型
    model = WireMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 加载数据
    df = pd.read_csv(CSV_FILE)

    # --- 阶段 1: 全局统计误差 ---
    print(f"🚀 正在评估 {len(df)} 个样本的物理误差...")
    errors = []

    for idx, row in df.iterrows():
        # 构造输入 (复现 Dataset 逻辑)
        a = np.array([row['a0'], row['a1'], row['a2']])
        b = np.array([row['b0'], row['b1'], row['b2']])
        diff = b - a
        diff = np.arctan2(np.sin(diff), np.cos(diff))  # 角度归一化

        features = np.concatenate([a, diff])
        input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(input_tensor).cpu().numpy()[0]

        # 解析预测值
        k_pred = out[0]
        phi_pred = np.arctan2(out[2], out[1])

        # 解析真值
        k_true = row['kappa']
        phi_true = row['phi']

        # 计算末端坐标误差
        tip_pred = pcc_forward(k_pred, phi_pred)[-1]
        tip_true = pcc_forward(k_true, phi_true)[-1]

        err = np.linalg.norm(tip_pred - tip_true)
        errors.append(err)

    avg_error = np.mean(errors)
    median_error = np.median(errors)

    print("\n" + "=" * 40)
    print("   纯线束特征 (Wire MLP) 验收报告")
    print("=" * 40)
    print(f"   平均末端误差: {avg_error:.2f} mm")
    print(f"   中位末端误差: {median_error:.2f} mm")

    if avg_error < 5.0:
        print("   ✅ 结果优秀！(误差 < 5mm)")
    elif avg_error < 15.0:
        print("   ⚠️ 结果可用 (5mm < 误差 < 15mm)，方向角可能有抖动")
    else:
        print("   ❌ 需要改进")

    # --- 阶段 2: 3D 可视化 (随机抽一个) ---
    rand_idx = np.random.randint(0, len(df))
    row = df.iloc[rand_idx]

    # 重新推理一次用于画图
    a = np.array([row['a0'], row['a1'], row['a2']])
    b = np.array([row['b0'], row['b1'], row['b2']])
    diff = b - a
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    feat = np.concatenate([a, diff])
    tensor = torch.from_numpy(feat).float().unsqueeze(0).to(DEVICE)
    out = model(tensor).cpu().detach().numpy()[0]

    k_p, phi_p = out[0], np.arctan2(out[2], out[1])
    k_t, phi_t = row['kappa'], row['phi']

    # 重建曲线
    pts_pred = pcc_forward(k_p, phi_p)
    pts_true = pcc_forward(k_t, phi_t)

    err_val = np.linalg.norm(pts_pred[-1] - pts_true[-1])

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画真值
    ax.plot(pts_true[:, 0], pts_true[:, 1], pts_true[:, 2], 'g-', linewidth=4, alpha=0.6, label='Ground Truth')
    # 画预测
    ax.plot(pts_pred[:, 0], pts_pred[:, 1], pts_pred[:, 2], 'r--', linewidth=2,
            label=f'Wire-MLP Pred (Err={err_val:.1f}mm)')

    # 标记末端
    ax.scatter(pts_true[-1, 0], pts_true[-1, 1], pts_true[-1, 2], c='g', s=100)
    ax.scatter(pts_pred[-1, 0], pts_pred[-1, 1], pts_pred[-1, 2], c='r', s=100, marker='x')

    # 设置比例
    ax.set_xlim([-40, 40]);
    ax.set_ylim([-40, 40]);
    ax.set_zlim([0, 100])
    ax.set_xlabel('X (mm)');
    ax.set_ylabel('Y (mm)');
    ax.set_zlabel('Z (mm)')
    ax.set_title(
        f"3D Reconstruction from Wire Angles\nKappa: {k_t:.4f} vs {k_p:.4f}\nPhi: {np.rad2deg(phi_t):.1f} vs {np.rad2deg(phi_p):.1f} deg")
    ax.legend()

    plt.savefig('wire_mlp_demo.png')
    print(f"✨ 3D 可视化图已保存: wire_mlp_demo.png")


if __name__ == "__main__":
    run_wire_demo()