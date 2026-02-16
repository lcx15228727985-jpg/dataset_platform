import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
from tqdm import tqdm


# ==========================================
# PCC 重建公式
# ==========================================
def pcc_forward_kinematics(kappa, phi, length=100.0):
    # 如果是直线
    if abs(kappa) < 1e-5:
        return np.array([0.0, 0.0, length])

    # 常曲率弧
    r = 1.0 / kappa
    theta = kappa * length

    # 局部坐标系下的末端 (在弯曲平面内)
    # x_loc = r(1-cos), z_loc = r*sin
    x_loc = r * (1 - np.cos(theta))
    z_loc = r * np.sin(theta)

    # 旋转到 3D 空间 (绕 Z 轴旋转 phi)
    # p = [x_loc * cos(phi), x_loc * sin(phi), z_loc]
    x = x_loc * np.cos(phi)
    y = x_loc * np.sin(phi)
    z = z_loc

    return np.array([x, y, z])


# ... (StackedResNet 类定义需复制过来) ...
class StackedResNet(nn.Module):
    def __init__(self):
        super(StackedResNet, self).__init__()
        resnet = models.resnet18(pretrained=False)  # 只需要结构
        self.new_conv = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = self.new_conv
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, x):
        return self.fc(self.backbone(x).view(x.size(0), -1))


def run_eval():
    MODEL_PATH = 'best_asym_pinn.pth'
    DATA_ROOT = "/root/asym_pinn_dataset"
    CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        print("❌ 模型不存在")
        return

    print("⏳ Loading model...")
    model = StackedResNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    df = pd.read_csv(CSV_PATH)
    sample_df = df.sample(200)

    errors = []

    print("🚀 开始 PCC 重建验收...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        npy_path = os.path.join(DATA_ROOT, row['filename'])
        stack = np.load(npy_path)
        tensor = torch.from_numpy(stack).float().unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            pred = model(tensor).cpu().numpy()[0]

        k_pred = pred[0]
        # 从 cos, sin 恢复 phi
        cos_p, sin_p = pred[1], pred[2]
        phi_pred = np.arctan2(sin_p, cos_p)

        # 计算预测坐标
        tip_pred = pcc_forward_kinematics(k_pred, phi_pred)

        # 计算真实坐标
        tip_true = pcc_forward_kinematics(row['kappa'], row['phi'])

        err = np.linalg.norm(tip_pred - tip_true)
        errors.append(err)

    avg_err = np.mean(errors)
    max_err = np.max(errors)

    print("\n" + "=" * 40)
    print("   PINN + Asym 验收报告")
    print("=" * 40)
    print(f"   平均末端误差: {avg_err:.4f} mm")
    print(f"   最大末端误差: {max_err:.4f} mm")

    if avg_err < 3.0:
        print("   ✅ 方案成功！(Error < 3mm)")
    else:
        print("   ⚠️ 还需要优化")


if __name__ == "__main__":
    run_eval()