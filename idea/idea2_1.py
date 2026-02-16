import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

# ==========================================
# 配置
# ==========================================
DATASET_NAME = "/root/unrolled_dataset"
TOTAL_SAMPLES = 5000
IMG_H = 200  # 对应机器人长度方向 (分辨率)
IMG_W = 360  # 对应角度方向 (1度1像素)


# ==========================================
# 物理引擎
# ==========================================
class RobotPhysicsEngine:
    def __init__(self, radius=4.0):
        self.radius = radius

    def generate_data(self):
        # 1. 随机生成姿态
        shape_type = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])

        if shape_type == 0:
            configs = [(100, 0.001, 0)]
            kappa, phi_bend = 0.0, 0.0
        elif shape_type == 1:
            theta = np.deg2rad(np.random.uniform(20, 90))
            phi_bend = np.random.uniform(0, 2 * np.pi)
            configs = [(100, theta, phi_bend)]
            kappa = theta / 100.0
        elif shape_type == 2:
            theta1 = np.deg2rad(np.random.uniform(30, 80))
            phi1 = np.random.uniform(0, 2 * np.pi)
            configs = [(100, theta1, phi1)]
            kappa = theta1 / 100.0;
            phi_bend = phi1

        # 2. 计算 Marker 在 "展开平面" 上的坐标
        # 我们不需要生成 3D 坐标再投影，直接生成参数空间坐标即可！
        # Stoll Pattern 方程: Phi(s) = 2*pi * f * s + Phi_0

        unrolled_img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        # 模拟 3 根螺旋线
        phases = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        turns = 3.0  # 缠绕圈数

        # 物理形变引入：
        # 当机器人弯曲时，不同角度的表面长度会发生变化
        # 内侧受压变短，外侧受拉变长
        # s_deformed = s * (1 - r * kappa * cos(theta - phi_bend))

        for m_i, phase in enumerate(phases):
            # 沿长度方向采样
            s_steps = np.linspace(0, 100, 500)  # 物理长度 0-100mm

            for s in s_steps:
                # 原始角度 (螺旋)
                angle_raw = (s / 100.0) * turns * 2 * np.pi + phase

                # === 关键物理：应变导致的坐标变换 ===
                # 计算当前角度相对于弯曲平面的相对角
                relative_angle = angle_raw - phi_bend

                # 表面拉伸/压缩率 (Strain)
                # strain = -radius * kappa * cos(relative_angle)
                # s_new = s * (1 + strain)
                strain = -self.radius * kappa * np.cos(relative_angle)
                s_deformed = s * (1 + strain)

                # 映射到图像坐标
                # Y轴: 变形后的长度 -> 像素行
                row = int((s_deformed / 100.0) * IMG_H)
                # X轴: 角度 -> 像素列 (0-360)
                col = int((np.rad2deg(angle_raw) % 360) * (IMG_W / 360.0))

                if 0 <= row < IMG_H and 0 <= col < IMG_W:
                    # 画点 (稍微画粗一点，方便CNN提取)
                    # 引入非对称性：Marker 0 更亮/更大
                    radius = 2 if m_i == 0 else 1
                    intensity = 255 if m_i == 0 else 150
                    cv2.circle(unrolled_img, (col, row), radius, intensity, -1)

        # 3. 计算末端坐标 (用于验证)
        tip_pos = self.calc_tip(kappa, phi_bend)

        return unrolled_img, kappa, phi_bend, tip_pos

    def calc_tip(self, kappa, phi):
        if abs(kappa) < 1e-5: return np.array([0, 0, 100.0])
        r = 1.0 / kappa;
        theta = kappa * 100.0
        x = r * (1 - np.cos(theta));
        z = r * np.sin(theta)
        return np.array([x * np.cos(phi), x * np.sin(phi), z])


if __name__ == "__main__":
    if os.path.exists(DATASET_NAME): shutil.rmtree(DATASET_NAME)
    os.makedirs(DATASET_NAME)

    engine = RobotPhysicsEngine()
    metadata = []

    print(f"🚀 生成 [展开图] 数据集: {TOTAL_SAMPLES} 张...")

    for i in tqdm(range(TOTAL_SAMPLES)):
        img, k, p, tip = engine.generate_data()

        filename = f"map_{i:05d}.png"
        cv2.imwrite(os.path.join(DATASET_NAME, filename), img)

        metadata.append([filename, k, p, tip[0], tip[1], tip[2]])

        # 可视化第一张看看长啥样
        if i == 0:
            plt.imshow(img, cmap='gray')
            plt.title(f"Unrolled Map (k={k:.4f})")
            plt.xlabel("Angle (0-360)")
            plt.ylabel("Length (0-100mm)")
            plt.savefig("unrolled_preview.png")
            print("✅ 预览图已保存: unrolled_preview.png")

    df = pd.DataFrame(metadata, columns=["filename", "kappa", "phi", "tx", "ty", "tz"])
    df.to_csv(os.path.join(DATASET_NAME, "metadata.csv"), index=False)
    print("✅ 完成")