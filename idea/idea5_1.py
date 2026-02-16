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
DATASET_NAME = "/root/cross_helix_dataset"
TOTAL_SAMPLES = 5000
IMG_H = 200  # 对应长度 s (0-100mm)
IMG_W = 360  # 对应角度 (0-360度)
ROBOT_RADIUS = 5.0


# ==========================================
# 物理引擎：双螺旋网格生成
# ==========================================
class CrossHelixEngine:
    def __init__(self):
        pass

    def generate_data(self):
        # 1. 随机姿态生成
        shape_type = np.random.choice([0, 1, 2], p=[0.1, 0.45, 0.45])

        if shape_type == 0:
            kappa, phi_bend = 0.0, 0.0
        elif shape_type == 1:
            # 大弯曲
            theta = np.deg2rad(np.random.uniform(10, 90))
            phi_bend = np.random.uniform(0, 2 * np.pi)
            kappa = theta / 100.0
        elif shape_type == 2:
            # 随机弯曲
            theta = np.deg2rad(np.random.uniform(20, 120))
            phi_bend = np.random.uniform(0, 2 * np.pi)
            kappa = theta / 100.0

        # 2. 生成展开图画布
        # 背景虽然是黑的，但为了模拟超声噪声，给一点底噪
        unrolled_img = np.random.normal(20, 5, (IMG_H, IMG_W)).astype(np.uint8)

        # 3. 定义双螺旋 (Double Helix)
        # Wire 1: 左旋, 3圈, 粗线 (模拟宽螺纹)
        # Wire 2: 右旋, 3圈, 细线 (模拟窄螺纹)
        # 方向 dir: +1 或 -1
        wires_config = [
            {"turns": 3.0, "dir": 1, "thickness": 5, "intensity": 255},  # 主相
            {"turns": 3.0, "dir": -1, "thickness": 2, "intensity": 180}  # 副相
        ]

        # 采样点 (越密越好，画出来连续)
        s_steps = np.linspace(0, 100, 1000)

        for cfg in wires_config:
            for s in s_steps:
                # === 物理形变核心公式 ===
                # 原始螺旋角度
                angle_raw = cfg["dir"] * (s / 100.0) * cfg["turns"] * 2 * np.pi

                # 计算应变 (Strain)
                # 当弯曲发生时，表面长度发生变化
                relative_angle = angle_raw - phi_bend
                strain = -ROBOT_RADIUS * kappa * np.cos(relative_angle)

                # 变形后的物理弧长 s_new
                s_deformed = s * (1 + strain)

                # 映射到像素坐标
                # Y轴: 长度 -> 像素行
                row = int((s_deformed / 100.0) * IMG_H)
                # X轴: 角度 -> 像素列 (0-360)
                col = int((np.rad2deg(angle_raw) % 360) * (IMG_W / 360.0))

                if 0 <= row < IMG_H and 0 <= col < IMG_W:
                    # 画点 (模拟超声回波)
                    # 粗细不同，亮度不同
                    cv2.circle(unrolled_img, (col, row),
                               radius=cfg["thickness"] // 2,
                               color=cfg["intensity"],
                               thickness=-1)

        # 4. 后处理：模拟超声伪影
        # 高斯模糊 (模拟波束宽度效应)
        unrolled_img = cv2.GaussianBlur(unrolled_img, (5, 5), 0)

        # 5. 计算真值标签 (末端坐标)
        tip_pos = self.calc_tip(kappa, phi_bend)

        return unrolled_img, kappa, phi_bend, tip_pos

    def calc_tip(self, kappa, phi):
        length = 100.0
        if abs(kappa) < 1e-5: return np.array([0, 0, length])
        r = 1.0 / kappa;
        theta = kappa * length
        x = r * (1 - np.cos(theta));
        z = r * np.sin(theta)
        return np.array([x * np.cos(phi), x * np.sin(phi), z])


if __name__ == "__main__":
    if os.path.exists(DATASET_NAME): shutil.rmtree(DATASET_NAME)
    os.makedirs(DATASET_NAME)

    engine = CrossHelixEngine()
    metadata = []

    print(f"🚀 生成 [反向双螺旋] 超声数据集: {TOTAL_SAMPLES} 张...")

    for i in tqdm(range(TOTAL_SAMPLES)):
        img, k, p, tip = engine.generate_data()

        filename = f"grid_{i:05d}.png"
        cv2.imwrite(os.path.join(DATASET_NAME, filename), img)
        metadata.append([filename, tip[0], tip[1], tip[2]])

        # 保存第一张预览图
        if i == 0:
            plt.figure(figsize=(10, 6))
            plt.imshow(img, cmap='gray', aspect='auto')
            plt.title(f"Simulated Ultrasound: Cross Helix Lattice\nKappa={k:.4f}, Phi={np.rad2deg(p):.1f}")
            plt.xlabel("Angle (0-360)")
            plt.ylabel("Length (0-100mm)")
            plt.savefig("cross_helix_preview.png")
            print("✅ 预览图: cross_helix_preview.png (请观察网格形变)")

    df = pd.DataFrame(metadata, columns=["filename", "tx", "ty", "tz"])
    df.to_csv(os.path.join(DATASET_NAME, "metadata.csv"), index=False)