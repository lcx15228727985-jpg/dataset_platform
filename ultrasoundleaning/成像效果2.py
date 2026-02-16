import matplotlib

# 强制使用 TkAgg 后端，解决 PyCharm 中不弹窗的问题
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np


def simulate_external_wrapped_ultrasound(probe_angle_deg):
    """
    模拟超声成像：线缆缠绕在圆柱体【外部】
    """
    # --- 1. 基础空间定义 (屏幕坐标系) ---
    width = 16
    depth = 16
    resolution = 300

    x = np.linspace(-width / 2, width / 2, resolution)
    y = np.linspace(0, depth, resolution)
    X, Y = np.meshgrid(x, y)

    # 极坐标
    probe_origin_y = -3
    dx = X
    dy = Y - probe_origin_y
    R = np.sqrt(dx ** 2 + dy ** 2)

    # --- 2. 旋转逻辑 ---
    # 保持"屏幕/探头"不动，旋转"物体"
    rad = np.deg2rad(-probe_angle_deg)
    cx, cy = 0, 7  # 圆柱中心在屏幕上的位置

    # 旋转矩阵：计算物体坐标系
    X_obj = (X - cx) * np.cos(rad) - (Y - cy) * np.sin(rad) + cx
    Y_obj = (X - cx) * np.sin(rad) + (Y - cy) * np.cos(rad) + cy

    # --- [修改点1] 半径设置 ---
    cyl_radius = 4.0  # 圆柱体本身半径
    wire_radius = 4.4  # 线缆半径 (比圆柱体大，表示在外部)

    # 圆柱体掩膜
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    cylinder_mask = dist_from_center <= cyl_radius

    # --- 3. 特征构建 ---

    # [特征A] 镜面反射 (Specular Reflection) - 圆柱表面
    ny_screen = (Y - cy) / (dist_from_center + 1e-6)
    # 为了让外部线缆看得更清楚，稍微让高光窄一点 (指数从40改到60)
    specular = (np.abs(ny_screen) ** 60) * (ny_screen < -0.9)
    specular *= (np.abs(dist_from_center - cyl_radius) < 0.15)  # 仅限表面环

    # [特征B] 外部线缆 (External Wires)
    np.random.seed(123)
    wire_signal = np.zeros_like(X)
    num_wires = 16  # 稍微增加一点线缆数量

    wire_angles = np.random.uniform(0, 2 * np.pi, num_wires)
    for w_theta in wire_angles:
        # 线缆在物体坐标系中的位置
        wx = cx + wire_radius * np.cos(w_theta)
        wy = cy + wire_radius * np.sin(w_theta)

        # --- [修改点3] 新增物理遮挡逻辑 ---
        # 如果线缆在圆柱体"背面" (深度比圆柱中心深太多)，应该被圆柱体挡住看不见。
        # 简单的几何判断：只显示上半圆和侧面一点点的线缆。
        # 给一点余量 (0.5cm)，让刚好在赤道附近的线缆也能看见一点。
        is_visible = (wy - cy) < 0.5

        if is_visible:
            # 计算距离并生成亮点
            dist = np.sqrt((X_obj - wx) ** 2 + (Y_obj - wy) ** 2)
            # 增加一点亮度 (从2.0到2.5)，因为外部没有背景纹理对比了
            wire_signal += np.exp(-dist ** 2 / 0.08) * 2.5

    # --- [修改点2] 移除内部限制 ---
    # 注释掉下面这行代码，允许线缆出现在圆柱mask外部
    # wire_signal *= cylinder_mask

    # [特征C] 内部杂波 (只在圆柱内部)
    internal_noise_raw = np.random.normal(0, 0.1, X.shape)
    internal_signal = np.abs(internal_noise_raw) * cylinder_mask * 0.2

    # --- 4. 物理限制模拟 ---

    # [衰减与声影]
    attenuation_map = np.exp(-0.05 * Y)
    shadow_mask = (Y > cy) & (np.abs(X - cx) < 3.5)
    shadow_factor = np.ones_like(X)
    shadow_factor[shadow_mask] = 0.15

    # --- 5. 合成 ---
    total_signal = specular + wire_signal + internal_signal
    total_signal *= attenuation_map
    total_signal *= shadow_factor

    # FOV 扇形裁切
    fov_mask = (np.abs(np.arctan2(X, Y - probe_origin_y)) < np.deg2rad(40)) & (R < 15) & (R > 2.5)

    # 对数压缩
    img_log = 20 * np.log10(np.abs(total_signal) + 1e-4)
    img_log = np.clip(img_log, -50, 0)
    img_out = (img_log - img_log.min()) / (img_log.max() - img_log.min())

    return X, Y, img_out * fov_mask


# --- 程序执行入口 ---
if __name__ == "__main__":
    angles = [-45, 0, 45]
    plt.figure(figsize=(15, 6))

    print("开始生成外部缠绕图像...")

    for i, ang in enumerate(angles):
        X, Y, img = simulate_external_wrapped_ultrasound(ang)

        ax = plt.subplot(1, 3, i + 1)
        ax.pcolormesh(X, Y, img, cmap='gray', shading='auto', vmin=0, vmax=1)

        ax.set_title(f"Probe Angle: {ang}° (External Wires)", color='white', backgroundcolor='black')
        ax.set_xlabel('Lateral (cm)')
        if i == 0: ax.set_ylabel('Depth (cm)')

        ax.invert_yaxis()
        ax.axis('equal')

        # 添加辅助线圈出重点
        if i == 1:
            # 画一个红圈示意线缆在外面
            circle = plt.Circle((0, 7), 4.4, color='red', fill=False, linestyle='--', linewidth=1, alpha=0.5)
            ax.add_patch(circle)
            ax.text(0, 2, "Wires are OUTSIDE\nthe bright ring", color='red', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()