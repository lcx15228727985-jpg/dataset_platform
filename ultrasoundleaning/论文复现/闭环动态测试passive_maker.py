import matplotlib

matplotlib.use('TkAgg')  # 确保动画流畅弹出

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 1. 核心数学模型 (The Math)
# 来源: Stoll & Dupont, MICCAI 2005
# ==========================================

# 论文第 172 行: 标记参数 [cite: 172]
ALPHA = 3.48  # 振幅 (mm)
BETA = 9.02  # 偏置 (mm)


def get_theoretical_pattern(theta_rad):
    """
    计算理论上的指纹 f(theta) [cite: 129, 130]
    Eq. 10: f(theta) = [p1, p2, p3]
    """
    # 这里的 Offset_A 设为 0，因为它是整体平移的一部分，会被 t 吸收
    p1 = ALPHA * np.sin(theta_rad)
    p2 = ALPHA * np.sin(theta_rad + 2 * np.pi / 3) + BETA
    p3 = ALPHA * np.sin(theta_rad + 4 * np.pi / 3) + 2 * BETA
    return np.array([p1, p2, p3])


def solve_pose_from_measurements(l_measured):
    """
    核心追踪算法: 逆向求解
    输入: l_measured (3个测量到的轴向位置)
    输出: theta_est (deg), t_est (mm)
    """
    n = 3

    # --- Step 1: 解算平移 t [cite: 107, 166] ---
    # 利用 sum(f(theta)) = const 的性质
    sum_f_theoretical = 3 * BETA
    t_est = (np.sum(l_measured) - sum_f_theoretical) / n

    # --- Step 2: 解算旋转 theta [cite: 110] ---
    # 暴力搜索最小误差角度 (ArgMin)
    search_space = np.linspace(0, 2 * np.pi, 360)  # 1度分辨率

    # 预计算所有可能的理论模式 (实际应用中这是查表法，速度极快)
    candidates = np.array([get_theoretical_pattern(a) for a in search_space])  # shape (360, 3)

    # 目标向量: l - t (去除平移分量，只看旋转分量)
    target_pattern = l_measured - t_est

    # 计算误差: || Candidate - Target ||
    # numpy 广播机制一次性算出360个误差
    errors = np.linalg.norm(candidates - target_pattern, axis=1)

    # 找到误差最小的那个角度
    best_idx = np.argmin(errors)
    theta_est = np.rad2deg(search_space[best_idx])

    return theta_est, t_est


# ==========================================
# 2. 仿真环境配置
# ==========================================

if __name__ == "__main__":
    # --- 初始化画布 ---
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Real-time Stoll Marker Tracking System", fontsize=16, fontweight='bold')

    # 布局:
    # Top Left: 物理状态 (仪表盘)
    # Top Right: 追踪结果 (仪表盘)
    # Bottom: 超声成像 (B-Mode)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

    ax_phys_dial = fig.add_subplot(gs[0, 0], projection='polar')
    ax_algo_dial = fig.add_subplot(gs[0, 1], projection='polar')
    ax_us = fig.add_subplot(gs[1, :])  # 占据下方整行

    # --- 设置仪表盘 1: 物理真值 (Ground Truth) ---
    ax_phys_dial.set_title("Physical Reality\n(Actual Rotation)", color='blue', fontweight='bold', pad=20)
    ax_phys_dial.set_yticklabels([])  # 隐藏径向刻度
    ax_phys_dial.set_theta_zero_location('N')  # 0度在北
    ax_phys_dial.set_theta_direction(-1)  # 顺时针旋转

    # 真实指针
    true_hand, = ax_phys_dial.plot([], [], 'b-', linewidth=4, label='True Angle')
    true_text = ax_phys_dial.text(0, 0, "", ha='center', va='center', fontsize=12, fontweight='bold', color='blue')

    # --- 设置仪表盘 2: 算法估计 (Algorithm Estimate) ---
    ax_algo_dial.set_title("Algorithm Output\n(Estimated Rotation)", color='green', fontweight='bold', pad=20)
    ax_algo_dial.set_yticklabels([])
    ax_algo_dial.set_theta_zero_location('N')
    ax_algo_dial.set_theta_direction(-1)

    # 估计指针
    est_hand, = ax_algo_dial.plot([], [], 'g-', linewidth=4, label='Est. Angle')
    est_text = ax_algo_dial.text(0, 0, "", ha='center', va='center', fontsize=12, fontweight='bold', color='green')

    # 误差显示
    error_text = fig.text(0.5, 0.52, "", ha='center', fontsize=12, color='red', backgroundcolor='#FFEEEE')

    # --- 设置超声视图 ---
    width = 40.0;
    depth = 25.0;
    resolution = 5
    x = np.linspace(-width / 2, width / 2, int(width * resolution))
    y = np.linspace(0, depth, int(depth * resolution))
    X, Y = np.meshgrid(x, y)
    center_shift = 9.0

    ax_us.set_title("Ultrasound B-Mode & Feature Extraction", backgroundcolor='black', color='white')
    ax_us.set_xlabel("Lateral Position (mm)")
    ax_us.set_ylabel("Depth (mm)")
    ax_us.invert_yaxis()
    ax_us.set_aspect('equal')

    # 初始化图像 Mesh
    img_mesh = ax_us.pcolormesh(X, Y, np.zeros_like(X), cmap='gray', vmin=0, vmax=1, shading='auto')

    # 标记检测到的点 (红色十字)
    detected_points_plot, = ax_us.plot([], [], 'r+', markersize=15, markeredgewidth=2, label='Measured Points (Input)')
    # 标记预测回投的点 (绿色圆圈) - 用于验证
    predicted_points_plot, = ax_us.plot([], [], 'go', markersize=10, fillstyle='none', markeredgewidth=2,
                                        label='Algorithm Re-projection')
    ax_us.legend(loc='lower right')


    # ==========================================
    # 3. 动画主循环 (Real-time Loop)
    # ==========================================

    def update(frame):
        # --- A. 物理模拟 (Simulation) ---
        # 1. 让器械旋转 (模拟手转动)
        true_angle_deg = (frame * 2) % 360
        true_angle_rad = np.deg2rad(true_angle_deg)

        # 2. 让器械平移 (模拟手抖动或插入深度变化)
        # 范围 -2mm 到 +2mm 之间的正弦运动
        true_t = 2.0 * np.sin(frame * 0.05)

        # 3. 计算真实的脊线位置 (Ground Truth)
        # l = f(theta) + t
        l_true = get_theoretical_pattern(true_angle_rad) + true_t

        # --- B. 测量模拟 (Measurement & Noise) ---
        # 论文指出测量误差约为 0.22mm [cite: 206]
        # 我们添加高斯白噪声模拟超声图像处理的误差
        measurement_noise = np.random.normal(0, 0.25, 3)
        l_measured = l_true + measurement_noise

        # --- C. 算法执行 (The "Black Box") ---
        # 只有 l_measured 被送入算法，真值被隐藏
        est_angle_deg, est_t = solve_pose_from_measurements(l_measured)

        # --- D. 视觉更新 (Visualization) ---

        # 1. 更新仪表盘
        true_hand.set_data([0, true_angle_rad], [0, 1])
        true_text.set_text(f"{true_angle_deg:.0f}°")

        est_angle_rad = np.deg2rad(est_angle_deg)
        est_hand.set_data([0, est_angle_rad], [0, 1])
        est_text.set_text(f"{est_angle_deg:.0f}°")

        # 计算误差 (处理 359度 vs 0度 的跳变)
        diff = abs(true_angle_deg - est_angle_deg)
        if diff > 180: diff = 360 - diff
        error_text.set_text(f"Tracking Error: {diff:.1f}° | Trans Error: {abs(true_t - est_t):.2f}mm")

        # 2. 更新超声图像
        # 生成含噪声的背景图像
        shaft_depth = 12.0
        surface_signal = np.exp(-((Y - shaft_depth) ** 2) / 0.2) * 0.6
        bumps_signal = np.zeros_like(X)

        for pos in l_true:  # 图像显示真实的物理位置
            x_img = pos - center_shift
            dist_sq = (X - x_img) ** 2 + (Y - (shaft_depth - 0.8)) ** 2 * 4.0
            bumps_signal += np.exp(-dist_sq / 1.5) * 2.5

        total_img = surface_signal + bumps_signal + np.random.normal(0, 0.1, X.shape) * 0.2

        # 归一化并更新
        total_img = np.clip(total_img, 0, None)  # 简单裁剪
        total_img /= np.max(total_img) + 1e-5
        img_mesh.set_array(total_img.ravel())

        # 3. 更新特征点标记
        # 红色十字: 算法"看到"的点 (含噪声)
        detected_points_plot.set_data(l_measured - center_shift, [shaft_depth - 0.8] * 3)

        # 绿色圆圈: 算法反算出的位置 (Re-projection)
        # 用解算出的 theta 和 t 重新生成理论位置，看是否与红点重合
        reprojected_l = get_theoretical_pattern(est_angle_rad) + est_t
        predicted_points_plot.set_data(reprojected_l - center_shift, [shaft_depth - 0.8] * 3)

        return img_mesh, true_hand, est_hand, detected_points_plot, predicted_points_plot, true_text, est_text, error_text


    # --- 启动动画 ---
    print("系统启动中...")
    print("模拟误差源: 图像噪声 (Gaussian sigma=0.25mm)")

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 1000),
                                  interval=30, blit=False)
    plt.tight_layout()
    plt.show()