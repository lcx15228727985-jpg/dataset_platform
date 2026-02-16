import matplotlib

# 强制使用 TkAgg 后端，确保动画弹窗正常工作
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np


# ==========================================
# 第一部分：物理与数学模型 (Physics & Math)
# 基于论文 Stoll & Dupont (MICCAI 2005)
# ==========================================

def calculate_stoll_positions(theta_rad):
    """
    根据论文 Equation (10) 计算三个脊线(Ridges)在轴向(x)上的位置
    参数: theta_rad (弧度)
    返回: positions (mm) - 包含三个脊线的轴向位置
    """
    # 论文第 172 行提供的精确参数 [cite: 172]
    alpha = 3.48  # 振幅 (Amplitude)
    beta = 9.02  # 脊线间的平均间隔偏置 (Offset)

    # 论文 Eq(10): 三条正弦波轨迹方程
    # 这是编码的核心：不同的角度对应唯一的三个位置组合
    p1 = alpha * np.sin(theta_rad)
    p2 = alpha * np.sin(theta_rad + 2 * np.pi / 3) + beta
    p3 = alpha * np.sin(theta_rad + 4 * np.pi / 3) + 2 * beta

    return np.array([p1, p2, p3])


# ==========================================
# 第二部分：超声成像仿真 (Ultrasound Engine)
# 将物理位置转换为 B-Mode 图像信号
# ==========================================

def generate_b_mode_image(X, Y, ridge_positions):
    """
    生成带有物理伪影的超声图像
    """
    # 参数设置
    shaft_depth = 15.0  # 器械杆深度 (mm)
    center_shift = 9.0  # 将图案中心对齐到图像中心

    # 1. 模拟器械表面回波 (Specular Reflection)
    # 表现为一条水平的高亮线
    surface_signal = np.exp(-((Y - shaft_depth) ** 2) / (0.4 ** 2)) * 0.8

    # 2. 模拟脊线/线缆回波 (Scattering)
    # 表现为表面上方的凸起亮点
    bumps_signal = np.zeros_like(X)
    for pos in ridge_positions:
        # 将物理坐标映射到图像像素坐标
        x_img = pos - center_shift

        # 模拟点扩散函数 (PSF): 超声并不是一个完美的点，而是一个高斯光斑
        # 纵向(X)分辨率略好于深度(Y)分辨率
        y_bump_pos = shaft_depth - 0.8  # 脊线略微突出于表面
        dist_sq = (X - x_img) ** 2 + (Y - y_bump_pos) ** 2 * 3.0

        # 叠加信号
        bumps_signal += np.exp(-dist_sq / 1.5) * 2.5

    # 合成基础信号
    total_signal = surface_signal + bumps_signal

    # 3. 添加声学噪声 (Acoustic Speckle)
    # 散斑噪声通常是乘性噪声，这里用简化模型模拟
    noise = np.random.normal(0, 0.15, X.shape)
    total_signal += noise * 0.2

    # 4. 信号处理链 (Signal Processing Chain)
    # 包络检测 -> 对数压缩 (Log Compression)
    # 这是让图像看起来像真实超声的关键步骤 [cite: 14]
    img_log = 20 * np.log10(np.abs(total_signal) + 1e-5)
    img_log = np.clip(img_log, -45, 0)  # 截断动态范围 (例如 45dB)

    # 归一化到 0-1 用于显示
    img_out = (img_log - img_log.min()) / (img_log.max() - img_log.min())
    return img_out


# ==========================================
# 第三部分：可视化与动画 (Visualization)
# 将上述所有过程整合在一个窗口中
# ==========================================

if __name__ == "__main__":
    # --- 1. 初始化画布布局 ---
    fig = plt.figure(figsize=(11, 10))
    fig.suptitle("Stoll Marker: Physics to Ultrasound Integration", fontsize=16, fontweight='bold')

    # 创建 3 行布局
    # Row 1: 物理视图 (Physical View)
    # Row 2: 超声视图 (Ultrasound View)
    # Row 3: 数学原理 (Mathematical Trajectory)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.4)
    ax_phys = fig.add_subplot(gs[0])
    ax_us = fig.add_subplot(gs[1])
    ax_math = fig.add_subplot(gs[2])

    # --- 2. 预计算与静态设置 ---

    # 定义超声成像区域 FOV
    width = 40.0;
    depth = 30.0;
    resolution = 8
    x = np.linspace(-width / 2, width / 2, int(width * resolution))
    y = np.linspace(0, depth, int(depth * resolution))
    X, Y = np.meshgrid(x, y)
    center_shift = 9.0

    # === 设置图 1: 物理视图 ===
    ax_phys.set_title("1. Physical Reality (Side View of Tool)", fontsize=12, color='blue', loc='left')
    ax_phys.set_xlim(-20, 20)
    ax_phys.set_ylim(-5, 10)
    ax_phys.set_aspect('equal')
    ax_phys.axis('off')

    # 画器械杆
    shaft_rect = patches.Rectangle((-20, -2), 40, 4, color='#DDDDDD', ec='black')
    ax_phys.add_patch(shaft_rect)
    ax_phys.text(-19, 0, "Surgical Instrument Shaft", va='center', fontsize=9)

    # 画超声探头示意
    ax_phys.text(0, 8, "[Ultrasound Probe Location]", ha='center', color='gray', fontweight='bold')
    ax_phys.annotate("", xy=(0, 3), xytext=(0, 7), arrowprops=dict(arrowstyle="->", color='cyan', lw=3))

    # 初始化物理脊线 (3个彩色方块)
    ridge_colors = ['red', 'green', 'blue']
    phys_ridges = []
    for c in ridge_colors:
        # 高度略高于杆表面
        rect = patches.Rectangle((0, 2), 1.5, 1.5, color=c, alpha=0.9)
        ax_phys.add_patch(rect)
        phys_ridges.append(rect)

    # === 设置图 2: 超声视图 ===
    # 初始化一张空白图
    img0 = generate_b_mode_image(X, Y, [0, 0, 0])
    mesh = ax_us.pcolormesh(X, Y, img0, cmap='gray', vmin=0, vmax=1, shading='auto')

    ax_us.set_title("2. Ultrasound Imaging (B-Mode)", fontsize=12, color='white', backgroundcolor='black')
    ax_us.set_ylabel("Depth (mm)")
    ax_us.invert_yaxis()  # 深度向下增加
    ax_us.set_aspect('equal')

    # 视觉辅助线：连接物理实体和图像亮点
    connectors = []
    for c in ridge_colors:
        line, = ax_phys.plot([], [], linestyle='--', linewidth=1, color=c, alpha=0.5)
        connectors.append(line)

    # === 设置图 3: 数学原理 ===
    ax_math.set_title("3. Mathematical Decoding (Eq. 10 from Paper)", fontsize=12, color='darkgreen', loc='left')
    ax_math.set_xlabel("Rotation Angle (degrees)")
    ax_math.set_ylabel("Axial Position (mm)")
    ax_math.set_xlim(0, 360)
    ax_math.set_ylim(-5, 25)
    ax_math.grid(True, linestyle=':', alpha=0.6)

    # 绘制理论轨迹曲线
    thetas = np.linspace(0, 2 * np.pi, 360)
    traj_data = np.array([calculate_stoll_positions(t) for t in thetas])  # Shape (360, 3)

    for i in range(3):
        # 减去 center_shift 以对齐坐标系
        ax_math.plot(np.rad2deg(thetas), traj_data[:, i] - center_shift, color=ridge_colors[i], alpha=0.4)

    # 动态指示器 (垂直线和圆点)
    math_indicator_line = ax_math.axvline(x=0, color='k', linestyle='--')
    math_dots, = ax_math.plot([], [], 'ko', markersize=6)


    # --- 3. 动画更新逻辑 ---

    def update(frame):
        # 计算当前角度 (0-360度循环)
        angle_deg = frame % 360
        angle_rad = np.deg2rad(angle_deg)

        # A. 计算核心位置 (The Core Logic)
        real_positions = calculate_stoll_positions(angle_rad)
        visual_positions = real_positions - center_shift  # 对齐到显示坐标系

        # B. 更新物理视图 (Layer 1)
        for i, pos in enumerate(visual_positions):
            # 更新方块位置
            phys_ridges[i].set_x(pos - 0.75)  # 居中

            # 更新连接线 (从物理层画到超声层)
            # 这里的坐标跨越了两个子图，虽然 matplotlib 可以处理，
            # 但为了简单，我们在物理图中画一条长线指向下方
            connectors[i].set_data([pos, pos], [2, -25])

        # C. 更新超声视图 (Layer 2)
        # 重新生成带噪声的图像
        new_img = generate_b_mode_image(X, Y, real_positions)
        mesh.set_array(new_img.ravel())

        # D. 更新数学视图 (Layer 3)
        math_indicator_line.set_xdata([angle_deg, angle_deg])
        # 更新轨迹上的黑点
        math_dots.set_data([angle_deg] * 3, visual_positions)

        return [mesh, math_indicator_line, math_dots] + phys_ridges + connectors


    # --- 4. 启动动画 ---
    print("正在启动 Stoll Marker 综合仿真...")
    print("红色/绿色/蓝色方块代表缠绕的线缆。")
    print("注意观察正弦运动如何转化为超声图像上的亮点移动。")

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2),
                                  interval=30, blit=False)

    plt.tight_layout()
    plt.show()