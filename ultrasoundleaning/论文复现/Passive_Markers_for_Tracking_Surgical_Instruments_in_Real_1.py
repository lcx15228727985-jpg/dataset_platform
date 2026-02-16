import matplotlib

matplotlib.use('TkAgg')  # 强制弹窗
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ==========================================
# 1. 数学模型 (Stoll Marker Physics)
# ==========================================
ALPHA = 3.48
BETA = 9.02


def get_f_theta(theta):
    """ 计算脊线位置 f(theta) """
    p1 = ALPHA * np.sin(theta)
    p2 = ALPHA * np.sin(theta + 2 * np.pi / 3) + BETA
    p3 = ALPHA * np.sin(theta + 4 * np.pi / 3) + 2 * BETA
    return np.array([p1, p2, p3])


def get_f_prime(theta):
    """ 计算导数 f'(theta) """
    d1 = ALPHA * np.cos(theta)
    d2 = ALPHA * np.cos(theta + 2 * np.pi / 3)
    d3 = ALPHA * np.cos(theta + 4 * np.pi / 3)
    return np.array([d1, d2, d3])


def calculate_instant_error(theta):
    """ 计算瞬时误差因子 """
    # 1. 导数向量
    f_prime = get_f_prime(theta)
    norm_f_prime = np.linalg.norm(f_prime)

    # 2. 平移向量 u (归一化)
    u = np.ones(3) / np.sqrt(3)

    # 3. 夹角正弦值
    dot = np.dot(f_prime, u)
    cos_phi = np.clip(dot / (norm_f_prime + 1e-9), -1, 1)
    sin_phi = np.sqrt(1 - cos_phi ** 2)
    if sin_phi < 1e-3: sin_phi = 1e-3  # 防止除零

    # 4. 计算因子
    e_t = 1.0 / sin_phi
    e_theta = 1.0 / (norm_f_prime * sin_phi)

    return e_t, e_theta


# ==========================================
# 2. 动画配置
# ==========================================
if __name__ == "__main__":
    # 预计算数据用于背景绘制
    thetas = np.linspace(0, 2 * np.pi, 360)
    traj_data = np.array([get_f_theta(t) for t in thetas]).T

    errors = np.array([calculate_instant_error(t) for t in thetas])
    e_t_data = errors[:, 0]
    e_theta_data = errors[:, 1]

    # 初始化画布
    fig, (ax_traj, ax_err) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Real-time Error Sensitivity Analysis", fontsize=16, fontweight='bold')

    # --- 上图：物理轨迹 ---
    ax_traj.set_title("1. Physical Pattern State (Ridge Positions)")
    ax_traj.set_ylabel("Axial Position (mm)")
    ax_traj.grid(True, linestyle=':', alpha=0.6)
    ax_traj.set_ylim(-5, 25)

    # 绘制静态背景线
    ax_traj.plot(np.rad2deg(thetas), traj_data[0], 'r--', alpha=0.3)
    ax_traj.plot(np.rad2deg(thetas), traj_data[1], 'g--', alpha=0.3)
    ax_traj.plot(np.rad2deg(thetas), traj_data[2], 'b--', alpha=0.3)

    # 动态指示器
    line_traj, = ax_traj.plot([], [], 'k-', linewidth=1.5)  # 垂直扫描线
    dots_traj, = ax_traj.plot([], [], 'ko', markersize=8)  # 当前点

    # --- 下图：误差监测 ---
    ax_err.set_title("2. System Health Monitor (Error Factor)")
    ax_err.set_ylabel("Error Sensitivity (Log Scale)")
    ax_err.set_xlabel("Rotation Angle (degrees)")
    ax_err.set_xlim(0, 360)
    ax_err.set_ylim(0, 10)  # 限制显示范围
    ax_err.grid(True)

    # 绘制静态背景误差线
    ax_err.plot(np.rad2deg(thetas), e_theta_data * 5, 'b-', alpha=0.3, label='Rotation Error')
    ax_err.plot(np.rad2deg(thetas), e_t_data, 'r-', alpha=0.3, label='Translation Error')
    ax_err.legend(loc='upper right')

    # 动态指示器
    line_err, = ax_err.plot([], [], 'k-', linewidth=1.5)
    dot_err_rot, = ax_err.plot([], [], 'bo', markersize=8)
    dot_err_trans, = ax_err.plot([], [], 'ro', markersize=8)

    # 状态文本框
    status_text = ax_err.text(180, 8, "", ha='center', fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle="round", fc="white", ec="black"))

    # 填充背景色块 (Danger Zones)
    # 预先找出高误差区域并涂红
    threshold = 4.0
    high_error_indices = np.where(e_theta_data * 5 > threshold)[0]


    # 简单处理：画出所有高误差时刻的垂直背景带
    # (为了动画流畅，这里不在循环里动态画矩形，而是依赖 status_text 变色)

    # --- 动画更新函数 ---
    def update(frame):
        angle_deg = frame % 360
        angle_rad = np.deg2rad(angle_deg)
        idx = int(angle_deg) if int(angle_deg) < 360 else 359

        # 1. 更新上图数据
        curr_pos = get_f_theta(angle_rad)
        line_traj.set_data([angle_deg, angle_deg], [-10, 30])
        dots_traj.set_data([angle_deg] * 3, curr_pos)

        # 2. 更新下图数据
        curr_e_t = e_t_data[idx]
        curr_e_theta = e_theta_data[idx]

        line_err.set_data([angle_deg, angle_deg], [0, 20])
        dot_err_rot.set_data([angle_deg], [curr_e_theta * 5])
        dot_err_trans.set_data([angle_deg], [curr_e_t])

        # 3. 状态判定逻辑 (核心)
        # 放大旋转误差以便观察
        scaled_rot_err = curr_e_theta * 5

        if scaled_rot_err > 5.0:
            status = "CRITICAL: DEAD ZONE"
            color = "#FFcccc"  # 浅红背景
            edge = "red"
        elif scaled_rot_err > 2.0:
            status = "WARNING: UNSTABLE"
            color = "#FFEEcc"  # 浅黄背景
            edge = "orange"
        else:
            status = "SYSTEM STABLE"
            color = "#ccFFcc"  # 浅绿背景
            edge = "green"

        status_text.set_text(f"Angle: {angle_deg}° | {status}\nError Factor: {scaled_rot_err:.2f}")
        status_text.set_bbox(dict(boxstyle="round", fc=color, ec=edge, linewidth=2))

        return line_traj, dots_traj, line_err, dot_err_rot, dot_err_trans, status_text


    # 启动动画
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=30, blit=False)

    print("动画正在播放... 注意观察状态框颜色的变化！")
    plt.tight_layout()
    plt.show()