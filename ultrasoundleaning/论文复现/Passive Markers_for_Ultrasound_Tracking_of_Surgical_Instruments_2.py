import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 定义标记参数 (与之前一致)
# ==========================================
ALPHA = 3.48  # 振幅 [cite: 172]
BETA = 9.02  # 偏置 [cite: 172]
OFFSET_A = 0  # 假设基础偏移为0


def get_theoretical_pattern(theta_rad):
    """
    计算给定角度下的理论指纹 f(theta) [cite: 130]
    """
    p1 = ALPHA * np.sin(theta_rad) + OFFSET_A
    p2 = ALPHA * np.sin(theta_rad + 2 * np.pi / 3) + BETA + OFFSET_A
    p3 = ALPHA * np.sin(theta_rad + 4 * np.pi / 3) + 2 * BETA + OFFSET_A
    return np.array([p1, p2, p3])


# ==========================================
# 2. 核心算法：逆向求解器
# ==========================================

def solve_pose(measured_positions):
    """
    输入: measured_positions (list of 3 floats) -> 图像上测量到的3个点的位置
    输出: estimated_theta (degrees), estimated_t (mm), cost_curve (用于绘图)
    """
    l = np.array(measured_positions)
    n = 3

    # --- 步骤 A: 解算平移 t  ---
    # 理论推导: sum(f(theta)) = 3*beta + 3*a (正弦项相加为0)
    # sum(l) = sum(f) + n*t
    # t = (sum(l) - sum(f)) / n
    sum_f_theoretical = 3 * BETA + 3 * OFFSET_A
    t_est = (np.sum(l) - sum_f_theoretical) / n

    # --- 步骤 B: 解算旋转 theta  ---
    # 我们使用暴力搜索法 (Brute Force) 扫描 0-360 度
    # 因为函数是非凸的，但在 0-2pi 内是简单的，暴力搜索最稳健且直观
    search_space = np.linspace(0, 2 * np.pi, 360)  # 1度分辨率
    costs = []

    # 既然 t 已经算出，我们寻找让误差最小的 alpha
    # Error = || l - f(alpha) - t ||
    for alpha in search_space:
        f_alpha = get_theoretical_pattern(alpha)
        predicted_l = f_alpha + t_est

        # 计算欧几里得距离 (L2 Norm)
        error = np.linalg.norm(l - predicted_l)
        costs.append(error)

    costs = np.array(costs)

    # 找到误差最小的索引
    min_idx = np.argmin(costs)
    theta_est_rad = search_space[min_idx]

    return np.rad2deg(theta_est_rad), t_est, costs


# ==========================================
# 3. 验证实验
# ==========================================

if __name__ == "__main__":
    # --- A. 设定真值 (Ground Truth) ---
    true_theta_deg = 145.0  # 假设器械转到了 145 度
    true_t_mm = 5.0  # 假设器械向后平移了 5 mm

    print(f"1. 设定真值: 角度={true_theta_deg}°, 平移={true_t_mm}mm")

    # --- B. 生成模拟观测数据 (Forward) ---
    # 生成理论位置
    l_clean = get_theoretical_pattern(np.deg2rad(true_theta_deg)) + true_t_mm

    # 添加测量噪声 (模拟超声测距误差)
    # 论文中提到测量误差约为 0.22 mm
    noise = np.random.normal(0, 0.22, 3)
    l_measured = l_clean + noise

    print(f"2. 模拟测量值 (带噪声): {np.round(l_measured, 2)}")

    # --- C. 执行解算 (Inverse) ---
    est_theta, est_t, cost_curve = solve_pose(l_measured)

    print(f"3. 解算结果: 角度={est_theta:.1f}°, 平移={est_t:.2f}mm")
    print(f"4. 误差: Δθ={abs(est_theta - true_theta_deg):.2f}°, Δt={abs(est_t - true_t_mm):.2f}mm")

    # --- D. 可视化误差曲线 (Cost Function) ---
    plt.figure(figsize=(10, 6))

    angles = np.linspace(0, 360, 360)
    plt.plot(angles, cost_curve, 'b-', linewidth=2, label='Matching Error')

    # 标记最低点
    min_val = np.min(cost_curve)
    plt.plot(est_theta, min_val, 'ro', markersize=10, label=f'Estimated Min ({est_theta:.1f}°)')

    # 标记真值
    plt.axvline(x=true_theta_deg, color='g', linestyle='--', label=f'True Angle ({true_theta_deg}°)')

    plt.title("Algorithm's 'Thinking' Process: Cost Function Analysis", fontsize=14)
    plt.xlabel("Candidate Angle (Degrees)")
    plt.ylabel("Matching Error (Residual)")
    plt.legend()
    plt.grid(True)
    plt.show()