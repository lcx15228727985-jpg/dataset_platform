import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 定义两种设计模式
# ==========================================

def get_continuous_error(theta):
    """ 连续设计 (正弦波): 有波峰波谷，导致斜率为0 """
    # 导数是 Cosine，会在 90度和270度归零
    slope = np.abs(np.cos(theta))
    # 误差因子与斜率成反比 (斜率越小，误差越大)
    # 加一个小常数防止除零爆炸
    error = 1.0 / (slope + 0.05)
    return error


def get_discontinuous_error(theta):
    """ 非连续设计 (锯齿波/平行线): 强制斜率恒定 """
    # 设计一种永远保持 45度斜率的线条
    # 无论角度怎么转，物理上我们都切换到了一根斜率陡峭的线上
    constant_slope = 0.8  # 假设一个理想的恒定高斜率
    error = 1.0 / constant_slope
    # 返回的是一个常数，不随 theta 变化
    return np.full_like(theta, error)


# ==========================================
# 可视化对比
# ==========================================
if __name__ == "__main__":
    thetas = np.linspace(0, 2 * np.pi, 360)

    err_cont = get_continuous_error(thetas)
    err_disc = get_discontinuous_error(thetas)

    plt.figure(figsize=(10, 6))

    # 绘制连续设计的误差 (红色)
    plt.plot(np.rad2deg(thetas), err_cont, 'r-', linewidth=2, alpha=0.6, label='Continuous (Sine Wave)')
    # 填充红色危险区
    plt.fill_between(np.rad2deg(thetas), 0, err_cont, color='red', alpha=0.1)

    # 绘制非连续设计的误差 (绿色)
    plt.plot(np.rad2deg(thetas), err_disc, 'g-', linewidth=3, label='Discontinuous (Optimized)')

    plt.title("Why Discontinuous Design is Better: Stability Analysis", fontsize=14)
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Tracking Error Sensitivity (Lower is Better)")
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)

    plt.text(180, 5, "Dead Zones Eliminated!", color='green', fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.show()

    """
    红线（连续）： 依然像过山车一样，有几个巨大的尖峰（误差爆炸）。

绿线（非连续）： 变成了一条又低又平的直线。这意味着无论器械转到哪个角度，追踪精度都一样高，完全消除了死区。
    """