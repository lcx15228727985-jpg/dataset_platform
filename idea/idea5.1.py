import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 物理配置
# ==========================================
ROBOT_LENGTH = 100.0
ROBOT_RADIUS = 5.0
NUM_POINTS = 600


# ==========================================
# 核心引擎 (PCC)
# ==========================================
def get_pcc_frame(s, kappa, phi):
    if abs(kappa) < 1e-6:
        return np.array([0, 0, s]), np.eye(3)

    r = 1.0 / kappa;
    theta = s * kappa
    x_loc = r * (1 - np.cos(theta));
    z_loc = r * np.sin(theta)

    cp, sp = np.cos(phi), np.sin(phi)
    R_phi = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])

    pos = R_phi @ np.array([x_loc, 0, z_loc])

    ct, st = np.cos(theta), np.sin(theta)
    tangent = R_phi @ np.array([st, 0, ct])
    normal = R_phi @ np.array([ct, 0, -st])
    binormal = np.cross(tangent, normal)
    rot = np.column_stack((normal, binormal, tangent))

    return pos, rot


def generate_cross_helix(kappa, phi):
    s_steps = np.linspace(0, ROBOT_LENGTH, NUM_POINTS)
    backbone = []

    # 定义两根反向螺旋
    # Helix 1: 左旋 (CCW), 3圈, 粗线
    # Helix 2: 右旋 (CW),  3圈, 细线
    # sign: +1 或 -1 控制旋转方向
    wires_config = [
        {"turns": 3.0, "dir": 1, "phase": 0},
        {"turns": 3.0, "dir": -1, "phase": 0}  # 镜面反向
    ]

    wires_points = [[] for _ in range(len(wires_config))]

    for s in s_steps:
        center, rot = get_pcc_frame(s, kappa, phi)
        backbone.append(center)

        for i, conf in enumerate(wires_config):
            # 核心修改：引入 direction (+1/-1)
            # angle = dir * (s/L * 2pi * N) + phase
            angle = conf["dir"] * (s / ROBOT_LENGTH) * 2 * np.pi * conf["turns"] + conf["phase"]

            local = np.array([
                ROBOT_RADIUS * np.cos(angle),
                ROBOT_RADIUS * np.sin(angle),
                0
            ])
            global_pt = center + rot @ local
            wires_points[i].append(global_pt)

    return np.array(backbone), [np.array(w) for w in wires_points]


# ==========================================
# 绘图
# ==========================================
def plot_cross_simulation():
    fig = plt.figure(figsize=(12, 6))

    # 场景: 空间弯曲
    kappa = 0.015
    phi = np.pi / 4  # 45度

    backbone, wires = generate_cross_helix(kappa, phi)

    # 1. 3D 视图
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(backbone[:, 0], backbone[:, 1], backbone[:, 2], 'k--', alpha=0.5, label='Backbone')

    # 绘制反向螺旋
    # Wire 1 (正向): 红色粗线
    ax.plot(wires[0][:, 0], wires[0][:, 1], wires[0][:, 2], 'r-', linewidth=3.0, label='Left Helix (Thick)')
    # Wire 2 (反向): 蓝色细线
    ax.plot(wires[1][:, 0], wires[1][:, 1], wires[1][:, 2], 'b-', linewidth=1.0, label='Right Helix (Thin)')

    # 标记交叉点 (示意)
    # 在 3D 图里，你会看到红蓝线构成了菱形网格

    ax.set_title(f"3D View: Mirror Reverse Helix\n(Note the Diamond Pattern)")
    ax.legend()

    # 强制比例
    limit = 60
    ax.set_xlim([-limit, limit]);
    ax.set_ylim([-limit, limit]);
    ax.set_zlim([0, 100])
    ax.view_init(elev=20, azim=30)

    # 2. 模拟展开图 (Unrolled Map) 示意
    ax2 = fig.add_subplot(1, 2, 2)

    # 简单的展开图模拟：Y轴是长度 s，X轴是角度 angle
    s_vals = np.linspace(0, 100, 600)

    # Helix 1 (Red, Thick) -> 斜率 k
    angle1 = (1 * (s_vals / 100) * 3.0 * 2 * np.pi) % (2 * np.pi)
    angle1 = np.rad2deg(angle1)
    ax2.scatter(angle1, s_vals, c='r', s=10, label='Left Helix')

    # Helix 2 (Blue, Thin) -> 斜率 -k
    angle2 = (-1 * (s_vals / 100) * 3.0 * 2 * np.pi) % (2 * np.pi)
    angle2 = np.rad2deg(angle2)
    ax2.scatter(angle2, s_vals, c='b', s=2, label='Right Helix')

    ax2.set_xlim([0, 360])
    ax2.set_ylim([0, 100])
    ax2.set_xlabel("Angle (0-360)")
    ax2.set_ylabel("Length (mm)")
    ax2.set_title("Unrolled Map: The 'X' Network")
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('cross_helix_demo.png')
    print("✨ 反向双螺旋仿真完成: cross_helix_demo.png")


if __name__ == "__main__":
    plot_cross_simulation()