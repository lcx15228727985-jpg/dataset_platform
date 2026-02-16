import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def simulate_ultrasound_scan(angle_deg, image_size=256):
    """
    模拟超声扫描
    angle_deg: 探头照射角度 (0度为水平从左向右照射)
    """
    # 1. 建立空间网格
    x = np.linspace(-10, 10, image_size)
    y = np.linspace(-10, 10, image_size)
    X, Y = np.meshgrid(x, y)

    # 2. 定义物体几何
    radius = 6.0
    # 圆柱体的隐函数方程: R - sqrt(x^2 + y^2)
    dist_map = np.sqrt(X ** 2 + Y ** 2)
    cylinder_mask = (dist_map <= radius)
    surface_mask = np.abs(dist_map - radius) < 0.15  # 圆柱表面

    # 3. 定义线缆 (随机点)
    np.random.seed(42)  # 固定随机种子以便对比
    wire_angles = np.random.uniform(0, 2 * np.pi, 12)  # 12根线缆
    wire_mask = np.zeros_like(X)
    for w_a in wire_angles:
        wx = radius * np.cos(w_a)
        wy = radius * np.sin(w_a)
        # 线缆是个小点
        wire_mask += np.exp(-((X - wx) ** 2 + (Y - wy) ** 2) / 0.2)

    # 4. 模拟波束方向 (Beam Direction)
    # 探头角度转换为弧度
    theta = np.deg2rad(angle_deg)
    beam_dir = np.array([-np.cos(theta), -np.sin(theta)])  # 波束方向向量

    # 5. 计算反射强度 (核心物理建模)

    # A. 圆柱体的镜面反射 (Specular Reflection)
    # 计算表面法向量
    norm_X = X / (dist_map + 1e-5)
    norm_Y = Y / (dist_map + 1e-5)

    # 计算 波束 与 法向量 的点积 (余弦值)
    # 只有当波束与法向量平行(垂直表面)时，反射最强
    alignment = -(norm_X * beam_dir[0] + norm_Y * beam_dir[1])
    alignment[alignment < 0] = 0  # 背对探头的面看不到

    # 镜面反射非常敏感，角度稍微偏一点，信号就指数级衰减
    specular_intensity = np.power(alignment, 20) * surface_mask

    # B. 线缆的散射 (Scattering)
    # 散射对角度不敏感，主要看是否被照亮
    # 简单的遮挡逻辑：我们在正面，所以都能看到
    scattering_intensity = wire_mask * 0.8  # 散射强度通常低于镜面反射

    # 6. 合成图像 (RF -> Envelope)
    raw_image = specular_intensity + scattering_intensity

    # 添加斑点噪声 (Speckle Noise) - 超声的典型特征
    noise = np.random.rayleigh(scale=0.1, size=raw_image.shape)
    b_mode = raw_image + raw_image * noise  # 乘性噪声

    # 模拟点扩散函数 (PSF)，使图像看起来更有"模糊感"
    b_mode = gaussian_filter(b_mode, sigma=1.0)

    return X, Y, b_mode


# --- 执行模拟 ---
angles = [0, 45, -45]  # 0度(左侧水平), 45度(左下), -45度(左上)
plt.figure(figsize=(15, 5))

for i, ang in enumerate(angles):
    X, Y, img = simulate_ultrasound_scan(ang)

    plt.subplot(1, 3, i + 1)
    plt.title(f"Probe Angle: {ang}°")
    plt.pcolormesh(X, Y, img, cmap='gray', shading='auto', vmin=0, vmax=1)

    # 画出探头位置示意
    rad_ang = np.deg2rad(ang)
    plt.arrow(9 * np.cos(rad_ang), 9 * np.sin(rad_ang),
              -3 * np.cos(rad_ang), -3 * np.sin(rad_ang),
              head_width=1, color='yellow', label='Beam Dir')

    plt.axis('equal')
    plt.xlabel('Lateral (cm)')
    plt.ylabel('Depth (cm)')
    if i == 0: plt.legend()

plt.tight_layout()
plt.show()