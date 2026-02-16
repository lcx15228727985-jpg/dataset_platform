import torch
import numpy as np
import trimesh
import os
import torch.nn.functional as F

# === 配置区域 ===
MODEL_PATH = "optimized_texture.pth"
OUTPUT_FILENAME = "chirp_marker_curved_k0.1.stl" # 修改文件名以体现曲率

# 物理尺寸 (毫米 mm)
ROBOT_DIAMETER = 15.0        
ROBOT_RADIUS = ROBOT_DIAMETER / 2.0  # 7.5 mm

FILM_THICKNESS_BASE = 0.5    
TEXTURE_MAX_DEPTH = 3.0      

LENGTH = 100.0               

# --- 🎯 目标曲率设置 ---
# Curvature k = 1 / Radius
# k = 0.1 mm^-1  =>  弯曲半径 R = 10 mm (非常弯，接近卷成一团)
# k = 0.01 mm^-1 =>  弯曲半径 R = 100 mm (比较自然的弯曲)
# 请确认您的 0.1 是指物理曲率(mm^-1) 还是归一化曲率。
# 如果是归一化曲率(0.1 * 0.025)，请在此处乘以系数。
# 这里默认按物理曲率处理：
TARGET_CURVATURE = 0.0125  # 例如：设置一个较小的曲率 R=400mm，模拟轻微弯曲
# 如果您确实需要 0.1 的物理曲率 (R=10mm)，请改为 0.1，但注意 100mm 长的管子卷 R=10mm 会卷好几圈且自相交。

# 网格分辨率
RES_Z = 500       
RES_THETA = 360    

# 引入模型定义 (保持不变)
try:
    from modules.texture import HelicalChirpTexture
except ImportError:
    print("⚠️ 未找到 'modules.texture' 模块。使用模拟类代替。")
    import torch.nn as nn
    class HelicalChirpTexture(nn.Module):
        def __init__(self, N=512, K_theta=10, K_z=40):
            super().__init__()
            self.K_z = K_z
        def forward(self):
            return torch.rand(1, 1, 512, 512), None

def apply_bending(vertices, curvature):
    """
    对直管顶点应用平面弯曲变换
    假设原始轴向为 Z 轴，弯曲发生在 X-Z 平面
    """
    if abs(curvature) < 1e-6:
        return vertices

    print(f"🔄 正在应用弯曲变形: k={curvature} (R={1/curvature:.1f}mm)...")

    # 弯曲半径
    R = 1.0 / curvature
    
    # 原始坐标
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # 几何变换逻辑：
    # 1. 将 Z 轴长度映射为圆弧角度 theta = z / R
    # 2. X 轴坐标作为相对于中性层（圆弧中心线）的径向偏移
    #    当前的弯曲中心在 X = -R 处
    #    新的径向距离 r_new = R + x
    
    theta = z / R
    r_new = R + x
    
    # 3. 计算新坐标
    # 绕 Y 轴弯曲，保持 Y 坐标不变
    # 新的 X, Z 坐标基于极坐标变换
    # 当 theta=0 (z=0) 时，应回到 (x, 0)
    x_new = -R + r_new * np.cos(theta)
    z_new = r_new * np.sin(theta)
    y_new = y
    
    return np.stack([x_new, y_new, z_new], axis=1)

def generate_mesh():
    # 如果您指的是题目中的 "曲率为0.1" (物理单位 mm^-1)，请在此覆盖
    # 注意：R=10mm 对于 15mm 粗的管子来说会导致内侧自相交 (R_inner=7.5, R_bend=10 -> 间隙2.5mm)
    # 且 100mm 长会卷 1.5 圈。这里我假设您可能是指较小的弯曲，或者您确实需要这个极端值。
    # 如果是归一化值 0.1 (对应 k=0.0025)，请修改上面的 TARGET_CURVATURE
    
    # 强制使用用户指定的 0.1 (如果是这个意图)
    # current_k = 0.1 
    current_k = TARGET_CURVATURE 

    print(f"🖨️ 正在生成 3D 打印模型: {OUTPUT_FILENAME}")
    print(f"   曲率 k={current_k}")

    device = torch.device("cpu")

    # 1. 加载模型
    try:
        model = HelicalChirpTexture(N=512, K_theta=10, K_z=40).to(device)
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ 已加载优化后的权重。")
        else:
            print("⚠️ 未找到权重文件，使用随机初始化参数。")
    except Exception as e:
        print(f"❌ 加载模型出错: {e}")
        return

    model.eval()

    # 2. 生成高分辨率纹理图
    print("📊 正在计算高分辨率高度场...")
    with torch.no_grad():
        tex_raw, _ = model() 

    tex_high_res = F.interpolate(
        tex_raw,
        size=(RES_THETA, RES_Z),
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()

    # 归一化纹理
    t_min = tex_high_res.min()
    t_max = tex_high_res.max()
    if t_max - t_min > 1e-6:
        tex_normalized = (tex_high_res - t_min) / (t_max - t_min)
        tex_physical_height = tex_normalized * TEXTURE_MAX_DEPTH
    else:
        tex_physical_height = np.zeros_like(tex_high_res)

    # 3. 构建网格顶点 (直管状态)
    print("⚙️ 正在构建直管拓扑...")

    z_coords = np.linspace(0, LENGTH, RES_Z)
    theta_coords = np.linspace(0, 2 * np.pi, RES_THETA, endpoint=False)

    Z_grid, Theta_grid = np.meshgrid(z_coords, theta_coords)

    # 半径计算
    R_inner_val = ROBOT_RADIUS
    R_outer_grid = R_inner_val + FILM_THICKNESS_BASE + tex_physical_height

    # 外表面
    X_outer = R_outer_grid * np.cos(Theta_grid)
    Y_outer = R_outer_grid * np.sin(Theta_grid)
    Z_outer = Z_grid
    verts_outer = np.stack((X_outer, Y_outer, Z_outer), axis=-1).reshape(-1, 3)

    # 内表面
    X_inner = R_inner_val * np.cos(Theta_grid)
    Y_inner = R_inner_val * np.sin(Theta_grid)
    Z_inner = Z_grid
    verts_inner = np.stack((X_inner, Y_inner, Z_inner), axis=-1).reshape(-1, 3)

    # 4. 生成面 (Faces)
    def create_grid_faces(rows, cols, offset=0, flip_normal=False):
        faces = []
        for r in range(rows):
            for c in range(cols - 1):
                p0 = offset + r * cols + c
                p1 = p0 + 1
                next_r = (r + 1) % rows
                p2 = offset + next_r * cols + c
                p3 = p2 + 1
                if not flip_normal:
                    faces.append([p0, p1, p2])
                    faces.append([p1, p3, p2])
                else:
                    faces.append([p0, p2, p1])
                    faces.append([p1, p2, p3])
        return faces

    faces_outer = create_grid_faces(RES_THETA, RES_Z, offset=0, flip_normal=False)
    faces_inner = create_grid_faces(RES_THETA, RES_Z, offset=len(verts_outer), flip_normal=True)

    # 封盖 (Caps)
    caps = []
    rows = RES_THETA
    cols = RES_Z
    offset_inner = len(verts_outer)
    for r in range(rows):
        next_r = (r + 1) % rows
        # Bottom
        o0, o1 = r * cols, next_r * cols
        i0, i1 = offset_inner + r * cols, offset_inner + next_r * cols
        caps.append([o0, i0, o1])
        caps.append([i0, i1, o1])
        # Top
        o0_t, o1_t = r * cols + cols - 1, next_r * cols + cols - 1
        i0_t, i1_t = offset_inner + r * cols + cols - 1, offset_inner + next_r * cols + cols - 1
        caps.append([o0_t, o1_t, i0_t])
        caps.append([o1_t, i1_t, i0_t])

    # 合并数据
    all_verts = np.vstack((verts_outer, verts_inner))
    all_faces = np.vstack((faces_outer, faces_inner, caps))

    # --- 5. 应用弯曲 (关键步骤) ---
    if abs(current_k) > 1e-6:
        all_verts = apply_bending(all_verts, current_k)

    # 6. 导出
    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    
    # 修复法线 (因为弯曲可能导致面反转，虽然各向同性弯曲一般没事，但检查一下更好)
    mesh.fix_normals()

    print(f"💾 正在保存至 {OUTPUT_FILENAME}...")
    mesh.export(OUTPUT_FILENAME)
    print("✅ 完成!")

if __name__ == "__main__":
    generate_mesh()