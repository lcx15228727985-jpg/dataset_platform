import torch
import torch.nn as nn
import numpy as np
import trimesh
import os
import torch.nn.functional as F

# === 配置区域 ===
MODEL_PATH = "optimized_texture.pth"
OUTPUT_FILENAME = "chirp_marker_15mm_dia.stl"

# 物理尺寸 (毫米 mm)
# 注意：这里是指机器人的内径 (套管内孔)
ROBOT_DIAMETER = 15.0        
ROBOT_RADIUS = ROBOT_DIAMETER / 2.0  # 7.5 mm

# 为了保证打印强度，通常加 0.5mm 基底
# 如果您希望总厚度严格为 3mm (含纹理)，可以将 BASE 设为 0，但这可能导致薄处破裂
FILM_THICKNESS_BASE = 0.5    
TEXTURE_MAX_DEPTH = 3.0      

LENGTH = 100.0               # 套管长度

# 网格分辨率 (越高越精细，但文件越大)
RES_Z = 500       
RES_THETA = 360    

# ==========================================
# [关键修复] 更新模型类定义以匹配训练代码
# ==========================================
class HelicalChirpTexture(nn.Module):
    def __init__(self, N=512, K_theta=10, K_z=40, max_height=3.0):
        super().__init__()
        self.N = N
        self.K_theta = K_theta
        self.K_z = K_z
        self.max_height = max_height 
        
        # 参数矩阵
        self.coeffs = nn.Parameter(torch.zeros(2 * K_theta + 1, K_z))
        
        # [修复] 必须包含 freq_scale，否则加载权重会报错
        self.freq_scale = nn.Parameter(torch.tensor(1.0))
        
        self._initialize_weights()
        self._precompute_basis()

    def _initialize_weights(self):
        # 初始化逻辑只需保持结构一致即可，数值会被 load_state_dict 覆盖
        pass 

    def _precompute_basis(self):
        n_z = torch.arange(self.N).float()
        k_z = torch.arange(self.K_z).float()
        grid_z = torch.outer(n_z + 0.5, k_z)
        self.register_buffer('basis_z', torch.cos((np.pi / self.N) * grid_z))
        
        theta_vals = torch.linspace(0, 2 * np.pi, self.N + 1)[:-1]
        basis_list = [torch.ones(self.N, 1)]
        for k in range(1, self.K_theta + 1):
            k_t = torch.tensor(float(k))
            basis_list.append(torch.cos(k_t * theta_vals).unsqueeze(1))
            basis_list.append(torch.sin(k_t * theta_vals).unsqueeze(1))
        self.register_buffer('basis_theta', torch.cat(basis_list, dim=1))

    def forward(self):
        # 1. 基础信号
        raw_base = self.basis_theta @ self.coeffs @ self.basis_z.T
        
        # 2. 频率缩放
        if self.freq_scale != 1.0:
            B, H, W = 1, self.N, self.N
            img = raw_base.view(1, 1, H, W)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=raw_base.device),
                torch.linspace(-1, 1, W, device=raw_base.device) * self.freq_scale
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
            raw = torch.nn.functional.grid_sample(
                img, grid, align_corners=True, padding_mode='border'
            ).squeeze()
        else:
            raw = raw_base

        # 3. 映射与四阶梯量化
        normalized = (torch.tanh(raw) + 1.0) / 2.0
        val = normalized * 4.0 
        sharpness = 30.0
        
        step1 = torch.sigmoid((val - 0.5) * sharpness)
        step2 = torch.sigmoid((val - 1.5) * sharpness)
        step3 = torch.sigmoid((val - 2.5) * sharpness)
        
        phys = (step1 + step2 + step3) * 1.0
        phys = torch.clamp(phys, 0, self.max_height)
        
        return phys.unsqueeze(0).unsqueeze(0), raw

# ==========================================
# 生成主逻辑
# ==========================================
def generate_mesh():
    print(f"🖨️ 正在生成 3D 打印模型: {OUTPUT_FILENAME}")
    print(f"   规格: 内径={ROBOT_DIAMETER}mm, 纹理深={TEXTURE_MAX_DEPTH}mm")

    device = torch.device("cpu")

    # 1. 加载模型
    try:
        model = HelicalChirpTexture(N=512, K_theta=10, K_z=40, max_height=TEXTURE_MAX_DEPTH).to(device)
        if os.path.exists(MODEL_PATH):
            # strict=False 可以容忍一些不匹配，但最好还是匹配
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ 已成功加载权重文件。")
        else:
            print("⚠️ 未找到权重文件，将生成随机纹理。")
    except Exception as e:
        print(f"❌ 加载模型出错: {e}")
        return

    model.eval()

    # 2. 计算高度场
    print("📊 计算高分辨率纹理...")
    with torch.no_grad():
        tex_raw, _ = model() # [1, 1, 512, 512]

    # 插值到打印分辨率
    tex_high_res = F.interpolate(
        tex_raw,
        size=(RES_THETA, RES_Z),
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()

    # 归一化高度: 确保最低点是0，最高点是3.0
    # 注意：这会忽略训练时的绝对高度，强制拉伸到 0~3mm
    t_min = tex_high_res.min()
    t_max = tex_high_res.max()
    if t_max - t_min > 1e-6:
        tex_normalized = (tex_high_res - t_min) / (t_max - t_min)
        tex_physical_height = tex_normalized * TEXTURE_MAX_DEPTH
    else:
        tex_physical_height = np.zeros_like(tex_high_res)

    # 3. 构建网格
    print("⚙️ 构建 3D 网格...")
    z_coords = np.linspace(0, LENGTH, RES_Z)
    theta_coords = np.linspace(0, 2 * np.pi, RES_THETA, endpoint=False)
    Z_grid, Theta_grid = np.meshgrid(z_coords, theta_coords)

    # 半径计算
    # R_inner = 7.5mm
    # R_outer = 7.5 + 0.5(Base) + Texture(0~3)
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

    # 4. 生成面片
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

    faces_outer = create_grid_faces(RES_THETA, RES_Z, offset=0)
    faces_inner = create_grid_faces(RES_THETA, RES_Z, offset=len(verts_outer), flip_normal=True)

    # 封盖 (Caps)
    caps = []
    rows = RES_THETA
    cols = RES_Z
    offset_inner = len(verts_outer)
    for r in range(rows):
        next_r = (r + 1) % rows
        # 底部 (Z=0)
        o0, o1 = r * cols, next_r * cols
        i0, i1 = offset_inner + r * cols, offset_inner + next_r * cols
        caps.append([o0, i0, o1])
        caps.append([i0, i1, o1])
        # 顶部 (Z=L)
        o0_t, o1_t = r * cols + cols - 1, next_r * cols + cols - 1
        i0_t, i1_t = offset_inner + r * cols + cols - 1, offset_inner + next_r * cols + cols - 1
        caps.append([o0_t, o1_t, i0_t])
        caps.append([o1_t, i1_t, i0_t])

    all_verts = np.vstack((verts_outer, verts_inner))
    all_faces = np.vstack((faces_outer, faces_inner, caps))

    # 5. 导出
    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    mesh.fix_normals()
    mesh.export(OUTPUT_FILENAME)
    print(f"✅ 完成! 已保存至 {OUTPUT_FILENAME}")

if __name__ == "__main__":
    generate_mesh()