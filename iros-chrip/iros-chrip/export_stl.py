import torch
import numpy as np
import trimesh
from tqdm import tqdm
import os

# 引入您的模型定义
from modules.texture import HelicalChirpTexture  # 或 EngravingTexture，取决于您当前的类名

# === 配置 ===
MODEL_PATH = "optimized_texture.pth" # 训练好的权重文件
OUTPUT_FILENAME = "chirp_marker_3d_print.stl"
BASE_RADIUS = 7.5       # 机器人内管半径 (mm)
# 注意：最终半径 = BASE_RADIUS + 纹理高度 (0~3mm)

# 网格分辨率 (越高越精细，但文件越大)
RES_Z = 1000   # Z轴方向的分段数 (对应 Chirp 的高频细节)
RES_THETA = 360 # 圆周方向的分段数 (1度一段)

def generate_mesh():
    print("🖨️ 正在准备 3D 打印模型生成...")
    
    device = torch.device("cpu") # 导出时用 CPU 显存更充裕
    
    # 1. 加载模型
    # 确保参数与训练时一致 (N=512, K=...)
    try:
        model = HelicalChirpTexture(N=512, K_theta=10, K_z=40).to(device)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✅ 已加载优化后的权重")
        else:
            print("⚠️ 未找到权重文件，使用随机初始化的纹理 (仅供测试)")
    except Exception as e:
        print(f"❌ 模型加载出错: {e}")
        return

    model.eval()
    
    # 2. 生成高分辨率纹理图
    print("📊 正在计算高分辨率高度场...")
    
    # 我们不使用模型默认的 N=512，而是手动生成更高分辨率的网格进行采样
    # 因为 3D 打印需要极高的物理平滑度
    
    # 手动构建高分辨率的基函数矩阵 (Basis)
    # 这部分逻辑是从 texture.py 提取并 Upsample 的
    z_vals = torch.linspace(0, 1, RES_Z)
    theta_vals = torch.linspace(0, 2 * np.pi, RES_THETA + 1)[:-1] # 去掉最后一个点避免重叠(闭合时处理)
    
    # 重新计算 Basis (Upsampled)
    # Z轴 (DCT)
    n_z = torch.arange(512).float() # 原模型 N
    k_z = torch.arange(model.K_z).float()
    # 这里有点 trick: 我们不能直接改 basis 的 N，必须插值。
    # 更简单的方法：直接生成 512x512 的图，然后用双线性插值放大到目标分辨率
    
    with torch.no_grad():
        tex_small, _ = model() # [1, 1, 512, 512]
        
    # 插值放大到打印分辨率
    tex_high_res = torch.nn.functional.interpolate(
        tex_small, 
        size=(RES_THETA, RES_Z), 
        mode='bicubic', 
        align_corners=True
    ).squeeze().numpy() # [Theta, Z]
    
    print(f"   高度场尺寸: {tex_high_res.shape}")
    
    # 3. 构建圆柱网格 (Vertices & Faces)
    print("⚙️ 正在构建拓扑网格...")
    
    vertices = []
    indices = []
    
    # 物理坐标范围
    L = 100.0 # 长度 100mm
    z_coords = np.linspace(0, L, RES_Z)
    theta_coords = np.linspace(0, 2 * np.pi, RES_THETA, endpoint=False) # 0 到 2pi (不含2pi)
    
    # --- 生成顶点 (Vertices) ---
    # 展平循环以加速
    Z_grid, Theta_grid = np.meshgrid(z_coords, theta_coords) # Shape: [RES_THETA, RES_Z]
    
    # 计算物理半径 R(z, theta)
    # 刻槽模式: H 是 0~3mm
    # 实际外径 = 基底半径 + H
    R_grid = BASE_RADIUS + tex_high_res
    
    # 柱坐标转笛卡尔坐标
    X_grid = R_grid * np.cos(Theta_grid)
    Y_grid = R_grid * np.sin(Theta_grid)
    Z_grid = Z_grid # Z轴不变
    
    # 堆叠顶点: [N_verts, 3]
    vertices = np.stack((X_grid, Y_grid, Z_grid), axis=-1).reshape(-1, 3)
    
    # --- 生成面 (Faces) ---
    # 我们需要把网格点连接成三角形。
    # 注意 Theta 轴是首尾相接的 (Seam)
    
    rows = RES_THETA
    cols = RES_Z
    
    faces = []
    
    for r in range(rows):
        for c in range(cols - 1):
            # 当前点索引
            p0 = r * cols + c
            p1 = p0 + 1
            # 下一行点索引 (注意处理 Theta 闭合)
            next_r = (r + 1) % rows
            p2 = next_r * cols + c
            p3 = p2 + 1
            
            # 两个三角形组成一个矩形
            # Tri 1: p0 -> p1 -> p2
            faces.append([p0, p1, p2])
            # Tri 2: p1 -> p3 -> p2
            faces.append([p1, p3, p2])
            
    faces = np.array(faces)
    
    # 4. 创建 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 5. 封口 (Capping)
    # 目前只是一个空心的管子壁，打印需要实心或者有厚度的管子。
    # 简单做法：如果只是测试纹理，我们把两头封上，做成实心棒。
    # Trimesh 并没有自动封圆柱的功能，我们手动或者让切片软件处理。
    # 更好的做法：生成一个内壁。
    
    print("🔧 正在进行网格修复与封口...")
    
    # 简单的封口尝试 (可能会失败，取决于拓扑)
    # 如果想做成套管 (Sleeve)，我们需要生成内壁。
    # 这里为了简便，我们做一个实心体，通过将两端的圆心加进去。
    
    # 方法：使用 convex hull 并不是个好主意因为表面是凹凸的。
    # 我们保留空心圆柱，但在切片软件(Chitubox/Cura)中设置为 "Solid" 或者手动加盖。
    # 或者，我们这里生成内壁。
    
    # --- 生成内壁 (Inner Wall) ---
    # 内壁半径 = BASE_RADIUS
    # 内壁不需要高分辨率，但为了顶点匹配，使用相同的 RES
    R_inner = BASE_RADIUS - 0.5 # 稍微小一点，或者直接用 BASE_RADIUS 作为一个 0mm 厚度的参考面
    # 这里假设我们要打印一个 2mm 厚的基底套管
    R_substrate = BASE_RADIUS - 2.0 
    
    X_inner = R_substrate * np.cos(Theta_grid)
    Y_inner = R_substrate * np.sin(Theta_grid)
    Z_inner = Z_grid
    
    verts_inner = np.stack((X_inner, Y_inner, Z_inner), axis=-1).reshape(-1, 3)
    
    # 内壁的面索引 (注意法线方向要反向，朝内)
    faces_inner = []
    offset = len(vertices)
    for r in range(rows):
        for c in range(cols - 1):
            p0 = offset + r * cols + c
            p1 = p0 + 1
            next_r = (r + 1) % rows
            p2 = offset + next_r * cols + c
            p3 = p2 + 1
            
            # 法线反向: p0 -> p2 -> p1
            faces_inner.append([p0, p2, p1])
            faces_inner.append([p1, p2, p3])
            
    faces_inner = np.array(faces_inner)
    
    # 合并内外壁
    all_verts = np.vstack((vertices, verts_inner))
    all_faces = np.vstack((faces, faces_inner))
    
    # --- 封顶 (Caps) ---
    # 连接外壁和内壁的顶端和底端环
    # Top ring (c = cols-1), Bottom ring (c = 0)
    caps = []
    
    for r in range(rows):
        next_r = (r + 1) % rows
        
        # Bottom Cap (c=0)
        # Outer: p0, p2; Inner: ip0, ip2
        o0 = r * cols + 0
        o1 = next_r * cols + 0
        i0 = offset + r * cols + 0
        i1 = offset + next_r * cols + 0
        
        # Face 1: o0 -> o1 -> i0
        caps.append([o0, o1, i0])
        # Face 2: i0 -> o1 -> i1
        caps.append([i0, o1, i1])
        
        # Top Cap (c = cols-1)
        o0_t = r * cols + (cols - 1)
        o1_t = next_r * cols + (cols - 1)
        i0_t = offset + r * cols + (cols - 1)
        i1_t = offset + next_r * cols + (cols - 1)
        
        # Face 1: o0 -> i0 -> o1 (Reverse normal)
        caps.append([o0_t, i0_t, o1_t])
        # Face 2: i0 -> i1 -> o1
        caps.append([i0_t, i1_t, o1_t])
        
    caps = np.array(caps)
    all_faces = np.vstack((all_faces, caps))
    
    # 最终网格
    final_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    
    # 6. 导出
    print(f"💾 正在保存至 {OUTPUT_FILENAME} ...")
    final_mesh.export(OUTPUT_FILENAME)
    print(f"✅ 成功! 文件大小: {os.path.getsize(OUTPUT_FILENAME)/1024/1024:.2f} MB")
    print("👉 请使用 Chitubox 或 Lychee Slicer 打开并进行 3D 打印。建议使用 0.05mm 或更低层高。")

if __name__ == "__main__":
    generate_mesh()