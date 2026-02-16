import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 引入模块
from modules.texture import HelicalChirpTexture
from modules.geometry import GeometryEngine
try:
    from modules.ultrasound import UltrasoundScanner
except ImportError:
    from modules.ultrasound import FastUltrasoundScanner as UltrasoundScanner

# 尝试引入 skimage
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠️ 未检测到 scikit-image，SSIM 将回退为 NCC 计算。建议 pip install scikit-image")

# --- 移植自 train_texture.py 的安全渲染函数 ---
def render_b_mode_safe(scanner, height_profile):
    """
    Args:
        height_profile: [B, W] 表面高度曲线
    Returns:
        intensity: [B, H_img, W]
    """
    B, W = height_profile.shape
    H_img = scanner.H_img
    device = height_profile.device
    
    depth_grid = torch.linspace(0, scanner.image_depth, H_img, device=device).view(1, -1, 1)
    surface_depth = scanner.probe_offset - height_profile.unsqueeze(1)
    diff = depth_grid - surface_depth
    
    thickness = scanner.res_axial * 1.5
    intensity = torch.exp(-(diff**2) / (2 * thickness**2))
    
    dz = torch.abs(height_profile[:, 1:] - height_profile[:, :-1])
    dz = torch.cat([dz, dz[:, -1:]], dim=1)
    reflectivity = 1.0 / (1.0 + 8.0 * (dz ** 2))
    
    return intensity * reflectivity.unsqueeze(1)

def calculate_ncc(img1, img2):
    v1 = img1.flatten()
    v2 = img2.flatten()
    v1 = v1 - np.mean(v1)
    v2 = v2 - np.mean(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def calculate_ssim_simple(img1, img2):
    if HAS_SKIMAGE:
        return ssim(img1, img2, data_range=img2.max() - img2.min())
    else:
        return calculate_ncc(img1, img2)

def check_global_uniqueness():
    print("🌍 启动全局唯一性验证 (Global Uniqueness Check)...")
    
    device = torch.device("cpu") # 测试通常用 CPU 即可
    
    # 1. 加载系统
    tex = HelicalChirpTexture(max_height=3.0).to(device)
    weights = ["optimized_texture.pth", "initial_texture.pth"]
    for w in weights:
        if os.path.exists(w):
            tex.load_state_dict(torch.load(w, map_location=device))
            print(f"✅ 已加载纹理权重: {w}")
            break
            
    geo = GeometryEngine().to(device)
    # 注意这里半径最好与训练一致
    scan = UltrasoundScanner(probe_width=30.0, image_depth=8.0, radius=10.5).to(device)
    
    # 2. 全局随机采样
    N_SAMPLES = 100 
    print(f"🎲 在全局空间随机采样 {N_SAMPLES} 个状态点...")
    
    # 随机生成参数并按 Z 排序
    z_vals = np.random.uniform(15.0, 85.0, N_SAMPLES)
    theta_vals = np.random.uniform(0, 360, N_SAMPLES)
    kappa_vals = np.random.uniform(0, 0.025, N_SAMPLES)
    phi_vals = np.random.uniform(0, 360, N_SAMPLES)
    
    sort_idx = np.argsort(z_vals)
    z_vals = z_vals[sort_idx]
    theta_vals = theta_vals[sort_idx]
    kappa_vals = kappa_vals[sort_idx]
    phi_vals = phi_vals[sort_idx]
    
    images = []
    print("📸 生成模拟图像中...")
    
    with torch.no_grad():
        full_tex, _ = tex()
        full_tex_batch = full_tex.expand(1, -1, -1, -1) # [1, 1, H, W]
        
        for i in tqdm(range(N_SAMPLES)):
            # 准备参数 [1]
            z_t = torch.tensor([z_vals[i]], dtype=torch.float32)
            th_t = torch.tensor([np.deg2rad(theta_vals[i])], dtype=torch.float32)
            k_t = torch.tensor([kappa_vals[i]], dtype=torch.float32).unsqueeze(1)
            p_t = torch.tensor([np.deg2rad(phi_vals[i])], dtype=torch.float32).unsqueeze(1)
            
            # [修复 1] 使用 get_slice_grid 而不是 get_scan_line_grid
            grid_z, grid_th = scan.get_slice_grid(z_t, th_t)
            
            # [修复 2] 展平网格后再传给 geo
            h_flat = geo(
                full_tex_batch, k_t, p_t, 
                grid_z.reshape(1, -1), 
                grid_th.reshape(1, -1)
            )
            
            # [修复 3] 恢复形状并取表面 [1, H_img, W_img] -> 取 dim 1 的第 0 个 (表面)
            h_surface = h_flat.reshape(1, scan.H_img, scan.W_img)[:, 0, :] # [1, W]
            
            # [修复 4] 使用安全渲染器
            img = render_b_mode_safe(scan, h_surface)
            
            images.append(img.squeeze().numpy())

    # 3. 计算相关性
    print("🧮 计算 N x N 相关性矩阵...")
    ncc_matrix = np.zeros((N_SAMPLES, N_SAMPLES))
    ssim_matrix = np.zeros((N_SAMPLES, N_SAMPLES))
    
    for i in range(N_SAMPLES):
        for j in range(N_SAMPLES):
            if i == j:
                ncc_matrix[i, j] = 1.0
                ssim_matrix[i, j] = 1.0
            else:
                ncc_matrix[i, j] = calculate_ncc(images[i], images[j])
                ssim_matrix[i, j] = calculate_ssim_simple(images[i], images[j])

    # 4. 统计与绘图
    mask = ~np.eye(N_SAMPLES, dtype=bool)
    avg_ncc = ncc_matrix[mask].mean()
    max_ncc = ncc_matrix[mask].max()
    
    print("\n📊 全局统计报告:")
    print(f"   > 平均互相关 (Avg NCC): {avg_ncc:.4f}")
    print(f"   > 最大互相关 (Max NCC): {max_ncc:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    im1 = axes[0].imshow(ncc_matrix, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[0].set_title(f"NCC Matrix (Sorted by Z)\nGlobal Uniqueness (N={N_SAMPLES})")
    axes[0].set_xlabel("Sample Index (Z sorted)")
    axes[0].set_ylabel("Sample Index")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(ssim_matrix, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title(f"SSIM Matrix (Sorted by Z)")
    axes[1].set_xlabel("Sample Index (Z sorted)")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("global_uniqueness.png")
    print("✅ 结果已保存至 global_uniqueness.png")

if __name__ == "__main__":
    check_global_uniqueness()