import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 引入模块
from modules.texture import HelicalChirpTexture
from modules.geometry import GeometryEngine
try:
    from modules.ultrasound import UltrasoundScanner
except ImportError:
    from modules.ultrasound import FastUltrasoundScanner as UltrasoundScanner

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠️ 未检测到 scikit-image，将使用简易 SSIM 计算。")

# --- 移植自 train_texture.py 的安全渲染函数 ---
def render_b_mode_safe(scanner, height_profile):
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

def check_uniqueness_matrix():
    print("🔬 启动高阶唯一性验证 (Matrix Analysis)...")
    
    device = torch.device("cpu")
    
    # 1. 加载系统
    tex = HelicalChirpTexture(max_height=3.0).to(device)
    if os.path.exists("optimized_texture.pth"):
        tex.load_state_dict(torch.load("optimized_texture.pth", map_location=device))
        print("✅ Loaded: optimized_texture.pth")
    else:
        print("⚠️ Warning: Using random weights.")
            
    geo = GeometryEngine().to(device)
    scan = UltrasoundScanner(probe_width=30.0, image_depth=8.0, radius=10.5).to(device)
    
    # 生成器函数
    def get_img(z, theta, kappa):
        with torch.no_grad():
            z_t = torch.tensor([z], dtype=torch.float32)
            th_t = torch.tensor([np.deg2rad(theta)], dtype=torch.float32)
            k_t = torch.tensor([kappa], dtype=torch.float32).unsqueeze(1)
            p_t = torch.tensor([0.0], dtype=torch.float32).unsqueeze(1)
            
            full, _ = tex()
            tex_in = full if full.dim() == 4 else full.unsqueeze(0).unsqueeze(0)
            
            # [修复] 完整的三步走：get_slice_grid -> flatten -> geo -> reshape -> render
            grid_z, grid_th = scan.get_slice_grid(z_t, th_t)
            
            h_flat = geo(
                tex_in, k_t, p_t, 
                grid_z.reshape(1, -1), 
                grid_th.reshape(1, -1)
            )
            
            h_surface = h_flat.reshape(1, scan.H_img, scan.W_img)[:, 0, :]
            img = render_b_mode_safe(scan, h_surface)
            return img.squeeze().numpy()

    # 2. 定义测试序列
    states = []
    labels = []
    
    base_z = 50.0
    base_th = 0.0
    
    # Z轴微变序列
    for dz in [0, 0.5, 1.0, 2.0, 5.0]:
        states.append({'z': base_z + dz, 'theta': base_th, 'kappa': 0.0})
        labels.append(f"Z+{dz}mm")
        
    # 角度微变序列
    for dth in [5, 10, 45]:
        states.append({'z': base_z, 'theta': base_th + dth, 'kappa': 0.0})
        labels.append(f"Th+{dth}°")

    # 混合变化
    states.append({'z': base_z + 1.0, 'theta': base_th + 5.0, 'kappa': 0.0})
    labels.append(f"Mix")

    num_states = len(states)
    images = []
    
    print(f"📸 Generating {num_states} sample images...")
    for s in states:
        images.append(get_img(s['z'], s['theta'], s['kappa']))
        
    # 3. 计算相关性
    ncc_matrix = np.zeros((num_states, num_states))
    ssim_matrix = np.zeros((num_states, num_states))
    
    print("🧮 Computing Similarity Matrices...")
    for i in range(num_states):
        for j in range(num_states):
            ncc_matrix[i, j] = calculate_ncc(images[i], images[j])
            ssim_matrix[i, j] = calculate_ssim_simple(images[i], images[j])

    # 4. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(ncc_matrix, cmap='viridis_r', vmin=0.0, vmax=1.0)
    axes[0].set_title("NCC (Lower is Better)")
    axes[0].set_xticks(range(num_states)); axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_yticks(range(num_states)); axes[0].set_yticklabels(labels)
    for i in range(num_states):
        for j in range(num_states):
            color = 'white' if ncc_matrix[i, j] < 0.5 else 'black'
            axes[0].text(j, i, f"{ncc_matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(ssim_matrix, cmap='viridis_r', vmin=0.0, vmax=1.0)
    axes[1].set_title("SSIM (Lower is Better)")
    axes[1].set_xticks(range(num_states)); axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_yticks(range(num_states)); axes[1].set_yticklabels(labels)
    for i in range(num_states):
        for j in range(num_states):
            color = 'white' if ssim_matrix[i, j] < 0.5 else 'black'
            axes[1].text(j, i, f"{ssim_matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("uniqueness_matrix.png")
    print("✅ 分析完成！结果已保存至 uniqueness_matrix.png")
    
    sim_small_step = ncc_matrix[0, 1]
    print("\n🧐 诊断报告:")
    print(f"   > Z+0.5mm 相似度: {sim_small_step:.3f}")

if __name__ == "__main__":
    check_uniqueness_matrix()