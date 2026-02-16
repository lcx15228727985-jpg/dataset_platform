import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

# --- 引入模块 ---
from modules.texture import HelicalChirpTexture
from modules.geometry import GeometryEngine
try:
    from modules.ultrasound import UltrasoundScanner
except ImportError:
    from modules.ultrasound import FastUltrasoundScanner as UltrasoundScanner

# ==========================================
# [关键修复] 自定义安全渲染函数
# ==========================================
def render_b_mode_safe(scanner, height_profile):
    """
    修正版 B-Mode 渲染器。
    解决 ultrasound.py 中 expand() 维度不匹配的 Bug。
    
    Args:
        scanner: UltrasoundScanner 实例
        height_profile: [B, W] 表面高度曲线
    Returns:
        intensity: [B, H_img, W] 生成的超声切面图像
    """
    B, W = height_profile.shape
    H_img = scanner.H_img
    device = height_profile.device
    
    # 1. 构建深度场坐标 Z [1, H_img, 1]
    # 例如 [1, 80, 1]
    depth_grid = torch.linspace(0, scanner.image_depth, H_img, device=device).view(1, -1, 1)
    
    # 2. 构建表面深度 map [B, 1, W]
    # probe_offset 是探头到基底的距离
    surface_depth = scanner.probe_offset - height_profile.unsqueeze(1)
    
    # 3. 利用广播机制计算距离差 [B, H_img, W]
    # [1, 80, 1] - [B, 1, W] -> [B, 80, W]
    # PyTorch 会自动处理，不需要手动 expand
    diff = depth_grid - surface_depth
    
    # 4. 高斯强度分布 (PSF)
    thickness = scanner.res_axial * 1.5
    intensity = torch.exp(-(diff**2) / (2 * thickness**2))
    
    # 5. [额外优化] 加入角度衰减 (模拟物理反射率)
    # 计算横向斜率
    dz = torch.abs(height_profile[:, 1:] - height_profile[:, :-1])
    dz = torch.cat([dz, dz[:, -1:]], dim=1) # 补齐长度
    
    # 坡度越陡，反射越弱 (Soft cutoff)
    reflectivity = 1.0 / (1.0 + 8.0 * (dz ** 2))
    
    # 将反射率应用到整列像素 [B, 1, W] * [B, 80, W]
    return intensity * reflectivity.unsqueeze(1)


def train_texture():
    print("🚀 启动纹理优化: 离散阶梯化与能量集中策略 (Discrete Staircase Optimization)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 1. 初始化纹理
    tex = HelicalChirpTexture(max_height=3.0).to(device)
    
    if not os.path.exists("initial_texture.pth"):
        torch.save(tex.state_dict(), "initial_texture.pth")
    
    geo = GeometryEngine().to(device)
    
    # 2. 初始化扫描仪 (注意半径修正)
    CONTACT_RADIUS = 10.5 
    scanner = UltrasoundScanner(
        probe_width=30.0, 
        image_depth=8.0, 
        radius=CONTACT_RADIUS 
    ).to(device)
    
    # 3. 优化器
    optimizer = optim.Adam(tex.parameters(), lr=0.002)
    
    epochs = 200
    batch_size = 64
    loss_history = []
    
    print("🎯 开始训练循环...")
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # --- A. 随机采样 ---
        z1 = torch.rand(batch_size, device=device) * 80.0 + 10.0
        offset = (torch.rand(batch_size, device=device) * 40.0 + 5.0)
        direction = torch.sign(torch.randn(batch_size, device=device))
        z2 = torch.clamp(z1 + offset * direction, 10.0, 90.0)
        
        theta = torch.zeros(batch_size, device=device)
        kappa = torch.zeros(batch_size, 1, device=device)
        phi = torch.zeros(batch_size, 1, device=device)
        
        # --- B. 前向传播 ---
        full_h, _ = tex()
        full_h_batch = full_h.expand(batch_size, -1, -1, -1)
        
        # 1. 几何采样 (使用 .reshape 避免 view 错误)
        grid_z1, grid_th1 = scanner.get_slice_grid(z1, theta)
        h1_flat = geo(
            full_h_batch, kappa, phi, 
            grid_z1.reshape(batch_size, -1), 
            grid_th1.reshape(batch_size, -1)
        )
        h1_profile = h1_flat.reshape(batch_size, scanner.H_img, scanner.W_img)[:, 0, :]
        
        grid_z2, grid_th2 = scanner.get_slice_grid(z2, theta)
        h2_flat = geo(
            full_h_batch, kappa, phi, 
            grid_z2.reshape(batch_size, -1), 
            grid_th2.reshape(batch_size, -1)
        )
        h2_profile = h2_flat.reshape(batch_size, scanner.H_img, scanner.W_img)[:, 0, :]
        
        # 2. [关键修改] 使用自定义的安全渲染函数
        img1 = render_b_mode_safe(scanner, h1_profile)
        img2 = render_b_mode_safe(scanner, h2_profile)
        
        # --- C. Loss 计算 ---
        
        # 1. 相似度 Loss
        v1 = img1.reshape(batch_size, -1)
        v2 = img2.reshape(batch_size, -1)
        v1_norm = torch.nn.functional.normalize(v1, p=2, dim=1)
        v2_norm = torch.nn.functional.normalize(v2, p=2, dim=1)
        similarity = torch.mean(torch.sum(v1_norm * v2_norm, dim=1))
        
        # 2. 平台平整度 Loss (Flatness)
        dz = torch.abs(h1_profile[:, 1:] - h1_profile[:, :-1])
        # 惩罚非0且非跳变的区域
        penalty_mask = torch.relu(dz - 0.05) * torch.relu(0.8 - dz)
        loss_plateau = torch.mean(penalty_mask)
        
        # 3. 能量保持 Loss
        std_val = torch.std(h1_profile)
        loss_energy = torch.relu(0.5 - std_val)
        
        total_loss = similarity + loss_plateau * 20.0 # + loss_energy * 2.0
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 10 == 0:
            pbar.set_description(f"Sim: {similarity.item():.4f} | Flat: {loss_plateau.item():.4f}")

    torch.save(tex.state_dict(), "optimized_texture.pth")
    print("\n✅ 训练完成！已保存 optimized_texture.pth")
    
    plt.figure()
    plt.plot(loss_history)
    plt.title("Staircase Optimization Loss")
    plt.savefig("train_loss.png")

if __name__ == "__main__":
    train_texture()