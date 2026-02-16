import torch
import numpy as np
import os
from tqdm import tqdm
import math

# --- 引入核心模块 ---
from modules.texture import HelicalChirpTexture
from modules.geometry import GeometryEngine
# 尝试兼容导入
try:
    from modules.ultrasound import UltrasoundScanner
except ImportError:
    from modules.ultrasound import FastUltrasoundScanner as UltrasoundScanner

def generate_dataset(
    output_path, 
    n_samples=20000, 
    batch_size=128, 
    robot_diameter=21.0, 
    probe_width=25.0
):
    print(f"🚀 开始生成数据集: {output_path}")
    print(f"   样本数: {n_samples} | Batch: {batch_size}")
    print(f"   物理参数: Robot Dia={robot_diameter}mm, Probe Width={probe_width}mm")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")

    # 1. 初始化仿真系统
    # ---------------------------------------------------------
    tex = HelicalChirpTexture(max_height=3.0).to(device)
    # 尝试加载优化后的纹理，如果不存在则使用默认/随机
    if os.path.exists("optimized_texture.pth"):
        tex.load_state_dict(torch.load("optimized_texture.pth", map_location=device))
        print("   ✅ 已加载优化纹理: optimized_texture.pth")
    else:
        print("   ⚠️ 未找到优化纹理，使用随机初始化")

    geo = GeometryEngine().to(device)
    
    # 初始化扫描仪 (设置物理参数)
    scan = UltrasoundScanner(
        probe_width=probe_width, 
        image_depth=8.0, 
        radius=robot_diameter / 2.0
    ).to(device)

    # 2. 准备数据容器
    # ---------------------------------------------------------
    data_images = []
    data_labels = []
    
    # 计算需要多少个 Batch
    n_batches = int(np.ceil(n_samples / batch_size))
    
    # 3. 循环生成
    # ---------------------------------------------------------
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Generating Batches"):
            # 动态调整最后一个 batch 的大小
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            if current_batch_size <= 0: break
            
            # --- A. 随机状态采样 (Random Sampling) ---
            
            # 1. Z轴位置 [10, 90] mm
            z_pos = torch.rand(current_batch_size, device=device) * 80.0 + 10.0
            
            # 2. 环向角度 Theta [0, 2pi]
            theta = torch.rand(current_batch_size, device=device) * 2 * np.pi
            
            # 3. 切面旋转 Yaw (Scan Angle) [-30, 30] 度
            # 转换为弧度: +/- 30 deg ~= +/- 0.52 rad
            yaw_range = np.deg2rad(30.0)
            yaw = (torch.rand(current_batch_size, device=device) * 2 - 1) * yaw_range
            
            # 4. 俯仰倾斜 Pitch (Tilt Angle) [-20, 20] 度
            # 转换为弧度: +/- 20 deg ~= +/- 0.35 rad
            pitch_range = np.deg2rad(20.0)
            pitch = (torch.rand(current_batch_size, device=device) * 2 - 1) * pitch_range
            
            # 5. 曲率 Kappa (混合采样策略)
            # 50% 明显弯曲 [0.005, 0.025]
            # 30% 轻微弯曲 [0.0, 0.005]
            # 20% 完全直线 [0.0]
            rand_k = torch.rand(current_batch_size, device=device)
            kappa = torch.zeros(current_batch_size, device=device)
            
            mask_large = rand_k < 0.5
            mask_small = (rand_k >= 0.5) & (rand_k < 0.8)
            # mask_zero = rand_k >= 0.8 (默认为0)
            
            kappa[mask_large] = torch.rand(mask_large.sum(), device=device) * 0.02 + 0.005
            kappa[mask_small] = torch.rand(mask_small.sum(), device=device) * 0.005
            
            # 6. 弯曲方向 Phi [0, 2pi]
            phi = torch.rand(current_batch_size, device=device) * 2 * np.pi
            
            # --- B. 仿真成像 (Simulation Pipeline) ---
            
            # 1. 准备纹理全图
            full_tex, _ = tex()
            # 扩展 Batch: [1, 1, H, W] -> [B, 1, H, W]
            tex_in = full_tex
            if tex_in.dim() == 2: tex_in = tex_in.unsqueeze(0).unsqueeze(0)
            elif tex_in.dim() == 3: tex_in = tex_in.unsqueeze(0)
            tex_in = tex_in.expand(current_batch_size, -1, -1, -1)
            
            # 2. 获取光线追踪网格 (Ray Casting Grid)
            # 输入需要 reshape 为 [B, 1]
            z_in = z_pos.unsqueeze(1)
            th_in = theta.unsqueeze(1)
            yaw_in = yaw.unsqueeze(1)
            pitch_in = pitch.unsqueeze(1)
            
            # grid_z, grid_th shape: [B, H_img, W_img]
            grid_z, grid_th = scan.get_slice_grid(z_in, th_in, scan_angle=yaw_in, tilt_angle=pitch_in)
            
            # 3. 几何采样
            # 需要展平为 [B, N] 喂给 GeometryEngine
            B, H, W = grid_z.shape
            grid_z_flat = grid_z.reshape(B, -1)
            grid_th_flat = grid_th.reshape(B, -1)
            
            # 几何计算
            # kappa, phi 也需要 [B, 1]
            kap_in = kappa.unsqueeze(1)
            phi_in = phi.unsqueeze(1)
            
            h_map_flat = geo(tex_in, kap_in, phi_in, grid_z_flat, grid_th_flat)
            h_map = h_map_flat.view(B, H, W)
            
            # 4. 渲染图像
            us_img = scan.render_slice(h_map) # [B, 1, H, W]
            
            # 5. 添加随机噪声 (Data Augmentation)
            # 随机噪声水平 0.2 ~ 0.6
            noise_levels = torch.rand(current_batch_size, 1, 1, 1, device=device) * 0.4 + 0.2
            noise = torch.randn_like(us_img)
            us_img_noisy = us_img + us_img * noise * noise_levels
            us_img_final = torch.clamp(us_img_noisy, 0, 1)
            
            # --- C. 标签编码 (Label Encoding) ---
            # 我们将构建一个 8维 标签向量
            # Y = [Norm_Z, sin_th, cos_th, sin_yaw, sin_pitch, Norm_Kappa, sin_phi, cos_phi]
            
            # 1. 归一化 Z: [10, 90] -> [-1, 1]
            norm_z = (z_pos - 50.0) / 40.0 
            
            # 2. 角度编码
            sin_th = torch.sin(theta)
            cos_th = torch.cos(theta)
            
            # Yaw 和 Pitch 范围较小，sin 值近似线性，但也可用
            sin_yaw = torch.sin(yaw)
            sin_pitch = torch.sin(pitch)
            
            # 3. 归一化 Kappa: [0, 0.025] -> [0, 1]
            norm_kappa = kappa / 0.025
            
            # 4. 弯曲方向
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            
            # 堆叠标签
            # Shape: [B, 8]
            labels = torch.stack([
                norm_z,      # 0
                sin_th,      # 1
                cos_th,      # 2
                sin_yaw,     # 3
                sin_pitch,   # 4
                norm_kappa,  # 5
                sin_phi,     # 6
                cos_phi      # 7
            ], dim=1)
            
            # 收集数据 (转回 CPU 节省显存)
            data_images.append(us_img_final.cpu().to(torch.float32)) # 也可以存 uint8 节省空间
            data_labels.append(labels.cpu().to(torch.float32))

    # 4. 合并与保存
    # ---------------------------------------------------------
    print("📦 正在打包数据...")
    all_images = torch.cat(data_images, dim=0)
    all_labels = torch.cat(data_labels, dim=0)
    
    # 截断多余的样本 (由于 batch 向上取整)
    all_images = all_images[:n_samples]
    all_labels = all_labels[:n_samples]
    
    print(f"   Images Shape: {all_images.shape}")
    print(f"   Labels Shape: {all_labels.shape}")
    print("   Label Definition: [Norm_Z, sin_th, cos_th, sin_yaw, sin_pitch, Norm_K, sin_phi, cos_phi]")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'images': all_images, 'labels': all_labels}, output_path)
    print(f"✅ 数据集已保存至: {output_path}")

if __name__ == "__main__":
    # 1. 生成训练集 (20,000 张)
    generate_dataset(
        "dataset/train_data_6dof.pt", 
        n_samples=20000, 
        batch_size=64 # 如果显存不够，调小这里
    )
    
    # 2. 生成验证集 (2,000 张)
    generate_dataset(
        "dataset/val_data_6dof.pt", 
        n_samples=2000, 
        batch_size=64
    )