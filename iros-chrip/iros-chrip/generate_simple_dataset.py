import torch
import numpy as np
import os
from tqdm import tqdm
import math

# --- 引入核心模块 ---
from modules.texture import HelicalChirpTexture
from modules.geometry import GeometryEngine
try:
    from modules.ultrasound import UltrasoundScanner
except ImportError:
    from modules.ultrasound import FastUltrasoundScanner as UltrasoundScanner
import torch.nn.functional as F

def generate_simple_dataset(
    output_path, 
    n_samples=10000, 
    batch_size=64, 
    robot_diameter=21.0, 
    probe_width=25.0
):
    print(f"🧪 开始生成[简化版]数据集 (3-DoF): {output_path}")
    print(f"   锁定: Yaw=0, Pitch=0")
    print(f"   变量: Z-axis, Theta, Curvature")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化系统
    tex = HelicalChirpTexture(max_height=3.0).to(device)
    geo = GeometryEngine().to(device)
    scan = UltrasoundScanner(probe_width=probe_width, image_depth=8.0, radius=robot_diameter/2.0).to(device)
    
    data_images = []
    data_labels = []
    
    n_batches = int(np.ceil(n_samples / batch_size))
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Generating 3-DoF Data"):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            if current_batch_size <= 0: break
            
            # --- A. 简化采样 (Simplified Sampling) ---
            
            # 1. Z轴位置: 随机
            z_pos = torch.rand(current_batch_size, device=device) * 80.0 + 10.0
            
            # 2. 环向角度 Theta: 随机
            theta = torch.rand(current_batch_size, device=device) * 2 * np.pi
            
            # 3. [固定] 切面旋转 Yaw = 0
            yaw = torch.zeros(current_batch_size, device=device)
            
            # 4. [固定] 俯仰倾斜 Pitch = 0 (无视差)
            pitch = torch.zeros(current_batch_size, device=device)
            
            # 5. 曲率 Kappa: 随机
            rand_k = torch.rand(current_batch_size, device=device)
            kappa = torch.zeros(current_batch_size, device=device)
            mask_curve = rand_k < 0.7 # 70% 弯曲，30% 直线
            kappa[mask_curve] = torch.rand(mask_curve.sum(), device=device) * 0.025
            
            # 6. 弯曲方向 Phi: 随机
            phi = torch.rand(current_batch_size, device=device) * 2 * np.pi
            
            # --- B. 仿真成像 ---
            full_tex, _ = tex()
            
            # [🔥 维度修复核心代码] 
            # 无论输入是 [H, W], [1, H, W] 还是 [1, 1, H, W]，都统一处理
            tex_in = full_tex
            if tex_in.dim() == 2:   # [H, W]
                tex_in = tex_in.unsqueeze(0).unsqueeze(0)
            elif tex_in.dim() == 3: # [C, H, W]
                tex_in = tex_in.unsqueeze(0)
            # 如果已经是 4维 [B, C, H, W]，就不用动了
            
            # 安全扩展到 Batch Size
            # 目标形状: [current_batch_size, 1, H, W]
            tex_in = tex_in.expand(current_batch_size, -1, -1, -1)
            
            # 获取网格 (Pitch=0, Yaw=0)
            grid_z, grid_th = scan.get_slice_grid(
                z_pos.unsqueeze(1), theta.unsqueeze(1), 
                scan_angle=yaw.unsqueeze(1), tilt_angle=pitch.unsqueeze(1)
            )
            
            B, H, W = grid_z.shape
            grid_z_flat = grid_z.reshape(B, -1)
            grid_th_flat = grid_th.reshape(B, -1)
            
            h_map = geo(
                tex_in, 
                kappa.unsqueeze(1), 
                phi.unsqueeze(1), 
                grid_z_flat, 
                grid_th_flat
            ).view(B, H, W)
            
            us_img = scan.render_slice(h_map)
            
            # 添加噪声
            noise = torch.randn_like(us_img)
            us_img = us_img + us_img * noise * 0.5 # 适中噪声
            us_img = torch.clamp(us_img, 0, 1)
            
            # --- C. 标签 ---
            norm_z = (z_pos - 50.0) / 40.0 
            norm_kappa = kappa / 0.025
            
            labels = torch.stack([
                norm_z,                  # 0: Z
                torch.sin(theta),        # 1: sin_th
                torch.cos(theta),        # 2: cos_th
                torch.zeros_like(yaw),   # 3: sin_yaw (Always 0)
                torch.zeros_like(pitch), # 4: sin_pitch (Always 0)
                norm_kappa,              # 5: Kappa
                torch.sin(phi),          # 6: sin_phi
                torch.cos(phi)           # 7: cos_phi
            ], dim=1)
            
            data_images.append(us_img.cpu().to(torch.float32))
            data_labels.append(labels.cpu().to(torch.float32))

    print("📦 打包数据...")
    all_images = torch.cat(data_images, dim=0)
    all_labels = torch.cat(data_labels, dim=0)
    
    save_dict = {'images': all_images, 'labels': all_labels}
    
    # 保存训练集
    torch.save(save_dict, output_path)
    print(f"✅ [简化版] 训练集已保存: {output_path}")
    
    # 顺便生成一个小验证集
    val_path = output_path.replace("train", "val")
    n_val = int(n_samples * 0.1)
    val_dict = {
        'images': all_images[-n_val:],
        'labels': all_labels[-n_val:]
    }
    train_dict = {
        'images': all_images[:-n_val],
        'labels': all_labels[:-n_val:] # 注意这里之前的代码有个小 typo，这里修正了切片
    }
    # 覆盖保存训练集（去除验证部分）
    torch.save(train_dict, output_path)
    # 保存验证集
    torch.save(val_dict, val_path)
    print(f"✅ [简化版] 验证集已保存: {val_path}")

if __name__ == "__main__":
    generate_simple_dataset("dataset/train_data_3dof.pt", n_samples=10000)