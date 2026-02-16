import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GeometryEngine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, texture_map, kappa, phi, grid_z, grid_theta):
        """
        几何变形引擎 (支持 Batch)
        
        Args:
            texture_map: [B, 1, N_theta, N_z] 原始纹理
            kappa: [B, 1] 曲率
            phi: [B, 1] 弯曲方向
            grid_z: [B, W_img] 扫描线 Z 坐标 (mm)
            grid_theta: [B, W_img] 扫描线 Theta 坐标 (rad)
            
        Returns:
            sampled_h: [B, W_img] 采样后的高度曲线
        """
        # 1. 确保输入维度正确
        if texture_map.dim() == 3:
            texture_map = texture_map.unsqueeze(0) # [1, C, H, W]
            
        batch_size = texture_map.shape[0]
        
        # 2. 坐标归一化 (Normalize to [-1, 1] for grid_sample)
        # 假设纹理 Z 轴范围 [0, 100mm] -> 映射到 [-1, 1]
        # Texture Z range: [0, 1] in normalized space, assuming length_mm=100
        length_mm = 100.0
        
        # Z normalization: 0 -> -1, 100 -> 1
        # z_norm = (z / 100.0) * 2 - 1
        grid_z_norm = (grid_z / length_mm) * 2.0 - 1.0
        
        # Theta normalization: 0 -> -1, 2pi -> 1
        # grid_theta is in radians [0, 2pi]
        grid_theta_norm = (grid_theta / (2 * np.pi)) * 2.0 - 1.0
        
        # 3. 构造采样网格
        # grid_sample expects (x, y) coordinates
        # In our texture: W=N_z (x-axis), H=N_theta (y-axis)
        # So we need (z, theta) pair
        
        # Stack: [B, W_img, 2]
        grid = torch.stack((grid_z_norm, grid_theta_norm), dim=-1)
        
        # Expand dims for grid_sample: [B, H_out, W_out, 2]
        # 我们是在采一条线，所以 H_out=1
        # Shape: [B, 1, W_img, 2]
        grid = grid.unsqueeze(1)
        
        # 4. 执行采样
        # texture_map: [B, 1, N_theta, N_z]
        # grid: [B, 1, W_img, 2]
        # Output: [B, 1, 1, W_img]
        sampled = F.grid_sample(
            texture_map, 
            grid, 
            mode='bilinear', 
            padding_mode='border', # 超出范围取边界值 (防止 nan)
            align_corners=False
        )
        
        # 5. 调整输出形状 -> [B, W_img]
        return sampled.view(batch_size, -1)

    def get_3d_mesh(self, kappa, phi, length=100, radius=7.5, resolution_z=100, resolution_theta=60):
        """
        生成 3D 实体网格 (仅用于可视化，CPU计算)
        """
        z = np.linspace(0, length, resolution_z)
        theta = np.linspace(0, 2*np.pi, resolution_theta)
        Z_grid, Theta_grid = np.meshgrid(z, theta)
        
        # 简单的弯曲逻辑 (PCC)
        if abs(kappa) < 1e-6:
            # 直线
            X = radius * np.cos(Theta_grid)
            Y = radius * np.sin(Theta_grid)
            Z = Z_grid
        else:
            # 弯曲
            r_bend = 1.0 / kappa
            alpha = Z_grid * kappa
            
            # 局部坐标 (在弯曲平面内)
            x_loc = (r_bend - radius * np.cos(Theta_grid)) * np.cos(alpha) - r_bend
            z_loc = (r_bend - radius * np.cos(Theta_grid)) * np.sin(alpha)
            y_loc = radius * np.sin(Theta_grid)
            
            # 旋转到 Phi 平面
            X = x_loc * np.cos(phi) - y_loc * np.sin(phi)
            Y = x_loc * np.sin(phi) + y_loc * np.cos(phi)
            Z = z_loc
            
        return X, Y, Z