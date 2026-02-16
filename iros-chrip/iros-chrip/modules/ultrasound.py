import torch
import torch.nn as nn
import numpy as np

class UltrasoundScanner(nn.Module):
    def __init__(self, probe_width=30.0, image_depth=10.0, res_axial=0.1, res_lateral=0.2, radius=7.5, device=None):
        """
        超声成像仿真器 (支持 6-DoF 姿态与光线追踪)
        """
        super().__init__()
        self.probe_width = probe_width
        self.image_depth = image_depth
        self.res_axial = res_axial
        self.res_lateral = res_lateral
        self.radius = radius # 机器人半径
        
        # 图像分辨率
        self.H_img = int(image_depth / res_axial)
        self.W_img = int(probe_width / res_lateral)
        
        # 探头偏移量 (Probe Offset):
        self.probe_offset = 4.0 
        
        self.device_param = device

    def get_slice_grid(self, z_center, theta_center, scan_angle=0.0, tilt_angle=0.0):
        """
        生成全图象素对应的物理坐标网格 (支持 Yaw/Scan 和 Pitch/Tilt)
        """
        device = z_center.device
        B = z_center.shape[0]
        
        # 1. 构建像素坐标系 (u, v)
        # u: 横向 (Lateral), shape [1, 1, W]
        u = torch.linspace(-self.probe_width/2, self.probe_width/2, self.W_img, device=device).view(1, 1, -1)
        # v: 深度 (Depth), shape [1, H, 1]
        v = torch.linspace(0, self.image_depth, self.H_img, device=device).view(1, -1, 1)
        
        # 2. 扩展参数维度 [B, 1, 1]
        def expand_dim(tensor):
            if not torch.is_tensor(tensor): tensor = torch.tensor(tensor, device=device)
            if tensor.dim() == 0: return tensor.view(1, 1, 1).expand(B, 1, 1)
            return tensor.view(B, 1, 1)
            
        scan_angle = expand_dim(scan_angle)
        tilt_angle = expand_dim(tilt_angle)
        z_center = z_center.view(B, 1, 1)
        theta_center = theta_center.view(B, 1, 1)
        
        # 3. 物理坐标映射 (Ray Casting Logic)
        # Z轴映射: [B, H, W]
        # v * sin(tilt) 引入了深度维度的变化，所以 z_grid 自然是 [B, H, W]
        z_grid = z_center + u * torch.cos(scan_angle) + v * torch.sin(tilt_angle)
        
        # Theta轴映射:
        # 初始形状为 [B, 1, W]
        d_theta = (u * torch.sin(scan_angle)) / self.radius
        theta_grid = theta_center + d_theta
        
        # [关键修复] 强制扩展 Theta 维度以匹配 Z 网格
        # 这样两个网格的形状都变成 [B, H, W]
        z_grid = z_grid.expand(B, self.H_img, self.W_img)
        theta_grid = theta_grid.expand(B, self.H_img, self.W_img)
        
        return z_grid, theta_grid

    def render_slice(self, texture_height_map):
        """
        基于采样的纹理高度图渲染 B-Mode 图像 (Volumetric Rendering 近似)
        Args:
            texture_height_map: [B, H, W] 
        Returns:
            image: [B, 1, H, W]
        """
        device = texture_height_map.device
        B, H, W = texture_height_map.shape
        
        # 1. 构建深度场 [1, H, 1] -> [B, H, W]
        depth_field = torch.linspace(0, self.image_depth, self.H_img, device=device).view(1, -1, 1).expand(B, H, W)
        
        # 2. 计算表面所在的深度
        surface_depth_map = self.probe_offset - texture_height_map
        
        # 3. 声波干涉/回波生成
        thickness = self.res_axial * 1.5 
        
        diff = depth_field - surface_depth_map
        intensity = torch.exp(- (diff**2) / (2 * thickness**2))
        
        return intensity.unsqueeze(1) 

    def add_speckle_noise(self, image, level=0.5):
        """添加乘性散斑噪声"""
        noise = torch.randn_like(image)
        noisy = image + image * noise * level
        noisy = noisy + torch.randn_like(image) * 0.02
        return torch.clamp(noisy, 0, 1)

# 兼容类名
class FastUltrasoundScanner(UltrasoundScanner):
    pass