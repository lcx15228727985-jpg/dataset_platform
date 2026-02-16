import torch
import torch.nn as nn
import numpy as np

class HelicalChirpTexture(nn.Module):
    def __init__(self, N=512, K_theta=10, K_z=40, max_height=3.0):
        """
        螺旋双频 Chirp 纹理生成器 (四阶梯量化版)
        
        输出高度: 离散的 0mm, 1mm, 2mm, 3mm
        """
        super().__init__()
        self.N = N
        self.K_theta = K_theta
        self.K_z = K_z
        self.max_height = max_height 
        
        # 参数矩阵
        self.coeffs = nn.Parameter(torch.zeros(2 * K_theta + 1, K_z))
        
        # [新增] 全局频率缩放 (可选，用于微调整体疏密)
        self.freq_scale = nn.Parameter(torch.tensor(1.0))
        
        self._initialize_weights()
        self._precompute_basis()

    def _initialize_weights(self):
        with torch.no_grad():
            z_space = torch.linspace(0, 1, self.N)
            # 初始化 Chirp 信号
            phi_fast = 25.0 * z_space + 0.5 * (45.0 - 15.0) * z_space**2
            phi_slow = 10.0 * z_space + 0.5 * (20.0 - 10.0) * z_space**2
            
            def project_to_dct(sig_1d, K):
                n = torch.arange(len(sig_1d)).float()
                k = torch.arange(K).float().unsqueeze(1)
                basis = torch.cos((np.pi/len(sig_1d))*(n+0.5)*k)
                return (basis @ sig_1d).squeeze() * 2.0 / len(sig_1d)

            coeffs_fast = project_to_dct(torch.cos(2*np.pi*phi_fast), self.K_z)
            coeffs_slow = project_to_dct(torch.cos(2*np.pi*phi_slow), self.K_z)
            
            self.coeffs.data[1, :] += coeffs_fast * 0.4
            self.coeffs.data[self.K_theta + 1, :] += coeffs_fast * 0.4 
            self.coeffs.data[1, :] += coeffs_slow * 0.3
            self.coeffs.data += torch.randn_like(self.coeffs) * 0.02

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
        # 1. 基础信号合成
        raw_base = self.basis_theta @ self.coeffs @ self.basis_z.T
        
        # 2. 频率缩放 (Freq Scale)
        if self.freq_scale != 1.0:
            B, H, W = 1, self.N, self.N
            img = raw_base.view(1, 1, H, W)
            # 仅在 Z 轴 (W) 方向缩放
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

        # 3. 映射与四阶梯量化 (4-Level Quantization)
        # 归一化到 [0, 1]
        normalized = (torch.tanh(raw) + 1.0) / 2.0
        
        # 映射到 [0, 4.0] 的虚拟高度，方便切割
        # 我们希望覆盖 0.5, 1.5, 2.5 这三个切割点
        val = normalized * 4.0 
        
        sharpness = 30.0 # 锐度控制：越高越像直角台阶
        
        # 叠加三个台阶函数
        # Step 1: 0mm -> 1mm (阈值 0.5)
        step1 = torch.sigmoid((val - 0.5) * sharpness)
        
        # Step 2: 1mm -> 2mm (阈值 1.5)
        step2 = torch.sigmoid((val - 1.5) * sharpness)
        
        # Step 3: 2mm -> 3mm (阈值 2.5)
        step3 = torch.sigmoid((val - 2.5) * sharpness)
        
        # 物理高度叠加: 0 + s1 + s2 + s3
        # 结果只可能是 0.0, 1.0, 2.0, 3.0
        phys = (step1 + step2 + step3) * 1.0 # 步长为 1mm
        
        # 强制限制最大高度 (防止数值溢出)
        phys = torch.clamp(phys, 0, self.max_height)
        
        return phys.unsqueeze(0).unsqueeze(0), raw.unsqueeze(0).unsqueeze(0)