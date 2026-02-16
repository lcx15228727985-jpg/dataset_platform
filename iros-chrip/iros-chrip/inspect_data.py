import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def inspect_dataset(file_path="dataset/train_data_6dof.pt", num_samples=16):
    """
    加载数据集并可视化，同时解码标签为物理单位
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到: {file_path}")
        return

    print(f"📦 正在加载: {file_path} ...")
    # map_location='cpu' 确保即使没有 GPU 也能运行
    data = torch.load(file_path, map_location='cpu')
    images = data['images']
    labels = data['labels']
    
    total_len = len(images)
    print(f"✅ 数据集加载成功! 总样本数: {total_len}")
    print(f"   图像尺寸: {images.shape[1:]}")
    
    # 随机采样
    indices = np.random.choice(total_len, num_samples, replace=False)
    
    # 设置画布
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten()
    
    print("🎨 正在绘图...")
    
    for i, idx in enumerate(indices):
        # 1. 获取图像 (移除通道维度 [1, H, W] -> [H, W])
        img = images[idx].squeeze().numpy()
        lbl = labels[idx].numpy()
        
        # 2. 解码标签 (还原归一化)
        # 根据 generate_dataset.py 中的公式反推
        
        # Z轴: norm = (z - 50) / 40  =>  z = norm * 40 + 50
        z_mm = lbl[0] * 40.0 + 50.0
        
        # Theta: atan2(sin, cos)
        theta_deg = np.degrees(np.arctan2(lbl[1], lbl[2]))
        if theta_deg < 0: theta_deg += 360
        
        # Yaw (切面): sin(yaw) => arcsin
        yaw_deg = np.degrees(np.arcsin(np.clip(lbl[3], -1, 1)))
        
        # Pitch (俯仰): sin(pitch) => arcsin
        pitch_deg = np.degrees(np.arcsin(np.clip(lbl[4], -1, 1)))
        
        # Kappa (曲率): norm = k / 0.025 => k = norm * 0.025
        kappa = lbl[5] * 0.025
        
        # 3. 绘制
        ax = axes[i]
        ax.imshow(img, cmap='gray', aspect='equal')
        
        # 标题显示关键参数
        # 重点观察 Pitch 是否导致图像拉伸/剪切，Noise 是否明显
        title_text = (f"Z:{z_mm:.1f}mm | K:{kappa:.3f}\n"
                      f"Yaw:{yaw_deg:.1f}° | Pit:{pitch_deg:.1f}°")
        
        ax.set_title(title_text, fontsize=9, color='blue')
        ax.axis('off')
        
        # 辅助文字：显示索引
        ax.text(5, 10, f"ID:{idx}", color='red', fontsize=8, fontweight='bold')

    plt.tight_layout()
    # 保存图片以便在服务器上查看
    save_path = "data_preview.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ 预览图已保存至: {save_path}")
    # 如果是在本地 IDE (VSCode) 运行，可以用 plt.show()
    # plt.show()

if __name__ == "__main__":
    # 检查训练集
    inspect_dataset("dataset/train_data_6dof.pt", num_samples=16)
    
    # 也可以取消注释检查验证集
    # inspect_dataset("dataset/val_data_6dof.pt", num_samples=9)