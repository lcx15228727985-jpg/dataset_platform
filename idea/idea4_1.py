import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 配置
# ==========================================
STACK_DATA_ROOT = "/root/asym_pinn_dataset"  # 原始切片数据
CSV_PATH = os.path.join(STACK_DATA_ROOT, "metadata.csv")
MODEL_PATH = "best_unrolled_model.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 核心修复：高精度投影算法
# ==========================================
def precise_stack_projection(stack):
    """
    将 (Seq_Len, 224, 224) 的切片堆叠 -> (200, 360) 的展开图
    """
    seq_len, h, w = stack.shape
    center = (w // 2, h // 2)
    max_radius = min(center)

    # 1. 极坐标变换 (对每一帧)
    # 结果: (Seq_Len, 360, Radius)
    polar_stack = []
    for i in range(seq_len):
        # 旋转 90 度以对齐相位 (根据之前数据生成的经验)
        # cv2.warpPolar 的 0 度通常在 3 点钟方向，我们需要对齐你的数学生成逻辑
        slice_img = stack[i]

        # 核心：极坐标变换
        # 输出尺寸: (360, max_radius) -> 行是角度，列是半径
        polar = cv2.warpPolar(slice_img, (max_radius, 360), center, max_radius, cv2.WARP_POLAR_LINEAR)
        polar_stack.append(polar)

    polar_stack = np.array(polar_stack)  # (5, 360, 112)

    # 2. 半径自动对焦 (Auto-Focus)
    # 我们不知道 Marker 确切在第几个像素半径上，但那里肯定最亮
    # 在半径维度上求最大值或者求和
    # energy_profile: (112,)
    energy_profile = np.mean(polar_stack, axis=(0, 1))

    # 找到能量峰值 (Marker 所在的半径环)
    # 也就是线束最亮的那个圈
    best_radius_idx = np.argmax(energy_profile)

    # 为了鲁棒，取峰值附近的 5 个像素取最大值 (Max Projection)
    # 这样能捕捉到稍有偏移的 Marker
    r_start = max(0, best_radius_idx - 3)
    r_end = min(polar_stack.shape[2], best_radius_idx + 3)

    # 提取展开图: Shape (Seq_Len, 360)
    # 在半径维度做 max pooling
    unrolled_raw = np.max(polar_stack[:, :, r_start:r_end], axis=2)

    # 3. 图像增强与尺寸对齐
    # 原始 stack 只有 5 帧，而模型需要 200 行
    # 必须使用 'nearest' 插值或者 'linear'，但要小心模糊
    # 这里我们用 Linear 插值模拟连续性

    # 转置一下变成 (H=Seq, W=Angle) -> (H=Angle, W=Seq) 以便 resize
    # 我们的目标是 (H=200, W=360)
    # cv2.resize dsize 是 (width, height)

    # 注意：unrolled_raw 是 (5, 360) -> 5行代表时间，360列代表角度
    unrolled_final = cv2.resize(unrolled_raw, (360, 200), interpolation=cv2.INTER_LINEAR)

    # 归一化到 0-255
    if unrolled_final.max() > 0:
        unrolled_final = (unrolled_final / unrolled_final.max() * 255).astype(np.uint8)

    # 4. 相位对齐修正
    # 此时可能存在相位偏差 (比如整体转了 90 度)，这取决于 warpPolar 的定义
    # 这是一个工程上的 Calibration 参数，通常是一个固定的像素偏移
    # 这里先不做，直接看效果，如果方向反了或偏了，只要加个 np.roll 即可

    return unrolled_final


# ==========================================
# 2. 模型定义
# ==========================================
class MapResNet(nn.Module):
    def __init__(self):
        super(MapResNet, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = self.new_conv
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, x):
        return self.fc(self.backbone(x).view(x.size(0), -1))


# ==========================================
# 3. 几何工具
# ==========================================
def pcc_forward(kappa, phi, length=100.0):
    s = np.linspace(0, length, 100)
    if abs(kappa) < 1e-5:
        return np.stack([np.zeros_like(s), np.zeros_like(s), s], axis=1)
    r = 1.0 / kappa;
    theta = s * kappa
    x = r * (1 - np.cos(theta)) * np.cos(phi)
    y = r * (1 - np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


# ==========================================
# 4. 主程序
# ==========================================
def run_fix_demo():
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型不存在")
        return

    model = MapResNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    df = pd.read_csv(CSV_PATH)
    # 找一个弯曲比较明显的样本，方便观察
    df_bend = df[df['kappa'] > 0.005]
    if len(df_bend) > 0:
        row = df_bend.sample(1).iloc[0]
    else:
        row = df.sample(1).iloc[0]

    npy_path = os.path.join(STACK_DATA_ROOT, row['filename'])
    stack_3d = np.load(npy_path)

    print(f"🔎 样本: {row['filename']} (True Kappa={row['kappa']:.4f})")

    # === 1. 生成推理用的图 ===
    inference_map = precise_stack_projection(stack_3d)

    # === 2. 加载一张训练集里的图(随便一张)做对比 ===
    # 为了对比，我们理想情况下应该重新用 generate_unrolled 生成这张图的真值
    # 但这里我们主要看 "画风" 是否一致
    # 我们临时从 unrolled_dataset 里读一张来看看风格
    train_style_img = None
    if os.path.exists("/root/unrolled_dataset/map_00000.png"):
        train_style_img = cv2.imread("/root/unrolled_dataset/map_00000.png", 0)

    # === 3. 推理 ===
    pil_img = Image.fromarray(inference_map).convert('L')
    transform = transforms.Compose([transforms.Resize((200, 360)), transforms.ToTensor()])
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy()[0]
        pred_tip = pred_norm * 100.0

    # === 4. 误差计算 ===
    gt_tip = pcc_forward(row['kappa'], row['phi'])[-1]
    error = np.linalg.norm(gt_tip - pred_tip)

    print(f"📊 误差: {error:.2f} mm")
    if error > 10:
        print("⚠️ 警告: 误差依然较大。请检查下方生成的对比图 'debug_comparison.png'")
        print("   如果 'Inference Input' 和 'Training Style' 看起来截然不同(比如黑白反了，或者全是噪音)，")
        print("   说明投影参数(Radius)还需要微调。")

    # === 5. 绘图诊断 ===
    fig = plt.figure(figsize=(15, 5))

    # 诊断 1: 我们生成送给网络的图
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(inference_map, cmap='gray', aspect='auto')
    ax1.set_title("1. Inference Input (From Stack)")

    # 诊断 2: 网络训练时看过的图 (风格参考)
    ax2 = fig.add_subplot(1, 3, 2)
    if train_style_img is not None:
        ax2.imshow(train_style_img, cmap='gray', aspect='auto')
        ax2.set_title("2. Training Data Style (Ideal)")
    else:
        ax2.text(0.5, 0.5, "No training data found", ha='center')

    # 诊断 3: 3D 结果
    pts_gt = pcc_forward(row['kappa'], row['phi'])
    # 逆解画图
    # 简单用直线连接原点和预测点来示意误差
    pts_pred_line = np.stack([
        [0, 0, 0],
        pred_tip
    ])

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], 'g-', linewidth=4, label='GT')
    ax3.plot(pts_pred_line[:, 0], pts_pred_line[:, 1], pts_pred_line[:, 2], 'r--', linewidth=2, label='Pred Tip Dir')
    ax3.scatter(gt_tip[0], gt_tip[1], gt_tip[2], c='g', s=100)
    ax3.scatter(pred_tip[0], pred_tip[1], pred_tip[2], c='r', s=100, marker='x')
    ax3.set_xlim([-40, 40]);
    ax3.set_ylim([-40, 40]);
    ax3.set_zlim([0, 100])
    ax3.set_title(f"Error: {error:.1f} mm")
    ax3.legend()

    plt.savefig("debug_comparison.png")
    print("✨ 诊断图已保存: debug_comparison.png")


if __name__ == "__main__":
    run_fix_demo()