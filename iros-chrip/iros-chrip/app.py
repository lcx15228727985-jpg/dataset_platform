import os

# Avoid OpenMP conflict (MKL vs PyTorch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# App root: run from script dir so modules and weights are found
_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != _APP_ROOT:
    os.chdir(_APP_ROOT)

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- 引入模块 ---
from modules.texture import HelicalChirpTexture
from modules.geometry import GeometryEngine

# 尝试导入 UltrasoundScanner
try:
    from modules.ultrasound import UltrasoundScanner
except ImportError:
    try:
        from modules.ultrasound import FastUltrasoundScanner as UltrasoundScanner
    except ImportError:
        st.error("❌ 严重错误: 无法导入 modules.ultrasound。请确保文件存在。")
        st.stop()

# --- 页面配置 ---
st.set_page_config(page_title="Engraving B-Mode Simulation", layout="wide")
st.title("🧬 连续体机器人 · B-Mode 刻槽仿真 (6-DoF 全姿态版)")
st.markdown("""
**仿真逻辑升级：全空间自由度与物理光线追踪**
* **物理参数:** 自定义机器人直径 (21mm) 与 探头 FOV。
* **俯仰 (Pitch/Tilt):** 模拟探头沿 Y 轴的前后倾斜，引入深度视差。
* **偏航 (Yaw/Scan):** 模拟探头绕 Z 轴的切面旋转。
""")

# --- 1. 侧边栏控制 ---
with st.sidebar:
    st.header("📂 模型加载")
    available_weights = ["None (随机初始化)"]
    w0 = os.path.join(_APP_ROOT, "initial_texture.pth")
    w1 = os.path.join(_APP_ROOT, "optimized_texture.pth")
    if os.path.exists(w0):
        available_weights.append("initial_texture.pth")
    if os.path.exists(w1):
        available_weights.append("optimized_texture.pth")
    selected_weight = st.selectbox("选择纹理权重", available_weights, index=len(available_weights)-1)
    
    st.divider()
    
    # --- 物理参数 ---
    st.header("⚙️ 物理参数")
    robot_diameter = st.number_input(
        "机器人直径 (mm)", 5.0, 50.0, 21.0, 0.5,
        help="包含外层薄膜后的总直径。"
    )
    robot_radius = robot_diameter / 2.0
    
    probe_width = st.number_input(
        "探头宽度 (FOV mm)", 10.0, 60.0, 25.0, 1.0,
        help="探头阵列的物理宽度。若大于机器人直径，横向扫描时两侧会显示黑边。"
    )
    
    st.divider()
    
    # --- 几何位置 ---
    st.header("🎮 几何位置")
    kappa = st.slider("曲率 (Kappa)", 0.0, 0.03, 0.015, format="%.4f")
    phi_deg = st.slider("弯曲方向 (Phi)", 0, 360, 0)
    z_pos = st.slider("探头 Z 位置 (mm)", 15.0, 85.0, 50.0)
    
    st.divider()
    
    # --- 探头姿态 (6-DoF) ---
    st.header("📷 探头姿态 (6-DoF)")
    probe_angle_deg = st.slider("1. 环向公转 (Theta)", 0, 360, 0)
    
    scan_angle_deg = st.slider(
        "2. 切面旋转 (Yaw/Scan)", -90, 90, 0,
        help="绕探头法线旋转。0°=纵向，90°=横向。"
    )
    
    tilt_angle_deg = st.slider(
        "3. 前后俯仰 (Pitch/Tilt)", -30, 30, 0,
        help="绕探头横轴倾斜。模拟探头未垂直贴合皮肤，产生视差。"
    )
    
    noise_level = st.slider("散斑噪声 (Speckle)", 0.0, 1.0, 0.4)

# --- 2. 系统初始化 ---
# 关键修复：去掉参数下划线，确保 Streamlit 正确响应参数变化
@st.cache_resource
def load_system(weight_path, probe_width, radius):
    device = torch.device("cpu")
    
    # A. 纹理
    tex = HelicalChirpTexture(max_height=3.0).to(device)
    msg = ""
    weight_full = os.path.join(_APP_ROOT, weight_path) if weight_path != "None (随机初始化)" else ""
    if weight_path != "None (随机初始化)" and os.path.exists(weight_full):
        try:
            tex.load_state_dict(torch.load(weight_full, map_location=device))
            msg = f"✅ 已加载: {weight_path}"
        except:
            msg = "⚠️ 加载失败，使用随机初始化"
    else:
        msg = "🎲 使用随机初始化"
            
    # B. 几何
    geo = GeometryEngine().to(device)
    
    # C. 扫描仪
    # 注入动态的物理参数
    try:
        scan = UltrasoundScanner(probe_width=probe_width, image_depth=8.0, radius=radius).to(device)
    except TypeError:
        st.error("UltrasoundScanner 版本不匹配，请更新 modules/ultrasound.py")
        st.stop()
    
    return tex, geo, scan, msg

# 加载
tex, geo, scan, status_msg = load_system(selected_weight, probe_width, robot_radius)
tex.eval()

if "✅" in status_msg: st.success(status_msg)
else: st.info(status_msg)

# --- 3. 仿真循环 ---
# --- 3. 仿真循环 ---
def run_simulation():
    with torch.no_grad():
        full_tex, _ = tex()
        
        # 1. 准备参数 (Float32)
        th_rad = np.deg2rad(probe_angle_deg)
        scan_rad = np.deg2rad(scan_angle_deg)
        tilt_rad = np.deg2rad(tilt_angle_deg) # [Pitch]
        phi_rad = np.deg2rad(phi_deg)
        
        z_t = torch.tensor([z_pos], dtype=torch.float32)
        th_t = torch.tensor([th_rad], dtype=torch.float32)
        scan_t = torch.tensor([scan_rad], dtype=torch.float32)
        tilt_t = torch.tensor([tilt_rad], dtype=torch.float32) # [Pitch]
        kap_t = torch.tensor([kappa], dtype=torch.float32).unsqueeze(1)
        phi_t = torch.tensor([phi_rad], dtype=torch.float32).unsqueeze(1)
        
        # 2. 获取全分辨率采样网格 (支持 Pitch 视差)
        # grid shape: [1, H, W]
        try:
            grid_z, grid_th = scan.get_slice_grid(z_t, th_t, scan_angle=scan_t, tilt_angle=tilt_t)
        except AttributeError:
            st.error("请更新 modules/ultrasound.py 以支持 get_slice_grid 方法")
            st.stop()
            
        # 3. 几何采样 (Geometry Sampling)
        # GeometryEngine 需要 flatten 输入
        B, H, W = grid_z.shape
        
        # [关键修复] 使用 .reshape() 代替 .view()
        # .view() 只能处理内存连续的张量，而 .expand() 生成的张量往往是不连续的
        grid_z_flat = grid_z.reshape(B, -1)
        grid_th_flat = grid_th.reshape(B, -1)
        
        # 扩展纹理维度
        tex_in = full_tex
        if tex_in.dim() == 2: tex_in = tex_in.unsqueeze(0).unsqueeze(0)
        elif tex_in.dim() == 3: tex_in = tex_in.unsqueeze(0)
            
        # 采样得到全像素纹理高度图
        h_map_flat = geo(tex_in, kap_t, phi_t, grid_z_flat, grid_th_flat)
        h_map = h_map_flat.view(B, H, W)
        
        # 4. 渲染切片 (Slice Rendering)
        # 基于光线追踪原理生成 B-Mode
        us_img = scan.render_slice(h_map)
        
        # 5. 添加噪声
        us_img = scan.add_speckle_noise(us_img, noise_level)
        
        # 辅助数据: 提取表面接触线用于显示 (第0层深度)
        h_prof = h_map[0, 0, :] 
        z_axis = grid_z[0, 0, :]
        
        return full_tex, h_prof, us_img.squeeze(), z_axis

full_tex, h_prof, us_img, z_axis = run_simulation()

# --- 4. 可视化 ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. B-Mode 超声切面")
    
    # 状态描述
    desc = []
    if abs(scan_angle_deg) > 0: desc.append(f"Yaw={scan_angle_deg}°")
    if abs(tilt_angle_deg) > 0: desc.append(f"Pitch={tilt_angle_deg}°")
    title_suffix = " | ".join(desc) if desc else "Normal"
    st.caption(f"Z={z_pos}mm | {title_suffix}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # 物理范围 extent
    w_half = probe_width / 2.0
    extent = [-w_half, w_half, scan.image_depth, 0]
    
    # [关键] aspect='equal' 保证真实物理比例 (不会被压扁)
    ax.imshow(us_img.numpy(), cmap='gray', aspect='equal', extent=extent, vmin=0, vmax=1)
    
    ax.set_xlabel("Lateral Distance (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"Simulated US (FOV={probe_width}mm)")
    
    # 辅助线：如果探头比机器人宽，画出机器人边界
    if probe_width > robot_diameter and abs(scan_angle_deg) > 45:
        ax.axvline(-robot_radius, color='yellow', linestyle='--', alpha=0.5)
        ax.axvline(robot_radius, color='yellow', linestyle='--', alpha=0.5)
        
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
    st.pyplot(fig)

with col2:
    st.subheader("2. 表面高度 (Input Profile)")
    st.caption("探头接触面处的纹理高度分布")
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x_lat = np.linspace(-probe_width/2, probe_width/2, len(h_prof))
    
    ax2.plot(x_lat, h_prof.numpy(), color='#ff4b4b', lw=2)
    ax2.fill_between(x_lat, h_prof.numpy(), 0, color='#ff4b4b', alpha=0.1)
    
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_ylabel("Texture Height (mm)")
    ax2.set_xlabel("Lateral (mm)")
    ax2.set_title("Surface Texture")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

st.divider()

# --- 5. 3D 预览 ---
st.subheader("3. 3D 实体预览")
st.caption(f"机器人直径: {robot_diameter}mm | 半径: {robot_radius}mm")

with torch.no_grad():
    # 准备纹理
    if full_tex.dim() == 2: tex_3d = full_tex.unsqueeze(0).unsqueeze(0)
    elif full_tex.dim() == 3: tex_3d = full_tex.unsqueeze(0)
    else: tex_3d = full_tex
    
    tex_small = torch.nn.functional.interpolate(tex_3d, size=(60, 100)).squeeze().numpy()
    
    # 获取网格 (传入 robot_radius)
    try:
        X, Y, Z = geo.get_3d_mesh(kappa, np.deg2rad(phi_deg), radius=robot_radius, resolution_z=100, resolution_theta=60)
    except TypeError:
        # Fallback
        X, Y, Z = geo.get_3d_mesh(kappa, np.deg2rad(phi_deg), resolution_z=100, resolution_theta=60)

surf = go.Surface(
    x=X, y=Y, z=Z, 
    surfacecolor=tex_small, 
    colorscale='Viridis', 
    cmin=0, cmax=3.0,
    opacity=0.9
)

fig3 = go.Figure(data=[surf])

# 探头位置指示 (红色球)
if kappa == 0:
    px = robot_radius * np.cos(np.deg2rad(probe_angle_deg))
    py = robot_radius * np.sin(np.deg2rad(probe_angle_deg))
    pz = z_pos
    
    fig3.add_trace(go.Scatter3d(
        x=[px], y=[py], z=[pz],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Probe Center'
    ))

fig3.update_layout(
    scene=dict(
        aspectmode='data', 
        zaxis=dict(title="Z (mm)"),
        xaxis=dict(title="X (mm)"),
        yaxis=dict(title="Y (mm)")
    ), 
    height=500, 
    margin=dict(l=0,r=0,b=0,t=0)
)
st.plotly_chart(fig3, width="stretch")