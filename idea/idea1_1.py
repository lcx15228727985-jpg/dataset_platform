import os
import cv2
import numpy as np
import pandas as pd  # 需要 pip install pandas
from tqdm import tqdm
import random


# ==========================================
# 物理引擎 (复用之前的，稍作修改以返回精确值)
# ==========================================
def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9: return np.zeros_like(v)
    return v / norm


class RobotPhysicsEngine:
    def __init__(self, radius=4.0):
        self.radius = radius

    def generate_sample(self):
        """随机生成一个样本，并返回精确的物理参数"""
        points_per_seg = 150
        L = 100.0

        # 随机生成曲率 kappa 和 弯曲平面 phi
        # 我们这里模拟单段弯曲，或者简化为主要弯曲分量
        # kappa = theta / length

        # 随机生成弯曲角度 (0 到 100度)
        bend_deg = np.random.uniform(0, 100)
        theta = np.deg2rad(bend_deg)
        if theta < 1e-4: theta = 1e-4

        # 真实的曲率 (Ground Truth Physics Parameter)
        kappa_gt = theta / L

        # 随机生成弯曲平面 (0 到 360度)
        phi_gt = np.random.uniform(0, 2 * np.pi)

        configs = [(L, theta, phi_gt)]

        # --- PCC 建模 ---
        points = [np.array([0., 0., 0.])]
        curr_pos = np.array([0., 0., 0.]);
        curr_rot = np.eye(3)

        for length, th, ph in configs:
            s = np.linspace(0, length, points_per_seg)
            r = length / th;
            t = (s / length) * th
            arc_x = r * (1 - np.cos(t));
            arc_z = r * np.sin(t)
            p_x = arc_x * np.cos(ph);
            p_y = arc_x * np.sin(ph);
            p_z = arc_z
            p_local = np.column_stack([p_x, p_y, p_z])

            c_p, s_p = np.cos(ph), np.sin(ph)
            c_t, s_t = np.cos(th), np.sin(th)
            R_phi = np.array([[c_p, -s_p, 0], [s_p, c_p, 0], [0, 0, 1]])
            R_bend = np.array([[c_t, 0, s_t], [0, 1, 0], [-s_t, 0, c_t]])
            R_local_end = R_phi @ R_bend @ R_phi.T

            for p in p_local[1:]: points.append(curr_pos + curr_rot @ p)
            curr_pos = points[-1];
            curr_rot = curr_rot @ R_local_end

        backbone = np.array(points)

        # --- Frenet & Markers ---
        tangents = np.gradient(backbone, axis=0)
        norms = np.linalg.norm(tangents, axis=1);
        norms[norms < 1e-9] = 1.0
        tangents /= norms[:, None]

        normals, binormals = [], []
        t0 = tangents[0]
        n0 = np.cross(t0, [0, 1, 0]) if abs(t0[0]) > 0.9 else np.cross(t0, [1, 0, 0])
        n0 = safe_normalize(n0)
        normals.append(n0);
        binormals.append(safe_normalize(np.cross(t0, n0)))

        for i in range(1, len(tangents)):
            t_c = tangents[i];
            n_p = normals[-1]
            n_c = safe_normalize(n_p - np.dot(n_p, t_c) * t_c)
            if np.linalg.norm(n_c) < 1e-9: n_c = n_p
            normals.append(n_c);
            binormals.append(safe_normalize(np.cross(t_c, n_c)))
        normals = np.array(normals);
        binormals = np.array(binormals)

        markers = []
        phases = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        turns = 12.0

        # 记录切面上的 Marker 角度 (Ground Truth Angles)
        marker_angles_at_slice = []

        for ph in phases:
            pts = []
            for i in range(len(backbone)):
                u = i / (len(backbone) - 1)
                # 物理公式：angle = u * turns * 2pi + phase
                ang = u * turns * 2 * np.pi + ph
                pos = backbone[i] + (self.radius + 0.5) * (np.cos(ang) * normals[i] + np.sin(ang) * binormals[i])
                pts.append(pos)
            markers.append(np.array(pts))

        return backbone, markers, normals, tangents, kappa_gt, phi_gt, turns


def render_sample(backbone, markers, slice_idx, normals, tangents):
    # 渲染单帧 (纯净版，方便学习)
    img_size = 224;
    w_mm = 30.0;
    res = img_size / w_mm
    image = np.zeros((img_size, img_size), dtype=np.uint8)

    target_pt = backbone[slice_idx]
    base_normal = normals[slice_idx]
    base_tangent = tangents[slice_idx]

    probe_pos = target_pt + base_normal * 20.0
    z_axis = -base_normal;
    y_axis = base_tangent;
    x_axis = safe_normalize(np.cross(y_axis, z_axis))
    plane_p = probe_pos;
    plane_n = y_axis

    # 渲染 Marker
    slice_marker_angles = []

    # 计算切面上的相对弧长 u (0~1)
    # 这对物理 Loss 至关重要
    u_slice = slice_idx / (len(backbone) - 1)

    for m_pts in markers:
        dists = np.dot(m_pts - plane_p, plane_n)
        crossings = np.where(np.diff(np.sign(dists)))[0]
        for idx in crossings:
            p1, p2 = m_pts[idx], m_pts[idx + 1]
            alpha = abs(dists[idx]) / (abs(dists[idx]) + abs(dists[idx + 1]) + 1e-9)
            pt = p1 + alpha * (p2 - p1)
            rel = pt - plane_p
            px = int((np.dot(rel, x_axis) + w_mm / 2) * res)
            py = int(np.dot(rel, z_axis) * res)
            if 0 < px < img_size and 0 < py < img_size:
                cv2.circle(image, (px, py), 4, 255, -1)

    return image, u_slice


# ==========================================
# 生成主程序
# ==========================================
if __name__ == "__main__":
    DATA_DIR = "pinn_dataset"
    TOTAL_SAMPLES = 20000

    if os.path.exists(DATA_DIR): shutil.rmtree(DATA_DIR)
    os.makedirs(os.path.join(DATA_DIR, "images"))

    engine = RobotPhysicsEngine()

    metadata = []  # 存储物理标签

    print(f"🚀 生成 PINN 数据集: {TOTAL_SAMPLES} 张...")

    for i in tqdm(range(TOTAL_SAMPLES)):
        # 生成
        backbone, markers, normals, tangents, kappa_gt, phi_gt, turns = engine.generate_sample()

        # 随机切片
        valid_len = len(backbone)
        slice_idx = np.random.randint(int(valid_len * 0.2), int(valid_len * 0.8))

        # 渲染
        img, u_slice = render_sample(backbone, markers, slice_idx, normals, tangents)

        # 保存图片
        img_name = f"img_{i:05d}.png"
        cv2.imwrite(os.path.join(DATA_DIR, "images", img_name), img)

        # 记录物理参数 (这就是 Physics 的来源！)
        # u_slice: 切片在机器人身上的归一化位置 (0~1)。这也是已知量(可以通过探头位置追踪获得)
        metadata.append([img_name, kappa_gt, phi_gt, u_slice])

    # 保存标签
    df = pd.DataFrame(metadata, columns=["filename", "kappa", "phi", "u_position"])
    df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)

    print("✅ 数据集生成完毕！包含 metadata.csv")