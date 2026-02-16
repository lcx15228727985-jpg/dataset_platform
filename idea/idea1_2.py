import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import shutil

# ==========================================
# 配置
# ==========================================
DATASET_NAME = "/root/pairs_dataset"
TOTAL_PAIRS = 20000
IMG_SIZE = 224


# ==========================================
# 物理引擎 (保持纯净版逻辑)
# ==========================================
def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9: return np.zeros_like(v)
    return v / norm


class RobotPhysicsEngine:
    def __init__(self, radius=4.0):
        self.radius = radius

    def generate_backbone_and_markers(self):
        # 随机生成一种弯曲状态
        shape_type = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
        points_per_seg = 200

        if shape_type == 0:  # Straight
            configs = [(100, 0.001, 0)]
            kappa_gt, phi_gt = 0.0, 0.0
        elif shape_type == 1:  # J-Curve
            theta = np.deg2rad(np.random.uniform(20, 90))
            phi = np.random.uniform(0, 2 * np.pi)
            configs = [(100, theta, phi)]
            kappa_gt = theta / 100.0;
            phi_gt = phi
        elif shape_type == 2:  # S-Curve
            theta1 = np.deg2rad(np.random.uniform(30, 80))
            phi1 = np.random.uniform(0, 2 * np.pi)
            # 简化：我们只取第一段作为"任意两切片"的采样源
            # 这样保证两张切片之间的曲率是恒定的，符合PCC假设
            configs = [(100, theta1, phi1)]
            kappa_gt = theta1 / 100.0;
            phi_gt = phi1

        # PCC 建模 (省略重复代码，与之前一致)
        points = [np.array([0., 0., 0.])]
        curr_pos = np.array([0., 0., 0.]);
        curr_rot = np.eye(3)
        for length, theta, phi in configs:
            s = np.linspace(0, length, points_per_seg)
            if abs(theta) < 1e-4: theta = 1e-4
            r = length / theta;
            t = (s / length) * theta
            arc_x = r * (1 - np.cos(t));
            arc_z = r * np.sin(t)
            p_x = arc_x * np.cos(phi);
            p_y = arc_x * np.sin(phi);
            p_z = arc_z
            p_local = np.column_stack([p_x, p_y, p_z])
            c_p, s_p = np.cos(phi), np.sin(phi);
            c_t, s_t = np.cos(theta), np.sin(theta)
            R_phi = np.array([[c_p, -s_p, 0], [s_p, c_p, 0], [0, 0, 1]])
            R_bend = np.array([[c_t, 0, s_t], [0, 1, 0], [-s_t, 0, c_t]])
            R_local_end = R_phi @ R_bend @ R_phi.T
            for p in p_local[1:]: points.append(curr_pos + curr_rot @ p)
            curr_pos = points[-1];
            curr_rot = curr_rot @ R_local_end
        backbone = np.array(points)

        # Frenet
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

        # Markers
        markers = []
        phases = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        turns = 12.0
        for ph in phases:
            pts = []
            for i in range(len(backbone)):
                u = i / (len(backbone) - 1)
                ang = u * turns * 2 * np.pi + ph
                pos = backbone[i] + (self.radius + 0.5) * (np.cos(ang) * normals[i] + np.sin(ang) * binormals[i])
                pts.append(pos)
            markers.append(np.array(pts))

        return backbone, markers, normals, tangents, kappa_gt, phi_gt


def render_slice(backbone, markers, slice_idx, normals, tangents):
    # 纯净渲染
    img_size = IMG_SIZE;
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

    # Markers only (Wall is less important for phase diff)
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
    return image


if __name__ == "__main__":
    if os.path.exists(DATASET_NAME): shutil.rmtree(DATASET_NAME)
    os.makedirs(os.path.join(DATASET_NAME, "images"))

    engine = RobotPhysicsEngine()
    metadata = []

    print(f"🚀 生成成对数据集: {TOTAL_PAIRS} 对...")

    for i in tqdm(range(TOTAL_PAIRS)):
        backbone, markers, normals, tangents, kappa, phi = engine.generate_backbone_and_markers()

        # 随机选取两个索引 (Idx A 和 Idx B)
        max_idx = len(backbone) - 1
        idx_a = np.random.randint(10, max_idx - 20)
        # 距离随机 (10 到 50 个单位) -> 模拟任意间距
        delta = np.random.randint(10, 50)
        idx_b = idx_a + delta

        if idx_b > max_idx: idx_b = max_idx

        # 归一化距离 (作为网络的辅助输入)
        dist_norm = (idx_b - idx_a) / len(backbone)

        # 渲染两张图
        img_a = render_slice(backbone, markers, idx_a, normals, tangents)
        img_b = render_slice(backbone, markers, idx_b, normals, tangents)

        # 保存 (拼接成一张宽图存储，或者分开存)
        # 这里分开存: 00001_a.png, 00001_b.png
        name_a = f"pair_{i:05d}_a.png"
        name_b = f"pair_{i:05d}_b.png"

        cv2.imwrite(os.path.join(DATASET_NAME, "images", name_a), img_a)
        cv2.imwrite(os.path.join(DATASET_NAME, "images", name_b), img_b)

        metadata.append([name_a, name_b, dist_norm, kappa, phi])

    df = pd.DataFrame(metadata, columns=["img_a", "img_b", "delta_dist", "kappa", "phi"])
    df.to_csv(os.path.join(DATASET_NAME, "metadata.csv"), index=False)
    print("✅ 成对数据集生成完毕！")