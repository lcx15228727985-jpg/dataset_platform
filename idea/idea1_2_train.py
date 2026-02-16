import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

# ==========================================
# 配置
# ==========================================
DATASET_NAME = "/root/pinn_seq_dataset"
TOTAL_SAMPLES = 5000
SEQ_LEN = 5  # 核心改动：一次看5帧
IMG_SIZE = 224


# ==========================================
# 物理引擎
# ==========================================
def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9: return np.zeros_like(v)
    return v / norm


class RobotPhysicsEngine:
    def __init__(self, radius=4.0):
        self.radius = radius

    def generate_robot(self):
        # 随机生成姿态
        shape_type = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
        points_per_seg = 200

        if shape_type == 0:
            configs = [(100, 0.001, 0)]
            kappa, phi = 0.0, 0.0
        elif shape_type == 1:
            theta = np.deg2rad(np.random.uniform(20, 90))
            phi = np.random.uniform(0, 2 * np.pi)
            configs = [(100, theta, phi)]
            kappa = theta / 100.0
        elif shape_type == 2:
            theta1 = np.deg2rad(np.random.uniform(30, 80))
            phi1 = np.random.uniform(0, 2 * np.pi)
            configs = [(100, theta1, phi1)]
            kappa = theta1 / 100.0;
            phi = phi1

        # PCC 建模
        points = [np.array([0., 0., 0.])]
        curr_pos = np.array([0., 0., 0.]);
        curr_rot = np.eye(3)
        for length, theta, p in configs:
            s = np.linspace(0, length, points_per_seg)
            if abs(theta) < 1e-4: theta = 1e-4
            r = length / theta;
            t = (s / length) * theta
            arc_x = r * (1 - np.cos(t));
            arc_z = r * np.sin(t)
            p_x = arc_x * np.cos(p);
            p_y = arc_x * np.sin(p);
            p_z = arc_z
            p_local = np.column_stack([p_x, p_y, p_z])
            c_p, s_p = np.cos(p), np.sin(p);
            c_t, s_t = np.cos(theta), np.sin(theta)
            R_phi = np.array([[c_p, -s_p, 0], [s_p, c_p, 0], [0, 0, 1]])
            R_bend = np.array([[c_t, 0, s_t], [0, 1, 0], [-s_t, 0, c_t]])
            R_local_end = R_phi @ R_bend @ R_phi.T
            for pt in p_local[1:]: points.append(curr_pos + curr_rot @ pt)
            curr_pos = points[-1];
            curr_rot = curr_rot @ R_local_end

        backbone = np.array(points)
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

        return backbone, markers, normals, tangents, kappa, phi


def render_slice(backbone, markers, idx, normals, tangents):
    img_size = IMG_SIZE;
    w_mm = 30.0;
    res = img_size / w_mm
    image = np.zeros((img_size, img_size), dtype=np.uint8)

    target_pt = backbone[idx];
    base_normal = normals[idx];
    base_tangent = tangents[idx]
    probe_pos = target_pt + base_normal * 20.0
    z_axis = -base_normal;
    y_axis = base_tangent;
    x_axis = safe_normalize(np.cross(y_axis, z_axis))
    plane_p = probe_pos;
    plane_n = y_axis

    for m_pts in markers:
        dists = np.dot(m_pts - plane_p, plane_n)
        crossings = np.where(np.diff(np.sign(dists)))[0]
        for c_idx in crossings:
            p1, p2 = m_pts[c_idx], m_pts[c_idx + 1]
            alpha = abs(dists[c_idx]) / (abs(dists[c_idx]) + abs(dists[c_idx + 1]) + 1e-9)
            pt = p1 + alpha * (p2 - p1)
            rel = pt - plane_p
            px = int((np.dot(rel, x_axis) + w_mm / 2) * res)
            py = int(np.dot(rel, z_axis) * res)
            if 0 < px < img_size and 0 < py < img_size:
                cv2.circle(image, (px, py), 4, 255, -1)
    return image


if __name__ == "__main__":
    if os.path.exists(DATASET_NAME): shutil.rmtree(DATASET_NAME)
    os.makedirs(DATASET_NAME)

    engine = RobotPhysicsEngine()
    metadata = []
    print(f"🚀 生成 5帧堆叠 PINN 数据集: {TOTAL_SAMPLES} 组...")

    for i in tqdm(range(TOTAL_SAMPLES)):
        backbone, markers, norms, tans, kappa, phi = engine.generate_robot()

        # 随机找个起始点
        max_start = len(backbone) - SEQ_LEN - 10
        start_idx = np.random.randint(10, max_start)

        frames = []
        for t in range(SEQ_LEN):
            img = render_slice(backbone, markers, start_idx + t, norms, tans)
            frames.append(img)

        # 堆叠: (5, 224, 224) -> 保存为 npy
        # 训练时我们会把这个当作一个 5通道(或15通道RGB) 的图
        stack = np.array(frames, dtype=np.uint8)

        filename = f"stack_{i:05d}.npy"
        np.save(os.path.join(DATASET_NAME, filename), stack)

        # 记录中心帧的物理参数
        # u_pos 归一化位置
        u_pos = (start_idx + SEQ_LEN // 2) / len(backbone)
        metadata.append([filename, kappa, phi, u_pos])

    df = pd.DataFrame(metadata, columns=["filename", "kappa", "phi", "u_position"])
    df.to_csv(os.path.join(DATASET_NAME, "metadata.csv"), index=False)
    print("✅ 数据生成完毕！")