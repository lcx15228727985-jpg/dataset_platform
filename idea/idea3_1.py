import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 配置
# ==========================================
# 使用之前的非对称数据集
DATA_ROOT = "/root/asym_pinn_dataset"
CSV_PATH = os.path.join(DATA_ROOT, "metadata.csv")
OUTPUT_CSV = "wire_features.csv"


def get_marker_angles(img_gray):
    """
    从灰度图中提取 3 个 Marker 的角度
    返回: [angle_0, angle_1, angle_2] (弧度, -pi 到 pi)
    其中 angle_0 对应最大的那个点 (主线束)
    """
    # 1. 二值化
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

    # 2. 找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 至少要找到 3 个点
    if len(contours) < 3:
        return None

    # 3. 分析轮廓 (找圆心和面积)
    markers = []
    img_center = np.array([img_gray.shape[1] / 2, img_gray.shape[0] / 2])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        # 计算相对于图像中心的角度
        # y轴向下，x轴向右。使用 arctan2(y, x)
        # 注意坐标系转换: y_img = -y_cartesian
        dx = x - img_center[0]
        dy = y - img_center[1]

        # 图像坐标系转极坐标角度 (y向下, 所以dy要反一下或者直接用)
        # 这里我们统一用 atan2(dy, dx)，只要前后一致即可
        angle = np.arctan2(dy, dx)

        markers.append({'area': area, 'angle': angle, 'center': (x, y)})

    # 4. 排序：按面积从大到小
    # area最大的就是 Marker 0 (我们设定的非对称主标记)
    markers.sort(key=lambda x: x['area'], reverse=True)

    # 取前3个 (防止有噪点)
    top_3 = markers[:3]

    # 提取角度
    # top_3[0] 是主标记，必须放在第一个位置
    # 后面的两个按角度排序，保证顺序一致性
    others = sorted(top_3[1:], key=lambda x: x['angle'])

    result_angles = [top_3[0]['angle']] + [m['angle'] for m in others]

    return result_angles


def process_dataset():
    if not os.path.exists(CSV_PATH):
        print("❌ 找不到数据集")
        return

    df = pd.read_csv(CSV_PATH)
    feature_data = []

    print("🚀 正在提取线束角度特征...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        npy_path = os.path.join(DATA_ROOT, row['filename'])
        try:
            stack = np.load(npy_path)  # (5, 224, 224)

            # 取首尾两帧 (Siamese)
            img_a = stack[0]
            img_b = stack[-1]

            angles_a = get_marker_angles(img_a)
            angles_b = get_marker_angles(img_b)

            if angles_a is None or angles_b is None:
                continue

            # 保存特征
            # 输入: 6个角度 + 1个距离
            # 输出: kappa, phi
            feature_row = {
                'a0': angles_a[0], 'a1': angles_a[1], 'a2': angles_a[2],
                'b0': angles_b[0], 'b1': angles_b[1], 'b2': angles_b[2],
                'delta_dist': 4.0,  # 假设间隔4帧，这里可以简化为固定值或从meta读取
                'kappa': row['kappa'],
                'phi': row['phi']
            }
            feature_data.append(feature_row)

        except Exception as e:
            continue

    # 保存为新的 CSV
    new_df = pd.DataFrame(feature_data)
    new_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 特征提取完毕，保存至: {OUTPUT_CSV}")
    print(f"   有效样本数: {len(new_df)} / {len(df)}")


if __name__ == "__main__":
    process_dataset()