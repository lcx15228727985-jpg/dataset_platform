# 🧬 CR-Chirp Digital Twin: 连续体机器人超声感知数字孪生系统

**CR-Chirp Digital Twin** 是一个基于 **“空间频率调制 (Spatial Frequency Modulation, SFM)”** 理论的连续体机器人纹理设计与仿真平台。

本项目旨在解决连续体机器人（Continuum Robots）在体内环境下的本体感知难题。通过在机器人表面设计特殊的 **“螺旋双频 Chirp (Helical Dual-Tone Chirp)”** 纹理，将机器人的形变状态（曲率、弯曲方向）编码为超声图像中的频率与相位变化，实现了单探头局部观测下的全局精确定位。

---

## 🌟 核心特性 (Key Features)

* **🌌 螺旋双频 Chirp 纹理 (Helical Dual-Tone Chirp):** * 采用 **双频拍频 (Beating)** 产生包络，解决局部视场下的定位模糊。
    * 采用 **螺旋结构 (Helical Structure)**，打破圆周对称性，解决旋转方向模糊。
    * 采用 **线性调频 (Chirp)**，利用频率梯度实现 Z 轴绝对定位。
* **📐 混合基函数生成器 (Hybrid Basis Generator):**
    * $\theta$ 轴使用 **傅里叶级数 (Fourier Series)**，保证圆周方向完美的拓扑连续性（无缝）。
    * $Z$ 轴使用 **DCT (离散余弦变换)**，适应有限长的轴向特征。
* **🚀 梯度流形优化 (Gradient Manifold Optimization):**
    * 引入 **Slope Loss**，强迫纹理生成高梯度的浮雕特征，避免“平坦死区”，对抗超声成像误差。
* **🖥️ 交互式数字孪生控制台:**
    * 基于 Streamlit 的 Web 界面，实时调节物理参数（曲率、位置），观测超声图像与频谱特性的动态响应。

---

## 🏗️ 系统架构 (Architecture)

```text
CR_Chirp_DigitalTwin/
├── app.py                  # [前端] Streamlit 仿真控制台 (用户交互界面)
├── train_texture.py        # [后端] 纹理优化训练脚本 (核心算法入口)
├── requirements.txt        # [配置] 项目依赖
├── optimized_texture.pth   # [模型] 训练好的纹理权重文件
└── modules/                # [核心库]
    ├── texture.py          # 纹理生成器 (混合基函数 + 螺旋初始化)
    ├── geometry.py         # 几何引擎 (PCC 运动学 + 3D 映射)
    └── ultrasound.py       # 超声仿真器 (扫描策略 + 散斑噪声)