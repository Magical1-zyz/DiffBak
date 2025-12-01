# DiffBake: Differentiable Texture Baking

![Teaser image](images/teaser.jpg) **DiffBake** 是一个基于 [nvdiffrec](https://github.com/NVlabs/nvdiffrec) 深度精简与改良的高性能纹理烘焙工具。

它利用可微渲染（Differentiable Rendering）技术，将高模（Reference Mesh）的**光影、材质和细节**精确地“烘焙”到低模（Base Mesh）的 Diffuse 贴图上。不同于传统的烘焙工具，DiffBake 通过梯度下降算法直接优化纹理像素，能够生成像素级对齐（Pixel-perfect）、边缘锐利且抗锯齿的高质量贴图。

## 核心特性 (Features)

- **🚀 极速显存预渲染 (VRAM Pre-rendering)**: 预先将高模的 Ground Truth 渲染至显存，训练过程无需重复渲染高模，速度提升 5x 以上。
- **🧩 自动 UV 重构 (Auto UV Unwrapping)**: 集成 `xatlas`，自动为低模生成无重叠的全局 UV，解决多材质合并时的 UV 冲突问题。
- **🎨 光影烘焙 (Lighting Baking)**: 支持将 HDR 环境光照（Lighting）直接烘焙进 BaseColor，输出 Unlit 风格的资产，完美还原高模光影。
- **🎥 斐波那契采样 (Fibonacci Sampling)**: 使用斐波那契半球分布算法生成相机视角，确保模型表面覆盖均匀，无死角。
- **💎 多材质支持 (Multi-Material Support)**: 支持将多个材质合并为一个 Atlas，或保留原始材质结构分别烘焙。
- **📉 鲁棒的损失函数**: 结合 Log-L1 Loss 与纹理平滑正则化（Smoothness Regularizer），有效抑制噪点并保留阴影细节。

## 安装 (Installation)

环境依赖与原版 `nvdiffrec` 类似，但更加轻量。

**系统要求**:
- OS: Windows / Linux
- GPU: NVIDIA GPU with CUDA 11.3+
- Python: 3.9+

**安装步骤**:

1. 创建环境:
```bash
conda create -n diffbake python=3.9
activate diffbake
```
2. 安装基础依赖:
```Bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
pip install ninja imageio xatlas gdown
```

3. 安装核心渲染库:
```Bash
pip install git+[https://github.com/NVlabs/nvdiffrast/](https://github.com/NVlabs/nvdiffrast/)
```

## 快速开始 (Usage)
1. 基础烘焙 (单材质合并)将高模的光影烘焙到低模的一张贴图上：
```Bash
python train_bake.py \
    --ref_mesh data/high_poly.gltf \
    --base_mesh data/low_poly.gltf \
    --texture_res 4096 4096 \
    --out-dir out/my_bake
```
2. 多材质烘焙 (保留原始材质槽)如果低模有多个材质 ID，且你希望保留这些材质（每个材质对应一张独立贴图）：
```Bash
python train_bake.py \
    --ref_mesh data/high_poly.obj \
    --base_mesh data/low_poly.obj \
    --multi_materials true \
    --out-dir out/multi_mat_bake
```
3. 使用配置文件 (推荐)
你可以通过 JSON 文件管理复杂参数：
```Bash
python train_bake.py --config configs/my_bake_config.json
```

## 📂 工程目录结构 (Project Structure)
本项目经过深度精简，仅保留了纹理烘焙所需的核心组件。
```text
DiffBake/
├── configs/                # 配置文件目录
├── data/                   # 存放输入模型 (.gltf/.obj) 和环境贴图 (.hdr)
├── dataset/                # 数据加载模块
│   ├── dataset.py          # 数据集基类
│   └── dataset_mesh.py     # [核心] 负责加载 Mesh，并使用斐波那契算法生成均匀的相机视角
├── geometry/               # 几何处理模块
│   └── dlmesh.py           # Mesh 容器，负责管理顶点数据和待优化的材质参数
├── render/                 # 核心可微渲染库
│   ├── renderutils/        # 底层 CUDA 加速算子 (光栅化、着色、损失函数计算)
│   ├── gltf.py             # glTF 格式读写支持 (含多材质导出逻辑)
│   ├── obj.py              # OBJ 格式读写支持 (含 MTL 解析)
│   ├── light.py            # 环境光 (Environment Map) 处理
│   ├── material.py         # 材质定义与管理
│   ├── mesh.py             # Mesh 数据结构与法线/切线计算
│   ├── render.py           # [核心] 可微渲染管线 (Rasterization -> Shading)
│   ├── texture.py          # 2D 纹理处理 (自动 Mipmap 生成、参数钳制)
│   └── util.py             # 通用工具 (图像读写、数学运算)
├── out/                    # 输出目录 (自动生成)
└── train_bake.py           # [主程序] 启动脚本，包含训练循环、UV 重构、预渲染和导出逻辑
```

## DiffBake 参数详解

DiffBake (`train_bake.py`) 支持多种参数来控制输入输出、烘焙质量、材质模式及训练过程。

### 📁 输入与输出 (Input/Output)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--ref_mesh` | `str` | **必填** | **参考高模路径 (Reference Mesh)**。<br>作为“真值”来源，支持带材质的 `.gltf` 或 `.obj` 格式。 |
| `--base_mesh` | `str` | **必填** | **目标低模路径 (Base Mesh)**。<br>作为烘焙目标，仅需几何体信息。程序会自动使用 `xatlas` 为其重新生成无重叠 UV。 |
| `--out-dir` | `str` | `out/baking_result` | **输出目录**。<br>最终烘焙好的模型及中间结果将保存在此目录下的 `baked_mesh` 子文件夹中。 |
| `--config` | `str` | `None` | **配置文件路径**。<br>指定一个 `.json` 文件来一次性覆盖多个参数。 |

### 🎨 烘焙选项 (Baking Options)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--multi_materials` | `bool` | `False` | **多材质模式开关**。<br>• `False` (默认): 将低模视为单一整体，所有部分合并烘焙到一张贴图上（适合游戏资产优化）。<br>• `True`: 保持低模原始的材质 ID 划分，为每个材质 ID 独立烘焙一张贴图（适合需要保留材质槽位的情况）。 |
| `--texture_res` | `int [2]` | `4096 4096` | **输出纹理分辨率** (宽 高)。<br>建议设置为 `2048 2048` 或 `4096 4096` 以保证清晰度。 |

### ⚙️ 训练与质量 (Training)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--iter` | `int` | `3000` | **迭代次数**。<br>优化过程的总步数。通常 2000-3000 步足以收敛。 |
| `--batch` | `int` | `1` | **批次大小**。<br>每次迭代渲染的视角数量。由于采用了显存预渲染技术，设为 `1` 即可获得极快的速度并大幅节省显存。 |
| `--lr` | `float` | `0.03` | **学习率**。<br>控制纹理像素优化的步长。默认 `0.03` 通常表现良好。 |
| `--train_res` | `int [2]` | `1024 1024` | **训练渲染分辨率**。<br>优化过程中内部渲染画布的大小。建议保持 `1024` 或 `512`，过高会导致显存溢出。 |
| `--spp` | `int` | `2` | **抗锯齿采样率 (Samples Per Pixel)**。<br>训练时的超采样倍数。`2` 表示 2x MSAA。 |

### 📉 损失与正则化 (Loss & Regularization)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--loss_type` | `str` | `logl1` | **主损失函数类型**。<br>• `logl1` (推荐): 对数 L1 Loss，对高光不敏感，能更好地还原阴影暗部细节。<br>• `l1`: 标准 L1 Loss，边缘锐利。<br>• `mse`: 均方误差，倾向于产生平滑/模糊的结果。 |
| `--smooth_weight` | `float` | `0.02` | **纹理平滑正则化权重**。<br>用于抑制噪点（Fireflies）。如果高模渲染图噪点较多，可适当调大此值（如 `0.05`）；如果纹理细节丢失，可调小或设为 `0`。 |

### 🌤️ 环境与相机 (Environment & Camera)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--envmap` | `str` | `None` | **HDR 环境贴图路径** (.hdr)。<br>指定用于照亮高模的 HDR 图像。如果不填，将使用纯白环境光（适合烘焙纯 BaseColor）。 |
| `--env_scale` | `float` | `1.0` | **环境光强度系数**。<br>调整 HDR 的亮度倍率。 |
| `--cam_radius_scale` | `float` | `2.0` | **相机距离缩放**。<br>基于模型包围球半径的倍数。`2.0` 表示相机距离物体中心 2 倍半径远。值越大透视畸变越小，但像素利用率可能降低。 |
| `--cam_near_far` | `float [2]` | `0.1 1000.0` | **相机裁剪平面** (近平面 远平面)。 |

### 🖥️ 显示与调试 (Misc)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--display-interval` | `int` | `50` | **可视化间隔**。<br>每隔多少步在屏幕上弹窗显示当前烘焙进度（包含对比图和差值热力图）。 |
| `--save-interval` | `int` | `100` | **保存间隔**。<br>（目前代码主要使用最终导出，此参数为预留）。 |
| `--background_rgb` | `float [3]` | `0 0 0` | **可视化背景色** (R G B)。<br>范围 `0.0 - 1.0`。例如 `1.0 1.0 1.0` 为白色背景。仅影响训练时的预览窗口，不影响烘焙结果。 |

## 引用与致谢 (Citation & Acknowledgement)

本项目修改自 NVIDIA 的开源项目 **nvdiffrec**。核心可微渲染逻辑归原作者所有。

如果您在研究中使用了本项目，请引用原始论文：

```bibtex
@inproceedings{Munkberg_2022_CVPR,
    author    = {Munkberg, Jacob and Hasselgren, Jon and Shen, Tianchang and Gao, Jun and Chen, Wenzheng 
                    and Evans, Alex and M\"uller, Thomas and Fidler, Sanja},
    title     = "{Extracting Triangular 3D Models, Materials, and Lighting From Images}",
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8280-8290}
}
```
特别感谢 xatlas 提供的优秀 UV 展开算法。