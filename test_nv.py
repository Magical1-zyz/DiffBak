import os
import argparse
import json
import torch
import nvdiffrast.torch as dr
import numpy as np

# 引用工程模块
from dataset.dataset_mesh import DatasetMesh
from render import mesh, render, light, util, texture


# ----------------------------------------------------------------------------
# 核心函数：渲染环境背景 (Ray Casting)
# ----------------------------------------------------------------------------

def render_env_background(envlight, resolution, mv, proj, fovy):
    """
    通过射线投射采样环境贴图生成背景
    """
    h, w = resolution

    # 1. 生成相机空间的射线方向 (NDC -> Camera Space)
    # PyTorch 的 meshgrid 生成 (y, x)
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device='cuda'),
        torch.linspace(-1.0, 1.0, w, device='cuda'),
        indexing='ij'
    )

    # 根据 FOV 和 Aspect Ratio 计算射线
    # OpenGL 坐标系: Looking down -Z, +Y is up, +X is right
    # util.perspective 中的投影矩阵 (1,1) 元素是 1/-tan(fov/2)，意味着它翻转了 Y 轴
    # 为了匹配 rasterizer，我们需要构建对应的射线
    aspect = w / h
    tan_half_fov = np.tan(fovy / 2.0)

    # 修正：根据 util.perspective 的逻辑，Y 轴在投影时被翻转了
    # 所以我们在生成射线时，屏幕空间的 Y 对应 Camera 空间的 -Y
    cam_x = gx * tan_half_fov * aspect
    cam_y = -gy * tan_half_fov  # 注意这里的负号，匹配 nvdiffrast/OpenGL 的投影行为
    cam_z = -torch.ones_like(gx)  # 图像平面在 z=-1

    cam_dirs = torch.stack((cam_x, cam_y, cam_z), dim=-1)  # [H, W, 3]
    cam_dirs = util.safe_normalize(cam_dirs)

    # 2. 转换到世界空间 (Camera Space -> World Space)
    # mv 是 World-to-Camera，我们需要 inverse(mv) 即 Camera-to-World
    # mv: [1, 4, 4]
    inv_mv = torch.linalg.inv(mv[0])
    rotation = inv_mv[:3, :3]  # 取旋转部分

    # [H, W, 3] @ [3, 3] -> [H, W, 3]
    # 注意矩阵乘法顺序：v @ R.T
    world_dirs = cam_dirs @ rotation.T

    # 3. 采样环境贴图 (Cubemap)
    # envlight.base 是 [6, Res, Res, 3] 的 Cubemap Tensor
    # nvdiffrast 的 texture 函数支持 boundary_mode='cube'
    # 输入需要是 [N, H, W, 3]
    env_col = dr.texture(
        envlight.base[None, ...],
        world_dirs[None, ...].contiguous(),
        filter_mode='linear',
        boundary_mode='cube'
    )

    return env_col  # [1, H, W, 3]


# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------

def _str2bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    s = str(v).strip().lower()
    if s in ("y", "yes", "true", "t", "1"): return True
    if s in ("n", "no", "false", "f", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


@torch.no_grad()
def composite_and_save(opt_img, bg_img, path, exposure=1.0):
    """
    合成：前景(opt_img) + 背景(bg_img, HDR)
    并进行 Tone Mapping 保存
    """
    # 1. 提取
    rgb_fg = opt_img[..., 0:3]
    alpha = opt_img[..., 3:4]
    rgb_bg = bg_img[..., 0:3]

    # 2. 合成 (Alpha Blending)
    # 注意：bg_img 也是 HDR 的，需要和前景一起进行后处理
    img_composite = rgb_fg * alpha + rgb_bg * (1.0 - alpha)

    # 3. 曝光调整 (可选)
    img_composite = img_composite * exposure

    # 4. sRGB Gamma 校正
    img_srgb = util.rgb_to_srgb(img_composite)

    # 5. 保存
    util.save_image(path, img_srgb[0].detach().cpu().numpy())


# ----------------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DiffBake Environment Test')

    parser.add_argument('--config', type=str, default=None, help='Config JSON file')
    parser.add_argument('-rm', '--ref_mesh', type=str, required=False, help='Mesh to render')
    parser.add_argument('-o', '--out-dir', type=str, default='out/env_test', help='Output folder')

    parser.add_argument('--train_res', nargs=2, type=int, default=[1024, 1024], help='Resolution')
    parser.add_argument('--spp', type=int, default=1, help='Samples per pixel')

    # 环境与相机
    parser.add_argument('--envmap', type=str, required=False, help='HDR environment map path (Required)')
    parser.add_argument('--env_scale', type=float, default=1.0, help='Environment brightness')
    parser.add_argument('--cam_radius_scale', type=float, default=2.0)
    parser.add_argument('--cam_near_far', nargs=2, type=float, default=[0.1, 1000.0])

    # 兼容参数
    parser.add_argument('--base_mesh', type=str, default=None)

    FLAGS = parser.parse_args()

    if FLAGS.config is not None:
        if not os.path.exists(FLAGS.config):
            raise FileNotFoundError(f"Config file not found: {FLAGS.config}")
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            if hasattr(FLAGS, key):
                FLAGS.__dict__[key] = data[key]

    print(f"\n=== Environment Rendering Test ===")
    print(f" Target Mesh: {FLAGS.ref_mesh}")
    print(f" Environment: {FLAGS.envmap} (Scale: {FLAGS.env_scale})")
    print(f" Output Dir:  {FLAGS.out_dir}")
    print(f"================================\n")

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()

    # 1. 加载模型
    print(f"[1/3] Loading Mesh...")
    ref_mesh = mesh.load_mesh(FLAGS.ref_mesh)
    ref_mesh = mesh.auto_normals(ref_mesh)
    ref_mesh = mesh.compute_tangents(ref_mesh)

    # 2. 初始化 Dataset (生成视角 + 加载环境光)
    print(f"[2/3] Preparing Environment & Views...")
    # validate=True 保证顺序渲染
    dataset = DatasetMesh(ref_mesh, base_mesh=None, glctx=glctx, FLAGS=FLAGS, validate=True)

    # 必须构建 Mipmaps，否则环境光不可用
    dataset.envlight.build_mips()

    views = dataset.precomputed_views
    n_views = len(views)

    # 3. 渲染循环
    print(f"[3/3] Rendering with Background...")

    with torch.no_grad():
        for i, view in enumerate(views):
            # A. 渲染前景物体
            if len(dataset.ref_meshes) == 1:
                out = render.render_mesh(glctx, dataset.ref_meshes[0], view['mvp'], view['campos'], dataset.envlight,
                                         FLAGS.train_res, spp=FLAGS.spp, msaa=True)
            else:
                out = render.render_meshes(glctx, dataset.ref_meshes, view['mvp'], view['campos'], dataset.envlight,
                                           FLAGS.train_res, spp=FLAGS.spp, msaa=True)

            fg_img = out['shaded']  # [1, H, W, 4]

            # B. 渲染环境背景 (核心步骤)
            # 注意：这里我们提取了视角中的 fovy 参数
            bg_img = render_env_background(
                dataset.envlight,
                FLAGS.train_res,
                view['mv'],
                view['mvp'],  # 这里其实没用到proj，只用了fovy，也可以传proj矩阵反推
                view['fovy']
            )
            # 对背景应用 env_scale (因为 light.py 加载时已经乘过一次 scale 存入 base，这里直接采样本体即可)
            # 但要注意 light.load_env 里 scale 已经乘进去了，所以这里采样的值已经是 Scale 过的。

            # C. 保存
            if i < dataset.n_top:
                name = f"env_top_{i:03d}.png"
            else:
                idx_bottom = i - dataset.n_top
                name = f"env_bottom_{idx_bottom:03d}.png"

            save_path = os.path.join(FLAGS.out_dir, name)

            # 合成并保存 (无需再乘 exposure，除非为了测试显示效果)
            composite_and_save(fg_img, bg_img, save_path)

            print(f"      Saved {i + 1}/{n_views}: {name}", end='\r')

    print(f"\n\n[Done] All images saved to {FLAGS.out_dir}")


if __name__ == "__main__":
    main()