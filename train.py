import os
import time
import argparse
import json
import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas
import matplotlib.cm
import cv2

# 核心模块引用
from dataset.dataset_mesh import DatasetMesh
from geometry.dlmesh import DLMesh
from render import obj, gltf, material, util, mesh, texture, light, render
from render.renderutils.loss import BakingLoss


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
def render_env_background(envlight, resolution, mv, fovy):
    """
    通过射线投射采样环境贴图生成背景
    """
    h, w = resolution

    # 1. 生成相机空间的射线 (NDC -> Camera Space)
    # PyTorch meshgrid 生成 (y, x)
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device='cuda'),
        torch.linspace(-1.0, 1.0, w, device='cuda'),
        indexing='ij'
    )

    aspect = w / h
    tan_half_fov = np.tan(fovy / 2.0)

    # 修正坐标系方向 (匹配 nvdiffrast/OpenGL)
    cam_x = gx * tan_half_fov * aspect
    cam_y = -gy * tan_half_fov
    cam_z = -torch.ones_like(gx)

    cam_dirs = torch.stack((cam_x, cam_y, cam_z), dim=-1)  # [H, W, 3]
    cam_dirs = util.safe_normalize(cam_dirs)

    # 2. 转换到世界空间 (Camera Space -> World Space)
    # mv 是 World-to-Camera [1, 4, 4]
    inv_mv = torch.linalg.inv(mv[0])
    rotation = inv_mv[:3, :3]

    world_dirs = cam_dirs @ rotation.T

    # 3. 采样环境贴图
    env_col = dr.texture(
        envlight.base[None, ...],
        world_dirs[None, ...].contiguous(),
        filter_mode='linear',
        boundary_mode='cube'
    )

    return env_col  # [1, H, W, 3]


@torch.no_grad()
def composite_background(image_rgba, bg_color):
    """合成背景色用于可视化"""
    # 1. 提取 RGB 和 Alpha
    if image_rgba.shape[-1] >= 4:
        rgb = image_rgba[..., 0:3]
        alpha = image_rgba[..., 3:4]
    else:
        # 如果只有3通道，假设完全不透明
        rgb = image_rgba[..., 0:3]
        alpha = torch.ones_like(rgb[..., 0:1])

    # 2. 处理背景
    # 如果背景是纯色 [3]，扩展维度以匹配广播
    if bg_color.ndim == 1:
        bg = bg_color.view(1, 1, 1, 3).to(image_rgba.device)
    # 如果背景是图片 [1, H, W, 3]，直接使用

    # 3. 合成
    return rgb * alpha + bg_color * (1.0 - alpha)


@torch.no_grad()
def generate_heatmap(opt_rgb, ref_rgb, mask, bg_color_tensor):
    """
    生成带标签的红蓝热力图，并与统一背景合成
    布局：[左下角] Max Err: X.XXXX [红----->蓝] 0
    """
    # 1. 计算误差
    diff = torch.abs(opt_rgb - ref_rgb) * mask

    # 转 Numpy
    diff_gray = torch.mean(diff, dim=-1)[0].detach().cpu().numpy()
    mask_np = mask[0, ..., 0].detach().cpu().numpy()
    bg_np = bg_color_tensor.detach().cpu().numpy()

    # 2. 归一化
    max_err = np.max(diff_gray)
    if max_err < 1e-6: max_err = 1.0
    norm_diff = np.clip(diff_gray / max_err, 0.0, 1.0)

    # 3. 应用 Jet Colormap (Blue->Red)
    heatmap_rgba = matplotlib.cm.jet(norm_diff)
    heatmap_rgb = heatmap_rgba[..., :3]

    # 4. 与背景合成
    heatmap_vis = heatmap_rgb * mask_np[..., None] + bg_np[None, None, :] * (1.0 - mask_np[..., None])

    # 5. 准备画布
    heatmap_u8 = (np.clip(heatmap_vis, 0, 1) * 255).astype(np.uint8)
    heatmap_u8 = np.ascontiguousarray(heatmap_u8)

    if heatmap_u8.ndim == 4: heatmap_u8 = heatmap_u8[0]
    h, w, _ = heatmap_u8.shape

    # 字体设置
    label = f"Max Err: {max_err:.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 字体颜色
    if np.mean(bg_np) > 0.5:
        text_color = (0, 0, 0);
        outline_color = (255, 255, 255)
    else:
        text_color = (255, 255, 255);
        outline_color = (0, 0, 0)

    # 自适应缩放
    scale = max(0.6, w / 1000.0)
    thickness = max(1, int(scale * 2))

    # 计算文字大小
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

    # ---------------------------------------------------------
    # 布局计算 (左下角)
    # ---------------------------------------------------------
    margin = int(20 * scale)
    x_text = margin
    y_text = h - margin  # 文字基线位置

    # 1. 绘制文字 "Max Err: X.XX"
    cv2.putText(heatmap_u8, label, (x_text, y_text), font, scale, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(heatmap_u8, label, (x_text, y_text), font, scale, text_color, thickness, cv2.LINE_AA)

    # 2. 绘制水平色带 (紧跟文字右侧)
    # 只有当有足够空间时才绘制
    bar_x = x_text + text_w + int(15 * scale)  # 文字右侧 15px
    bar_y = y_text - text_h  # 与文字顶部对齐
    bar_h = text_h  # 高度与文字一致
    bar_w = int(w * 0.25)  # 宽度占画布 25%

    if bar_x + bar_w < w - margin:
        # 生成水平渐变: 1.0(Red) -> 0.0(Blue)
        gradient = np.linspace(1, 0, bar_w)
        bar_color = matplotlib.cm.jet(gradient)[:, :3]  # [W, 3]
        bar_color = (bar_color * 255).astype(np.uint8)

        # 扩展高度: [W, 3] -> [1, W, 3] -> [H, W, 3]
        bar_img = np.tile(bar_color[None, :, :], (bar_h, 1, 1))

        # 贴图
        heatmap_u8[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = bar_img

        # 边框
        cv2.rectangle(heatmap_u8, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), outline_color, 1)

        # 3. 绘制末端 "0" 标签
        label_0 = "0"
        cv2.putText(heatmap_u8, label_0, (bar_x + bar_w + 5, y_text), font, scale, outline_color, thickness + 2,
                    cv2.LINE_AA)
        cv2.putText(heatmap_u8, label_0, (bar_x + bar_w + 5, y_text), font, scale, text_color, thickness, cv2.LINE_AA)

    return torch.from_numpy(heatmap_u8.astype(np.float32) / 255.0).to(opt_rgb.device).unsqueeze(0)


def process_mesh_uvs(base_mesh, multi_materials=False, use_custom_uv=False):
    """
    处理 UV 和 材质分组逻辑。
    """
    # 标准转换
    v_pos = base_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = base_mesh.t_pos_idx.detach().cpu().numpy()

    # 使用原始法线
    v_nrm_orig = None
    if base_mesh.v_nrm is not None:
        v_nrm_orig = base_mesh.v_nrm.detach().cpu().numpy()
    else:
        # 如果原始模型本身没法线，先在拓扑完整的原始模型上算一遍平滑法线
        print("      [Mesh] Computing auto_normals on original topology...")
        temp_mesh = mesh.auto_normals(base_mesh)
        v_nrm_orig = temp_mesh.v_nrm.detach().cpu().numpy()

    has_uv = base_mesh.v_tex is not None and base_mesh.t_tex_idx is not None

    mat_indices = np.zeros(t_pos_idx.shape[0], dtype=np.int32)
    if base_mesh.face_material_idx is not None:
        mat_indices = base_mesh.face_material_idx.detach().cpu().numpy()

    # 模式 A: 使用原有 UV
    if use_custom_uv:
        if not has_uv:
            raise ValueError("Requested --use_custom_uv but the mesh has no UVs!")
        print(f"      [UV] Mode: Using Original UVs (Skip xatlas)")

        # 即使不重分UV，也要确保切线空间正确
        if base_mesh.v_tng is None:
            base_mesh = mesh.compute_tangents(base_mesh)

        if multi_materials:
            unique_mats = np.unique(mat_indices)
            return base_mesh, unique_mats
        else:
            base_mesh.face_material_idx = torch.zeros_like(base_mesh.face_material_idx)
            return base_mesh, [0]

    # 模式 B: xatlas 自动展开

    # 内部辅助函数：构建新 Mesh 并映射法线
    def build_new_mesh(v_pos_in, t_pos_idx_in, v_nrm_in, vmapping, indices, uvs, materials_list=None,
                       face_mat_idx=None):
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

        # 1. 映射顶点位置
        new_v_pos = torch.from_numpy(v_pos_in[vmapping]).to(device='cuda', dtype=torch.float32)

        # 2. 映射法线
        # vmapping[i] 表示新顶点 i 对应原始模型的哪个顶点。
        # 我们直接把原始模型的平滑法线拿过来，这样即使 UV 接缝处顶点断开了，法线依然是连续的。
        new_v_nrm = torch.from_numpy(v_nrm_in[vmapping]).to(device='cuda', dtype=torch.float32)

        # 3. 构建 Mesh
        new_mesh_obj = mesh.Mesh(
            v_pos=new_v_pos,
            t_pos_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            v_nrm=new_v_nrm,  # 使用映射过来的正确法线
            t_nrm_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),  # 法线索引与位置对齐
            v_tex=torch.from_numpy(uvs).to(device='cuda', dtype=torch.float32),
            t_tex_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            material=None,
            materials=materials_list,
            face_material_idx=face_mat_idx
        )

        # 4. 只计算切线 (切线依赖 UV，必须重算，但基础法线是正确的)
        new_mesh_obj = mesh.compute_tangents(new_mesh_obj)
        return new_mesh_obj

    if not multi_materials:
        print(f"      [UV] Mode: Single Material Atlas (Running xatlas...)")
        start_x = time.time()
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)
        print(
            f"           xatlas finished in {time.time() - start_x:.2f}s. Tris: {t_pos_idx.shape[0]} -> {indices.shape[0]}")

        # 这里的 v_nrm_orig 就是最上面准备好的正确法线
        return build_new_mesh(v_pos, t_pos_idx, v_nrm_orig, vmapping, indices, uvs), [0]

    else:
        unique_mats = np.unique(mat_indices)
        print(f"      [UV] Mode: Multi-Material ({len(unique_mats)} materials) (Running xatlas...)")

        final_v, final_n, final_f, final_uv, final_mid = [], [], [], [], []
        global_v_offset = 0
        active_mat_ids = []

        start_x = time.time()
        for m_id in unique_mats:
            mask = (mat_indices == m_id)
            sub_faces_global = t_pos_idx[mask]

            # 提取子网格
            used_v, sub_faces_local = np.unique(sub_faces_global, return_inverse=True)
            sub_faces_local = sub_faces_local.reshape(-1, 3)
            sub_v_pos = v_pos[used_v]
            sub_v_nrm = v_nrm_orig[used_v]  # 提取对应的正确法线

            vmapping, indices, uvs = xatlas.parametrize(sub_v_pos, sub_faces_local)

            # 映射属性
            final_v.append(sub_v_pos[vmapping])
            final_n.append(sub_v_nrm[vmapping])  # 映射法线
            final_uv.append(uvs)
            final_f.append(indices + global_v_offset)
            final_mid.append(np.full(indices.shape[0], m_id, dtype=np.int64))

            global_v_offset += vmapping.shape[0]
            active_mat_ids.append(m_id)
        print(f"           xatlas finished in {time.time() - start_x:.2f}s")

        indices_int64 = np.concatenate(final_f, axis=0).astype(np.uint64, casting='same_kind').view(np.int64)

        new_mesh = mesh.Mesh(
            v_pos=torch.from_numpy(np.concatenate(final_v, axis=0)).to(device='cuda', dtype=torch.float32),
            t_pos_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            v_nrm=torch.from_numpy(np.concatenate(final_n, axis=0)).to(device='cuda', dtype=torch.float32),
            t_nrm_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            v_tex=torch.from_numpy(np.concatenate(final_uv, axis=0)).to(device='cuda', dtype=torch.float32),
            t_tex_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            face_material_idx=torch.from_numpy(np.concatenate(final_mid, axis=0)).to(device='cuda', dtype=torch.int64),
            material=None, materials=[]
        )
        new_mesh = mesh.compute_tangents(new_mesh)
        return new_mesh, active_mat_ids


# ----------------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Texture Baking Tool')

    # 基础参数
    parser.add_argument('--config', type=str, default=None, help='Config JSON file')
    parser.add_argument('-rm', '--ref_mesh', type=str, default=None, help='High-poly reference')
    parser.add_argument('-bm', '--base_mesh', type=str, default=None, help='Low-poly target')
    parser.add_argument('-o', '--out-dir', type=str, default='out/baking_result')

    # 烘焙选项
    parser.add_argument('--multi_materials', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_custom_uv', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--texture_res', nargs=2, type=int, default=[4096, 4096])
    parser.add_argument('--use_opt_pbr', type=_str2bool, nargs='?', const=True, default=False,
                        help='Use PBR shading for the optimized mesh (Default: Unlit/KD only)')
    parser.add_argument('--bake_mode', type=str, default='full_pbr', choices=['color_only', 'full_pbr'],
                        help='color_only: Optimize Kd only; full_pbr: Optimize Kd, Ks, Normal')

    # 训练参数
    parser.add_argument('--iter', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--train_res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('--spp', type=int, default=2)
    parser.add_argument('--loss_type', type=str, default='logl1')
    parser.add_argument('--smooth_weight', type=float, default=0.02)
    parser.add_argument('--coarse_to_fine', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--amp', type=_str2bool, nargs='?', const=True, default=True)

    # 环境与相机
    parser.add_argument('--envmap', type=str, default=None)
    parser.add_argument('--env_scale', type=float, default=1.0)
    parser.add_argument('--cam_radius_scale', type=float, default=2.0)
    parser.add_argument('--cam_near_far', nargs=2, type=float, default=[0.1, 1000.0])
    parser.add_argument('--background_rgb', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument('--cam_views_top', type=int, default=64, help='Number of views for top hemisphere')
    parser.add_argument('--cam_views_bottom', type=int, default=16, help='Number of views for bottom hemisphere')

    # 显示与导出
    parser.add_argument('--display_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--max_display_width', type=int, default=1600)

    # 是否渲染 HDR 环境背景
    parser.add_argument('--render_env_bg', type=_str2bool, nargs='?', const=True, default=False,
                        help='Render HDR environment map as background in visualization')

    FLAGS = parser.parse_args()

    if FLAGS.config is not None:
        if not os.path.exists(FLAGS.config):
            raise FileNotFoundError(f"Config file not found: {FLAGS.config}")
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            if hasattr(FLAGS, key):
                FLAGS.__dict__[key] = data[key]

    if FLAGS.ref_mesh is None: raise ValueError("Reference mesh required.")
    if FLAGS.base_mesh is None: raise ValueError("Base mesh required.")

    # 逻辑判断：如果开了 use_opt_pbr 或者 bake_mode 是 full_pbr，通常都得用 pbr shader
    target_bsdf = 'pbr' if (FLAGS.use_opt_pbr or FLAGS.bake_mode == 'full_pbr') else 'kd'

    print(f"\n=== Texture Baking Config ===")
    print(f" Ref: {FLAGS.ref_mesh}")
    print(f" Base: {FLAGS.base_mesh}")
    print(f" Output: {FLAGS.out_dir}")
    print(f" Loss: {FLAGS.loss_type} (Smooth: {FLAGS.smooth_weight})")
    print(f" Multi-Mat: {FLAGS.multi_materials}")
    print(f" Custom UV: {FLAGS.use_custom_uv}")
    print(
        f" Bake Mode: {FLAGS.bake_mode.upper()} (Optimizing: {'Kd' if FLAGS.bake_mode == 'color_only' else 'Kd, Ks, Normal'})")
    print(f" Texture Res: {FLAGS.texture_res}")
    print(f" Background: {FLAGS.background_rgb}")
    print(f" Coarse-to-Fine: {FLAGS.coarse_to_fine}")
    print(f" AMP: {FLAGS.amp}")
    print(f"=============================\n")

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()
    bg_tensor = torch.tensor(FLAGS.background_rgb, dtype=torch.float32, device='cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=FLAGS.amp)

    # 1. Load & Process Meshes
    print(f"[1/5] Processing Meshes...")
    ref_mesh = mesh.load_mesh(FLAGS.ref_mesh)
    ref_mesh = mesh.auto_normals(ref_mesh)
    ref_mesh = mesh.compute_tangents(ref_mesh)

    temp_base = mesh.load_mesh(FLAGS.base_mesh)
    base_mesh, active_mat_ids = process_mesh_uvs(temp_base, FLAGS.multi_materials, FLAGS.use_custom_uv)

    with torch.no_grad():
        ref_min, ref_max = mesh.aabb(ref_mesh)
        ref_center = (ref_min + ref_max) * 0.5
        ref_scale = torch.linalg.norm(ref_max - ref_min)

        base_min, base_max = mesh.aabb(base_mesh)
        base_center = (base_min + base_max) * 0.5
        base_scale = torch.linalg.norm(base_max - base_min)

        s = ref_scale / (base_scale + 1e-8)
        base_mesh.v_pos = (base_mesh.v_pos - base_center) * s + ref_center
        print(f"      Aligned scale factor: {s:.4f}")

    # 2. Setup Lighting
    if FLAGS.envmap:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
    else:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=1.0)
    with torch.no_grad():
        lgt.build_mips()

    # 3. Pre-render Ground Truth
    print(f"[2/5] Pre-rendering Ground Truth...")
    dummy_dataset = DatasetMesh(ref_mesh, base_mesh, glctx, FLAGS=FLAGS, validate=False)
    views = dummy_dataset.precomputed_views
    target_images = []

    ref_spp = FLAGS.spp

    with torch.no_grad():
        for i, view in enumerate(views):
            if len(dummy_dataset.ref_meshes) == 1:
                out = render.render_mesh(glctx, dummy_dataset.ref_meshes[0], view['mvp'], view['campos'], lgt,
                                         FLAGS.train_res, spp=ref_spp, msaa=True)
            else:
                out = render.render_meshes(glctx, dummy_dataset.ref_meshes, view['mvp'], view['campos'], lgt,
                                           FLAGS.train_res, spp=ref_spp, msaa=True)
            # 存到cpu
            target_images.append(out['shaded'].detach().cpu())
            if (i + 1) % 10 == 0: print(f"      Rendered {i + 1}/{len(views)}")

    # 4. Setup Optimization
    print(f"[3/5] Setup Optimization...")
    geometry = DLMesh(base_mesh, FLAGS)
    train_mats, params = [], []

    # 默认值
    ks_val = [0.0, 0.5, 0.0]
    ks_init = torch.tensor(ks_val, dtype=torch.float32, device='cuda')

    # 法线默认值 (Flat Normal [0.5, 0.5, 1.0])
    normal_init = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32, device='cuda')

    for m_id in active_mat_ids:
        # Kd (Base Color) 始终优化
        kd_init = torch.full((FLAGS.texture_res[0], FLAGS.texture_res[1], 3), 0.5, dtype=torch.float32, device='cuda')

        m = material.Material({
            'name': f'baked_mat_{m_id}',
            'bsdf': target_bsdf,
            'kd': texture.Texture2D(kd_init)
        })
        params += list(m['kd'].parameters())  # 加入优化器

        # 根据 bake_mode 决定 Ks 和 Normal
        if FLAGS.bake_mode == 'full_pbr':
            # 创建可训练的 Texture2D
            m['ks'] = texture.Texture2D(
                ks_init.view(1, 1, 3).repeat(FLAGS.texture_res[0], FLAGS.texture_res[1], 1))
            m['normal'] = texture.Texture2D(
                normal_init.view(1, 1, 3).repeat(FLAGS.texture_res[0], FLAGS.texture_res[1], 1))

            # 加入优化器
            params += list(m['ks'].parameters())
            params += list(m['normal'].parameters())
        else:
            # color_only: 使用固定值 Texture2D (不加 parameters 到 params)
            m['ks'] = texture.Texture2D(ks_init)
            # Normal 设为 None，渲染器会自动使用几何法线
            m['normal'] = None

        train_mats.append(m)

    if len(train_mats) == 1 and not FLAGS.multi_materials:
        geometry.mesh.material = train_mats[0]
        geometry.mesh.materials = None
    else:
        geometry.mesh.materials = train_mats

    print(f"[Info] Exporting initial mesh state...")
    init_save_path = os.path.join(FLAGS.out_dir, "initial_mesh")
    os.makedirs(init_save_path, exist_ok=True)
    gltf.save_gltf(init_save_path, geometry.mesh, diffuse_only=True)  # 初始只存diffuse

    optimizer = torch.optim.Adam(params, lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10 ** (-x * 0.0002)))
    criterion = BakingLoss(loss_type=FLAGS.loss_type, smooth_weight=FLAGS.smooth_weight)

    # 5. Training Loop
    print(f"[4/5] Baking...")
    start_t = time.time()

    def get_batch(n, b):
        while True:
            perm = np.random.permutation(n)
            for i in range(0, n, b):
                yield perm[i:i + b]

    idx_gen = get_batch(len(views), FLAGS.batch)

    for it in range(FLAGS.iter + 1):
        optimizer.zero_grad()
        idxs = next(idx_gen)

        curr_res = FLAGS.train_res
        curr_spp = FLAGS.spp
        if FLAGS.coarse_to_fine:
            if it < int(FLAGS.iter * 0.3):
                curr_res = [r // 4 for r in FLAGS.train_res]
                curr_spp = 1
            elif it < int(FLAGS.iter * 0.6):
                curr_res = [r // 2 for r in FLAGS.train_res]
                curr_spp = max(1, FLAGS.spp // 2)

        mvp = torch.cat([views[i]['mvp'] for i in idxs])
        campos = torch.cat([views[i]['campos'] for i in idxs])
        # 临时搬运回 GPU
        target = torch.cat([target_images[i].to('cuda', non_blocking=True) for i in idxs])

        if target.shape[1] != curr_res[0] or target.shape[2] != curr_res[1]:
            target_nchw = target.permute(0, 3, 1, 2)
            target_down = torch.nn.functional.interpolate(target_nchw, size=curr_res, mode='bilinear',
                                                          align_corners=False)
            target = target_down.permute(0, 2, 3, 1)

        with torch.cuda.amp.autocast(enabled=FLAGS.amp):
            buffers = render.render_mesh(glctx, geometry.mesh, mvp, campos, lgt, curr_res, spp=curr_spp, msaa=True,
                                         bsdf=target_bsdf)
            loss, stats = criterion(buffers['shaded'], target, buffers.get('kd_grad', None))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            for m in train_mats:
                m['kd'].clamp_()
                if FLAGS.bake_mode == 'full_pbr':
                    m['ks'].clamp_()
                    # normal map 不需要 clamp，因为 texture.py 里有专门的处理或者它本身是无限范围
                    # 但为了安全通常不需要 clamp (0,1)

        if it % 50 == 0:
            print(
                f"      Iter {it:04d} [{curr_res[0]}x{curr_res[1]}] | Loss: {loss.item():.6f} (Main: {stats['main']:.6f}, Reg: {stats['reg']:.6f})")

        do_save = (FLAGS.save_interval and it % FLAGS.save_interval == 0)
        do_display = (FLAGS.display_interval and it % FLAGS.display_interval == 0)

        if do_save or do_display:
            with torch.no_grad():
                idx = 0
                opt_v = composite_background(buffers['shaded'][idx:idx + 1], bg_tensor)
                ref_v = composite_background(target[idx:idx + 1], bg_tensor)

                # 动态决定背景内容
                if FLAGS.render_env_bg and FLAGS.envmap:
                    # 使用当前视角的参数渲染环境背景
                    view_id = idxs[idx]  # 获取当前样本对应的全局视角ID
                    view_data = views[view_id]
                    bg_img = render_env_background(lgt, curr_res, view_data['mv'], view_data['fovy'])
                else:
                    # 使用纯色背景
                    bg_img = bg_tensor.view(1, 1, 1, 3)

                # 合成
                vis_opt = composite_background(opt_v, bg_img)
                vis_ref = composite_background(ref_v, bg_img)

                # Reinhard 色调映射，解决过曝问题
                vis_opt = vis_opt / (1.0 + vis_opt)
                vis_ref = vis_ref / (1.0 + vis_ref)

                # 转为 sRGB
                opt_v_srgb = util.rgb_to_srgb(vis_opt)
                ref_v_srgb = util.rgb_to_srgb(vis_ref)

                mask_v = target[idx:idx + 1, ..., 3:4]
                # 热力图使用 Linear 空间数据计算
                diff_v = generate_heatmap(buffers['shaded'][idx:idx + 1, ..., 0:3], target[idx:idx + 1, ..., 0:3],
                                          mask_v, bg_tensor)

                vis = torch.cat([opt_v_srgb, ref_v_srgb, diff_v], dim=2)

                if do_save:
                    for i, m in enumerate(train_mats):
                        suffix = f"_mat{i}" if len(train_mats) > 1 else ""
                        filename = os.path.join(FLAGS.out_dir, f"progress_kd{suffix}_{it:04d}.png")
                        texture.save_texture2D(filename, texture.rgb_to_srgb(m['kd']))
                        # 只在 full_pbr 模式下保存 ks 和 normal
                        if FLAGS.bake_mode == 'full_pbr':
                            texture.save_texture2D(os.path.join(FLAGS.out_dir, f"progress_ks{suffix}_{it:04d}.png"),
                                                   m['ks'])
                            texture.save_texture2D(os.path.join(FLAGS.out_dir, f"progress_nrm{suffix}_{it:04d}.png"),
                                                   m['normal'])

                    comp_path = os.path.join(FLAGS.out_dir, f"progress_render_{it:04d}.png")
                    util.save_image(comp_path, vis[0].detach().cpu().numpy())

                if do_display:
                    max_w = FLAGS.max_display_width
                    curr_w = vis.shape[2]
                    if curr_w > max_w:
                        scale = max_w / curr_w
                        new_h = int(vis.shape[1] * scale)
                        vis_resized = util.scale_img_nhwc(vis, [new_h, max_w])
                        img_to_show = vis_resized[0].detach().cpu().numpy()
                    else:
                        img_to_show = vis[0].detach().cpu().numpy()
                    util.display_image(img_to_show, title=f"Iter {it}")

    # 6. Final Export
    print(f"\n[5/5] Exporting...")
    total_psnr = 0.0
    count = 0
    final_mesh = geometry.getMesh(None)

    # 重新组装材质给final_mesh
    if len(train_mats) == 1 and not FLAGS.multi_materials:
        final_mesh.material = train_mats[0]
        final_mesh.materials = None
    else:
        final_mesh.materials = train_mats

    with torch.no_grad():
        for i, view in enumerate(views):
            buffers = render.render_mesh(glctx, final_mesh, view['mvp'], view['campos'], lgt, FLAGS.train_res,
                                         spp=FLAGS.spp, msaa=True, bsdf=target_bsdf)
            opt_rgb = torch.clamp(buffers['shaded'][..., 0:3], 0.0, 1.0)

            # 临时从CPU取回ground truth图片到GPU
            target_gpu = target_images[i].to('cuda')

            ref_rgb = torch.clamp(target_gpu[..., 0:3], 0.0, 1.0)
            mask = target_gpu[..., 3:4] > 0.0

            valid = torch.sum(mask) * 3.0
            if valid < 1.0: continue
            mse = torch.sum(((opt_rgb - ref_rgb) * mask) ** 2) / valid
            psnr = -10.0 * torch.log10(mse + 1e-8)
            total_psnr += psnr.item()
            count += 1

    avg_psnr = total_psnr / count if count > 0 else 0.0
    print(f"      Final Average PSNR: {avg_psnr:.2f} dB")

    save_path = os.path.join(FLAGS.out_dir, "baked_mesh")

    # 动态判断是否只导出 Diffuse (避免 color_only 模式下导出无用贴图)
    is_diffuse_only = (FLAGS.bake_mode == 'color_only')

    if len(train_mats) > 1:
        gltf.save_gltf_multi(save_path, final_mesh, diffuse_only=is_diffuse_only)
    else:
        gltf.save_gltf(save_path, final_mesh, diffuse_only=is_diffuse_only)

    with torch.no_grad():
        # 最后一张对比图
        last_idx = len(views) - 1
        opt_img = buffers['shaded'][0:1]  # 复用上面循环最后一次渲染结果 (正好是最后一张)
        # 取回 GPU
        ref_img = target_images[last_idx].to('cuda')

        # 最终导出也支持环境背景
        if FLAGS.render_env_bg and FLAGS.envmap:
            view_data = views[last_idx]
            bg_img = render_env_background(lgt, FLAGS.train_res, view_data['mv'], view_data['fovy'])
        else:
            bg_img = bg_tensor.view(1, 1, 1, 3)

        vis_opt = composite_background(opt_img, bg_img)
        vis_ref = composite_background(ref_img, bg_img)

        # Reinhard 色调映射
        vis_opt = vis_opt / (1.0 + vis_opt)
        vis_ref = vis_ref / (1.0 + vis_ref)

        # 转 sRGB
        vis_opt_srgb = util.rgb_to_srgb(vis_opt)
        vis_ref_srgb = util.rgb_to_srgb(vis_ref)

        vis_diff = generate_heatmap(opt_img[..., 0:3], ref_img[..., 0:3], ref_img[..., 3:4], bg_tensor)

        final_comp = torch.cat([vis_opt_srgb, vis_ref_srgb, vis_diff], dim=2)
        util.save_image(os.path.join(FLAGS.out_dir, "final_comparison.png"), final_comp[0].detach().cpu().numpy())

    print(f"Saved to {save_path}")
    print(f"Total time: {time.time() - start_t:.2f}s")
    print("[Done]")


if __name__ == "__main__":
    main()
