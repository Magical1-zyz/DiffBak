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
def composite_background(image_rgba, bg_color):
    """合成背景色用于可视化"""
    rgb = image_rgba[..., 0:3]
    alpha = image_rgba[..., 3:4]
    bg = bg_color.view(1, 1, 1, 3).to(image_rgba.device)
    return rgb * alpha + bg * (1.0 - alpha)


@torch.no_grad()
def generate_heatmap(opt_rgb, ref_rgb, mask, bg_color_tensor):
    """
    生成带标签的红蓝热力图，并与统一背景合成 (标签在底部)
    """
    # 1. 计算误差
    diff = torch.abs(opt_rgb - ref_rgb) * mask

    # 转 Numpy (注意：这里取 [0] 是假设 batch=1 或只取第一个)
    diff_gray = torch.mean(diff, dim=-1)[0].detach().cpu().numpy()
    mask_np = mask[0, ..., 0].detach().cpu().numpy()
    bg_np = bg_color_tensor.detach().cpu().numpy()

    # 2. 归一化
    max_err = np.max(diff_gray)
    if max_err < 1e-6: max_err = 1.0
    norm_diff = np.clip(diff_gray / max_err, 0.0, 1.0)

    # 3. 颜色映射 (Jet: Blue->Red)
    heatmap_rgba = matplotlib.cm.jet(norm_diff)
    heatmap_rgb = heatmap_rgba[..., :3]

    # 4. 背景合成
    heatmap_vis = heatmap_rgb * mask_np[..., None] + bg_np[None, None, :] * (1.0 - mask_np[..., None])

    # 5. 绘制标签 (使用 OpenCV)
    heatmap_u8 = (np.clip(heatmap_vis, 0, 1) * 255).astype(np.uint8)
    heatmap_u8 = np.ascontiguousarray(heatmap_u8)

    # [修复点] 确保维度正确后再解包
    if heatmap_u8.ndim == 3:
        h, w, _ = heatmap_u8.shape
    else:
        # 如果意外传入了 batch 维，尝试压缩
        heatmap_u8 = heatmap_u8.squeeze()
        h, w, _ = heatmap_u8.shape

    label = f"Max Err: {max_err:.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    if np.mean(bg_np) > 0.5:
        text_color = (0, 0, 0);
        outline_color = (255, 255, 255)
    else:
        text_color = (255, 255, 255);
        outline_color = (0, 0, 0)

    scale = max(0.6, w / 800.0)
    thickness = max(1, int(scale * 2))

    margin_x = int(20 * scale)
    margin_y = int(20 * scale)
    x_pos = margin_x
    y_pos = h - margin_y

    cv2.putText(heatmap_u8, label, (x_pos, y_pos), font, scale, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(heatmap_u8, label, (x_pos, y_pos), font, scale, text_color, thickness, cv2.LINE_AA)

    return torch.from_numpy(heatmap_u8.astype(np.float32) / 255.0).to(opt_rgb.device).unsqueeze(0)


def process_mesh_uvs(base_mesh, multi_materials=False, use_custom_uv=False):
    """
    处理 UV 和 材质分组逻辑。
    """
    v_pos = base_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = base_mesh.t_pos_idx.detach().cpu().numpy()

    has_uv = base_mesh.v_tex is not None and base_mesh.t_tex_idx is not None

    mat_indices = np.zeros(t_pos_idx.shape[0], dtype=np.int32)
    if base_mesh.face_material_idx is not None:
        mat_indices = base_mesh.face_material_idx.detach().cpu().numpy()

    if use_custom_uv:
        if not has_uv:
            raise ValueError("Requested --use_custom_uv but the mesh has no UVs!")
        print(f"      [UV] Mode: Using Original UVs (Skip xatlas)")
        if multi_materials:
            unique_mats = np.unique(mat_indices)
            return base_mesh, unique_mats
        else:
            base_mesh.face_material_idx = torch.zeros_like(base_mesh.face_material_idx)
            return base_mesh, [0]

    if not multi_materials:
        print(f"      [UV] Mode: Single Material Atlas (Running xatlas...)")
        start_x = time.time()
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)
        print(f"           xatlas finished in {time.time() - start_x:.2f}s")

        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

        new_mesh = mesh.Mesh(
            v_pos=torch.from_numpy(v_pos[vmapping]).to(device='cuda', dtype=torch.float32),
            t_pos_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            v_tex=torch.from_numpy(uvs).to(device='cuda', dtype=torch.float32),
            t_tex_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            material=None
        )
        new_mesh = mesh.auto_normals(new_mesh)
        new_mesh = mesh.compute_tangents(new_mesh)
        return new_mesh, [0]
    else:
        unique_mats = np.unique(mat_indices)
        print(f"      [UV] Mode: Multi-Material ({len(unique_mats)} materials) (Running xatlas...)")

        final_v, final_f, final_uv, final_mid = [], [], [], []
        global_v_offset = 0
        active_mat_ids = []

        start_x = time.time()
        for m_id in unique_mats:
            mask = (mat_indices == m_id)
            sub_faces_global = t_pos_idx[mask]

            used_v, sub_faces_local = np.unique(sub_faces_global, return_inverse=True)
            sub_faces_local = sub_faces_local.reshape(-1, 3)
            sub_v_pos = v_pos[used_v]

            vmapping, indices, uvs = xatlas.parametrize(sub_v_pos, sub_faces_local)

            remapped_v_pos = sub_v_pos[vmapping]
            final_v.append(remapped_v_pos)
            final_uv.append(uvs)
            final_f.append(indices + global_v_offset)
            final_mid.append(np.full(indices.shape[0], m_id, dtype=np.int64))

            global_v_offset += remapped_v_pos.shape[0]
            active_mat_ids.append(m_id)
        print(f"           xatlas finished in {time.time() - start_x:.2f}s")

        indices_int64 = np.concatenate(final_f, axis=0).astype(np.uint64, casting='same_kind').view(np.int64)

        new_mesh = mesh.Mesh(
            v_pos=torch.from_numpy(np.concatenate(final_v, axis=0)).to(device='cuda', dtype=torch.float32),
            t_pos_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            v_tex=torch.from_numpy(np.concatenate(final_uv, axis=0)).to(device='cuda', dtype=torch.float32),
            t_tex_idx=torch.from_numpy(indices_int64).to(device='cuda', dtype=torch.int64),
            face_material_idx=torch.from_numpy(np.concatenate(final_mid, axis=0)).to(device='cuda', dtype=torch.int64),
            material=None, materials=[]
        )
        new_mesh = mesh.auto_normals(new_mesh)
        new_mesh = mesh.compute_tangents(new_mesh)
        return new_mesh, active_mat_ids


# ----------------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Texture Baking Tool')

    parser.add_argument('--config', type=str, default=None, help='Config JSON file')
    parser.add_argument('-rm', '--ref_mesh', type=str, default=None, help='High-poly reference')
    parser.add_argument('-bm', '--base_mesh', type=str, default=None, help='Low-poly target')
    parser.add_argument('-o', '--out-dir', type=str, default='out/baking_result')

    parser.add_argument('--multi_materials', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_custom_uv', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--texture_res', nargs=2, type=int, default=[4096, 4096])

    parser.add_argument('--iter', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--train_res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('--spp', type=int, default=2)

    parser.add_argument('--loss_type', type=str, default='logl1')
    parser.add_argument('--smooth_weight', type=float, default=0.02)

    parser.add_argument('--envmap', type=str, default=None)
    parser.add_argument('--env_scale', type=float, default=1.0)
    parser.add_argument('--cam_radius_scale', type=float, default=2.0)
    parser.add_argument('--cam_near_far', nargs=2, type=float, default=[0.1, 1000.0])
    parser.add_argument('--background_rgb', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument('--display_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--max_display_width', type=int, default=1600)

    parser.add_argument('--coarse_to_fine', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--amp', type=_str2bool, nargs='?', const=True, default=True)

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

    print(f"\n=== Texture Baking Config ===")
    print(f" Ref: {FLAGS.ref_mesh}")
    print(f" Base: {FLAGS.base_mesh}")
    print(f" Output: {FLAGS.out_dir}")
    print(f" Loss: {FLAGS.loss_type} (Smooth: {FLAGS.smooth_weight})")
    print(f" Multi-Mat: {FLAGS.multi_materials}")
    print(f" Custom UV: {FLAGS.use_custom_uv}")
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
            target_images.append(out['shaded'].detach())
            if (i + 1) % 10 == 0: print(f"      Rendered {i + 1}/{len(views)}")

    # 4. Setup Optimization
    print(f"[3/5] Setup Optimization...")
    geometry = DLMesh(base_mesh, FLAGS)
    train_mats, params = [], []
    ks_init = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda')

    for m_id in active_mat_ids:
        kd_init = torch.full((FLAGS.texture_res[0], FLAGS.texture_res[1], 3), 0.5, dtype=torch.float32, device='cuda')
        m = material.Material({
            'name': f'baked_mat_{m_id}',
            'bsdf': 'kd',
            'kd': texture.Texture2D(kd_init),
            'ks': texture.Texture2D(ks_init)
        })
        train_mats.append(m)
        params += list(m['kd'].parameters())

    if len(train_mats) == 1 and not FLAGS.multi_materials:
        geometry.mesh.material = train_mats[0]
        geometry.mesh.materials = None
    else:
        geometry.mesh.materials = train_mats

    print(f"[Info] Exporting initial mesh state...")
    init_save_path = os.path.join(FLAGS.out_dir, "initial_mesh")
    os.makedirs(init_save_path, exist_ok=True)
    if len(train_mats) > 1:
        gltf.save_gltf_multi(init_save_path, geometry.mesh, diffuse_only=True)
    else:
        gltf.save_gltf(init_save_path, geometry.mesh, diffuse_only=True)

    optimizer = torch.optim.Adam(params, lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10 ** (-x * 0.0002)))
    criterion = BakingLoss(loss_type=FLAGS.loss_type, smooth_weight=FLAGS.smooth_weight)

    # 5. Training Loop
    print(f"[4/5] Baking...")
    start_t = time.time()

    def get_batch(n, b):
        while True:
            perm = np.random.permutation(n)
            for i in range(0, n, b): yield perm[i:i + b]

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
        target = torch.cat([target_images[i] for i in idxs])

        if target.shape[1] != curr_res[0] or target.shape[2] != curr_res[1]:
            target_nchw = target.permute(0, 3, 1, 2)
            target_down = torch.nn.functional.interpolate(target_nchw, size=curr_res, mode='bilinear',
                                                          align_corners=False)
            target = target_down.permute(0, 2, 3, 1)

        with torch.cuda.amp.autocast(enabled=FLAGS.amp):
            buffers = render.render_mesh(glctx, geometry.mesh, mvp, campos, lgt, curr_res, spp=curr_spp, msaa=True,
                                         bsdf='kd')
            loss, stats = criterion(buffers['shaded'], target, buffers.get('kd_grad', None))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            for m in train_mats: m['kd'].clamp_()

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
                mask_v = target[idx:idx + 1, ..., 3:4]
                diff_v = generate_heatmap(buffers['shaded'][idx:idx + 1, ..., 0:3], target[idx:idx + 1, ..., 0:3],
                                          mask_v, bg_tensor)

                vis = torch.cat([opt_v, ref_v, diff_v], dim=2)

                if do_save:
                    for i, m in enumerate(train_mats):
                        suffix = f"_mat{i}" if len(train_mats) > 1 else ""
                        filename = os.path.join(FLAGS.out_dir, f"progress_kd{suffix}_{it:04d}.png")
                        texture.save_texture2D(filename, texture.rgb_to_srgb(m['kd']))

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
    if len(train_mats) == 1 and not FLAGS.multi_materials:
        final_mesh.material = train_mats[0]
        final_mesh.materials = None
    else:
        final_mesh.materials = train_mats

    with torch.no_grad():
        for i, view in enumerate(views):
            buffers = render.render_mesh(glctx, final_mesh, view['mvp'], view['campos'], lgt, FLAGS.train_res,
                                         spp=FLAGS.spp, msaa=True, bsdf='kd')
            opt_rgb = torch.clamp(buffers['shaded'][..., 0:3], 0.0, 1.0)
            ref_rgb = torch.clamp(target_images[i][..., 0:3], 0.0, 1.0)
            mask = target_images[i][..., 3:4] > 0.0

            valid = torch.sum(mask) * 3.0
            if valid < 1.0: continue
            mse = torch.sum(((opt_rgb - ref_rgb) * mask) ** 2) / valid
            psnr = -10.0 * torch.log10(mse + 1e-8)
            total_psnr += psnr.item()
            count += 1

    avg_psnr = total_psnr / count if count > 0 else 0.0
    print(f"      Final Average PSNR: {avg_psnr:.2f} dB")

    save_path = os.path.join(FLAGS.out_dir, "baked_mesh")
    if len(train_mats) > 1:
        gltf.save_gltf_multi(save_path, final_mesh, diffuse_only=True)
    else:
        gltf.save_gltf(save_path, final_mesh, diffuse_only=True)

    # 修复后的最后一张对比图保存逻辑
    with torch.no_grad():
        # 取出最后一张 Ground Truth (已是 4D 张量 [1, H, W, 4])
        ref_img_last = target_images[len(views) - 1]

        vis_opt = composite_background(buffers['shaded'][0:1], bg_tensor)
        vis_ref = composite_background(ref_img_last, bg_tensor)

        vis_diff = generate_heatmap(
            buffers['shaded'][..., 0:3],
            ref_img_last[..., 0:3],
            ref_img_last[..., 3:4],
            bg_tensor
        )
        final_comp = torch.cat([vis_opt, vis_ref, vis_diff], dim=2)
        util.save_image(os.path.join(FLAGS.out_dir, "final_comparison.png"), final_comp[0].detach().cpu().numpy())

    print(f"Saved to {save_path}")
    print("[Done]")


if __name__ == "__main__":
    main()