import os
import time
import argparse
import json
import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# 核心模块引用
from dataset.dataset_mesh import DatasetMesh
from geometry.dlmesh import DLMesh
from render import obj, gltf, material, util, mesh, texture, light, render
from render.renderutils.loss import BakingLoss  # 引用我们刚写好的 Loss 类


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


def process_mesh_uvs(base_mesh, multi_materials=False):
    """
    处理 UV 和 材质分组逻辑 (xatlas)
    """
    v_pos = base_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = base_mesh.t_pos_idx.detach().cpu().numpy()

    mat_indices = np.zeros(t_pos_idx.shape[0], dtype=np.int32)
    if base_mesh.face_material_idx is not None:
        mat_indices = base_mesh.face_material_idx.detach().cpu().numpy()

    if not multi_materials:
        print(f"      [UV] Mode: Single Material Atlas")
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

        new_mesh = mesh.Mesh(
            v_pos=torch.tensor(v_pos[vmapping], dtype=torch.float32, device='cuda'),
            t_pos_idx=torch.tensor(indices_int64, dtype=torch.int64, device='cuda'),
            v_tex=torch.tensor(uvs, dtype=torch.float32, device='cuda'),
            t_tex_idx=torch.tensor(indices_int64, dtype=torch.int64, device='cuda'),
            material=None
        )
        return new_mesh, [0]
    else:
        unique_mats = np.unique(mat_indices)
        print(f"      [UV] Mode: Multi-Material ({len(unique_mats)} materials)")

        final_v, final_f, final_uv, final_mid = [], [], [], []
        global_v_offset = 0
        active_mat_ids = []

        for m_id in unique_mats:
            mask = (mat_indices == m_id)
            sub_faces_global = t_pos_idx[mask]

            # 提取子网格
            used_v, sub_faces_local = np.unique(sub_faces_global, return_inverse=True)
            sub_faces_local = sub_faces_local.reshape(-1, 3)
            sub_v_pos = v_pos[used_v]

            # 独立 Parameterize
            vmapping, indices, uvs = xatlas.parametrize(sub_v_pos, sub_faces_local)

            # 映射回原始位置并收集
            remapped_v_pos = sub_v_pos[vmapping]
            final_v.append(remapped_v_pos)
            final_uv.append(uvs)
            final_f.append(indices + global_v_offset)
            final_mid.append(np.full(indices.shape[0], m_id, dtype=np.int64))

            global_v_offset += remapped_v_pos.shape[0]
            active_mat_ids.append(m_id)

        indices_int64 = np.concatenate(final_f, axis=0).astype(np.uint64, casting='same_kind').view(np.int64)

        new_mesh = mesh.Mesh(
            v_pos=torch.tensor(np.concatenate(final_v, axis=0), dtype=torch.float32, device='cuda'),
            t_pos_idx=torch.tensor(indices_int64, dtype=torch.int64, device='cuda'),
            v_tex=torch.tensor(np.concatenate(final_uv, axis=0), dtype=torch.float32, device='cuda'),
            t_tex_idx=torch.tensor(indices_int64, dtype=torch.int64, device='cuda'),
            face_material_idx=torch.tensor(np.concatenate(final_mid, axis=0), dtype=torch.int64, device='cuda'),
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

    # [Input/Output]
    parser.add_argument('--config', type=str, default=None, help='Config JSON file')
    parser.add_argument('-rm', '--ref_mesh', type=str, required=True, help='High-poly reference')
    parser.add_argument('-bm', '--base_mesh', type=str, required=True, help='Low-poly target')
    parser.add_argument('-o', '--out-dir', type=str, default='out/baking_result')

    # [Baking Options]
    parser.add_argument('--multi_materials', type=_str2bool, nargs='?', const=True, default=False,
                        help='Enable multi-material baking')
    parser.add_argument('--texture_res', nargs=2, type=int, default=[4096, 4096], help='Output texture resolution')

    # [Training Params]
    parser.add_argument('--iter', type=int, default=3000, help='Total iterations')
    parser.add_argument('--batch', type=int, default=1, help='Batch size (keep low for high res)')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--train_res', nargs=2, type=int, default=[1024, 1024],
                        help='Rendering resolution during training')
    parser.add_argument('--spp', type=int, default=2, help='Samples per pixel (Anti-aliasing)')

    # [Loss Control] - 暴露给用户的参数
    parser.add_argument('--loss_type', type=str, default='logl1', choices=['l1', 'logl1', 'mse'],
                        help='Main reconstruction loss type')
    parser.add_argument('--smooth_weight', type=float, default=0.02,
                        help='Weight for texture smoothness regularization')

    # [Environment & Camera]
    parser.add_argument('--envmap', type=str, default=None, help='HDR environment map path')
    parser.add_argument('--env_scale', type=float, default=1.0)
    parser.add_argument('--cam_radius_scale', type=float, default=2.0)
    parser.add_argument('--background_rgb', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument('--display_interval', type=int, default=50)

    FLAGS = parser.parse_args()

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    print(f"\n=== Texture Baking Config ===")
    print(f" Ref: {FLAGS.ref_mesh}")
    print(f" Base: {FLAGS.base_mesh}")
    print(f" Output: {FLAGS.out_dir}")
    print(f" Loss: {FLAGS.loss_type} (Smooth: {FLAGS.smooth_weight})")
    print(f" Multi-Mat: {FLAGS.multi_materials}")
    print(f"=============================\n")

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()
    bg_tensor = torch.tensor(FLAGS.background_rgb, dtype=torch.float32, device='cuda')

    # 1. Load & Process Meshes
    print(f"[1/5] Processing Meshes...")
    ref_mesh = mesh.load_mesh(FLAGS.ref_mesh)
    temp_base = mesh.load_mesh(FLAGS.base_mesh)

    # UV Parametrization
    base_mesh, active_mat_ids = process_mesh_uvs(temp_base, FLAGS.multi_materials)

    # Auto Alignment
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

    # Render loop
    ref_spp = max(FLAGS.spp, 4)  # Higher quality for GT
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

    for m_id in active_mat_ids:
        # Gray initialization
        kd_init = torch.full((FLAGS.texture_res[0], FLAGS.texture_res[1], 3), 0.5, dtype=torch.float32, device='cuda')
        m = material.Material({
            'name': f'baked_mat_{m_id}',
            'bsdf': 'kd',  # Unlit shader
            'kd': texture.Texture2D(kd_init)
        })
        train_mats.append(m)
        params += list(m.parameters())

    if len(train_mats) == 1 and not FLAGS.multi_materials:
        geometry.mesh.material = train_mats[0]
        geometry.mesh.materials = None
    else:
        geometry.mesh.materials = train_mats

    optimizer = torch.optim.Adam(params, lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10 ** (-x * 0.0002)))

    # 使用我们封装好的 Loss Manager
    criterion = BakingLoss(loss_type=FLAGS.loss_type, smooth_weight=FLAGS.smooth_weight)

    # 5. Training Loop
    print(f"[4/5] Baking...")
    start_t = time.time()

    # Simple random batch generator
    def get_batch(n, b):
        while True:
            perm = np.random.permutation(n)
            for i in range(0, n, b): yield perm[i:i + b]

    idx_gen = get_batch(len(views), FLAGS.batch)

    for it in range(FLAGS.iter + 1):
        optimizer.zero_grad()
        idxs = next(idx_gen)

        # Batch data
        mvp = torch.cat([views[i]['mvp'] for i in idxs])
        campos = torch.cat([views[i]['campos'] for i in idxs])
        target = torch.cat([target_images[i] for i in idxs])

        # Render
        buffers = render.render_mesh(glctx, geometry.mesh, mvp, campos, lgt, FLAGS.train_res, spp=FLAGS.spp, msaa=True,
                                     bsdf='kd')

        # Loss Calculation
        loss, stats = criterion(buffers['shaded'], target, buffers.get('kd_grad', None))

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp texture
        with torch.no_grad():
            for m in train_mats: m['kd'].clamp_()

        if it % 50 == 0:
            print(f"      Iter {it:04d} | Loss: {loss.item():.6f} (Main: {stats['main']:.6f}, Reg: {stats['reg']:.6f})")

        if FLAGS.display_interval and it % FLAGS.display_interval == 0:
            with torch.no_grad():
                # Visualize first item in batch
                opt_v = composite_background(buffers['shaded'][0:1], bg_tensor)
                ref_v = composite_background(target[0:1], bg_tensor)
                diff = torch.abs(opt_v - ref_v)
                vis = torch.cat([opt_v, ref_v, diff], dim=2)
                util.display_image(vis[0].detach().cpu().numpy(), title=f"Iter {it}")

    # 6. Final Export
    print(f"\n[5/5] Exporting...")
    save_path = os.path.join(FLAGS.out_dir, "baked_mesh")

    if len(train_mats) > 1:
        gltf.save_gltf_multi(save_path, geometry.mesh, diffuse_only=True)
    else:
        gltf.save_gltf(save_path, geometry.mesh, diffuse_only=True)

    print(f"Saved to {save_path}")
    print("[Done]")


if __name__ == "__main__":
    main()