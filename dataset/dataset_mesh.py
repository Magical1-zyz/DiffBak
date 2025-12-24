import numpy as np
import torch
from render import util
from render import mesh as mesh_mod
from render import render
from render import light
from .dataset import Dataset


class DatasetMesh(Dataset):
    def __init__(self, ref_mesh, base_mesh=None, glctx=None, cam_radius=3.0, FLAGS=None, validate=False):
        self.FLAGS = FLAGS
        self.validate = validate
        self.glctx = glctx
        self.aspect = FLAGS.train_res[1] / FLAGS.train_res[0]

        # 1. 处理模型
        self.ref_mesh = mesh_mod.compute_tangents(ref_mesh)
        if base_mesh is not None:
            # 如果没有法线，计算一下
            if base_mesh.v_nrm is None:
                self.base_mesh = mesh_mod.auto_normals(base_mesh)
            else:
                self.base_mesh = base_mesh
            # 始终确保有切线
            if self.base_mesh.v_tng is None:
                self.base_mesh = mesh_mod.compute_tangents(self.base_mesh)
        else:
            self.base_mesh = self.ref_mesh

        if isinstance(ref_mesh, list):
            self.ref_meshes = ref_mesh
        else:
            self.ref_meshes = [self.ref_mesh]

        # 2. 加载环境光
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

        # 3. 计算包围盒中心和半径
        # 3. 计算包围盒 (AABB)
        # 我们需要保留 AABB 的 8 个顶点用于后续投影计算
        with torch.no_grad():
            vmins = []
            vmaxs = []
            for m in self.ref_meshes:
                vmin, vmax = mesh_mod.aabb(m)
                vmins.append(vmin)
                vmaxs.append(vmax)
            self.aabb_min = torch.min(torch.stack(vmins, dim=0), dim=0).values
            self.aabb_max = torch.max(torch.stack(vmaxs, dim=0), dim=0).values

            self.center = ((self.aabb_min + self.aabb_max) * 0.5).detach()
            bbox_extent = (self.aabb_max - self.aabb_min).detach()
            self.bound_radius = (torch.linalg.norm(bbox_extent) * 0.5).item()
            self.aabb_size = bbox_extent.cpu().numpy()

            # 生成 AABB 的 8 个角点 (用于计算紧凑 FOV)
            # x: min, max | y: min, max | z: min, max
            corners = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner = torch.stack([
                            self.aabb_min[0] if dx == 0 else self.aabb_max[0],
                            self.aabb_min[1] if dy == 0 else self.aabb_max[1],
                            self.aabb_min[2] if dz == 0 else self.aabb_max[2]
                        ])
                        corners.append(corner)
            self.aabb_corners = torch.stack(corners).to(device='cuda')  # [8, 3]

        # 4. 生成相机视角
        self.n_top = getattr(self.FLAGS, 'cam_views_top', 64)
        self.n_bottom = getattr(self.FLAGS, 'cam_views_bottom', 16)
        self.n_views = self.n_top + self.n_bottom

        self.precomputed_views = self._generate_scanning_views(self.n_top, self.n_bottom)

        print(f"DatasetMesh: Generated {self.n_views} views (Macro Scanning Mode).")

    # 生成扫描视角 (Macro Scanning Mode)
    def _generate_scanning_views(self, n_top, n_bottom):
        views = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        # 基础轨道半径
        radius_scale = getattr(self.FLAGS, 'cam_radius_scale', 2.0)
        global_dist = self.bound_radius * radius_scale

        # Near/Far (由于我们会 zoom in，near 需要更近一点)
        safe_near = max(0.01, global_dist * 0.1)
        safe_far = global_dist * 3.0

        # 定义 "视野覆盖率" (Zoom Factor)
        # 0.5 表示每一张图只保证看清物体 50% 大小的区域，从而获得 2x 的放大倍率
        # 对于长条物体，这会产生极好的特写效果
        ZOOM_COVERAGE = 0.55

        def add_view(x, y, z):
            # 1. 基础相机方向 (Fibonacci Sphere)
            cam_dir = np.array([x, y, z])
            cam_dir = cam_dir / np.linalg.norm(cam_dir)

            # 2. [关键] 计算随机 Look-At 目标点 (扫描模型)
            # 在 AABB 内部随机取一个点，作为相机的“焦点”
            # 范围收缩一点(0.8)，避免盯着极端的边角看
            jitter_range = self.aabb_size * 0.4
            random_offset = np.random.uniform(-jitter_range, jitter_range)

            # 目标点 = 中心 + 随机偏移
            target_pos_np = self.center.cpu().numpy() + random_offset
            target_pos = torch.tensor(target_pos_np, dtype=torch.float32, device='cuda')

            # 3. 计算相机位置
            # 为了安全，相机依然在“球壳”上运动，避免穿插到模型内部
            # 但它会旋转去盯着 target_pos 看
            campos = self.center + torch.tensor(cam_dir * global_dist, dtype=torch.float32, device='cuda')

            # 构建 LookAt 矩阵
            up = self._robust_up(campos - target_pos)  # up 向量基于视线方向
            mv = util.lookAt(campos, target_pos, up)

            # 4. [关键] 计算微距 FOV
            # 目标：让 target_pos 周围 radius * ZOOM_COVERAGE 大小的区域撑满屏幕
            # 这样就实现了“放大”效果

            # 计算相机到目标的实际距离
            dist_to_target = torch.linalg.norm(campos - target_pos).item()

            # 想要覆盖的局部半径
            visible_radius = self.bound_radius * ZOOM_COVERAGE

            # 简单的三角函数计算 FOV
            fovy = 2.0 * np.arctan(visible_radius / dist_to_target)

            # 限制 FOV 防止过小或过大
            fovy = float(np.clip(fovy, np.deg2rad(15.0), np.deg2rad(100.0)))

            # 投影矩阵
            proj_mtx = util.perspective(fovy, self.aspect, safe_near, safe_far, device='cuda')
            mvp = proj_mtx @ mv

            views.append({
                'mv': mv[None, ...],
                'mvp': mvp[None, ...],
                'campos': campos[None, ...],
                'fovy': fovy
            })

        # 循环生成 (保持 Fibonacci 分布)
        for i in range(n_top):
            y = 1 - (i / float(n_top - 1)) if n_top > 1 else 1.0
            radius_at_y = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            add_view(x, y, z)

        for i in range(n_bottom):
            y = - (i / float(n_bottom - 1)) if n_bottom > 1 else -1.0
            radius_at_y = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            add_view(x, y, z)

        return views

    # 生成标准视角 (Standard Mode)
    def _generate_fibonacci_views(self, n_top, n_bottom):
        views = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        # 基础距离
        radius_scale = getattr(self.FLAGS, 'cam_radius_scale', 2.0)
        dist = self.bound_radius * radius_scale

        # 动态计算 Near/Far (安全范围)
        safe_near = max(0.01, dist - self.bound_radius * 1.5)
        safe_far = dist + self.bound_radius * 1.5

        # 辅助函数
        def add_view(x, y, z):
            # 1. 计算相机位置和基础矩阵
            jitter = np.random.normal(0, 0.02, 3)
            cam_dir = np.array([x, y, z]) + jitter
            cam_dir = cam_dir / np.linalg.norm(cam_dir)

            campos = self.center + torch.tensor(cam_dir * dist, dtype=torch.float32, device='cuda')
            up = self._robust_up(campos)
            mv = util.lookAt(campos, self.center, up)

            # 2. 自适应 FOV 计算
            # 将 AABB 的 8 个角点转换到相机空间
            # mv: [4, 4], corners: [8, 3] -> 需扩充为 [8, 4]
            corners_homo = torch.cat([self.aabb_corners, torch.ones((8, 1), device='cuda')], dim=1)  # [8, 4]
            corners_cam = (mv @ corners_homo.T).T  # [8, 4]
            corners_cam = corners_cam[..., :3]  # [8, 3] (Camera Space: X right, Y up, -Z forward)

            # 计算相机空间下的包围范围 (最大 abs X 和 abs Y)
            # 此时物体中心大概在 (0, 0, -dist)
            # 我们需要看物体相对于 Z轴张开了多大角度

            # 简单估算：找到所有点中，|Y| / |Z| 的最大值，即为 tan(half_fov_y)
            # 注意：Z 坐标在相机空间通常是负数，所以取 abs
            max_tan_y = torch.max(torch.abs(corners_cam[:, 1]) / torch.abs(corners_cam[:, 2]))

            # 为了保险，加上一点 margin (例如 1.1 倍，即留 10% 空隙)
            # 允许用户通过 config 调整 margin，默认更紧凑一点 (1.05)
            fov_margin = 1.05
            fovy = 2.0 * float(np.arctan(max_tan_y.item() * fov_margin))

            # 限制 FOV 范围，防止过大或过小
            fovy = float(np.clip(fovy, np.deg2rad(5.0), np.deg2rad(120.0)))

            # 3. 投影矩阵
            proj_mtx = util.perspective(fovy, self.aspect, safe_near, safe_far, device='cuda')
            mvp = proj_mtx @ mv

            views.append({
                'mv': mv[None, ...],
                'mvp': mvp[None, ...],
                'campos': campos[None, ...],
                'fovy': fovy
            })

        # 上半球
        for i in range(n_top):
            y = 1 - (i / float(n_top - 1)) if n_top > 1 else 1.0
            radius_at_y = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            add_view(x, y, z)

        # 下半球
        for i in range(n_bottom):
            y = - (i / float(n_bottom - 1)) if n_bottom > 1 else -1.0
            radius_at_y = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            add_view(x, y, z)

        return views

    def _robust_up(self, forward):
        # forward = (self.center - eye)
        # forward 已经是 (eye - target) 或者反过来，这里我们需要它是 Z 轴方向
        # lookAt 函数期望 up 向量。我们简单选取 Y 或 Z
        forward = forward / torch.linalg.norm(forward)
        up_y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.center.device)
        up_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.center.device)

        # 如果视线太接近 Y 轴，就用 Z 轴当 Up，否则用 Y
        cross_len = torch.linalg.norm(torch.linalg.cross(up_y, forward))
        return up_y if cross_len > 1e-1 else up_z

    def __len__(self):
        return (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        if not self.validate:
            idx = np.random.randint(0, self.n_views)
        else:
            idx = itr % self.n_views

        view_data = self.precomputed_views[idx]
        mvp = view_data['mvp']
        campos = view_data['campos']
        mv = view_data['mv']
        iter_res = self.FLAGS.train_res
        iter_spp = self.FLAGS.spp

        if len(self.ref_meshes) == 1:
            ref_out = render.render_mesh(self.glctx, self.ref_meshes[0], mvp, campos, self.envlight,
                                         iter_res, spp=iter_spp, num_layers=1,
                                         msaa=True, background=None)
        else:
            ref_out = render.render_meshes(self.glctx, self.ref_meshes, mvp, campos, self.envlight,
                                           iter_res, spp=iter_spp, num_layers=1,
                                           msaa=True, background=None)

        return {
            'mv': mv,
            'mvp': mvp,
            'campos': campos,
            'resolution': iter_res,
            'spp': iter_spp,
            'img': ref_out['shaded'],
            'base_mesh': self.base_mesh,
        }

    def collate(self, batch):
        return super().collate(batch)