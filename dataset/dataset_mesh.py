import numpy as np
import torch
from render import util
from render import mesh as mesh_mod
from render import render
from render import light
from .dataset import Dataset


class DatasetMesh(Dataset):
    def __init__(self, ref_mesh, base_mesh=None, glctx=None, cam_radius=3.0, FLAGS=None, validate=False):
        # ... (保留原有的初始化代码) ...
        self.ref_mesh = mesh_mod.compute_tangents(ref_mesh)
        self.base_mesh = mesh_mod.compute_tangents(base_mesh) if base_mesh is not None else self.ref_mesh
        self.glctx = glctx
        self.FLAGS = FLAGS
        self.validate = validate
        self.aspect = FLAGS.train_res[1] / FLAGS.train_res[0]

        # 统一处理 mesh 列表
        if isinstance(ref_mesh, list):
            self.ref_meshes = ref_mesh
        else:
            self.ref_meshes = [self.ref_mesh]

        # 加载环境光
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

        # --- 核心修改：计算包围盒中心和半径 ---
        with torch.no_grad():
            vmins = []
            vmaxs = []
            for m in self.ref_meshes:
                vmin, vmax = mesh_mod.aabb(m)
                vmins.append(vmin)
                vmaxs.append(vmax)
            vmin = torch.min(torch.stack(vmins, dim=0), dim=0).values
            vmax = torch.max(torch.stack(vmaxs, dim=0), dim=0).values
            self.center = ((vmin + vmax) * 0.5).detach()
            bbox_extent = (vmax - vmin).detach()
            # 包围球半径
            self.bound_radius = (torch.linalg.norm(bbox_extent) * 0.5).item()

        # --- 核心修改：生成64个斐波那契半球分布的相机位姿 ---
        self.n_views = 64  # 固定64个点位
        self.precomputed_views = self._generate_fibonacci_views(self.n_views)

        print(f"DatasetMesh: Generated {self.n_views} optimized baking views.")

    def _generate_fibonacci_views(self, n_points):
        """
        生成均匀分布在半球面的相机位姿列表
        """
        views = []

        # 黄金角度
        golden_angle = np.pi * (3 - np.sqrt(5))

        # 为了更清晰的烘焙，相机需要离物体稍远一点，避免透视畸变过大
        # 使用 FLAGS 中的 cam_radius_scale，默认建议 2.0 左右
        radius_scale = getattr(self.FLAGS, 'cam_radius_scale', 2.0)
        dist = self.bound_radius * radius_scale

        for i in range(n_points):
            # 1. 计算斐波那契点位 (y 从 1 到 0，覆盖上半球)
            # y 坐标分布：从顶部(1)到底部(0，即赤道)
            y = 1 - (i / float(n_points - 1))
            radius_at_y = np.sqrt(1 - y * y)
            theta = golden_angle * i

            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y

            # 这是一个单位球上的方向向量 (x, y, z)
            # 注意：nvdiffrec通常假设Y轴向上。如果你的模型底面朝下，y >= 0 就是上半球。
            # 如果模型朝向不同，可能需要交换坐标轴，例如 z >= 0

            # 加入微小的随机扰动 (Jitter)，防止完全死板的网格，增加烘焙鲁棒性
            jitter = np.random.normal(0, 0.02, 3)
            cam_dir = np.array([x, y, z]) + jitter
            cam_dir = cam_dir / np.linalg.norm(cam_dir)  # 重新归一化

            # 2. 计算相机位置
            # 相机位置 = 物体中心 + 方向向量 * 距离
            campos = self.center + torch.tensor(cam_dir * dist, dtype=torch.float32, device='cuda')

            # 3. 计算 LookAt 矩阵 (看向物体中心)
            up = self._robust_up(campos)
            mv = util.lookAt(campos, self.center, up)

            # 4. 自适应 FOV 计算
            # 确保物体在画面内占比合适 (fov_margin控制留白，1.1表示留10%边距)
            fov_margin = 1.1
            # 简单估算：tan(fov/2) = radius / dist
            # 考虑到 margin 和 aspect ratio
            half_fov = np.arctan((self.bound_radius * fov_margin) / dist)
            fovy = 2 * half_fov

            # 防止FOV过大或过小
            fovy = float(np.clip(fovy, np.deg2rad(10.0), np.deg2rad(90.0)))

            # 构建投影矩阵
            proj_mtx = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1],
                                        device='cuda')
            mvp = proj_mtx @ mv

            views.append({
                'mv': mv[None, ...],
                'mvp': mvp[None, ...],
                'campos': campos[None, ...],
                'fovy': fovy  # 保存FOV以备查
            })

        return views

    def _robust_up(self, eye):
        # 计算一个稳健的 Up 向量，防止相机在头顶时翻转
        forward = (self.center - eye)
        forward = forward / torch.linalg.norm(forward)
        # 默认 Y 轴向上
        up_y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.center.device)
        up_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.center.device)

        # 如果视线与Y轴过于平行（例如在正上方），改用Z轴作为参考Up
        cross_len = torch.linalg.norm(torch.linalg.cross(up_y, forward))
        up = up_y if cross_len > 1e-1 else up_z
        return up

    def __len__(self):
        # 这里的长度决定了一个 epoch 跑多少次
        # 如果是烘焙模式，我们希望多跑几轮优化，不仅仅是64次
        # 所以这里的长度应该基于 FLAGS.iter
        return (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # --- 核心修改：从预计算的64个视角中循环或随机采样 ---

        # 方案A：如果是 Validation，按顺序展示
        # 方案B：如果是 Training (Baking)，随机从64个最佳视角中选，确保每个batch都在优化有效区域

        idx = itr % self.n_views  # 简单循环，或者用 random.randint(0, self.n_views-1)

        # 为了更好的SGD优化效果，训练时建议随机打乱顺序，而不是固定循环
        if not self.validate:
            idx = np.random.randint(0, self.n_views)

        view_data = self.precomputed_views[idx]

        mv = view_data['mv']
        mvp = view_data['mvp']
        campos = view_data['campos']
        iter_res = self.FLAGS.train_res
        iter_spp = self.FLAGS.spp

        # 渲染 Reference (多材质高模)
        if len(self.ref_meshes) == 1:
            ref_out = render.render_mesh(self.glctx, self.ref_meshes[0], mvp, campos, self.envlight,
                                         iter_res, spp=iter_spp, num_layers=self.FLAGS.layers,
                                         msaa=True, background=None)
        else:
            ref_out = render.render_meshes(self.glctx, self.ref_meshes, mvp, campos, self.envlight,
                                           iter_res, spp=iter_spp, num_layers=self.FLAGS.layers,
                                           msaa=True, background=None)

        # 注意：Baking模式下，target['img'] 应该是 Reference 渲染出的图像
        return {
            'mv': mv,
            'mvp': mvp,
            'campos': campos,
            'resolution': iter_res,
            'spp': iter_spp,
            'img': ref_out['shaded'],  # 这是Ground Truth
            'base_mesh': self.base_mesh,
        }