import os
import io
import json
import numpy as np
import torch
import imageio.v2 as imageio  # 使用 v2 接口读取内存流
from pygltflib import GLTF2

from . import mesh
from . import material
from . import texture
from . import util


# ==============================================================================================
#  Load glTF / GLB (支持嵌入式纹理)
# ==============================================================================================

@torch.no_grad()
def load_gltf(filename, mtl_override=None, merge_materials=False):
    """
    加载 .gltf 或 .glb 文件。
    支持读取外部 .bin/.png/.jpg 或 GLB 内部嵌入的二进制数据。
    """
    gltf_path = os.path.dirname(filename)

    # 1. 使用 pygltflib 加载
    try:
        doc = GLTF2.load(filename)
    except Exception as e:
        raise RuntimeError(f"Failed to load GLTF/GLB file {filename}: {e}")

    gltf = json.loads(doc.to_json())

    # 2. 提取二进制 Buffers (修复 Method/Bytes 问题)
    buffers = []
    for i, buf in enumerate(doc.buffers):
        # 情况 A: GLB 内部嵌入的二进制块
        if buf.uri is None:
            blob = doc.binary_blob
            # 防御性编程：某些版本 blob 可能是一个方法
            if callable(blob): blob = blob()
            buffers.append(blob if blob is not None else b'')

        # 情况 B: 外部 .bin 文件
        elif buf.uri is not None:
            try:
                # 跳过 data: 协议的 buffer (极其罕见)
                if buf.uri.startswith("data:"):
                    buffers.append(b'')
                else:
                    bin_path = os.path.join(gltf_path, buf.uri)
                    with open(bin_path, 'rb') as bf:
                        buffers.append(bf.read())
            except Exception as e:
                print(f"Warning: Failed to load buffer uri {buf.uri}: {e}")
                buffers.append(b'')
        else:
            buffers.append(b'')

    # 3. 辅助函数：读取几何数据 (Accessor)
    def read_accessor(acc_idx):
        if acc_idx is None or acc_idx < 0: return None
        if acc_idx >= len(gltf['accessors']): return None

        acc = gltf['accessors'][acc_idx]
        if 'bufferView' not in acc: return None

        buf_view = gltf['bufferViews'][acc['bufferView']]
        buffer_idx = buf_view.get('buffer', 0)

        if buffer_idx >= len(buffers): return None
        data = buffers[buffer_idx]

        if not isinstance(data, (bytes, bytearray)): return None

        base_offset = buf_view.get('byteOffset', 0)
        acc_offset = acc.get('byteOffset', 0)
        total_offset = base_offset + acc_offset

        count = acc['count']
        comp_type = acc['componentType']
        type_str = acc['type']

        num_comp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}.get(type_str, 1)

        if comp_type == 5126:
            dtype = np.float32;
            comp_size = 4
        elif comp_type == 5123:
            dtype = np.uint16;
            comp_size = 2
        elif comp_type == 5125:
            dtype = np.uint32;
            comp_size = 4
        elif comp_type == 5121:
            dtype = np.uint8;
            comp_size = 1
        elif comp_type == 5122:
            dtype = np.int16;
            comp_size = 2
        else:
            return None

        expected_bytes = count * num_comp * comp_size
        if total_offset + expected_bytes > len(data):
            return None

        byte_stride = buf_view.get('byteStride', None)
        elem_size = num_comp * comp_size

        if byte_stride is None or byte_stride == elem_size:
            arr = np.frombuffer(data, dtype=dtype, count=count * num_comp, offset=total_offset)
            return arr.reshape((count, num_comp))
        else:
            out = np.empty((count, num_comp), dtype=dtype)
            for i in range(count):
                start = total_offset + i * byte_stride
                out[i] = np.frombuffer(data, dtype=dtype, count=num_comp, offset=start)
            return out

    # 4. 从 Image Index 加载纹理 (支持 BufferView)
    def load_texture_from_img_idx(img_idx, channels=None, lambda_fn=None):
        if img_idx is None or img_idx < 0 or img_idx >= len(gltf['images']):
            return None

        img_def = gltf['images'][img_idx]

        # 路径 A: 外部文件 (URI)
        uri = img_def.get('uri', None)
        if uri:
            if not uri.startswith('data:'):
                img_path = os.path.join(gltf_path, uri)
                if os.path.exists(img_path):
                    return texture.load_texture2D(img_path, lambda_fn=lambda_fn, channels=channels)

        # 路径 B: 内部嵌入数据 (BufferView) -> 这就是 GLB 缺少的逻辑！
        if 'bufferView' in img_def:
            bv_idx = img_def['bufferView']
            if 0 <= bv_idx < len(gltf['bufferViews']):
                bv = gltf['bufferViews'][bv_idx]
                buf_idx = bv.get('buffer', 0)
                if 0 <= buf_idx < len(buffers):
                    blob = buffers[buf_idx]
                    offset = bv.get('byteOffset', 0)
                    length = bv.get('byteLength', 0)
                    # 从大二进制块中切出图片的字节流
                    img_bytes = blob[offset: offset + length]

                    try:
                        # 使用 imageio 直接从内存解码 (像打开文件一样)
                        img_np = imageio.imread(io.BytesIO(img_bytes))

                        # 归一化到 [0, 1] float32
                        if img_np.dtype == np.uint8:
                            img_np = img_np.astype(np.float32) / 255.0
                        elif img_np.dtype == np.uint16:
                            img_np = img_np.astype(np.float32) / 65535.0

                        # 处理通道 (RGB/RGBA)
                        if channels is not None:
                            img_np = img_np[..., :channels]

                        # 转 Tensor
                        img_tensor = torch.tensor(img_np, dtype=torch.float32, device='cuda')

                        # 后处理 (如 Normal Map 需要 *2 -1)
                        if lambda_fn is not None:
                            img_tensor = lambda_fn(img_tensor)

                        return texture.Texture2D(img_tensor)
                    except Exception as e:
                        print(f"Warning: Failed to decode embedded image {img_idx}: {e}")
                        return None

        return None

    # 5. 解析材质
    gltf_textures = gltf.get('textures', [])
    all_materials = []

    for mat_def in gltf.get('materials', []):
        name = mat_def.get('name', f'mat_{len(all_materials)}')
        m = material.Material({'name': name})
        m['bsdf'] = 'pbr'

        # 默认值
        m['kd'] = texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda'))
        # 默认 Roughness=0.5, Metallic=0.0 (避免变成黑镜子)
        m['ks'] = texture.Texture2D(torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32, device='cuda'))

        pbr = mat_def.get('pbrMetallicRoughness', {})

        # --- Base Color ---
        bc_factor = pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])
        bc_factor_t = torch.tensor(bc_factor[:3], dtype=torch.float32, device='cuda')

        if 'baseColorTexture' in pbr:
            tex_idx = pbr['baseColorTexture'].get('index', -1)
            if 0 <= tex_idx < len(gltf_textures):
                img_idx = gltf_textures[tex_idx].get('source', None)
                # 调用我们的超级加载器
                kd_tex = load_texture_from_img_idx(img_idx)
                if kd_tex is not None:
                    kd_tex = texture.srgb_to_rgb(kd_tex)  # sRGB -> Linear
                    # 简单乘因子 (仅支持 mip0)
                    try:
                        base = kd_tex.getMips()[0] * bc_factor_t.view(1, 1, 1, 3)
                        m['kd'] = texture.Texture2D(base)
                    except:
                        m['kd'] = kd_tex
                else:
                    m['kd'] = texture.Texture2D(bc_factor_t)
        else:
            m['kd'] = texture.Texture2D(bc_factor_t)

        # --- Metallic Roughness ---
        met_factor = pbr.get('metallicFactor', 1.0)
        rgh_factor = pbr.get('roughnessFactor', 1.0)

        if 'metallicRoughnessTexture' in pbr:
            tex_idx = pbr['metallicRoughnessTexture'].get('index', -1)
            if 0 <= tex_idx < len(gltf_textures):
                img_idx = gltf_textures[tex_idx].get('source', None)
                # G=Roughness, B=Metallic
                mr_tex = load_texture_from_img_idx(img_idx, channels=3)
                if mr_tex is not None:
                    # 必须处理数据！
                    # 1. 强制清零 R 通道 (AO)，防止模型变黑
                    # 2. 应用因子
                    try:
                        base = mr_tex.getMips()[0]
                        # 创建一个全0的R通道
                        zero_r = torch.zeros_like(base[..., 0:1])
                        # 提取 G (Roughness) 和 B (Metallic)
                        g_rough = base[..., 1:2] * rgh_factor
                        b_metal = base[..., 2:3] * met_factor

                        # 重新组合为 [0, Roughness, Metallic]
                        new_ks = torch.cat([zero_r, g_rough, b_metal], dim=-1)
                        m['ks'] = texture.Texture2D(new_ks)
                    except:
                        # Fallback
                        m['ks'] = mr_tex

        # --- Normal Map ---
        if 'normalTexture' in mat_def:
            tex_idx = mat_def['normalTexture'].get('index', -1)
            if 0 <= tex_idx < len(gltf_textures):
                img_idx = gltf_textures[tex_idx].get('source', None)
                n_tex = load_texture_from_img_idx(img_idx, channels=3, lambda_fn=lambda x: x * 2.0 - 1.0)
                if n_tex is not None:
                    m['normal'] = n_tex

        all_materials.append(m)

    if not all_materials:
        default_mat = material.Material({'name': 'default'})
        default_mat['bsdf'] = 'pbr'
        default_mat['kd'] = texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda'))
        default_mat['ks'] = texture.Texture2D(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda'))
        all_materials.append(default_mat)

    # 6. 解析网格 (保持之前逻辑不变)
    pos_list = []
    nrm_list = []
    tan_list = []
    uv_list = []
    idx_list = []
    mat_idx_list = []

    global_v_offset = 0

    for mesh_def in gltf.get('meshes', []):
        for prim in mesh_def.get('primitives', []):
            mat_idx = prim.get('material', 0)
            if mat_idx >= len(all_materials): mat_idx = 0

            attrs = prim.get('attributes', {})
            pos = read_accessor(attrs.get('POSITION'))
            nrm = read_accessor(attrs.get('NORMAL'))
            tan = read_accessor(attrs.get('TANGENT'))
            uv0 = read_accessor(attrs.get('TEXCOORD_0'))
            indices = read_accessor(prim.get('indices'))

            if pos is None: continue

            v_pos = torch.tensor(pos, dtype=torch.float32, device='cuda')
            v_nrm = torch.tensor(nrm, dtype=torch.float32, device='cuda') if nrm is not None else None
            v_tan = torch.tensor(tan[:, :3], dtype=torch.float32, device='cuda') if tan is not None else None
            v_uv = torch.tensor(uv0, dtype=torch.float32, device='cuda') if uv0 is not None else None

            if indices is not None:
                idx = torch.tensor(indices.astype(np.int64).flatten(), dtype=torch.int64, device='cuda').reshape(-1, 3)
            else:
                count = v_pos.shape[0]
                idx = torch.arange(count, dtype=torch.int64, device='cuda').reshape(-1, 3)

            idx += global_v_offset

            pos_list.append(v_pos)
            if v_nrm is not None: nrm_list.append(v_nrm)
            if v_tan is not None: tan_list.append(v_tan)
            if v_uv is not None: uv_list.append(v_uv)
            idx_list.append(idx)

            num_faces = idx.shape[0]
            mat_idx_list.append(torch.full((num_faces,), mat_idx, dtype=torch.int64, device='cuda'))

            global_v_offset += v_pos.shape[0]

    if not pos_list:
        raise RuntimeError("No valid geometry found in glTF")

    V = torch.cat(pos_list, dim=0)
    F = torch.cat(idx_list, dim=0)
    N = torch.cat(nrm_list, dim=0) if len(nrm_list) == len(pos_list) else None
    T = torch.cat(tan_list, dim=0) if len(tan_list) == len(pos_list) else None
    UV = torch.cat(uv_list, dim=0) if len(uv_list) == len(pos_list) else None
    MF = torch.cat(mat_idx_list, dim=0)

    if len(uv_list) > 0 and len(uv_list) != len(pos_list):
        print("Warning: Some primitives are missing UVs. UVs will be discarded.")
        UV = None

    mesh_obj = mesh.Mesh(
        v_pos=V, t_pos_idx=F,
        v_nrm=N, t_nrm_idx=F if N is not None else None,
        v_tex=UV, t_tex_idx=F if UV is not None else None,
        v_tng=T, t_tng_idx=F if T is not None else None,
        material=all_materials[0],
        materials=all_materials,
        face_material_idx=MF
    )

    return mesh_obj


# ==============================================================================================
#  Save glTF Functions
# ==============================================================================================

@torch.no_grad()
def save_gltf(folder, mesh_obj, diffuse_only=False):
    return _save_gltf_impl(folder, mesh_obj, diffuse_only, multi=False)


@torch.no_grad()
def save_gltf_multi(folder, mesh_obj, diffuse_only=False):
    return _save_gltf_impl(folder, mesh_obj, diffuse_only, multi=True)


@torch.no_grad()
def _save_gltf_impl(folder, mesh_obj, diffuse_only, multi):
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(os.path.normpath(folder))
    bin_name = base_name + '.bin'
    gltf_name = base_name + '.gltf'
    bin_filename = os.path.join(folder, bin_name)
    gltf_filename = os.path.join(folder, gltf_name)

    V = mesh_obj.v_pos.detach().cpu().numpy().astype(np.float32)
    UV = mesh_obj.v_tex.detach().cpu().numpy().astype(np.float32) if mesh_obj.v_tex is not None else None
    F = mesh_obj.t_pos_idx.detach().cpu().numpy()

    buffer_data = bytearray()
    buffer_views = []
    accessors = []

    def add_buffer_view(data_bytes, target):
        view_idx = len(buffer_views)
        offset = len(buffer_data)
        buffer_data.extend(data_bytes)
        while len(buffer_data) % 4 != 0: buffer_data.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes), "target": target})
        return view_idx

    def add_accessor(view_idx, comp_type, count, type_str, min_val=None, max_val=None):
        acc = {"bufferView": view_idx, "byteOffset": 0, "componentType": comp_type, "count": count, "type": type_str}
        if min_val is not None: acc["min"] = min_val
        if max_val is not None: acc["max"] = max_val
        accessors.append(acc)
        return len(accessors) - 1

    pos_view = add_buffer_view(V.tobytes(), 34962)
    pos_idx = add_accessor(pos_view, 5126, V.shape[0], "VEC3", V.min(axis=0).tolist(), V.max(axis=0).tolist())

    uv_idx = None
    if UV is not None:
        uv_view = add_buffer_view(UV.tobytes(), 34962)
        uv_idx = add_accessor(uv_view, 5126, UV.shape[0], "VEC2")

    images = []
    textures = []
    materials = []
    primitives = []

    mats_to_export = mesh_obj.materials if multi and hasattr(mesh_obj, 'materials') and mesh_obj.materials else [
        mesh_obj.material]

    for i, mat in enumerate(mats_to_export):
        # 1. Base Color (Albeddo)  -必需
        kd_name = f"mat_{i}_baseColor.png" if multi else "baseColor.png"
        kd_path = os.path.join(folder, kd_name)
        kd_map = mat['kd']
        texture.save_texture2D(kd_path, texture.rgb_to_srgb(kd_map))

        images.append({"uri": kd_name})
        textures.append({"sampler": 0, "source": len(images)-1})
        kd_tex_idx = len(textures) - 1

        # 2. Metallic Roughness -可选
        # 需要 Linear (不要转 sRGB)
        mr_tex_idx = None
        if not diffuse_only and 'ks' in mat and mat['ks'] is not None:
            ks_name = f"mat_{i}_metallicRoughness.png" if multi else "metallicRoughness.png"
            ks_path = os.path.join(folder, ks_name)
            texture.save_texture2D(ks_path, mat['ks'])  # Ks 存储的是 [R=Occ, G=Rough, B=Metal]

            images.append({"uri": ks_name})
            textures.append({"sampler": 0, "source": len(images) - 1})
            mr_tex_idx = len(textures) - 1

        # 3. Normal Map - 可选
        # 需要 Linear
        nrm_tex_idx = None
        if not diffuse_only and 'normal' in mat and mat['normal'] is not None:
            nrm_name = f"mat_{i}_normal.png" if multi else "normal.png"
            nrm_path = os.path.join(folder, nrm_name)
            texture.save_texture2D(nrm_path, mat['normal'])

            images.append({"uri": nrm_name})
            textures.append({"sampler": 0, "source": len(images) - 1})
            nrm_tex_idx = len(textures) - 1

        # 构建 glTF 材质定义
        pbr_def = {
            "baseColorTexture": {"index": kd_tex_idx},
            "metallicFactor": 1.0,  # 贴图控制，因子设为1
            "roughnessFactor": 1.0
        }

        # 如果有 Ks 贴图，添加引用
        if mr_tex_idx is not None:
            pbr_def["metallicRoughnessTexture"] = {"index": mr_tex_idx}

        mat_def = {
            "name": getattr(mat, 'name', f"mat_{i}"),
            "pbrMetallicRoughness": pbr_def,
            "doubleSided": True
        }

        # 如果有 Normal 贴图，添加引用
        if nrm_tex_idx is not None:
            mat_def["normalTexture"] = {"index": nrm_tex_idx}

        materials.append(mat_def)

    if multi and mesh_obj.face_material_idx is not None:
        MF = mesh_obj.face_material_idx.detach().cpu().numpy()
        unique_mats = np.unique(MF)
        for m_id in unique_mats:
            mask = (MF == m_id)
            sub_faces = F[mask].flatten()
            if sub_faces.size == 0: continue

            if sub_faces.max() < 65536:
                ind_bytes = sub_faces.astype(np.uint16).tobytes();
                comp = 5123
            else:
                ind_bytes = sub_faces.astype(np.uint32).tobytes();
                comp = 5125
            ind_idx = add_accessor(add_buffer_view(ind_bytes, 34963), comp, sub_faces.shape[0], "SCALAR")

            primitives.append({
                "attributes": {"POSITION": pos_idx, "TEXCOORD_0": uv_idx} if uv_idx is not None else {
                    "POSITION": pos_idx},
                "indices": ind_idx,
                "material": int(m_id),
                "mode": 4
            })
    else:
        indices_flat = F.flatten()
        if indices_flat.max() < 65536:
            ind_bytes = indices_flat.astype(np.uint16).tobytes();
            comp = 5123
        else:
            ind_bytes = indices_flat.astype(np.uint32).tobytes();
            comp = 5125
        ind_idx = add_accessor(add_buffer_view(ind_bytes, 34963), comp, indices_flat.shape[0], "SCALAR")
        primitives.append({
            "attributes": {"POSITION": pos_idx, "TEXCOORD_0": uv_idx} if uv_idx is not None else {"POSITION": pos_idx},
            "indices": ind_idx,
            "material": 0,
            "mode": 4
        })

    gltf_json = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": bin_name, "byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": images,
        "textures": textures,
        "materials": materials,
        "meshes": [{"name": "baked_mesh", "primitives": primitives}],
        "nodes": [{"name": "root", "mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0
    }

    with open(gltf_filename, 'w') as f:
        json.dump(gltf_json, f, indent=2)
    with open(bin_filename, 'wb') as f:
        f.write(buffer_data)
