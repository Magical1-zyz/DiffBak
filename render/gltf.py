import os
import json
import numpy as np
import torch
from pygltflib import GLTF2, Buffer, BufferView, Sampler

from . import mesh
from . import material
from . import texture
from . import util


@torch.no_grad()
def load_gltf(filename, mtl_override=None, merge_materials=False):
    # ... (保持你现有的 load_gltf 代码不变，或者使用之前的版本) ...
    # 鉴于我们需要关注的是 save 部分，这里为了节省篇幅，假设 load 部分你已经有了
    # 如果没有，可以使用之前回答中的版本
    pass


# ... [load_gltf 的代码] ...

@torch.no_grad()
def save_gltf(folder, mesh_obj, diffuse_only=False):
    """
    保存单材质 Mesh 到 glTF
    """
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(os.path.normpath(folder))  # Use folder name as base
    bin_name = base_name + '.bin'
    gltf_name = base_name + '.gltf'
    bin_filename = os.path.join(folder, bin_name)
    gltf_filename = os.path.join(folder, gltf_name)

    # 提取数据
    V = mesh_obj.v_pos.detach().cpu().numpy().astype(np.float32)
    UV = mesh_obj.v_tex.detach().cpu().numpy().astype(np.float32) if mesh_obj.v_tex is not None else None
    F = mesh_obj.t_pos_idx.detach().cpu().numpy()

    # 强制单材质：即使有 face_material_idx，也忽略，因为是 save_gltf (Single)
    # 如果需要多材质，请调用 save_gltf_multi

    # 准备 Buffer
    buffer_data = bytearray()
    buffer_views = []
    accessors = []

    def add_buffer_view(data_bytes, target):
        view_idx = len(buffer_views)
        offset = len(buffer_data)
        buffer_data.extend(data_bytes)
        # padding 4 bytes
        while len(buffer_data) % 4 != 0: buffer_data.append(0)

        buffer_views.append({
            "buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes), "target": target
        })
        return view_idx

    def add_accessor(view_idx, comp_type, count, type_str, min_val=None, max_val=None):
        acc = {
            "bufferView": view_idx, "byteOffset": 0, "componentType": comp_type,
            "count": count, "type": type_str
        }
        if min_val is not None: acc["min"] = min_val
        if max_val is not None: acc["max"] = max_val
        accessors.append(acc)
        return len(accessors) - 1

    # POSITION
    pos_idx = add_accessor(add_buffer_view(V.tobytes(), 34962), 5126, V.shape[0], "VEC3", V.min(axis=0).tolist(),
                           V.max(axis=0).tolist())

    # TEXCOORD_0
    uv_idx = None
    if UV is not None:
        uv_idx = add_accessor(add_buffer_view(UV.tobytes(), 34962), 5126, UV.shape[0], "VEC2")

    # INDICES
    indices_flat = F.flatten()
    if indices_flat.max() < 65536:
        ind_bytes = indices_flat.astype(np.uint16).tobytes()
        comp_type = 5123
    else:
        ind_bytes = indices_flat.astype(np.uint32).tobytes()
        comp_type = 5125
    ind_idx = add_accessor(add_buffer_view(ind_bytes, 34963), comp_type, indices_flat.shape[0], "SCALAR")

    # Images & Textures
    images = []
    textures = []
    materials = []

    # 处理材质
    mat = mesh_obj.material
    tex_path = os.path.join(folder, "texture_base.png")

    # 保存图片
    texture.save_texture2D(tex_path, texture.rgb_to_srgb(mat['kd']))

    images.append({"uri": "texture_base.png"})
    textures.append({"sampler": 0, "source": 0})

    materials.append({
        "name": "baked_material",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0  # Unlit feel
        },
        "doubleSided": True
    })

    # 构建 GLTF JSON
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": bin_name, "byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": images,
        "textures": textures,
        "materials": materials,
        "meshes": [{
            "name": "mesh",
            "primitives": [{
                "attributes": {"POSITION": pos_idx, "TEXCOORD_0": uv_idx},
                "indices": ind_idx,
                "material": 0,
                "mode": 4
            }]
        }],
        "nodes": [{"name": "root", "mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0
    }

    with open(gltf_filename, 'w') as f:
        json.dump(gltf, f, indent=2)
    with open(bin_filename, 'wb') as f:
        f.write(buffer_data)


@torch.no_grad()
def save_gltf_multi(folder, mesh_obj, diffuse_only=False):
    """
    保存多材质 Mesh 到 glTF。
    会将 mesh_obj 根据 face_material_idx 拆分为多个 primitives，每个引用不同的材质。
    """
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(os.path.normpath(folder))
    bin_name = base_name + '.bin'
    gltf_name = base_name + '.gltf'
    bin_filename = os.path.join(folder, bin_name)
    gltf_filename = os.path.join(folder, gltf_name)

    # 提取数据
    V = mesh_obj.v_pos.detach().cpu().numpy().astype(np.float32)
    UV = mesh_obj.v_tex.detach().cpu().numpy().astype(np.float32)
    F = mesh_obj.t_pos_idx.detach().cpu().numpy()
    MF = mesh_obj.face_material_idx.detach().cpu().numpy()

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

    # 1. 写入公共顶点属性 (所有 Primitive 共享顶点Buffer，或者根据索引拆分)
    # 为了简单，我们让所有 Primitive 共享 POSITION 和 TEXCOORD BufferView
    # 只是 INDEX Buffer 不同
    pos_view = add_buffer_view(V.tobytes(), 34962)
    pos_idx = add_accessor(pos_view, 5126, V.shape[0], "VEC3", V.min(axis=0).tolist(), V.max(axis=0).tolist())

    uv_view = add_buffer_view(UV.tobytes(), 34962)
    uv_idx = add_accessor(uv_view, 5126, UV.shape[0], "VEC2")

    # 2. 导出材质贴图
    images = []
    textures = []
    materials = []

    for i, mat in enumerate(mesh_obj.materials):
        tex_name = f"mat_{i}_diffuse.png"
        tex_path = os.path.join(folder, tex_name)
        # 保存图片
        texture.save_texture2D(tex_path, texture.rgb_to_srgb(mat['kd']))

        images.append({"uri": tex_name})
        textures.append({"sampler": 0, "source": i})
        materials.append({
            "name": getattr(mat, 'name', f"material_{i}"),
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": i},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            "doubleSided": True
        })

    # 3. 构建 Primitives
    primitives = []
    unique_mats = np.unique(MF)

    for m_id in unique_mats:
        # 提取属于该材质的面索引
        mask = (MF == m_id)
        sub_faces = F[mask].flatten()

        if sub_faces.max() < 65536:
            ind_bytes = sub_faces.astype(np.uint16).tobytes()
            comp = 5123
        else:
            ind_bytes = sub_faces.astype(np.uint32).tobytes()
            comp = 5125

        ind_view = add_buffer_view(ind_bytes, 34963)
        ind_acc = add_accessor(ind_view, comp, sub_faces.shape[0], "SCALAR")

        primitives.append({
            "attributes": {"POSITION": pos_idx, "TEXCOORD_0": uv_idx},
            "indices": ind_acc,
            "material": int(m_id),
            "mode": 4
        })

    # 4. JSON
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