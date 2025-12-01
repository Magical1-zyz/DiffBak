# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import torch

from . import texture
from . import mesh
from . import material


######################################################################################
# Create mesh object from objfile
######################################################################################

def load_obj(filename, clear_ks=True, mtl_override=None):
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load materials
    all_materials = []
    if mtl_override is not None:
        all_materials = material.load_mtl(mtl_override, clear_ks)
    else:
        # Scan for mtllib
        for line in lines:
            if len(line.split()) == 0: continue
            if line.split()[0] == 'mtllib':
                mtl_path = os.path.join(obj_path, line.split()[1])
                if os.path.exists(mtl_path):
                    all_materials += material.load_mtl(mtl_path, clear_ks)

    # If no materials loaded, create a default one
    if not all_materials:
        all_materials = [
            material.Material({
                'name': 'default',
                'bsdf': 'pbr',
                'kd': texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
                'ks': texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
            })
        ]

    # Map material names to indices
    mat_map = {m['name']: i for i, m in enumerate(all_materials)}

    # Load geometry
    vertices, texcoords, normals = [], [], []
    faces, tfaces, nfaces = [], [], []
    face_mat_indices = []

    current_mat_idx = 0

    for line in lines:
        if len(line.split()) == 0: continue
        prefix = line.split()[0].lower()

        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])
        elif prefix == 'usemtl':
            mat_name = line.split()[1]
            if mat_name in mat_map:
                current_mat_idx = mat_map[mat_name]
        elif prefix == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1

            for i in range(nv - 2):  # Triangulate
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1

                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
                face_mat_indices.append(current_mat_idx)

    # Convert to tensors
    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')

    # Handle optional attributes
    if len(texcoords) > 0 and len(tfaces) > 0:
        texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda')
        tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda')
    else:
        texcoords = None
        tfaces = None

    if len(normals) > 0 and len(nfaces) > 0:
        normals = torch.tensor(normals, dtype=torch.float32, device='cuda')
        nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda')
    else:
        normals = None
        nfaces = None

    if len(face_mat_indices) > 0:
        face_mat_idx = torch.tensor(face_mat_indices, dtype=torch.int64, device='cuda')
    else:
        face_mat_idx = None

    # Construct Mesh with multi-material support
    # Note: materials=all_materials list is passed, not a single material
    return mesh.Mesh(
        v_pos=vertices, t_pos_idx=faces,
        v_nrm=normals, t_nrm_idx=nfaces,
        v_tex=texcoords, t_tex_idx=tfaces,
        material=all_materials[0],  # Default single material access
        materials=all_materials,  # Multi-material list
        face_material_idx=face_mat_idx
    )


######################################################################################
# Save mesh object to objfile
######################################################################################

def write_obj(folder, mesh, save_material=True):
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))

        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))

        # Simple material handling for export
        # If it's a baked mesh (single material), we just use defaultMat
        # If it's a multi-material mesh, we technically should write usemtl per face,
        # but for this specific 'baking result export', we usually just want the single baked material.
        f.write("usemtl defaultMat\n")

        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                v_idx = str(t_pos_idx[i][j] + 1)
                vt_idx = str(t_tex_idx[i][j] + 1) if t_tex_idx is not None else ''
                vn_idx = str(t_nrm_idx[i][j] + 1) if t_nrm_idx is not None else ''
                f.write(' %s/%s/%s' % (v_idx, vt_idx, vn_idx))
            f.write("\n")

    if save_material and mesh.material is not None:
        mtl_file = os.path.join(folder, 'mesh.mtl')
        print("Writing material: ", mtl_file)
        material.save_mtl(mtl_file, mesh.material)

    print("Done exporting mesh")