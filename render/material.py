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
from . import util


######################################################################################
# Wrapper to make materials behave like a python dict, but register textures as
# torch.nn.Module parameters.
######################################################################################
class Material(torch.nn.Module):
    def __init__(self, mat_dict):
        super(Material, self).__init__()
        self.mat_keys = set()
        for key in mat_dict.keys():
            self.mat_keys.add(key)
            self[key] = mat_dict[key]

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        self.mat_keys.add(key)
        setattr(self, key, val)

    def __delitem__(self, key):
        self.mat_keys.remove(key)
        delattr(self, key)

    def keys(self):
        return self.mat_keys


######################################################################################
# .mtl material format loading / storing
######################################################################################
@torch.no_grad()
def load_mtl(fn, clear_ks=True):
    import re
    mtl_path = os.path.dirname(fn)

    # Read file
    with open(fn, 'r') as f:
        lines = f.readlines()

    # Parse materials
    materials = []
    current_mat = None

    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]

        if 'newmtl' in prefix:
            current_mat = Material({'name': data[0]})
            materials.append(current_mat)
        elif current_mat is not None:
            if 'bsdf' in prefix or 'map_kd' in prefix or 'map_ks' in prefix or 'bump' in prefix:
                current_mat[prefix] = data[0]
            else:
                # Handle standard mtl properties like Kd, Ks, etc.
                try:
                    current_mat[prefix] = torch.tensor(tuple(float(d) for d in data), dtype=torch.float32,
                                                       device='cuda')
                except:
                    pass

    # Post-process: Convert paths to Texture objects
    for mat in materials:
        if 'bsdf' not in mat:
            mat['bsdf'] = 'pbr'

        if 'map_kd' in mat:
            mat['kd'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_kd']))
        else:
            # Fallback color if map not present but Kd is
            val = mat['kd'] if 'kd' in mat else torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')
            mat['kd'] = texture.Texture2D(val)

        if 'map_ks' in mat:
            mat['ks'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_ks']), channels=3)
        else:
            val = mat['ks'] if 'ks' in mat else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
            mat['ks'] = texture.Texture2D(val)

        if 'bump' in mat:
            mat['normal'] = texture.load_texture2D(os.path.join(mtl_path, mat['bump']), lambda_fn=lambda x: x * 2 - 1,
                                                   channels=3)

        # Convert Kd from sRGB to linear RGB for rendering
        mat['kd'] = texture.srgb_to_rgb(mat['kd'])

        if clear_ks:
            # Override ORM occlusion (red) channel by zeros.
            for mip in mat['ks'].getMips():
                mip[..., 0] = 0.0

    return materials


@torch.no_grad()
def save_mtl(fn, material):
    folder = os.path.dirname(fn)
    with open(fn, "w") as f:
        f.write('newmtl defaultMat\n')
        if material is not None:
            f.write('bsdf   %s\n' % material['bsdf'])
            if 'kd' in material.keys():
                f.write('map_Kd texture_kd.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_kd.png'), texture.rgb_to_srgb(material['kd']))
            if 'ks' in material.keys():
                f.write('map_Ks texture_ks.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_ks.png'), material['ks'])
            if 'normal' in material.keys():
                f.write('bump texture_n.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_n.png'), material['normal'],
                                       lambda_fn=lambda x: (util.safe_normalize(x) + 1) * 0.5)
        else:
            f.write('Kd 1 1 1\n')
            f.write('Ks 0 0 0\n')
            f.write('Ka 0 0 0\n')
            f.write('Tf 1 1 1\n')
            f.write('Ni 1\n')
            f.write('Ns 0\n')