# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 基础 Loss 实现 (Python Native)
# ----------------------------------------------------------------------------

def log_l1_loss(opt, ref, mask, eps=1e-8):
    """
    Log-Space L1 Loss.
    对高光不敏感，能显著提升阴影和暗部的烘焙细节。
    """
    # log(x + 1) 变换
    opt_log = torch.log(torch.clamp(opt, min=0, max=65535) + 1)
    ref_log = torch.log(torch.clamp(ref, min=0, max=65535) + 1)
    diff = torch.abs(opt_log - ref_log)
    return torch.sum(diff * mask) / (torch.sum(mask) + eps)


def l1_loss(opt, ref, mask, eps=1e-8):
    """标准 L1 Loss"""
    diff = torch.abs(opt - ref)
    return torch.sum(diff * mask) / (torch.sum(mask) + eps)


def mse_loss(opt, ref, mask, eps=1e-8):
    """标准 MSE Loss"""
    diff = (opt - ref) ** 2
    return torch.sum(diff * mask) / (torch.sum(mask) + eps)


def smoothness_loss(kd_grad, weight=1.0):
    """
    纹理平滑损失 (Total Variation 变体)。
    kd_grad: 渲染器输出的纹理梯度图 [N, H, W, C]
    """
    if weight <= 0.0:
        return torch.tensor(0.0, device=kd_grad.device)
    return torch.mean(kd_grad) * weight


# ----------------------------------------------------------------------------
# 统一管理类 (Engineering Style)
# ----------------------------------------------------------------------------

class BakingLoss(nn.Module):
    def __init__(self, loss_type='logl1', smooth_weight=0.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.smooth_weight = smooth_weight

        # 路由表
        self.loss_fns = {
            'l1': l1_loss,
            'logl1': log_l1_loss,
            'mse': mse_loss
        }

        if self.loss_type not in self.loss_fns:
            raise ValueError(f"Unsupported loss type: {self.loss_type}. Supported: {list(self.loss_fns.keys())}")

        print(f"[Loss Config] Main: {self.loss_type.upper()} | Smoothness Weight: {self.smooth_weight}")

    def forward(self, opt_img, ref_img, texture_grad=None):
        """
        Args:
            opt_img: [N, H, W, 4] 渲染结果 (RGBA)
            ref_img: [N, H, W, 4] 真值图 (RGBA)
            texture_grad: [N, H, W, C] 纹理梯度图 (用于平滑正则化)
        Returns:
            total_loss, stats_dict
        """
        # 1. 准备 Mask 和 RGB
        # Alpha 通道作为 Mask，确保只计算物体表面
        mask = ref_img[..., 3:4]
        # 确保 RGB 在合法范围内
        opt_rgb = torch.clamp(opt_img[..., 0:3], 0.0, 65535.0)
        ref_rgb = torch.clamp(ref_img[..., 0:3], 0.0, 65535.0)

        # 2. 计算主 Loss
        main_loss_fn = self.loss_fns[self.loss_type]
        reconstruction_loss = main_loss_fn(opt_rgb, ref_rgb, mask)

        # 3. 计算正则化 Loss
        reg_loss = torch.tensor(0.0, device=opt_img.device)
        if self.smooth_weight > 0.0 and texture_grad is not None:
            reg_loss = smoothness_loss(texture_grad, self.smooth_weight)

        # 4. 总 Loss
        total_loss = reconstruction_loss + reg_loss

        stats = {
            'main': reconstruction_loss.item(),
            'reg': reg_loss.item()
        }

        return total_loss, stats