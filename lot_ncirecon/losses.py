"""Reconstruction and adversarial loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))



def weighted_charbonnier(pred, target, weight, eps=1e-3):
    """
    Per-sample weighted Charbonnier loss.
    weight: [B] or [B,1], representing patch-level OT reliability S.
    """
    diff = torch.sqrt((pred - target) ** 2 + eps ** 2)
    per_sample = diff.flatten(1).mean(dim=1)
    weight = weight.view(-1).to(pred.device).clamp(0.0, 1.0)
    return (per_sample * weight).sum() / (weight.sum() + 1e-6)


class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)


def lsgan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def discriminator_lsgan_loss(discriminator, real_img, fake_img, source_img=None):
    real_enc, real_dec = discriminator(real_img)
    fake_enc, fake_dec = discriminator(fake_img.detach())

    loss = (
        lsgan_loss(real_enc, True)
        + lsgan_loss(real_dec, True)
        + lsgan_loss(fake_enc, False)
        + lsgan_loss(fake_dec, False)
    )

    if source_img is not None:
        src_enc, src_dec = discriminator(source_img.detach())
        loss = loss + lsgan_loss(src_enc, False) + lsgan_loss(src_dec, False)

    return loss


def generator_lsgan_loss(discriminator, fake_img):
    fake_enc, fake_dec = discriminator(fake_img)
    return lsgan_loss(fake_enc, True) + lsgan_loss(fake_dec, True)


def ssim_loss_simple(pred, target, window_size=7):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, 1, pad)
    mu_y = F.avg_pool2d(target, window_size, 1, pad)

    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return 1.0 - ssim_map.mean()


