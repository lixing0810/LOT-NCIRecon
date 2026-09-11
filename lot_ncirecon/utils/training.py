"""Reproducibility, checkpoint, EMA, and optimization utilities."""

import math
import os
import random
from collections import OrderedDict

import numpy as np
import torch
def set_seed(seed=2026, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# =============================================================================
# File and data utilities
# =============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


# =============================================================================
# EMA
# =============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = OrderedDict()
        self.backup = None
        for name, value in model.state_dict().items():
            self.shadow[name] = value.detach().clone()

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for name, value in state.items():
            if name not in self.shadow:
                self.shadow[name] = value.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        self.backup = OrderedDict()
        state = model.state_dict()
        for name, value in state.items():
            self.backup[name] = value.detach().clone()
            value.copy_(self.shadow[name])

    def restore(self, model):
        if self.backup is None:
            return
        state = model.state_dict()
        for name, value in state.items():
            value.copy_(self.backup[name])
        self.backup = None

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# =============================================================================
# Model output parser
# =============================================================================
def parse_generator_outputs(outputs):
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Generator should return tuple/list outputs.")

    if len(outputs) == 4:
        n_hat, l_hat, low_hat, l_hat_2 = outputs
        return n_hat, l_hat_2, low_hat, l_hat

    if len(outputs) == 3:
        n_hat, l_hat, low_hat = outputs
        return n_hat, l_hat, low_hat, None

    if len(outputs) == 2:
        n_hat, l_hat = outputs
        return n_hat, l_hat, None, None

    raise RuntimeError(f"Unsupported generator output length: {len(outputs)}")


# =============================================================================
# Checkpoint utilities
# =============================================================================
def strip_module_prefix(state_dict):
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def save_checkpoint(path, step, generator, img_d, grad_d, g_opt, img_d_opt, grad_d_opt, scaler, ema, args):
    ckpt = {
        "step": step,
        "generator": generator.state_dict(),
        "img_discriminator": img_d.state_dict() if img_d is not None else None,
        "grad_discriminator": grad_d.state_dict() if grad_d is not None else None,
        "g_optimizer": g_opt.state_dict() if g_opt is not None else None,
        "img_d_optimizer": img_d_opt.state_dict() if img_d_opt is not None else None,
        "grad_d_optimizer": grad_d_opt.state_dict() if grad_d_opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, generator, img_d, grad_d, g_opt, img_d_opt, grad_d_opt, scaler, ema, device):
    ckpt = torch.load(path, map_location=device)

    generator.load_state_dict(strip_module_prefix(ckpt["generator"]), strict=True)

    if img_d is not None and ckpt.get("img_discriminator") is not None:
        img_d.load_state_dict(strip_module_prefix(ckpt["img_discriminator"]), strict=True)
    if grad_d is not None and ckpt.get("grad_discriminator") is not None:
        grad_d.load_state_dict(strip_module_prefix(ckpt["grad_discriminator"]), strict=True)

    if g_opt is not None and ckpt.get("g_optimizer") is not None:
        g_opt.load_state_dict(ckpt["g_optimizer"])
    if img_d_opt is not None and ckpt.get("img_d_optimizer") is not None:
        img_d_opt.load_state_dict(ckpt["img_d_optimizer"])
    if grad_d_opt is not None and ckpt.get("grad_d_optimizer") is not None:
        grad_d_opt.load_state_dict(ckpt["grad_d_optimizer"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])

    return int(ckpt.get("step", 0))


# =============================================================================
# Monitor visualization
# =============================================================================
def adjust_lr(optimizer, base_lr, step, warmup_steps, max_iter, min_lr_ratio=0.05):
    if warmup_steps > 0 and step <= warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, max_iter - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        min_lr = base_lr * min_lr_ratio
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# =============================================================================
# Build objects
