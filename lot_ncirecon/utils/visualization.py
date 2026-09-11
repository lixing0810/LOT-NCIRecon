"""Training monitor visualization utilities."""

import os

import numpy as np
import torch

from .training import ensure_dir, parse_generator_outputs
def make_uint8(img):
    img = np.asarray(img, dtype=np.float32)
    img = np.squeeze(img)

    if img.min() >= -0.05 and img.max() <= 1.05:
        v = np.clip(img, 0, 1)
    else:
        v = np.clip(img, -160, 240)
        v = (v + 160) / 400.0

    return np.clip(v * 255, 0, 255).astype(np.uint8)


def save_visual_grid(path, ncct, low, pred, target):
    from PIL import Image, ImageDraw

    ncct = make_uint8(ncct)
    low = make_uint8(low)
    pred = make_uint8(pred)
    target = make_uint8(target)

    h, w = pred.shape
    gap = 8
    label_h = 22
    canvas = np.ones((h + label_h, w * 4 + gap * 3), dtype=np.uint8) * 255

    imgs = [ncct, low, pred, target]
    labels = ["NCCT", "LD-CECT", "Recon", "Target"]

    for i, im in enumerate(imgs):
        x0 = i * (w + gap)
        canvas[label_h:label_h + h, x0:x0 + w] = im

    pil = Image.fromarray(canvas).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for i, label in enumerate(labels):
        x0 = i * (w + gap) + 4
        draw.text((x0, 3), label, fill=(0, 0, 0))

    pil.save(path)


@torch.no_grad()
def save_monitor_images(generator, monitor_loader, device, out_dir, step, ema=None, num_samples=6):
    ensure_dir(out_dir)

    if ema is not None:
        ema.apply_to(generator)

    generator.eval()
    saved = 0

    for low, target, ncct, _weight in monitor_loader:
        low = low.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        ncct = ncct.to(device, non_blocking=True)

        outputs = generator(ncct, low)
        _, pred_cect, _, _ = parse_generator_outputs(outputs)
        pred_cect = pred_cect.clamp(0, 1)

        b = low.size(0)
        for i in range(b):
            if saved >= num_samples:
                break
            save_visual_grid(
                os.path.join(out_dir, f"step_{step:06d}_sample_{saved:02d}.png"),
                ncct[i, 0].detach().cpu().numpy(),
                low[i, 0].detach().cpu().numpy(),
                pred_cect[i, 0].detach().cpu().numpy(),
                target[i, 0].detach().cpu().numpy(),
            )
            saved += 1

        if saved >= num_samples:
            break

    if ema is not None:
        ema.restore(generator)

    generator.train()


# =============================================================================
# LR schedule
# =============================================================================
