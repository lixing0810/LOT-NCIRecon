"""Independent per-phase reconstruction training."""

import csv
import os

import torch
from torch.utils.data import DataLoader

from ..data import OTTripletPatchDataset, _default_manifest_path, list_npy_files
from ..losses import (
    CharbonnierLoss, SobelGrad, discriminator_lsgan_loss, generator_lsgan_loss,
    ssim_loss_simple, weighted_charbonnier,
 )
from ..models.discriminator import UNet, turn_on_spectral_norm
from ..models.reconstruction import LOTNCIRecon
from ..utils.training import (
    ModelEMA, adjust_lr, ensure_dir, load_checkpoint, parse_generator_outputs,
    save_checkpoint, set_requires_grad, set_seed,
 )
from ..utils.visualization import save_monitor_images
def build_dataloader(args):
    # For very large patch datasets, build/read a manifest instead of recursively
    # globbing and sorting every time the training script starts.
    train_manifest = args.train_manifest
    if train_manifest is None:
        train_manifest = _default_manifest_path(args.train_data_dir, args.output_root, tag="train")

    train_files = list_npy_files(
        args.train_data_dir,
        recursive=True,
        manifest_path=train_manifest,
        rebuild_manifest=args.rebuild_manifest,
        max_files=args.max_train_files,
        sort_files=args.sort_files,
    )

    train_set = OTTripletPatchDataset(
        root=args.train_data_dir,
        crop_size=args.crop_size,
        recursive=True,
        random_flip=args.random_flip,
        files=train_files,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    monitor_root = args.monitor_data_dir if args.monitor_data_dir else args.train_data_dir
    if args.monitor_data_dir:
        monitor_manifest = args.monitor_manifest
        if monitor_manifest is None:
            monitor_manifest = _default_manifest_path(monitor_root, args.output_root, tag="monitor")

        monitor_files = list_npy_files(
            monitor_root,
            recursive=True,
            manifest_path=monitor_manifest,
            rebuild_manifest=args.rebuild_manifest,
            max_files=args.num_monitor_files,
            sort_files=args.sort_files,
        )
    else:
        # Avoid scanning the 500k-file training directory a second time just for monitor images.
        monitor_files = train_files[:args.num_monitor_files]

    monitor_set = OTTripletPatchDataset(
        root=monitor_root,
        crop_size=args.monitor_crop_size,
        recursive=True,
        random_flip=False,
        files=monitor_files,
    )

    monitor_loader = DataLoader(
        monitor_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, monitor_loader
def build_models(args, device):
    generator = LOTNCIRecon().to(device)

    img_d = UNet(
        repeat_num=args.repeat_num,
        use_discriminator=True,
        conv_dim=args.disc_dim,
        use_sigmoid=False,
    )
    img_d = turn_on_spectral_norm(img_d).to(device)

    grad_d = UNet(
        repeat_num=args.repeat_num,
        use_discriminator=True,
        conv_dim=args.disc_dim,
        use_sigmoid=False,
    )
    grad_d = turn_on_spectral_norm(grad_d).to(device)

    return generator, img_d, grad_d


# =============================================================================
# Train
# =============================================================================
def train(args):
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = ensure_dir(args.output_root)
    ckpt_dir = ensure_dir(os.path.join(output_root, "checkpoints"))
    vis_dir = ensure_dir(os.path.join(output_root, "monitor_images"))
    log_dir = ensure_dir(os.path.join(output_root, "logs"))

    log_csv = os.path.join(log_dir, "train_log.csv")
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "g_lr", "d_lr",
            "g_total", "g_recon", "g_aux", "g_ncct", "g_low", "g_grad", "g_ssim", "g_adv_img", "g_adv_grad",
            "d_img", "d_grad",
        ])

    train_loader, monitor_loader = build_dataloader(args)
    generator, img_d, grad_d = build_models(args, device)

    g_opt = torch.optim.AdamW(
        generator.parameters(),
        lr=args.g_lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    img_d_opt = torch.optim.Adam(
        img_d.parameters(),
        lr=args.d_lr,
        betas=(0.5, 0.999),
    )

    grad_d_opt = torch.optim.Adam(
        grad_d.parameters(),
        lr=args.d_lr,
        betas=(0.5, 0.999),
    )

    charbonnier = CharbonnierLoss(eps=args.charb_eps).to(device)
    sobel = SobelGrad().to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    ema = ModelEMA(generator, decay=args.ema_decay) if args.use_ema else None

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            generator,
            img_d,
            grad_d,
            g_opt,
            img_d_opt,
            grad_d_opt,
            scaler,
            ema,
            device,
        )
        print(f"Resumed from {args.resume}, start_step={start_step}")

    data_iter = iter(train_loader)

    running = {
        "g_total": 0.0,
        "g_recon": 0.0,
        "g_aux": 0.0,
        "g_ncct": 0.0,
        "g_low": 0.0,
        "g_grad": 0.0,
        "g_ssim": 0.0,
        "g_adv_img": 0.0,
        "g_adv_grad": 0.0,
        "d_img": 0.0,
        "d_grad": 0.0,
    }

    print("=" * 100)
    print("Stable dual-domain GAN training for OT-selected unfolded reconstruction network")
    print(f"phase               : {args.phase}")
    print(f"device              : {device}")
    print(f"train_data_dir      : {args.train_data_dir}")
    print(f"monitor_data_dir    : {args.monitor_data_dir if args.monitor_data_dir else args.train_data_dir}")
    print(f"output_root         : {args.output_root}")
    print(f"max_iter/save_freq  : {args.max_iter}/{args.save_freq}")
    print(f"g_lr/d_lr           : {args.g_lr}/{args.d_lr}")
    print(f"batch_size          : {args.batch_size}")
    print("Kept: image discriminator + gradient discriminator + dual-domain GAN.")
    print("=" * 100)

    generator.train()
    img_d.train()
    grad_d.train()

    for step in range(start_step + 1, args.max_iter + 1):
        try:
            low, target, ncct, sim_weight = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            low, target, ncct, sim_weight = next(data_iter)

        low = low.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        ncct = ncct.to(device, non_blocking=True)
        sim_weight = sim_weight.to(device, non_blocking=True).view(-1)

        g_lr = adjust_lr(g_opt, args.g_lr, step, args.warmup_steps, args.max_iter, args.min_lr_ratio)
        d_lr = adjust_lr(img_d_opt, args.d_lr, step, args.warmup_steps, args.max_iter, args.min_lr_ratio)
        _ = adjust_lr(grad_d_opt, args.d_lr, step, args.warmup_steps, args.max_iter, args.min_lr_ratio)

        # ---------------------------------------------------------------------
        # Forward generator once for discriminator update.
        # ---------------------------------------------------------------------
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            with torch.no_grad():
                outputs = generator(ncct, low)
                _, pred_cect, _, _ = parse_generator_outputs(outputs)
                pred_cect_det = pred_cect.clamp(0, 1).detach()

        # ---------------------------------------------------------------------
        # Update image discriminator.
        # ---------------------------------------------------------------------
        set_requires_grad(img_d, True)
        img_d_opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            d_img_loss = discriminator_lsgan_loss(
                discriminator=img_d,
                real_img=target,
                fake_img=pred_cect_det,
                source_img=low if args.use_source_as_fake else None,
            )

        scaler.scale(args.w_d_img * d_img_loss).backward()
        if args.d_grad_clip > 0:
            scaler.unscale_(img_d_opt)
            torch.nn.utils.clip_grad_norm_(img_d.parameters(), args.d_grad_clip)
        scaler.step(img_d_opt)

        # ---------------------------------------------------------------------
        # Update gradient discriminator.
        # ---------------------------------------------------------------------
        set_requires_grad(grad_d, True)
        grad_d_opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            grad_real = sobel(target)
            grad_fake = sobel(pred_cect_det)
            grad_source = sobel(low) if args.use_source_as_fake else None

            d_grad_loss = discriminator_lsgan_loss(
                discriminator=grad_d,
                real_img=grad_real,
                fake_img=grad_fake,
                source_img=grad_source,
            )

        scaler.scale(args.w_d_grad * d_grad_loss).backward()
        if args.d_grad_clip > 0:
            scaler.unscale_(grad_d_opt)
            torch.nn.utils.clip_grad_norm_(grad_d.parameters(), args.d_grad_clip)
        scaler.step(grad_d_opt)

        # ---------------------------------------------------------------------
        # Update generator.
        # ---------------------------------------------------------------------
        set_requires_grad(img_d, False)
        set_requires_grad(grad_d, False)

        g_opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            outputs = generator(ncct, low)
            pred_ncct, pred_cect, pred_low, aux_cect = parse_generator_outputs(outputs)

            pred_cect = pred_cect.clamp(0, 1)
            pred_ncct = pred_ncct.clamp(0, 1)
            pred_low = pred_low.clamp(0, 1) if pred_low is not None else None
            aux_cect = aux_cect.clamp(0, 1) if aux_cect is not None else None

            recon_loss = weighted_charbonnier(pred_cect, target, sim_weight, eps=args.charb_eps)

            aux_loss = torch.tensor(0.0, device=device)
            if aux_cect is not None and args.w_aux > 0:
                aux_loss = charbonnier(aux_cect, target)

            ncct_loss = torch.tensor(0.0, device=device)
            if args.w_ncct > 0:
                ncct_loss = charbonnier(pred_ncct, ncct)

            low_loss = torch.tensor(0.0, device=device)
            if pred_low is not None and args.w_low > 0:
                low_loss = charbonnier(pred_low, low)

            grad_recon_loss = torch.tensor(0.0, device=device)
            if args.w_grad > 0:
                grad_recon_loss = charbonnier(sobel(pred_cect), sobel(target))

            ssim_l = torch.tensor(0.0, device=device)
            if args.w_ssim > 0:
                ssim_l = ssim_loss_simple(pred_cect, target)

            adv_img_loss = torch.tensor(0.0, device=device)
            if args.w_adv_img > 0:
                adv_img_loss = generator_lsgan_loss(img_d, pred_cect)

            adv_grad_loss = torch.tensor(0.0, device=device)
            if args.w_adv_grad > 0:
                adv_grad_loss = generator_lsgan_loss(grad_d, sobel(pred_cect))

            g_total = (
                args.w_recon * recon_loss
                + args.w_aux * aux_loss
                + args.w_ncct * ncct_loss
                + args.w_low * low_loss
                + args.w_grad * grad_recon_loss
                + args.w_ssim * ssim_l
                + args.w_adv_img * adv_img_loss
                + args.w_adv_grad * adv_grad_loss
            )

        if not torch.isfinite(g_total):
            print(f"[Warning] Non-finite generator loss at step={step}. Skip G update.")
            scaler.update()
            continue

        scaler.scale(g_total).backward()
        if args.g_grad_clip > 0:
            scaler.unscale_(g_opt)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.g_grad_clip)
        scaler.step(g_opt)
        scaler.update()

        set_requires_grad(img_d, True)
        set_requires_grad(grad_d, True)

        if ema is not None:
            ema.update(generator)

        # ---------------------------------------------------------------------
        # Logging.
        # ---------------------------------------------------------------------
        running["g_total"] += float(g_total.detach().cpu())
        running["g_recon"] += float(recon_loss.detach().cpu())
        running["g_aux"] += float(aux_loss.detach().cpu())
        running["g_ncct"] += float(ncct_loss.detach().cpu())
        running["g_low"] += float(low_loss.detach().cpu())
        running["g_grad"] += float(grad_recon_loss.detach().cpu())
        running["g_ssim"] += float(ssim_l.detach().cpu())
        running["g_adv_img"] += float(adv_img_loss.detach().cpu())
        running["g_adv_grad"] += float(adv_grad_loss.detach().cpu())
        running["d_img"] += float(d_img_loss.detach().cpu())
        running["d_grad"] += float(d_grad_loss.detach().cpu())

        if step % args.log_freq == 0:
            denom = args.log_freq
            row = {k: running[k] / denom for k in running}

            print(
                f"step {step:06d}/{args.max_iter} | "
                f"g_lr={g_lr:.3e} d_lr={d_lr:.3e} | "
                f"G={row['g_total']:.5f} recon={row['g_recon']:.5f} "
                f"grad={row['g_grad']:.5f} advI={row['g_adv_img']:.5f} advG={row['g_adv_grad']:.5f} | "
                f"D_img={row['d_img']:.5f} D_grad={row['d_grad']:.5f}"
            )

            with open(log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, g_lr, d_lr,
                    row["g_total"],
                    row["g_recon"],
                    row["g_aux"],
                    row["g_ncct"],
                    row["g_low"],
                    row["g_grad"],
                    row["g_ssim"],
                    row["g_adv_img"],
                    row["g_adv_grad"],
                    row["d_img"],
                    row["d_grad"],
                ])

            for k in running:
                running[k] = 0.0

        # ---------------------------------------------------------------------
        # Fixed interval checkpoint and monitor images.
        # Do not select the best checkpoint using non-registered test metrics.
        # ---------------------------------------------------------------------
        if step % args.save_freq == 0 or step == args.max_iter:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step:06d}.pth")
            save_checkpoint(
                ckpt_path,
                step,
                generator,
                img_d,
                grad_d,
                g_opt,
                img_d_opt,
                grad_d_opt,
                scaler,
                ema,
                args,
            )
            print(f"[Checkpoint] saved: {ckpt_path}")

            latest_path = os.path.join(ckpt_dir, "latest.pth")
            save_checkpoint(
                latest_path,
                step,
                generator,
                img_d,
                grad_d,
                g_opt,
                img_d_opt,
                grad_d_opt,
                scaler,
                ema,
                args,
            )

            torch.save(generator.state_dict(), os.path.join(ckpt_dir, f"generator_raw_{step:06d}.pth"))

            if ema is not None:
                ema.apply_to(generator)
                torch.save(generator.state_dict(), os.path.join(ckpt_dir, f"generator_ema_{step:06d}.pth"))
                ema.restore(generator)

            save_monitor_images(
                generator=generator,
                monitor_loader=monitor_loader,
                device=device,
                out_dir=vis_dir,
                step=step,
                ema=ema if args.use_ema_for_monitor else None,
                num_samples=args.num_monitor_images,
            )
