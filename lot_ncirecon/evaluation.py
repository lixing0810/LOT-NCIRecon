"""Patient-aware checkpoint evaluation for multi-phase CECT reconstruction."""
import csv
import json
import os
import re
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

from .data import iter_npy_files, to_float01
from .models.reconstruction import LOTNCIRecon
from .utils.training import ensure_dir, parse_generator_outputs, strip_module_prefix


PHASE_KEYS = {
    "A": ("A_LOW", "A_NORMAL"),
    "V": ("V_LOW", "V_NORMAL"),
    # The clinical files use L for the delayed phase.
    "D": ("L_LOW", "L_NORMAL"),
}


def load_generator(checkpoint_path, device, use_ema=False):
    model = LOTNCIRecon().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and use_ema and checkpoint.get("ema"):
        state = checkpoint["ema"]["shadow"]
    elif isinstance(checkpoint, dict) and "generator" in checkpoint:
        state = checkpoint["generator"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(strip_module_prefix(state), strict=True)
    model.eval()
    return model


def _patient_token(filename):
    """Extract a stable grouping token without writing PHI to result files."""
    match = re.match(r"data_(.+)_\d+\.mat\.npy$", os.path.basename(filename))
    return match.group(1) if match else os.path.dirname(filename)


def group_files(data_dir, max_files=None):
    # Match the original test script: prefer immediate patient subdirectories.
    # Some exports also duplicate the same slices at the dataset root.
    patient_dirs = [
        entry for entry in sorted(os.scandir(data_dir), key=lambda item: item.name)
        if entry.is_dir() and any(iter_npy_files(entry.path, recursive=False))
    ]
    raw_groups = defaultdict(list)
    if patient_dirs:
        for entry in patient_dirs:
            raw_groups[entry.name].extend(sorted(iter_npy_files(entry.path, recursive=False)))
    else:
        for path in sorted(iter_npy_files(data_dir, recursive=False)):
            raw_groups[_patient_token(path)].append(path)

    if max_files is not None:
        remaining = max_files
        limited = defaultdict(list)
        for token in sorted(raw_groups):
            limited[token] = raw_groups[token][:remaining]
            remaining -= len(limited[token])
            if remaining <= 0:
                break
        raw_groups = limited
    groups = {}
    for index, token in enumerate(sorted(raw_groups), start=1):
        groups["patient_{:02d}".format(index)] = raw_groups[token]
    return groups


class InceptionV3FeatureExtractor(nn.Module):
    """Return pool3 features for FID/KID and Mixed_6e GAP features for sFID."""

    def __init__(self):
        super().__init__()
        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
            transform_input=False,
        )
        self.inception.eval()

    def forward(self, x):
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        mixed_6e = self.inception.Mixed_6e(x)
        sfid = F.adaptive_avg_pool2d(mixed_6e, (1, 1)).flatten(1)
        x = self.inception.Mixed_7a(mixed_6e)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        pool3 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return pool3, sfid


_INCEPTION_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_inception(images):
    tensors = []
    for image in images:
        image = np.clip(np.squeeze(image).astype(np.float32), 0, 1)
        tensors.append(_INCEPTION_PREPROCESS((image * 255).astype(np.uint8)))
    return torch.stack(tensors)


def calculate_fid(real_features, fake_features, eps=1e-6):
    mu_real, cov_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_fake, cov_fake = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    cov_real += np.eye(cov_real.shape[0]) * eps
    cov_fake += np.eye(cov_fake.shape[0]) * eps
    covariance_mean, _ = linalg.sqrtm(cov_real.dot(cov_fake), disp=False)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = mu_real - mu_fake
    return float(difference.dot(difference) + np.trace(
        cov_real + cov_fake - 2 * covariance_mean
    ))


def calculate_kid(real_features, fake_features, num_subsets=100, subset_size=1000,
                  seed=2026):
    count = min(len(real_features), len(fake_features))
    if count < 10:
        raise ValueError("KID requires at least 10 samples")
    subset_size = min(subset_size, count // 2)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(num_subsets):
        indices = rng.permutation(count)
        real = real_features[indices[:subset_size]]
        fake = fake_features[indices[subset_size:2 * subset_size]]
        features = np.vstack([real, fake])
        kernel = polynomial_kernel(
            features, degree=3, gamma=1.0 / features.shape[1], coef0=1.0
        )
        xx = kernel[:subset_size, :subset_size]
        yy = kernel[subset_size:, subset_size:]
        xy = kernel[:subset_size, subset_size:]
        score = (
            (xx.sum() - np.diag(xx).sum()) / (subset_size * (subset_size - 1))
            + (yy.sum() - np.diag(yy).sum()) / (subset_size * (subset_size - 1))
            - 2 * xy.mean()
        )
        scores.append(score)
    return float(np.mean(scores)), float(np.std(scores))


def _mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _load_batch(paths, phase):
    low_key, normal_key = PHASE_KEYS[phase]
    lows, targets, nccts = [], [], []
    for path in paths:
        sample = np.load(path, allow_pickle=True).item()
        for key in (low_key, normal_key, "N"):
            if key not in sample:
                raise KeyError("{} is missing from {}".format(key, path))
        lows.append(to_float01(sample[low_key]))
        targets.append(to_float01(sample[normal_key]))
        nccts.append(to_float01(sample["N"]))
    return (
        torch.from_numpy(np.stack(lows)[:, None]).float(),
        np.stack(targets).astype(np.float32),
        torch.from_numpy(np.stack(nccts)[:, None]).float(),
    )


@torch.no_grad()
def evaluate_patient(patient_id, paths, model, phase, device, batch_size,
                     feature_extractor=None):
    rows = []
    real_pool3, fake_pool3, real_sfid, fake_sfid = [], [], [], []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        low, targets, ncct = _load_batch(batch_paths, phase)
        low, ncct = low.to(device), ncct.to(device)
        _, prediction, _, _ = parse_generator_outputs(model(ncct, low))
        predictions = prediction.clamp(0, 1)[:, 0].cpu().numpy()

        for offset, (target, predicted) in enumerate(zip(targets, predictions)):
            rows.append({
                "patient_id": patient_id,
                "sample_id": start + offset + 1,
                "psnr": float(peak_signal_noise_ratio(target, predicted, data_range=1.0)),
                "ssim": float(structural_similarity(target, predicted, data_range=1.0)),
                "rmse": float(np.sqrt(np.mean((target - predicted) ** 2))),
            })

        if feature_extractor is not None:
            real_input = preprocess_inception(targets).to(device)
            fake_input = preprocess_inception(predictions).to(device)
            real_p, real_s = feature_extractor(real_input)
            fake_p, fake_s = feature_extractor(fake_input)
            real_pool3.append(real_p.cpu().numpy())
            fake_pool3.append(fake_p.cpu().numpy())
            real_sfid.append(real_s.cpu().numpy())
            fake_sfid.append(fake_s.cpu().numpy())

    summary = {
        "num_slices": len(rows),
        "PSNR": _mean_std([row["psnr"] for row in rows]),
        "SSIM": _mean_std([row["ssim"] for row in rows]),
        "RMSE": _mean_std([row["rmse"] for row in rows]),
    }
    if feature_extractor is not None:
        real_pool3, fake_pool3 = np.vstack(real_pool3), np.vstack(fake_pool3)
        real_sfid, fake_sfid = np.vstack(real_sfid), np.vstack(fake_sfid)
        kid_mean, kid_std = calculate_kid(real_pool3, fake_pool3)
        summary.update({
            "FID": calculate_fid(real_pool3, fake_pool3),
            "sFID": calculate_fid(real_sfid, fake_sfid),
            "KID": {"mean": kid_mean, "std": kid_std},
        })
    return rows, summary


def evaluate(data_dir, checkpoint_path, output_dir, phase="A", batch_size=2,
             max_files=None, distribution_metrics=True, use_ema=False):
    if phase not in PHASE_KEYS:
        raise ValueError("phase must be A, V, or D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(checkpoint_path, device, use_ema=use_ema)
    groups = group_files(data_dir, max_files=max_files)
    if not groups:
        raise RuntimeError("No .npy test files found in {}".format(data_dir))

    feature_extractor = None
    if distribution_metrics:
        feature_extractor = InceptionV3FeatureExtractor().to(device).eval()

    all_rows, patient_summaries = [], {}
    for patient_id, paths in groups.items():
        rows, summary = evaluate_patient(
            patient_id, paths, model, phase, device, batch_size, feature_extractor
        )
        all_rows.extend(rows)
        patient_summaries[patient_id] = summary

    pooled = {
        "num_patients": len(patient_summaries),
        "num_slices": len(all_rows),
        "PSNR": _mean_std([row["psnr"] for row in all_rows]),
        "SSIM": _mean_std([row["ssim"] for row in all_rows]),
        "RMSE": _mean_std([row["rmse"] for row in all_rows]),
    }
    macro = {}
    for metric in ("PSNR", "SSIM", "RMSE"):
        macro[metric] = _mean_std([
            patient_summaries[patient][metric]["mean"] for patient in patient_summaries
        ])
    if distribution_metrics:
        for metric in ("FID", "sFID"):
            macro[metric] = _mean_std([
                patient_summaries[patient][metric] for patient in patient_summaries
            ])
        macro["KID"] = _mean_std([
            patient_summaries[patient]["KID"]["mean"] for patient in patient_summaries
        ])

    result = {
        "phase": phase,
        "intensity_scale": "[0, 1]",
        "phase_keys": {"low": PHASE_KEYS[phase][0], "target": PHASE_KEYS[phase][1]},
        "patient_grouping": "immediate subdirectories preferred; root files used only as fallback",
        "checkpoint": os.path.basename(checkpoint_path),
        "pooled_slice_statistics": pooled,
        "macro_patient_statistics": macro,
        "patients": patient_summaries,
    }
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "per_slice_metrics.csv"), "w", newline="",
              encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["patient_id", "sample_id", "psnr", "ssim", "rmse"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return result
