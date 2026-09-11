"""CT-specific patch similarity and triplet quality metrics."""

import numpy as np
from skimage.metrics import structural_similarity as ssim

PATCH_SIZE = 64
HU_SCALE = 4096.0
HU_SHIFT = -1024.0
DENSITY_BINS = np.array([0, 64, 128, 256], dtype=np.float32)
BETA = 0.5
BETA_PRIME = 0.5
EPS = 1e-8


def _get_patch(data, key):
    aliases = {
        "A_LOW": ["A_LOW", "L_LOW"],
        "A_NORMAL": ["A_NORMAL", "A_T", "L_NORMAL"],
        "N": ["N", "NCCT"],
    }
    for name in aliases[key]:
        if name in data:
            return data[name]
    raise KeyError("Cannot find {} in {}".format(key, list(data)))
def extract_patch(img, x, y, size=PATCH_SIZE):
    half = size // 2
    patch = img[y - half:y + half, x - half:x + half]
    if patch.shape != (size, size):
        return None
    return patch.astype(np.float32)


def to_norm01(patch):
    patch = np.asarray(patch, dtype=np.float32)
    if patch.max() > 2.0 or patch.min() < -1.0:
        patch = (patch - HU_SHIFT) / HU_SCALE
    return np.clip(patch, 0.0, 1.0).astype(np.float32)


def patch_to_flat(patch):
    return to_norm01(patch).reshape(-1).astype(np.float32)


def to_uint8_fixed_scale(patch):
    patch01 = to_norm01(patch)
    return np.clip(np.round(patch01 * 255.0), 0, 255).astype(np.uint8)


def discretize_radiodensity(patch):
    patch_u8 = to_uint8_fixed_scale(patch)
    mask = np.digitize(patch_u8, DENSITY_BINS[1:-1], right=False)
    return mask.astype(np.uint8)


def calc_ssim(p1, p2):
    a = to_norm01(p1)
    b = to_norm01(p2)
    try:
        value = ssim(a, b, data_range=1.0)
    except Exception:
        value = 0.0
    return float(np.clip(value, 0.0, 1.0))


def calc_mask_overlap(p1, p2):
    m1 = discretize_radiodensity(p1)
    m2 = discretize_radiodensity(p2)
    return float(np.mean(m1 == m2))


def composite_similarity(patch_a, patch_b, beta=BETA, beta_prime=BETA_PRIME):
    """
    Same score scale as the local OT script:
        S = beta * S_ssim + beta_prime * S_mask
        cost = 1 - S
    """
    if abs((beta + beta_prime) - 1.0) > 1e-6:
        raise ValueError("beta + beta_prime should be 1.")
    s_ssim = calc_ssim(patch_a, patch_b)
    s_mask = calc_mask_overlap(patch_a, patch_b)
    sim = beta * s_ssim + beta_prime * s_mask
    sim = float(np.clip(sim, 0.0, 1.0))
    cost = 1.0 - sim
    return cost, sim, s_ssim, s_mask


def composite_similarity_from_flat(src_flat, tgt_flat):
    p1 = src_flat.reshape(PATCH_SIZE, PATCH_SIZE)
    p2 = tgt_flat.reshape(PATCH_SIZE, PATCH_SIZE)
    _, sim, _, _ = composite_similarity(p1, p2, beta=BETA, beta_prime=BETA_PRIME)
    return float(sim)


def calc_balanced_consistency_pair(p1, p2):
    s_ssim = calc_ssim(p1, p2)
    s_mask = calc_mask_overlap(p1, p2)
    s_bal = 2.0 * s_ssim * s_mask / (s_ssim + s_mask + EPS)
    return float(s_ssim), float(s_mask), float(s_bal)


def calc_triplet_quality_metrics(A_LOW, N, A_NORMAL):
    pairs = [(A_LOW, N), (A_LOW, A_NORMAL), (N, A_NORMAL)]
    ssim_values, mask_values, bal_values = [], [], []
    for p1, p2 in pairs:
        s_ssim, s_mask, s_bal = calc_balanced_consistency_pair(p1, p2)
        ssim_values.append(s_ssim)
        mask_values.append(s_mask)
        bal_values.append(s_bal)
    return {
        "mean_ssim": float(np.mean(ssim_values)),
        "mean_mask": float(np.mean(mask_values)),
        "mean_balance_consistency": float(np.mean(bal_values)),
        "pair_low_n_ssim": float(ssim_values[0]),
        "pair_low_a_ssim": float(ssim_values[1]),
        "pair_n_a_ssim": float(ssim_values[2]),
        "pair_low_n_mask": float(mask_values[0]),
        "pair_low_a_mask": float(mask_values[1]),
        "pair_n_a_mask": float(mask_values[2]),
        "pair_low_n_balance": float(bal_values[0]),
        "pair_low_a_balance": float(bal_values[1]),
        "pair_n_a_balance": float(bal_values[2]),
    }


def get_pair_similarity_from_triplet(data, pair_name):
    """
    Read pair-level local-OT similarity from saved triplet if available.

    The local OT script saves:
        data['similarity']['N']
        data['similarity']['A_NORMAL']

    If missing, recompute the same composite similarity with beta=0.5.
    """
    if "similarity" in data and isinstance(data["similarity"], dict):
        if pair_name in data["similarity"]:
            return float(data["similarity"][pair_name])

    A_LOW = _get_patch(data, "A_LOW")
    if pair_name == "N":
        target = _get_patch(data, "N")
    elif pair_name == "A_NORMAL":
        target = _get_patch(data, "A_NORMAL")
    else:
        raise ValueError(f"Unknown pair_name: {pair_name}")

    _, sim, _, _ = composite_similarity(A_LOW, target, beta=BETA, beta_prime=BETA_PRIME)
    return float(sim)


# =========================
# Similarity regression dataset
