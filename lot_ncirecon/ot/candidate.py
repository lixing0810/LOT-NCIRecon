"""Stage-1 local-OT candidate selection.

This module implements the first patch-selection pass used by LOT-NCIRecon:
scan consecutive CT slices, solve local OT with the handcrafted
SSIM/radiodensity cost, and save the first high-quality 10,000 triplets as
training candidates for the neural OT matcher.
"""

import csv
import os

import numpy as np
from tqdm import tqdm

from .metrics import (
    BETA,
    BETA_PRIME,
    DENSITY_BINS,
    PATCH_SIZE,
    composite_similarity,
    extract_patch,
)
from ..utils.training import ensure_dir

STRIDE = 16
SEARCH_RADIUS = 20
SEARCH_STEP = 4
TOP_K = 10000
QUALITY_THRESHOLD = 0.735
ENTROPY_REG = 0.05
SINKHORN_MAX_ITER = 200
SINKHORN_TOL = 1e-6
INVALID_COST = 1e6
EPS = 1e-12

PHASE_KEY_PREFIX = {"A": "A", "V": "V", "D": "L"}


def phase_keys(phase):
    prefix = PHASE_KEY_PREFIX.get(phase.upper())
    if prefix is None:
        raise ValueError("phase must be one of A, V, or D")
    return "{}_LOW".format(prefix), "{}_NORMAL".format(prefix)


def get_raw_patch(data, key):
    if key in data:
        return data[key]
    aliases = {
        "D_LOW": ["L_LOW"],
        "D_NORMAL": ["L_NORMAL"],
        "A_NORMAL": ["A_T"],
        "N": ["NCCT"],
    }
    for alias in aliases.get(key, []):
        if alias in data:
            return data[alias]
    raise KeyError("Cannot find key {}. Available keys: {}".format(key, list(data)))


def generate_source_patches(low_img, z, filename, stride=STRIDE):
    items = []
    half = PATCH_SIZE // 2
    y_start = 100
    y_end = min(300, low_img.shape[0] - half)
    x_start = 100
    x_end = min(300, low_img.shape[1] - half)

    for y in range(y_start, y_end, stride):
        for x in range(x_start, x_end, stride):
            patch = extract_patch(low_img, x, y)
            if patch is None:
                continue
            items.append({
                "patch": patch,
                "position": (x, y, z),
                "filename": filename,
            })
    return items


def build_target_pool(layers, source_items, search_radius=SEARCH_RADIUS,
                      search_step=SEARCH_STEP):
    pool = {}
    half = PATCH_SIZE // 2
    for src in source_items:
        sx, sy, _ = src["position"]
        for z_idx, img in layers:
            h, w = img.shape[:2]
            for dy in range(-search_radius, search_radius + 1, search_step):
                for dx in range(-search_radius, search_radius + 1, search_step):
                    x = sx + dx
                    y = sy + dy
                    if x - half < 0 or x + half > w:
                        continue
                    if y - half < 0 or y + half > h:
                        continue
                    key = (z_idx, x, y)
                    if key in pool:
                        continue
                    patch = extract_patch(img, x, y)
                    if patch is None:
                        continue
                    pool[key] = {"patch": patch, "position": (x, y, z_idx)}
    return list(pool.values())


def build_local_cost_matrix(source_items, target_items, beta=BETA,
                            beta_prime=BETA_PRIME,
                            search_radius=SEARCH_RADIUS):
    n = len(source_items)
    m = len(target_items)
    cost = np.full((n, m), INVALID_COST, dtype=np.float32)
    sim = np.zeros((n, m), dtype=np.float32)
    sim_ssim = np.zeros((n, m), dtype=np.float32)
    sim_mask = np.zeros((n, m), dtype=np.float32)

    for i, src in enumerate(source_items):
        sx, sy, _ = src["position"]
        for j, tgt in enumerate(target_items):
            tx, ty, _ = tgt["position"]
            if abs(tx - sx) > search_radius or abs(ty - sy) > search_radius:
                continue
            c, s, s_ssim, s_mask = composite_similarity(
                src["patch"], tgt["patch"], beta=beta, beta_prime=beta_prime
            )
            cost[i, j] = c
            sim[i, j] = s
            sim_ssim[i, j] = s_ssim
            sim_mask[i, j] = s_mask
    return cost, sim, sim_ssim, sim_mask


def sinkhorn_ot(cost_matrix, reg=ENTROPY_REG, max_iter=SINKHORN_MAX_ITER,
                tol=SINKHORN_TOL):
    cost_matrix = cost_matrix.astype(np.float64)
    valid = cost_matrix < INVALID_COST / 2
    valid_cols = valid.any(axis=0)
    if valid_cols.sum() == 0:
        return None, valid_cols

    cost = cost_matrix[:, valid_cols]
    valid = valid[:, valid_cols]
    n, m = cost.shape
    if n == 0 or m == 0:
        return None, valid_cols

    kernel = np.exp(-cost / reg)
    kernel[~valid] = 0.0
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)

    for _ in range(max_iter):
        u_prev = u.copy()
        u = a / (kernel @ v + EPS)
        v = b / (kernel.T @ u + EPS)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    transport = (u[:, None] * kernel) * v[None, :]
    return transport.astype(np.float32), valid_cols


def solve_local_ot_matching(source_items, target_items, beta=BETA,
                            beta_prime=BETA_PRIME,
                            search_radius=SEARCH_RADIUS):
    if len(source_items) == 0 or len(target_items) == 0:
        return {}

    cost, sim, sim_ssim, sim_mask = build_local_cost_matrix(
        source_items, target_items, beta=beta, beta_prime=beta_prime,
        search_radius=search_radius
    )
    transport, valid_cols = sinkhorn_ot(cost)
    if transport is None:
        return {}

    target_items = [t for t, keep in zip(target_items, valid_cols) if keep]
    cost = cost[:, valid_cols]
    sim = sim[:, valid_cols]
    sim_ssim = sim_ssim[:, valid_cols]
    sim_mask = sim_mask[:, valid_cols]

    matches = {}
    for i in range(len(source_items)):
        valid_js = np.where(cost[i] < INVALID_COST / 2)[0]
        if len(valid_js) == 0:
            continue
        j = int(valid_js[int(np.argmax(transport[i, valid_js]))])
        matches[i] = {
            "target": target_items[j],
            "transport_mass": float(transport[i, j]),
            "cost": float(cost[i, j]),
            "similarity": float(sim[i, j]),
            "ssim": float(sim_ssim[i, j]),
            "mask_overlap": float(sim_mask[i, j]),
        }
    return matches


def displacement_2d(pos_a, pos_b):
    xa, ya, _ = pos_a
    xb, yb, _ = pos_b
    return float(np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2))


def collect_triplets_for_slice(data_folder, files, z, phase="A", beta=BETA,
                               beta_prime=BETA_PRIME,
                               search_radius=SEARCH_RADIUS,
                               search_step=SEARCH_STEP):
    low_key, normal_key = phase_keys(phase)
    paths = [os.path.join(data_folder, files[idx]) for idx in (z - 1, z, z + 1)]
    try:
        data_prev, data_curr, data_next = [
            np.load(path, allow_pickle=True).item() for path in paths
        ]
        low_img = get_raw_patch(data_curr, low_key)
        n_layers = [
            (z - 1, get_raw_patch(data_prev, "N")),
            (z, get_raw_patch(data_curr, "N")),
            (z + 1, get_raw_patch(data_next, "N")),
        ]
        normal_layers = [
            (z - 1, get_raw_patch(data_prev, normal_key)),
            (z, get_raw_patch(data_curr, normal_key)),
            (z + 1, get_raw_patch(data_next, normal_key)),
        ]
    except Exception as exc:
        print("[Warning] Failed to read {}: {}".format(files[z], exc))
        return []

    if low_img is None or low_img.shape[0] < PATCH_SIZE or low_img.shape[1] < PATCH_SIZE:
        return []

    source_items = generate_source_patches(low_img, z, files[z])
    n_targets = build_target_pool(n_layers, source_items, search_radius, search_step)
    normal_targets = build_target_pool(normal_layers, source_items,
                                       search_radius, search_step)
    n_matches = solve_local_ot_matching(source_items, n_targets, beta, beta_prime,
                                        search_radius)
    normal_matches = solve_local_ot_matching(source_items, normal_targets, beta,
                                             beta_prime, search_radius)

    triplets = []
    for i, src in enumerate(source_items):
        if i not in n_matches or i not in normal_matches:
            continue
        n_match = n_matches[i]
        normal_match = normal_matches[i]
        sim_n = n_match["similarity"]
        sim_normal = normal_match["similarity"]
        ssim_n = n_match["ssim"]
        ssim_normal = normal_match["ssim"]
        mask_n = n_match["mask_overlap"]
        mask_normal = normal_match["mask_overlap"]
        disp_n = displacement_2d(src["position"], n_match["target"]["position"])
        disp_normal = displacement_2d(src["position"], normal_match["target"]["position"])
        mean_disp = 0.5 * (disp_n + disp_normal)

        triplets.append({
            "A_LOW": src["patch"],
            "N": n_match["target"]["patch"],
            "A_NORMAL": normal_match["target"]["patch"],
            "positions": {
                "A_LOW": src["position"],
                "N": n_match["target"]["position"],
                "A_NORMAL": normal_match["target"]["position"],
            },
            "filename": src["filename"],
            "phase": phase.upper(),
            "quality_score": float(min(sim_n, sim_normal)),
            "table_stats": {
                "mean_ssim": float(0.5 * (ssim_n + ssim_normal)),
                "mean_mask": float(0.5 * (mask_n + mask_normal)),
                "mean_S": float(0.5 * (sim_n + sim_normal)),
                "mean_displacement": float(mean_disp),
            },
            "similarity": {
                "N": float(sim_n),
                "A_NORMAL": float(sim_normal),
                "N_ssim": float(ssim_n),
                "N_mask": float(mask_n),
                "A_NORMAL_ssim": float(ssim_normal),
                "A_NORMAL_mask": float(mask_normal),
            },
            "displacement": {
                "N": float(disp_n),
                "A_NORMAL": float(disp_normal),
                "mean": float(mean_disp),
            },
            "ot_costs": {
                "N": float(n_match["cost"]),
                "A_NORMAL": float(normal_match["cost"]),
                "total": float(n_match["cost"] + normal_match["cost"]),
            },
            "transport_mass": {
                "N": float(n_match["transport_mass"]),
                "A_NORMAL": float(normal_match["transport_mass"]),
                "total": float(n_match["transport_mass"] + normal_match["transport_mass"]),
            },
            "beta": float(beta),
            "beta_prime": float(beta_prime),
            "patch_size": PATCH_SIZE,
            "search_radius": int(search_radius),
        })
    return triplets


def summarize_selected(selected, phase, beta, beta_prime, threshold, used_slices,
                       total_candidates, total_valid):
    row = {
        "phase": phase.upper(),
        "beta": float(beta),
        "beta_prime": float(beta_prime),
        "num_selected": int(len(selected)),
        "quality_threshold": float(threshold),
        "used_slices": int(used_slices),
        "total_candidates": int(total_candidates),
        "total_valid_above_threshold": int(total_valid),
    }
    if len(selected) == 0:
        row.update({
            "effective_tau": np.nan,
            "mean_ssim": np.nan,
            "mean_mask": np.nan,
            "mean_S": np.nan,
            "mean_displacement": np.nan,
        })
        return row

    quality = np.asarray([x["quality_score"] for x in selected], dtype=np.float64)
    for key, out_key in (
        ("mean_ssim", "mean_ssim"),
        ("mean_mask", "mean_mask"),
        ("mean_S", "mean_S"),
        ("mean_displacement", "mean_displacement"),
    ):
        row[out_key] = float(np.mean([x["table_stats"][key] for x in selected]))
    row["effective_tau"] = float(np.min(quality))
    return row


def save_selected_triplets(selected, save_folder, beta, beta_prime):
    beta_folder = ensure_dir(os.path.join(
        save_folder,
        "beta_{:.2f}_betaprime_{:.2f}".format(beta, beta_prime).replace(".", "p"),
    ))
    for rank, data in enumerate(selected, start=1):
        x, y, z = data["positions"]["A_LOW"]
        base = os.path.splitext(data["filename"])[0]
        quality = data["quality_score"]
        save_name = "rank{:05d}_{}_z{}_x{}_y{}_q{:.4f}.npy".format(
            rank, base, z, x, y, quality
        )
        np.save(os.path.join(beta_folder, save_name), data)

    summary = {
        "num_selected": int(len(selected)),
        "top_k": TOP_K,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "search_radius": SEARCH_RADIUS,
        "search_step": SEARCH_STEP,
        "density_bins": DENSITY_BINS.tolist(),
        "beta": float(beta),
        "beta_prime": float(beta_prime),
    }
    np.save(os.path.join(beta_folder, "selection_summary.npy"), summary)
    return beta_folder


def write_summary_table(rows, output_root):
    csv_path = os.path.join(output_root, "candidate_summary_table.csv")
    fields = [
        "phase", "beta", "beta_prime", "num_selected", "effective_tau",
        "mean_ssim", "mean_mask", "mean_S", "mean_displacement",
        "quality_threshold", "used_slices", "total_candidates",
        "total_valid_above_threshold",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def select_candidates_until_enough(data_folder, output_root, phase="A", top_k=TOP_K,
                                   quality_threshold=QUALITY_THRESHOLD,
                                   beta=BETA, beta_prime=BETA_PRIME,
                                   search_radius=SEARCH_RADIUS,
                                   search_step=SEARCH_STEP):
    output_root = ensure_dir(output_root)
    files = sorted([name for name in os.listdir(data_folder) if name.endswith(".npy")])
    if len(files) < 3:
        raise RuntimeError("At least 3 continuous slice files are required.")

    selected = []
    total_candidates = 0
    total_valid = 0
    used_slices = 0

    print("Stage 1: local-OT candidate selection")
    print("phase={}, target={}, threshold>{:.4f}".format(
        phase.upper(), top_k, quality_threshold
    ))
    for z in tqdm(range(1, len(files) - 1), desc="First select {}".format(phase.upper())):
        triplets = collect_triplets_for_slice(
            data_folder=data_folder,
            files=files,
            z=z,
            phase=phase,
            beta=beta,
            beta_prime=beta_prime,
            search_radius=search_radius,
            search_step=search_step,
        )
        used_slices += 1
        total_candidates += len(triplets)
        valid = [x for x in triplets if x["quality_score"] > quality_threshold]
        valid = sorted(valid, key=lambda x: x["quality_score"], reverse=True)
        total_valid += len(valid)
        selected.extend(valid[:max(top_k - len(selected), 0)])
        print("{}: collected={}, valid={}, selected={}/{}".format(
            files[z], len(triplets), len(valid), len(selected), top_k
        ))
        if len(selected) >= top_k:
            break

    row = summarize_selected(
        selected=selected,
        phase=phase,
        beta=beta,
        beta_prime=beta_prime,
        threshold=quality_threshold,
        used_slices=used_slices,
        total_candidates=total_candidates,
        total_valid=total_valid,
    )
    selected_root = save_selected_triplets(selected, output_root, beta, beta_prime)
    row["selected_root"] = selected_root
    write_summary_table([row], output_root)
    return row
