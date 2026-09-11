"""Full-slice local candidate search and OT-consistent triplet selection."""

import os

import numpy as np
import torch
from tqdm import tqdm

from .data import get_patch
from ..utils.training import ensure_dir
from .metrics import PATCH_SIZE, calc_triplet_quality_metrics, extract_patch, patch_to_flat

STRIDE = 16
SEARCH_STEP = 4
SPATIAL_CONSTRAINT = 50
PHASE_KEY_PREFIX = {"A": "A", "V": "V", "D": "L"}


def phase_keys(phase):
    prefix = PHASE_KEY_PREFIX.get(phase.upper())
    if prefix is None:
        raise ValueError("phase must be one of A, V, or D")
    return "{}_LOW".format(prefix), "{}_NORMAL".format(prefix)


def generate_source_patch_items(A_LOW, z, filename):
    items = []
    half = PATCH_SIZE // 2
    y_start = 100
    y_end = min(300, A_LOW.shape[0] - half)
    x_start = 100
    x_end = min(300, A_LOW.shape[1] - half)

    for y in range(y_start, y_end, STRIDE):
        for x in range(x_start, x_end, STRIDE):
            patch = extract_patch(A_LOW, x, y)
            if patch is None:
                continue
            items.append({
                "patch": patch,
                "flat": patch_to_flat(patch),
                "position": (x, y, z),
                "filename": filename,
            })
    return items


def find_candidate_patches(search_img, center_x, center_y, z_idx, spatial_constraint,
                           search_step=SEARCH_STEP):
    candidates = []
    half = PATCH_SIZE // 2
    h, w = search_img.shape[:2]

    for dy in range(-spatial_constraint, spatial_constraint + 1, search_step):
        for dx in range(-spatial_constraint, spatial_constraint + 1, search_step):
            # Paper-consistent local constraint: ||r_j - r_i||_2 <= C.
            if dx * dx + dy * dy > spatial_constraint * spatial_constraint:
                continue
            x = center_x + dx
            y = center_y + dy
            if x - half < 0 or x + half > w or y - half < 0 or y + half > h:
                continue
            patch = extract_patch(search_img, x, y)
            if patch is None:
                continue
            candidates.append({
                "patch": patch,
                "flat": patch_to_flat(patch),
                "position": (x, y, z_idx),
            })
    return candidates


def predict_scores(model, source_flat, candidate_flats, device, infer_batch_size=2048):
    if len(candidate_flats) == 0:
        return np.array([], dtype=np.float32)

    source = np.asarray(source_flat, dtype=np.float32).reshape(1, -1)
    targets = np.stack(candidate_flats, axis=0).astype(np.float32)
    sources = np.repeat(source, repeats=len(targets), axis=0)
    combined = np.concatenate([sources, targets], axis=1).astype(np.float32)

    scores_all = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(combined), infer_batch_size):
            batch = combined[start:start + infer_batch_size]
            x = torch.from_numpy(batch).to(device)
            scores = model(x).squeeze(1).detach().cpu().numpy()
            scores_all.append(scores)

    return np.concatenate(scores_all, axis=0)


def process_one_slice(data_folder, files, z, model, device, score_threshold, infer_batch_size,
                      save_folder, global_counter, metric_accumulator, spatial_constraint,
                      search_step=SEARCH_STEP, phase="A"):
    low_key, normal_key = phase_keys(phase)
    try:
        data_prev = np.load(os.path.join(data_folder, files[z - 1]), allow_pickle=True).item()
        data_curr = np.load(os.path.join(data_folder, files[z]), allow_pickle=True).item()
        data_next = np.load(os.path.join(data_folder, files[z + 1]), allow_pickle=True).item()

        A_LOW = get_patch(data_curr, low_key)
        N_layers = [
            (z - 1, get_patch(data_prev, "N")),
            (z, get_patch(data_curr, "N")),
            (z + 1, get_patch(data_next, "N")),
        ]
        A_layers = [
            (z - 1, get_patch(data_prev, normal_key)),
            (z, get_patch(data_curr, normal_key)),
            (z + 1, get_patch(data_next, normal_key)),
        ]
    except Exception as e:
        print(f"[Warning] Failed at {files[z]}: {e}")
        return global_counter

    source_items = generate_source_patch_items(A_LOW, z, files[z])
    saved_this_slice = 0

    for src in source_items:
        x, y, _ = src["position"]

        N_candidates, A_candidates = [], []
        for z_idx, img in N_layers:
            N_candidates.extend(find_candidate_patches(img, x, y, z_idx, spatial_constraint, search_step))
        for z_idx, img in A_layers:
            A_candidates.extend(find_candidate_patches(img, x, y, z_idx, spatial_constraint, search_step))

        if len(N_candidates) == 0 or len(A_candidates) == 0:
            continue

        N_scores = predict_scores(
            model=model,
            source_flat=src["flat"],
            candidate_flats=[c["flat"] for c in N_candidates],
            device=device,
            infer_batch_size=infer_batch_size
        )
        A_scores = predict_scores(
            model=model,
            source_flat=src["flat"],
            candidate_flats=[c["flat"] for c in A_candidates],
            device=device,
            infer_batch_size=infer_batch_size
        )

        if len(N_scores) == 0 or len(A_scores) == 0:
            continue

        best_N_idx = int(np.argmax(N_scores))
        best_A_idx = int(np.argmax(A_scores))
        best_N = N_candidates[best_N_idx]
        best_A = A_candidates[best_A_idx]

        score_N = float(N_scores[best_N_idx])
        score_A = float(A_scores[best_A_idx])

        # Same triplet confidence definition as local OT:
        # both pairwise correspondences should be reliable.
        quality_score = min(score_N, score_A)

        if quality_score <= score_threshold:
            continue

        metrics = calc_triplet_quality_metrics(src["patch"], best_N["patch"], best_A["patch"])

        global_counter += 1
        saved_this_slice += 1

        save_data = {
            "A_LOW": src["patch"],
            "N": best_N["patch"],
            "A_NORMAL": best_A["patch"],
            "positions": {
                "A_LOW": src["position"],
                "N": best_N["position"],
                "A_NORMAL": best_A["position"],
            },
            "filename": files[z],
            "score_N": score_N,
            "score_A": score_A,
            "quality_score": quality_score,
            "method": "calibrated_deep_ot_similarity_regressor",
            "metrics": metrics,
        }

        save_name = (
            f"rank{global_counter:06d}_"
            f"{os.path.splitext(files[z])[0]}_"
            f"z{z}_x{x}_y{y}_q{quality_score:.4f}.npy"
        )
        np.save(os.path.join(save_folder, save_name), save_data)

        metric_accumulator["count"] += 1
        metric_accumulator["sum_ssim"] += metrics["mean_ssim"]
        metric_accumulator["sum_mask"] += metrics["mean_mask"]
        metric_accumulator["sum_balance"] += metrics["mean_balance_consistency"]

    print(f"{files[z]}: saved={saved_this_slice}, total_saved={global_counter}")
    return global_counter


def run_full_dataset_inference(data_folder, save_folder, model, score_threshold,
                               max_test_slices=None, start_index=1, infer_batch_size=2048,
                               spatial_constraint=SPATIAL_CONSTRAINT, search_step=SEARCH_STEP,
                               phase="A"):
    ensure_dir(save_folder)
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".npy")])
    if len(files) < 3:
        raise RuntimeError(f"At least 3 continuous slices are required in {data_folder}")

    end_index = len(files) - 1
    if max_test_slices is not None:
        end_index = min(end_index, start_index + max_test_slices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metric_accumulator = {"count": 0, "sum_ssim": 0.0, "sum_mask": 0.0, "sum_balance": 0.0}
    global_counter = 0

    for z in tqdm(range(start_index, end_index), desc="Full-dataset inference"):
        global_counter = process_one_slice(
            data_folder=data_folder,
            files=files,
            z=z,
            model=model,
            device=device,
            score_threshold=score_threshold,
            infer_batch_size=infer_batch_size,
            save_folder=save_folder,
            global_counter=global_counter,
            metric_accumulator=metric_accumulator,
            spatial_constraint=spatial_constraint,
            search_step=search_step,
            phase=phase
        )

    if metric_accumulator["count"] > 0:
        return {
            "num_saved": metric_accumulator["count"],
            "mean_ssim": metric_accumulator["sum_ssim"] / metric_accumulator["count"],
            "mean_mask": metric_accumulator["sum_mask"] / metric_accumulator["count"],
            "mean_balance_consistency": metric_accumulator["sum_balance"] / metric_accumulator["count"],
        }

    return {"num_saved": 0, "mean_ssim": np.nan, "mean_mask": np.nan, "mean_balance_consistency": np.nan}


# =========================
# Summary output
