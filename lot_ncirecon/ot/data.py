"""Training pairs and file handling for the neural OT matcher."""

import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .metrics import composite_similarity_from_flat, get_pair_similarity_from_triplet, patch_to_flat
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def list_npy_files(root, recursive=True):
    files = []
    if recursive:
        for cur_root, _, names in os.walk(root):
            for name in names:
                if name.endswith(".npy") and not name.startswith("selection_summary"):
                    files.append(os.path.join(cur_root, name))
    else:
        for name in os.listdir(root):
            if name.endswith(".npy") and not name.startswith("selection_summary"):
                files.append(os.path.join(root, name))
    return sorted(files)


def rank_key(path):
    name = os.path.basename(path)
    m = re.search(r"rank(\d+)_", name)
    if m is None:
        return 10 ** 12
    return int(m.group(1))


def get_patch(data, key):
    if key in data:
        return data[key]
    aliases = {
        "A_LOW": ["L_LOW"],
        "A_NORMAL": ["A_T", "L_NORMAL"],
        "V_LOW": ["V_LOW"],
        "V_NORMAL": ["V_NORMAL"],
        "D_LOW": ["L_LOW"],
        "D_NORMAL": ["L_NORMAL"],
        "N": ["NCCT"],
    }
    for alias in aliases.get(key, []):
        if alias in data:
            return data[alias]
    raise KeyError(f"Cannot find key {key}. Available keys: {list(data.keys())}")


# =========================
# Patch and metric utilities
class PairRegressionDataset(Dataset):
    """
    Pair regression dataset.

    Positive pairs:
        A_LOW_i -> N_i        label = local-OT pair similarity_N
        A_LOW_i -> A_NORMAL_i label = local-OT pair similarity_A_NORMAL

    Mismatched pairs:
        A_LOW_i -> N_j / A_NORMAL_j, j != i
        label = computed composite similarity, not hard zero by default.

    This makes the output calibrated to the local OT similarity scale.
    """
    def __init__(self, A_arr, N_arr, AN_arr, N_label_arr, AN_label_arr,
                 pair_index, compute_negative_label=True):
        super().__init__()
        self.A_arr = A_arr
        self.N_arr = N_arr
        self.AN_arr = AN_arr
        self.N_label_arr = N_label_arr
        self.AN_label_arr = AN_label_arr
        self.pair_index = pair_index
        self.compute_negative_label = compute_negative_label

    def __len__(self):
        return len(self.pair_index)

    def __getitem__(self, idx):
        src_idx, tgt_idx, tgt_type, is_positive = self.pair_index[idx]
        src = self.A_arr[src_idx]

        if tgt_type == 0:
            tgt = self.N_arr[tgt_idx]
            if is_positive:
                label = self.N_label_arr[src_idx]
            else:
                label = composite_similarity_from_flat(src, tgt) if self.compute_negative_label else 0.0
        else:
            tgt = self.AN_arr[tgt_idx]
            if is_positive:
                label = self.AN_label_arr[src_idx]
            else:
                label = composite_similarity_from_flat(src, tgt) if self.compute_negative_label else 0.0

        x = np.concatenate([src, tgt], axis=0).astype(np.float32)
        y = np.float32(np.clip(label, 0.0, 1.0))
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def load_selected_triplets_for_training(selected_root, train_size, seed):
    all_files = list_npy_files(selected_root, recursive=True)
    all_files = [f for f in all_files if os.path.basename(f).startswith("rank")]
    all_files = sorted(all_files, key=rank_key)

    if len(all_files) == 0:
        raise RuntimeError(f"No selected triplet npy files found under: {selected_root}")
    if train_size > len(all_files):
        raise ValueError(f"train_size={train_size} exceeds available files={len(all_files)}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_files, size=train_size, replace=False)

    A_list, N_list, AN_list = [], [], []
    N_label_list, AN_label_list = [], []

    print(f"Loading {train_size} selected triplets for calibrated similarity-regression training...")
    for path in tqdm(chosen, desc="Load selected triplets"):
        data = np.load(path, allow_pickle=True).item()

        A_LOW = get_patch(data, "A_LOW")
        N = get_patch(data, "N")
        A_NORMAL = get_patch(data, "A_NORMAL")

        sim_N = get_pair_similarity_from_triplet(data, "N")
        sim_A = get_pair_similarity_from_triplet(data, "A_NORMAL")

        A_list.append(patch_to_flat(A_LOW))
        N_list.append(patch_to_flat(N))
        AN_list.append(patch_to_flat(A_NORMAL))
        N_label_list.append(float(np.clip(sim_N, 0.0, 1.0)))
        AN_label_list.append(float(np.clip(sim_A, 0.0, 1.0)))

    A_arr = np.stack(A_list, axis=0).astype(np.float32)
    N_arr = np.stack(N_list, axis=0).astype(np.float32)
    AN_arr = np.stack(AN_list, axis=0).astype(np.float32)
    N_label_arr = np.asarray(N_label_list, dtype=np.float32)
    AN_label_arr = np.asarray(AN_label_list, dtype=np.float32)

    print("Training label statistics on local-OT scale:")
    print(f"  N similarity       : min={N_label_arr.min():.4f}, mean={N_label_arr.mean():.4f}, max={N_label_arr.max():.4f}")
    print(f"  A_NORMAL similarity: min={AN_label_arr.min():.4f}, mean={AN_label_arr.mean():.4f}, max={AN_label_arr.max():.4f}")

    return A_arr, N_arr, AN_arr, N_label_arr, AN_label_arr


def build_pair_indices(num_triplets, negative_ratio=1.0, seed=2026):
    rng = np.random.default_rng(seed)
    pair_index = []

    # Positive pairs.
    for i in range(num_triplets):
        pair_index.append((i, i, 0, True))
        pair_index.append((i, i, 1, True))

    num_pos = len(pair_index)
    num_neg = int(num_pos * negative_ratio)

    # Mismatched pairs. They are not forced to label=0 in calibrated mode.
    for _ in range(num_neg):
        src_idx = int(rng.integers(0, num_triplets))
        tgt_idx = int(rng.integers(0, num_triplets - 1))
        if tgt_idx >= src_idx:
            tgt_idx += 1
        tgt_type = int(rng.integers(0, 2))
        pair_index.append((src_idx, tgt_idx, tgt_type, False))

    rng.shuffle(pair_index)
    return pair_index

