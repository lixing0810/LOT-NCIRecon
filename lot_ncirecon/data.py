"""Dataset and normalization utilities for OT-selected CT triplets."""

import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
def iter_npy_files(root, recursive=True):
    """Yield .npy sample paths lazily. This avoids building a huge glob list first."""
    bad_prefix = (
        "selection_summary",
        "recomputed_metrics",
        "network_plus_similarity_filter_summary",
        "filtered_patch_metrics",
    )

    if recursive:
        for cur_root, _dirs, names in os.walk(root):
            for name in names:
                if name.endswith(".npy") and not name.startswith(bad_prefix):
                    yield os.path.join(cur_root, name)
    else:
        for name in os.listdir(root):
            if name.endswith(".npy") and not name.startswith(bad_prefix):
                yield os.path.join(root, name)


def _default_manifest_path(root, output_root=None, tag="train"):
    """Create a stable manifest path outside the huge dataset directory by default."""
    base = os.path.abspath(root).rstrip(os.sep).replace(os.sep, "__").replace(":", "")
    if output_root is None:
        return os.path.join(root, f"_{tag}_npy_manifest.txt")
    return os.path.join(output_root, "file_manifests", f"{tag}_{base}.txt")


def list_npy_files(
    root,
    recursive=True,
    manifest_path=None,
    rebuild_manifest=False,
    max_files=None,
    sort_files=False,
):
    """
    Fast file listing for large patch datasets.

    For hundreds of thousands of small .npy patch files, recursive glob + sorting can make
    dataset initialization look stuck. This function supports a manifest cache:

      1) First run with --rebuild_manifest: scan the directory once and save paths.
      2) Later runs: read the manifest directly, avoiding repeated directory traversal.

    Notes:
      - max_files is applied during scanning, so debug runs do not need to scan all files.
      - sort_files=False is recommended for training because DataLoader(shuffle=True)
        already randomizes sample order.
    """
    if manifest_path is not None and os.path.exists(manifest_path) and not rebuild_manifest:
        print(f"[Dataset] Loading file list from manifest: {manifest_path}")
        with open(manifest_path, "r") as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        print(f"[Dataset] Scanning npy files under: {root}")
        files = []
        for i, path in enumerate(iter_npy_files(root, recursive=recursive), start=1):
            files.append(path)
            if i % 50000 == 0:
                print(f"[Dataset] scanned {i} npy files...")
            if max_files is not None and len(files) >= max_files:
                break

        if sort_files:
            def key_fn(p):
                name = os.path.basename(p)
                m = re.search(r"rank(\d+)", name)
                return int(m.group(1)) if m else name
            files = sorted(files, key=key_fn)

        if manifest_path is not None:
            manifest_dir = os.path.dirname(manifest_path)
            if manifest_dir:
                os.makedirs(manifest_dir, exist_ok=True)
            print(f"[Dataset] Saving manifest to: {manifest_path}")
            with open(manifest_path, "w") as f:
                for path in files:
                    f.write(path + "\n")

    if max_files is not None:
        files = files[:max_files]

    return files


def get_array(data, key):
    aliases = {
        "A_LOW": [
            "A_LOW", "L_LOW", "V_LOW", "D_LOW",
            "low_dose", "low", "LD_CECT", "ld_cect",
        ],
        "A_NORMAL": [
            "A_NORMAL", "L_NORMAL", "V_NORMAL", "D_NORMAL",
            "A_T", "target", "normal_dose", "ND_CECT", "nd_cect",
        ],
        "N": [
            "N", "NCCT", "non_contrast", "noncontrast",
            "full_dose4", "baseline_ncct",
        ],
    }

    for k in aliases[key]:
        if k in data and data[k] is not None:
            return data[k]

    raise KeyError(f"Cannot find {key}. Available keys: {list(data.keys())}")



def get_similarity_weight(data):
    """
    Patch-level reliability weight S(p^A,p^B) used in the manuscript loss.
    It is usually saved as 'quality_score' by the local OT / matcher+similarity filtering stage.
    If unavailable, use 1.0 for backward compatibility.
    """
    for key in ["quality_score", "composite_quality_score", "mean_composite_similarity", "net_confidence"]:
        if key in data:
            try:
                return float(data[key])
            except Exception:
                pass
    if "similarity" in data and isinstance(data["similarity"], dict):
        vals = []
        for k in ["N", "A_NORMAL"]:
            if k in data["similarity"]:
                vals.append(float(data["similarity"][k]))
        if vals:
            return float(min(vals))
    return 1.0


def to_float01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.min() >= -0.05 and arr.max() <= 1.05:
        return np.clip(arr, 0.0, 1.0)
    if arr.min() >= 0 and arr.max() <= 255:
        return np.clip(arr / 255.0, 0.0, 1.0)

    arr = (arr + 1024.0) / 4096.0
    return np.clip(arr, 0.0, 1.0)


def to_tensor_chw(arr):
    arr = to_float01(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] != 1:
            arr = arr[:1]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return torch.from_numpy(arr.astype(np.float32))


def random_crop_triplet(low, target, ncct, crop_size):
    _, h, w = low.shape

    if crop_size <= 0 or (h == crop_size and w == crop_size):
        return low, target, ncct

    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        pad = (0, pad_w, 0, pad_h)
        low = F.pad(low, pad, mode="reflect")
        target = F.pad(target, pad, mode="reflect")
        ncct = F.pad(ncct, pad, mode="reflect")
        _, h, w = low.shape

    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)

    return (
        low[:, y:y + crop_size, x:x + crop_size],
        target[:, y:y + crop_size, x:x + crop_size],
        ncct[:, y:y + crop_size, x:x + crop_size],
    )


def maybe_flip(low, target, ncct, p=0.5):
    if random.random() < p:
        low = torch.flip(low, dims=[2])
        target = torch.flip(target, dims=[2])
        ncct = torch.flip(ncct, dims=[2])
    if random.random() < p:
        low = torch.flip(low, dims=[1])
        target = torch.flip(target, dims=[1])
        ncct = torch.flip(ncct, dims=[1])
    return low, target, ncct


class OTTripletPatchDataset(Dataset):
    def __init__(
        self,
        root,
        crop_size=64,
        recursive=True,
        random_flip=False,
        max_files=None,
        manifest_path=None,
        rebuild_manifest=False,
        sort_files=False,
        files=None,
    ):
        self.root = root

        if files is not None:
            self.files = list(files)
            if max_files is not None:
                self.files = self.files[:max_files]
        else:
            self.files = list_npy_files(
                root,
                recursive=recursive,
                manifest_path=manifest_path,
                rebuild_manifest=rebuild_manifest,
                max_files=max_files,
                sort_files=sort_files,
            )

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy samples found in {root}")

        self.crop_size = crop_size
        self.random_flip = random_flip

        print(f"[Dataset] root={root}")
        print(f"[Dataset] num_files={len(self.files)}")
        print(f"[Dataset] crop_size={crop_size}, random_flip={random_flip}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        low = to_tensor_chw(get_array(data, "A_LOW"))
        target = to_tensor_chw(get_array(data, "A_NORMAL"))
        ncct = to_tensor_chw(get_array(data, "N"))

        low, target, ncct = random_crop_triplet(low, target, ncct, self.crop_size)

        if self.random_flip:
            low, target, ncct = maybe_flip(low, target, ncct)

        weight = torch.tensor([get_similarity_weight(data)], dtype=torch.float32)
        return low, target, ncct, weight


# =============================================================================
# Loss functions
