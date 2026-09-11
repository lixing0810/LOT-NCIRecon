"""Training loop for the calibrated neural OT matcher."""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.matcher import LightDeepOTMatcher
from ..ot.data import PairRegressionDataset, build_pair_indices, load_selected_triplets_for_training
from ..ot.metrics import BETA, BETA_PRIME, PATCH_SIZE
from ..utils.training import ensure_dir
def train_matcher(selected_root, model_dir, train_size, epochs=50, batch_size=256,
                  lr=1e-4, hidden_size=256, num_layers=2,
                  negative_ratio=1.0, seed=2026, num_workers=0,
                  compute_negative_label=True):
    ensure_dir(model_dir)

    A_arr, N_arr, AN_arr, N_label_arr, AN_label_arr = load_selected_triplets_for_training(
        selected_root=selected_root,
        train_size=train_size,
        seed=seed
    )

    pair_index = build_pair_indices(
        num_triplets=train_size,
        negative_ratio=negative_ratio,
        seed=seed
    )

    dataset = PairRegressionDataset(
        A_arr=A_arr,
        N_arr=N_arr,
        AN_arr=AN_arr,
        N_label_arr=N_label_arr,
        AN_label_arr=AN_label_arr,
        pair_index=pair_index,
        compute_negative_label=compute_negative_label
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightDeepOTMatcher(PATCH_SIZE * PATCH_SIZE * 2, hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Regression loss. This is the key difference from the BCE version.
    criterion = nn.SmoothL1Loss(beta=0.05)

    print(f"Training calibrated similarity regressor with train_size={train_size}, pairs={len(dataset)}, device={device}")
    print(f"negative_ratio={negative_ratio}, compute_negative_label={compute_negative_label}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_num = 0.0, 0
        pred_values, label_values = [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x).squeeze(1)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_num += x.size(0)

            if epoch == 1 or epoch == epochs or epoch % 10 == 0:
                pred_values.append(pred.detach().cpu().numpy())
                label_values.append(y.detach().cpu().numpy())

            pbar.set_postfix(loss=total_loss / max(total_num, 1))

        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            preds = np.concatenate(pred_values) if len(pred_values) else np.array([])
            labels = np.concatenate(label_values) if len(label_values) else np.array([])
            if len(preds) > 0:
                print(
                    f"Epoch {epoch}: loss={total_loss / max(total_num, 1):.6f}, "
                    f"pred[min/mean/max]={preds.min():.4f}/{preds.mean():.4f}/{preds.max():.4f}, "
                    f"label[min/mean/max]={labels.min():.4f}/{labels.mean():.4f}/{labels.max():.4f}"
                )

    save_path = os.path.join(model_dir, f"calibrated_deep_ot_matcher_train{train_size}.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_size": train_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "patch_size": PATCH_SIZE,
            "negative_ratio": negative_ratio,
            "seed": seed,
            "mode": "similarity_regression",
            "beta": BETA,
            "beta_prime": BETA_PRIME,
        },
        save_path
    )
    print(f"Calibrated matcher saved to: {save_path}")
    return model, save_path


# =========================
# Inference on full dataset
