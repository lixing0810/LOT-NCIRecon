#!/usr/bin/env python
"""Train the neural OT matcher and select paper-consistent patch triplets."""
import torch
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lot_ncirecon.config import load_config
from lot_ncirecon.ot.reporting import save_selection_summary
from lot_ncirecon.ot.selector import run_full_dataset_inference
from lot_ncirecon.trainers.matcher import train_matcher
from lot_ncirecon.utils.training import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", default=None, choices=["A", "V", "D"])
    parser.add_argument("--selected-root",default="/public/home/lixing/ct_cross_modal_ten_dose/tip_test/A/beta_0p50_betaprime_0p50")
    parser.add_argument("--data-folder",default="/public/home/lixing/ct_cross_modal_ten_dose/train_reg_N_new_all_same")
    parser.add_argument("--output-root",default="/public/home/lixing/ct_cross_modal_ten_dose/selected_patches_onlyA_public")
    parser.add_argument("--spatial-constraint", type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--max-test-slices", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg = dict(config.get("selection", {}))
    candidate_cfg = dict(config.get("candidate", {}))
    phase = args.phase or cfg.get("phase") or candidate_cfg.get("phase", "A")
    for key, value in {
        "selected_root": args.selected_root,
        "data_folder": args.data_folder,
        "output_root": args.output_root,
        "spatial_constraint": args.spatial_constraint,
        "train_size": args.train_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "score_threshold": args.score_threshold,
        "max_test_slices": args.max_test_slices,
    }.items():
        if value is not None:
            cfg[key] = value

    if not cfg.get("selected_root"):
        beta = candidate_cfg.get("beta", 0.5)
        beta_prime = candidate_cfg.get("beta_prime", 0.5)
        candidate_dir = "beta_{:.2f}_betaprime_{:.2f}".format(
            beta, beta_prime
        ).replace(".", "p")
        cfg["selected_root"] = os.path.join(
            candidate_cfg.get("output_root", "outputs/candidates"),
            phase,
            candidate_dir,
        )

    for required in ("selected_root", "data_folder"):
        if not cfg.get(required):
            raise ValueError("Set selection.{} in YAML or pass --{}.".format(
                required, required.replace("_", "-")
            ))
    if cfg["spatial_constraint"] <= 0:
        raise ValueError("spatial_constraint must be positive")
    if cfg["search_step"] <= 0:
        raise ValueError("search_step must be positive")

    set_seed(cfg["seed"])
    run_dir = ensure_dir(os.path.join(cfg["output_root"], "train_{}".format(cfg["train_size"])))
    model, _ = train_matcher(
        selected_root=cfg["selected_root"],
        model_dir=ensure_dir(os.path.join(run_dir, "models")),
        train_size=cfg["train_size"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        negative_ratio=cfg["negative_ratio"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        compute_negative_label=cfg["negative_label_mode"] == "similarity",
    )
    selected_dir = ensure_dir(os.path.join(run_dir, "selected_triplets"))
    summary = run_full_dataset_inference(
        data_folder=cfg["data_folder"],
        save_folder=selected_dir,
        model=model,
        score_threshold=cfg["score_threshold"],
        max_test_slices=cfg["max_test_slices"],
        start_index=cfg["start_index"],
        infer_batch_size=cfg["infer_batch_size"],
        spatial_constraint=cfg["spatial_constraint"],
        search_step=cfg["search_step"],
        phase=phase,
    )
    path = save_selection_summary(summary, run_dir, cfg["train_size"], cfg["score_threshold"])
    print("Selection summary saved to {}".format(path))


if __name__ == "__main__":
    main()
