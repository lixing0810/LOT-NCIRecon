#!/usr/bin/env python
"""Select 10,000 local-OT candidate triplets before matcher training."""
import torch
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lot_ncirecon.config import load_config
from lot_ncirecon.ot.candidate import select_candidates_until_enough
from lot_ncirecon.utils.training import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/home/lixing/LOT-NCIRecon/configs/default.yaml")
    parser.add_argument("--phase", default=None, choices=["A", "V", "D"])
    parser.add_argument("--data-folder",default="/public/home/lixing/ct_cross_modal_ten_dose/train_reg_N_new_all_same")
    parser.add_argument("--output-root",default="/public/home/lixing/ct_cross_modal_ten_dose/tip_test")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--quality-threshold", type=float)
    parser.add_argument("--search-radius", type=int)
    parser.add_argument("--search-step", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg = dict(config.get("candidate", {}))
    phase = args.phase or cfg.get("phase", "A")

    for key, value in {
        "data_folder": args.data_folder,
        "output_root": args.output_root,
        "top_k": args.top_k,
        "quality_threshold": args.quality_threshold,
        "search_radius": args.search_radius,
        "search_step": args.search_step,
    }.items():
        if value is not None:
            cfg[key] = value

    if not cfg.get("data_folder"):
        raise ValueError("Set candidate.data_folder in YAML or pass --data-folder.")
    if not cfg.get("output_root"):
        raise ValueError("Set candidate.output_root in YAML or pass --output-root.")

    set_seed(cfg.get("seed", 2026))
    phase_output = os.path.join(cfg["output_root"], phase)
    summary = select_candidates_until_enough(
        data_folder=cfg["data_folder"],
        output_root=phase_output,
        phase=phase,
        top_k=cfg.get("top_k", 10000),
        quality_threshold=cfg.get("quality_threshold", 0.735),
        beta=cfg.get("beta", 0.5),
        beta_prime=cfg.get("beta_prime", 0.5),
        search_radius=cfg.get("search_radius", 20),
        search_step=cfg.get("search_step", 4),
    )
    print("Candidate root for stage 2: {}".format(summary["selected_root"]))


if __name__ == "__main__":
    main()
