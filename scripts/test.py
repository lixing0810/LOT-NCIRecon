#!/usr/bin/env python
"""Evaluate trained LOT-NCIRecon checkpoints for one or more CECT phases."""
import torch
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lot_ncirecon.config import load_config
from lot_ncirecon.evaluation import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phases", nargs="+",default="A", choices=["A", "V", "D"])
    parser.add_argument("--data-template")
    parser.add_argument("--checkpoint-template")
    parser.add_argument("--output-template")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--pixel-metrics-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg = dict(config.get("test", {}))
    phases = args.phases or config.get("phases", ["A", "V", "D"])
    for key in ("data_template", "checkpoint_template", "output_template", "batch_size", "max_files"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if len(phases) > 1:
        for key in ("data_template", "checkpoint_template", "output_template"):
            if "{phase}" not in cfg[key]:
                raise ValueError("test.{} must contain {{phase}} for multi-phase testing".format(key))

    for phase in phases:
        result = evaluate(
            data_dir=cfg["data_template"].format(phase=phase),
            checkpoint_path=cfg["checkpoint_template"].format(phase=phase),
            output_dir=cfg["output_template"].format(phase=phase),
            phase=phase,
            batch_size=cfg.get("batch_size", 2),
            max_files=cfg.get("max_files"),
            distribution_metrics=cfg.get("distribution_metrics", True)
            and not args.pixel_metrics_only,
            use_ema=cfg.get("use_ema", False),
        )
        print("[Phase {}] {}".format(phase, result["macro_patient_statistics"]))


if __name__ == "__main__":
    main()
