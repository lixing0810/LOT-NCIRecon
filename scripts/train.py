#!/usr/bin/env python
"""Train independent arterial, venous, and delayed reconstruction models."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lot_ncirecon.config import build_phase_config, load_config
from lot_ncirecon.trainers.reconstruction import train


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phases", nargs="+", default="A",choices=["A", "V", "D"])
    parser.add_argument("--train-data-template")
    parser.add_argument("--output-template")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    phases = args.phases or config.get("phases", ["A", "V", "D"])
    if args.train_data_template:
        config.setdefault("data", {})["train_template"] = args.train_data_template
    if args.output_template:
        config.setdefault("output", {})["template"] = args.output_template
    if len(phases) > 1:
        if "{phase}" not in config["data"]["train_template"]:
            raise ValueError("data.train_template must contain {phase} for multi-phase training")
        if "{phase}" not in config["output"]["template"]:
            raise ValueError("output.template must contain {phase} for multi-phase training")

    for phase in phases:
        print("\n[Phase {}] Starting independent training".format(phase))
        train(build_phase_config(config, phase))


if __name__ == "__main__":
    main()
