#!/usr/bin/env python
"""Smoke-test the two-stage OT patch pipeline on a single raw slice file."""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lot_ncirecon.data import OTTripletPatchDataset
from lot_ncirecon.ot.candidate import select_candidates_until_enough
from lot_ncirecon.ot.reporting import save_selection_summary
from lot_ncirecon.ot.selector import run_full_dataset_inference
from lot_ncirecon.trainers.matcher import train_matcher
from lot_ncirecon.utils.training import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, help="One raw .npy slice dictionary.")
    parser.add_argument("--work-dir", default="outputs/smoke_ot")
    parser.add_argument("--phase", default="A", choices=["A", "V", "D"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--candidate-threshold", type=float, default=-1.0)
    parser.add_argument("--selection-threshold", type=float, default=-1.0)
    parser.add_argument("--spatial-constraint", type=int, default=50)
    return parser.parse_args()


def prepare_three_slice_folder(sample_path, raw_dir):
    ensure_dir(raw_dir)
    for i in range(3):
        dst = os.path.join(raw_dir, "slice_{:03d}.npy".format(i))
        shutil.copyfile(sample_path, dst)


def main():
    args = parse_args()
    set_seed(2026)

    work_dir = ensure_dir(args.work_dir)
    raw_dir = os.path.join(work_dir, "raw_three_slices")
    prepare_three_slice_folder(args.sample, raw_dir)

    candidate_summary = select_candidates_until_enough(
        data_folder=raw_dir,
        output_root=os.path.join(work_dir, "candidates", args.phase),
        phase=args.phase,
        top_k=args.top_k,
        quality_threshold=args.candidate_threshold,
    )
    selected_root = candidate_summary["selected_root"]
    print("Stage-1 candidates: {}".format(candidate_summary["num_selected"]))
    if candidate_summary["num_selected"] < args.train_size:
        raise RuntimeError("Not enough candidate triplets for matcher training.")

    model, _ = train_matcher(
        selected_root=selected_root,
        model_dir=ensure_dir(os.path.join(work_dir, "matcher")),
        train_size=args.train_size,
        epochs=args.epochs,
        batch_size=args.train_size,
        num_workers=0,
    )

    final_dir = ensure_dir(os.path.join(work_dir, "selected_triplets"))
    selection_summary = run_full_dataset_inference(
        data_folder=raw_dir,
        save_folder=final_dir,
        model=model,
        score_threshold=args.selection_threshold,
        max_test_slices=1,
        infer_batch_size=256,
        spatial_constraint=args.spatial_constraint,
        phase=args.phase,
    )
    save_selection_summary(selection_summary, work_dir, args.train_size,
                           args.selection_threshold)
    print("Stage-2 selected patches: {}".format(selection_summary["num_saved"]))

    dataset = OTTripletPatchDataset(final_dir, crop_size=64, max_files=1)
    low, target, ncct, weight = dataset[0]
    print("Reconstruction dataset tensors:")
    print("  ncct={}, low={}, target={}, weight={}".format(
        tuple(ncct.shape),
        tuple(low.shape),
        tuple(target.shape),
        float(weight),
    ))


if __name__ == "__main__":
    main()
