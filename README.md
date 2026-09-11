# LOT-NCIRecon

Official PyTorch implementation of "Non-Contrast-Informed Low-Dose Multi-Phase Contrast-Enhanced CT Reconstruction via Local Optimal Transport."

LOT-NCIRecon uses a two-stage patch-selection pipeline before reconstruction training:

1. `scripts/first_select.py` solves handcrafted local OT and saves 10,000 high-quality candidate NCCT/LD-CECT/ND-CECT patch triplets.
2. `scripts/select_patches.py` trains a calibrated neural OT matcher on those candidates and applies it to the full raw slice set to output all selected patch triplets.

The final triplets are used to train the unfolded reconstruction network independently for arterial (A), venous (V), and delayed (D) phases.

## Project Structure

```text
LOT-NCIRecon/
|-- configs/
|   `-- default.yaml                 # data, OT, training, and test settings
|-- lot_ncirecon/
|   |-- config.py                    # YAML loading and phase path resolution
|   |-- data.py                      # reconstruction dataset and normalization
|   |-- evaluation.py                # checkpoint loading and test metrics
|   |-- losses.py                    # reconstruction, gradient, SSIM, and GAN losses
|   |-- models/
|   |   |-- discriminator.py         # image/gradient discriminators
|   |   |-- matcher.py               # calibrated neural OT matcher
|   |   `-- reconstruction.py        # unfolded LOT-NCIRecon generator
|   |-- ot/
|   |   |-- candidate.py             # stage-1 local-OT candidate selection
|   |   |-- data.py                  # OT matcher pair dataset
|   |   |-- metrics.py               # SSIM/radiodensity similarity
|   |   |-- reporting.py             # selection summary export
|   |   `-- selector.py              # stage-2 full-data patch selection
|   |-- trainers/
|   |   |-- matcher.py               # neural OT matcher trainer
|   |   `-- reconstruction.py        # reconstruction trainer
|   `-- utils/
|       |-- training.py              # seed, EMA, checkpoint, and LR utilities
|       `-- visualization.py         # monitor image generation
|-- scripts/
|   |-- first_select.py              # select 10,000 candidate triplets
|   |-- select_patches.py            # train matcher and output all selected patches
|   |-- smoke_ot_pipeline.py         # one-sample end-to-end OT smoke test
|   |-- train.py                     # independent A/V/D reconstruction training
|   `-- test.py                      # reconstruction evaluation
|-- tests/
|   `-- test_config.py
|-- results/A/summary.json           # anonymized A-phase test summary
|-- MODEL.md                         # checkpoint metadata
|-- pyproject.toml
`-- requirements.txt
```

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/lixing0810/LOT-NCIRecon.git
cd LOT-NCIRecon
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build matching your CUDA runtime if the default wheel is not appropriate.

## Configuration

All reproducibility settings are collected in `configs/default.yaml`. Important defaults include:

- phases: `A`, `V`, and `D`;
- stage-1 candidate count: `10000`;
- stage-1 quality threshold: `0.735`;
- stage-1 search radius: `20` pixels;
- stage-2 local spatial constraint: `C = 50` pixels;
- patch size: `64 x 64`;
- reconstruction training iterations per phase: `100000`.

The default raw training path is `/public/home/lixing/ct_cross_modal_ten_dose/train_reg_N_new_all_same`. Delayed phase `D` maps to `L_LOW/L_NORMAL` in the clinical `.npy` files.

## Data Format

Each raw slice `.npy` file stores a Python dictionary with phase-specific arrays, for example:

```python
{
    "N": np.ndarray,
    "A_LOW": np.ndarray,
    "A_NORMAL": np.ndarray,
    "V_LOW": np.ndarray,
    "V_NORMAL": np.ndarray,
    "L_LOW": np.ndarray,
    "L_NORMAL": np.ndarray,
}
```

Saved patch triplets use generic training keys:

```python
{
    "A_LOW": np.ndarray,       # low-dose CECT patch for the selected phase
    "A_NORMAL": np.ndarray,    # normal-dose CECT target for the selected phase
    "N": np.ndarray,           # NCCT guidance patch
    "quality_score": float,
}
```

Clinical patient data are not distributed because of privacy and institutional restrictions.

## 1. Select Candidate Patches

Run the first local-OT pass to obtain 10,000 high-quality candidate triplets:

```bash
python scripts/first_select.py \
  --config configs/default.yaml \
  --phase A
```

Default output:

```text
outputs/candidates/A/beta_0p50_betaprime_0p50/
```

## 2. Train Matcher And Output Patches

Use the 10,000 candidate triplets to train the neural OT matcher, then run full-data selection:

```bash
python scripts/select_patches.py \
  --config configs/default.yaml \
  --phase A \
  --selected-root outputs/candidates/A/beta_0p50_betaprime_0p50 \
  --spatial-constraint 50
```

If `--selected-root` is omitted, `select_patches.py` derives it from `candidate.output_root`, `phase`, `beta`, and `beta_prime`.

The second-stage candidate centers satisfy the paper-consistent constraint:

```text
||r_j - r_i||_2 <= C,  C = 50 pixels.


## 3. Train Reconstruction Models

Organize final selected triplets as follows, or change `data.train_template`:

```text
data/selected/
|-- A/
|-- V/
`-- D/
```

Train all three phases independently:

```bash
python scripts/train.py --config configs/default.yaml
```

Train a subset of phases:

```bash
python scripts/train.py --config configs/default.yaml --phases A
python scripts/train.py --config configs/default.yaml --phases V D
```

Each phase writes checkpoints, CSV logs, and monitor images under `outputs/reconstruction/<phase>/`.

## 4. Test Checkpoints

Set `test.data_template` and `test.checkpoint_template`, then run:

```bash
python scripts/test.py --config configs/default.yaml
```

The evaluator writes anonymized per-slice PSNR/SSIM/RMSE and a JSON summary. With `test.distribution_metrics: true`, it also computes patient-level FID, sFID, and KID using ImageNet Inception-v3 features.

## Reproducibility Notes

- Default random seed: `2026`.
- Set GPU visibility externally, e.g. `CUDA_VISIBLE_DEVICES=0`.
- Set `data.rebuild_manifest: true` after adding or removing `.npy` files.
- Dataset paths, patient data, generated outputs, checkpoints, and credentials are excluded by `.gitignore`.

## Citation

Please cite the associated paper if this repository is useful in your research.

```bibtex
@ARTICLE{11674387,
  author={Li, Xing and Wang, Miaomiao and Zhang, Baoping and Zhu, Shumeng and Jin, Chao and Yang, Yan and Yang, Jian and Ma, Jianhua},
  journal={IEEE Transactions on Image Processing},
  title={Non-Contrast-Informed Low-Dose Multi-Phase Contrast-Enhanced CT Reconstruction via Local Optimal Transport},
  year={2026},
  pages={1-1},
  doi={10.1109/TIP.2026.3727893}
}
```

## License

This project is released under the MIT License. See `LICENSE`.
