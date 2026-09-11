# Reproduced arterial-phase results

These metrics were recomputed from the validated `latest.pth` checkpoint on 2026-08-19. The evaluation used two anonymized test subjects and 657 unique 512×512 slices.

| Aggregation | PSNR (dB) | SSIM | RMSE |
|---|---:|---:|---:|
| Pooled slices | 36.4908 ± 3.2144 | 0.9583 ± 0.0446 | 0.01627 ± 0.01064 |
| Macro patients | 36.4616 ± 0.5815 | 0.9582 ± 0.0019 | 0.01629 ± 0.00030 |

Patient-macro distribution metrics were FID `20.8698 ± 1.5559`, sFID `0.03740 ± 0.00558`, and KID `0.002767 ± 0.000216`.

PSNR, SSIM and RMSE were computed after clipping predictions to `[0,1]`; PSNR and SSIM used `data_range=1.0`. FID/KID used ImageNet Inception-v3 pool3 features, while sFID used global-average-pooled Mixed_6e features.

The source directory contained both patient subdirectories and duplicate files at its root. Following the original test script, the evaluator prioritizes patient subdirectories and ignores duplicate root copies. The first uncorrected recursive scan contained 1,314 paths and was discarded; the reported results use exactly 657 unique slices.

No patient names, clinical identifiers, or source paths are included in the repository artifacts.
