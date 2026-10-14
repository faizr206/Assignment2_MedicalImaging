# ENG 2440 Assignment 2 — Tumor Segmentation in MRI Volumes

**Student:** Faiz Rizki Ramadhan (25011106)
**Course:** ENG 2440 Medical Imaging & AI in Healthcare — Fall 2026
**Report:** submitted separately as a PDF via Moodle (GitHub link included in the report)

End-to-end, patch-based PyTorch pipeline for **binary whole-tumor segmentation** on the
[Medical Segmentation Decathlon Task01_BrainTumour](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar)
(484 labelled cases, 4 co-registered MRI channels). The notebook covers: NIfTI geometry inspection,
case-level splitting, geometry-safe preprocessing, uniform vs. tumor-aware patch sampling with a
lightweight 3D U-Net, overlapping sliding-window inference with probability averaging, connected-component
postprocessing, three-plane error analysis, and two bonus experiments (missing-modality stress test,
sliding-window seam test).

Headline result (final pipeline on the 73-case test split): **mean Dice 0.828**, median HD95 **5.0 mm**,
**0/73** empty predictions.

---

## Repository contents

```
Assignment2_MedicalImaging/
├── Assignment2_Faiz.ipynb              # Fully executed notebook (all outputs included)
├── assignment2_split.csv               # Supplied case-level split (train / validation / test)
├── final_summary_evaluation_on_test.csv# Final test-set metrics (also in output/)
├── requirements.txt                    # Python environment
├── ai_use_declaration.txt              # Required AI-use disclosure
├── README.md                           # This file
└── output/                             # All figures and CSVs referenced in the report
```

All implementation code lives in the single notebook (no separate `.py` modules are required).
**No MRI volumes, masks, or patches are stored in this repository** — download them as described below.

---

## 1. Environment

- Python **3.13.5**, CUDA-capable GPU (experiments ran on an NVIDIA RTX 5000 Ada; a smaller GPU works but is slower)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Data setup

1. Download `Task01_BrainTumour.tar` (~**7.6 GB**) from the
   [official MSD page](https://medicaldecathlon.com/dataaws/) or the
   [MONAI-hosted mirror](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar).
2. Extract it so the layout looks like this:

```
Task01_BrainTumour/
├── imagesTr/        # 4-channel MRI NIfTI volumes
├── labelsTr/        # segmentation NIfTI volumes
└── dataset.json
```

3. Keep `assignment2_split.csv` next to the notebook.

## 3. ⚠️ DATA_ROOT — the only line you need to change

Open `Assignment2_Faiz.ipynb` and edit the **first code cell** (under "### Loading Data"):

```python
DATA_ROOT = Path("/l/users/faiz.ramadhan/brain_tumor/Task01_BrainTumour")   # <-- CHANGE THIS to your extracted Task01_BrainTumour folder
```

This is the single marked `DATA_ROOT` variable; every later path is derived from it.
Then `Kernel → Restart & Run All`.

## 4. Exact run order

Execute the notebook **top-to-bottom**; the sections are already in this order:

1. Setup and split verification (paths, 484 cases, disjoint splits)
2. Volume inspection (shape, channel order, affine, orientation, spacing, label values)
3. Split statistics (case counts and reference tumor-volume distributions)
4. DataLoader batch verification (prints one `[C, D, H, W]` / `[1, D, H, W]` batch)
5. Training — uniform random patches
6. Training — tumor-aware patch sampling (same seed, init, and budget)
7. Validation evaluation of both samplers (full-volume Dice / HD95 / sensitivity / FP volume)
8. Sliding-window stride ablation (0% / 50% overlap vs. stride 24)
9. Postprocessing ablation (LCC, Morph-1 + LCC, Morph-2 + LCC) and final configuration selection
10. Final test evaluation + three-plane error analysis (best / median / worst cases)
11. Bonus experiments — missing-modality stress test, sliding-window seam test

## 5. Reproducibility

- **Random seed: 42** (Python `random`, NumPy, and PyTorch; deterministic cuDNN selected).
- Both sampling conditions use **identical initialization and training budget** (10 epochs × 50 updates, batch size 1).
- Model selection uses **validation evidence only**; test labels are loaded only in stage 10.

## 6. Hardware & runtime

| Stage | Approx. time (NVIDIA RTX 5000 Ada) |
|---|---|
| Training, each sampler (500 updates) | **[FILL IN — e.g., ~10 min]** |
| Sliding-window ablation, 3 strides × 73 volumes | 149.9 s / 166.7 s / 267.6 s |
| Postprocessing ablation, 4 configs × 73 volumes | ~150–190 s each |
| Final test inference (73 cases) | ~186 s total (~2.54 s/case) |
| Bonus: seam test (3 overlap settings) | 392.4 / 392.8 / 499.2 s |
| Bonus: missing-modality test (5 settings) | **[FILL IN]** |
| **Total notebook run** | **[FILL IN — e.g., ~45 min on RTX 5000 Ada]** |

CPU-only execution is possible but will take considerably longer.

## 7. Outputs

Everything the report references is regenerated into `output/`, e.g.:

- `loss_curve.png` — training/validation curves for both samplers (report Fig. 2)
- `stride_summary.csv`, `postprocessing_summary.csv` — ablation tables
- `final_metrics.csv`, `final_summary_evaluation_on_test.csv` — final test metrics (report Table VI)
- `best_median_worst_test_cases.png` — three-plane error analysis (report Fig. 4)
- `missing_modality_report.csv`, `overlap_report.csv`, `patch_border_prob_diff.png` — bonus results
- `sample_case.png`, `all_angle_view.png`, `split_volume_distribution.png`, `flair_augmentation_before_after.png` — inspection/preprocessing figures

## 8. Data licence & attribution

Task01_BrainTumour is derived from BraTS 2016/2017 (Menze et al., 2015) and distributed via the
Medical Segmentation Decathlon (Antonelli et al., 2022). Follow the dataset licence; cite both when
using this work. Do not redistribute the imaging data.

## 9. AI use

See [`ai_use_declaration.txt`](ai_use_declaration.txt) for the required disclosure of AI tools,
purposes, and affected sections.
