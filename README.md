# 🩻📊 Train–Infer–Visualize (Pediatric **Lateral** CXR Classification with Paired Reader Analysis + EigenCAM)

**One notebook, end-to-end** for research on **pediatric lateral chest X-ray (CXR)** classification: **train** with k-fold cross-validation (k=5), **select** the best fold by **lowest validation loss**, **infer** on a **locked independent test set**, **explain** with **EigenCAM**, and **compare** model performance against three radiologists using **paired** analysis across three cohorts: (All / Confirmed / Unconfirmed). 

> ⚠️ **Clinical use disclaimer:** Research use only. Not for diagnostic or therapeutic decisions.

---

## 🗂️ Table of Contents
- [🧭 Overview](#-overview)
- [⚙️ Install & Environment](#️-install--environment)
- [🚀 Quick Start](#-quick-start)
- [🗃️ Data & Layout](#️-data--layout)
- [🧪 Configuration](#-configuration)
- [▶️ Usage](#️-usage)
- [📁 Outputs & Artifacts](#-outputs--artifacts)
- [🎯 Reproducibility](#-reproducibility)
- [🛠️ Troubleshooting](#️-troubleshooting)

---

## 🧭 Overview
- 🧱 **Frameworks:** PyTorch, PyTorch Lightning, `timm`, Albumentations, scikit-learn, matplotlib.
- 🧠 **Backbone:** DPN-68 (`dpn68_*`) for binary **lateral** CXR classification.
- 🔂 **Cross-Validation:** 5-fold with EarlyStopping + ModelCheckpoint (best by `val_loss`).
- ✅ **Best-fold selection:** choose the checkpoint with the **lowest `val_loss`** among the five fold models.
- ✅ **Locked test inference:** run the chosen best model on the independent/locked test set.
- 📈 **Metrics:** Balanced Accuracy, Sensitivity/Specificity, Precision, F1, MCC, Cohen’s κ, Youden’s J, ROC-AUC.
- 🩻 **Reader Study (paired):** compare **model vs each radiologist** on the **same test cases** under **All / Confirmed / Unconfirmed** cohorts.
- 🧾 **Paired stats:** McNemar (overall + TB-positive subset), paired bootstrap 95% CI for ΔYouden’s J at τ=0.5, Holm–Bonferroni for secondary comparisons.
- 📈 **Clinical utility:** Decision Curve Analysis (DCA) for Model vs Treat-All vs Treat-None (+ readers).
- 🤝 **Reliability:** Pairwise Cohen’s κ (reader↔reader and model↔reader) + Fleiss’ κ among the three radiologists.
- 🔥 **Explainability:** EigenCAM-based visualizations (thresholded CAM masks + overlays + contours/bboxes) on the **test set**.

---

## ⚙️ Install & Environment
```bash
# 1) Create and activate environment (example with conda)
conda create -n pediatric-lateral-cxr python=3.11 -y
conda activate pediatric-lateral-cxr

# 2) Install requirements
pip install -r requirements.txt
```

---
## 🚀 Quick Start
1. 📓 Open the notebook: train_infer_visualize.ipynb and run cells in order.
2. ⚙️ Edit hyperparameters: update the h dictionary (model, img_size, batch_size, n_splits=5, output_dir, test_dir, etc.).
3. 🏋️ Train with 5-fold CV: best checkpoint per fold saved by lowest validation loss (val_loss).
4. ✅ Select best model: choose the checkpoint with the lowest val_loss across the five folds.
5. ✅ Locked test inference: saves metrics and ROC curves for the locked test set.
6. 🩻 Paired reader comparison: computes metrics for model vs each radiologist on the same test cases under All / Confirmed / Unconfirmed cohorts.
7. 🧩 Discordance analysis: exports discordance squares for accuracy (all cases) and sensitivity (TB+ cases).
8. 📈 Clinical utility (DCA): exports net-benefit curves for Model vs Treat-All vs Treat-None (+ readers).
9. 🤝 Reliability: computes pairwise Cohen’s κ (model↔reader, reader↔reader) + Fleiss’ κ among radiologists.
10. 🔥 EigenCAM (test-set only): exports CAM heatmaps/contours/bounding boxes at original resolution.

✅ Metadata CSV: rad_predictions.csv (replaces image_data_github.csv from the frontal study).

---

## 🗃️ Data & Layout
```
project/
├─ train_infer_visualize.ipynb
├─ data/
│  ├─ train/
│  │  ├─ normal/
│  │  └─ normal_not/
│  └─ test/
│     ├─ normal/
│     └─ normal_not/
└─ rad_predictions.csv   # metadata (filenames, cohorts, reader cols)
```
- 🩻 Images are lateral CXRs.
- 📄 CSV fields typically include: original, Type (Cases/Controls), MC_status (Confirmed/Unconfirmed/Blanks), and reader predictions: alopezca_prediction, lamorenog_prediction, rdmontoyac_prediction.
- 🧭 Class convention: class_names = ["normal", "normal_not"]; positive class = normal_not (TB-related abnormality). Ensure folders and CSV agree.

---

## 🧪 Configuration
Edit the h dictionary near the top of the notebook:
- 🧠 Model: model (DPN-68), dropout, num_classes=2
- 🏋️ Training: img_size, batch_size, num_workers, lr, max_epochs, patience, n_splits=5
- ⚖️ Imbalance: balance=True → WeightedRandomSampler
- 💾 IO: test_dir, output_dir
- 🔬 EigenCAM: CAM_METHOD="eigencam", heatmap alpha, contour thickness, save masks/bboxes

---

## ▶️ Usage
- **Notebook flow:**
  - 🏋️ CV training → ✅ best-fold selection (min val_loss) → ✅ locked test inference
  - 🩻 paired model↔reader analysis (All / Confirmed / Unconfirmed)
  - 📈 DCA → 🤝 reliability (Cohen’s κ + Fleiss’ κ) → 🔥 EigenCAM (test only) → 🔥 Explanation → 👩‍⚕️👨‍⚕️ Reader comparison.

**Script mode (optional):**
  ```bash
  jupyter nbconvert --to script train_infer_visualize.ipynb
  python train_infer_visualize.py
  ```

---

## 📁 Outputs & Artifacts
```
{output_dir}/
├─ fold_*/{model}/
│  ├─ *.ckpt
│  ├─ validation_inference/ (confusion_matrix.png, metrics.csv, roc.png)
│  └─ test_inference/       (confusion_matrix.png, metrics.csv, roc.png)
├─ {model}/
│  ├─ all_folds_validation_metrics.csv
│  ├─ combined_validation_roc.png
│  ├─ test_metrics.csv
│  ├─ test_roc.png
│  ├─ eigencam/…            # heatmaps, masks, contours, bboxes (TEST ONLY)
│  └─ comparison_to_radiologists/
│     ├─ All/
│     ├─ Confirmed/
│     └─ Unconfirmed/
│        ├─ paired_metrics_tables.csv
│        ├─ mcnemar_stats.csv
│        ├─ bootstrap_dyoudenj_ci.csv
│        ├─ discordance_squares_accuracy.png
│        ├─ discordance_squares_sensitivity.png
│        ├─ dca_curves.png
│        ├─ kappa_pairwise.csv
│        └─ fleiss_kappa.csv
```

---

## 🎯 Reproducibility
- 🎲 Global seeds for Python/NumPy/PyTorch (CPU & CUDA).
- ⚙️ Deterministic cuDNN where appropriate; TF32 enabled for matmuls.
- 🧮 Mixed precision (bf16) during training/inference where supported.
- 🧴 Consistent Albumentations transforms (CLAHE for train; ImageNet mean/std).
- 🔒 Locked test set: no threshold tuning on test; select best fold using validation only.

---

## 🛠️ Troubleshooting
- ❓ No checkpoints found: verify fold setup and that training completed; filenames follow the template in the notebook.
- 🔁 CSV–folder mismatch: ensure original names align with files in data/*/* and cases map to normal_not.
- 🧪 Cohort filtering wrong: confirm MC_status values are exactly Confirmed / Unconfirmed (case-insensitive handling recommended).
- 🔬 EigenCAM target not found: ensure the DPN backbone exposes Conv2d layers; the helper selects the last Conv2d in model.backbone.
- 🧠 Out-of-memory: reduce batch_size / img_size, disable contour/bbox export, or run EigenCAM on a subset first.
---