# Project Details: Methodology, Full Results, and Critical Assessment

Deep-dive companion to the [main README](../README.md). All modeling numbers here are produced by [`scripts/run_leakage_experiment.py`](../scripts/run_leakage_experiment.py) (which writes [`results/metrics/model_comparison.csv`](../results/metrics/model_comparison.csv) and the stage-4 figures); EDA numbers come from the notebooks in [`notebooks/`](../notebooks/).

---

## 1. Dataset

### Origin

The Poqemon QoE dataset comes from the **PoQeMoN Project** (Platform Quality Evaluation of Mobile Networks), a crowdsourcing campaign by the LiSSi laboratory, Paris Est Creteil University (UPEC), France:

- **Participants**: 181 testers (researchers, students, families), ages 19-38
- **Networks**: 4 French mobile operators (Orange, SFR, Bouygues, Free)
- **Platforms**: 9 Android devices with VLC media player
- **Collection**: Around Paris (lab, train stations, walking)

### Contents

- **Samples**: 1,543 video viewing sessions in the distributed CSV (original documentation cites 1,560; the -1% discrepancy is documented in notebook 01)
- **Features**: 23 columns across 5 categories
- **Target**: MOS (Mean Opinion Score), 1-5
- **Missing values**: none

| MOS | Label | Count | Share |
| ----- | ------- | ------- | ------- |
| 1 | Bad | 93 | 6.0% |
| 2 | Poor | 118 | 7.6% |
| 3 | Fair | 246 | 15.9% |
| 4 | Good | 784 | 50.8% (majority class) |
| 5 | Excellent | 302 | 19.6% |

The 50.8% majority class defines the baseline every model must beat.

### Feature categories

**QoA — Video Quality of Application (8 features, from VLC)**
`QoA_VLCresolution` (240p/360p), `QoA_VLCbitrate` (kbps), `QoA_VLCframerate` (fps), `QoA_VLCdropped` (dropped frames), `QoA_VLCaudiorate` (kbps), `QoA_VLCaudioloss` (lost packets), `QoA_BUFFERINGcount`, `QoA_BUFFERINGtime` (ms). The two buffering features turn out to be the most important predictors.

**QoS — Network Quality of Service (2 features)**
`QoS_type` (1=EDGE, 2=UMTS, 3=HSPA, 4=HSPAP, 5=LTE), `QoS_operator` (1=SFR, 2=Bouygues, 3=Orange, 4=Free).

**QoD — Device (3 features)**
`QoD_model`, `QoD_os-version`, `QoD_api-level`.

**QoU — User (3 features)**
`QoU_sex`, `QoU_age`, `QoU_Ustedy` (education; 92.4% identical values, so near-zero variance).

**QoF — User Feedback (4 features) — the data-leakage risk**
`QoF_begin`, `QoF_shift`, `QoF_audio`, `QoF_video`. These are subjective ratings the user gave during the session. `QoF_audio` correlates r=0.841 with MOS and `QoF_video` r=0.689 — they measure the same subjective perception as the target. Including them builds a "predict subjective from subjective" model that cannot be deployed, because in production you do not have the user's ratings.

---

## 2. Stage-by-Stage Methodology

### Stage 1 — Data understanding (notebook 01)

Validated the CSV against the official documentation: 1,543 rows (vs 1,560 documented), 0 missing values, severe class imbalance (MOS=4 at 50.8%).

### Stage 2 — Exploratory data analysis (notebook 02)

**Top correlations with MOS:**

| Feature | r | Reading |
| --------- | ----- | --------- |
| QoF_audio | +0.841 | Suspiciously high — leakage signal |
| QoF_video | +0.689 | Leakage signal |
| QoA_VLCframerate | +0.544 | Moderate positive |
| QoA_BUFFERINGtime | -0.482 | Strong negative — key predictor |
| QoA_BUFFERINGcount | -0.411 | Moderate negative — key predictor |

**Network type effect** (one-way ANOVA F=37.47, p<0.001):

| Network | Mean MOS | n |
| --------- | ---------- | ----- |
| HSPA+ (HSPAP, code 4) | 3.84 | 572 |
| LTE (code 5) | 3.78 | 473 |
| UMTS (code 2) | 3.64 | 399 |
| HSPA (code 3) | 3.26 | 72 |
| EDGE (code 1) | 1.56 | 27 |

*Labels follow the official code mapping in `data/raw/pokemon.names` (1=EDGE, 2=UMTS, 3=HSPA, 4=HSPAP, 5=LTE); per-code means recomputed directly from the raw CSV. Note the original notebook writeup labeled code 4 as "HSPA" — the mapping above is the verified one.*

**Buffering threshold**: sessions with 0-2 buffering events average MOS >= 3.6; sessions with 3+ events fall below MOS 2.4.

**Counterintuitive findings** worth noting: video bitrate correlates only r=0.090 with MOS, and resolution slightly negatively (r=-0.022) — plausibly because adaptive streaming pushes higher resolution onto connections that then buffer more.

### Stage 3 — Preprocessing (notebook 03, reproduced by the script)

**Removed 4 columns**: `id`, `user_id` (identifiers), `QoD_model`, `QoD_os-version` (high cardinality). `QoS_operator` is one-hot encoded into 4 operator columns.

**Engineered 4 features** from EDA insights:

- `Buffering_Severity` = count x time / 1000
- `Network_Generation`: 5 network types grouped into 2G/3G/4G
- `Video_Quality_Index`: weighted composite of bitrate (0.4), framerate (0.3), resolution (0.3)
- `Audio_Quality`: audio rate adjusted by loss percentage

Result: 25 features (tracked in `data/processed/feature_names.txt`; the script asserts its engineered columns match that file exactly).

**Two feature variants** — this is the leakage experiment design:

- **Objective (19 features)**: excludes all four `QoF_*` ratings (leakage) plus two near-constant columns — `QoU_Ustedy` (92.4% identical values) and `QoA_VLCresolution` (95.6% of sessions at 360p) — what a deployed system would see
- **Full (25 features)**: everything included — benchmark to quantify the inflation

**Split and scaling**: 80/20 stratified split (1,234 train / 309 test), seed 42, identical row assignment for both variants, StandardScaler fitted on training data only (used by Logistic Regression; trees consume unscaled features).

### Stage 4 — Modeling (scripts/run_leakage_experiment.py)

Models: majority-class baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting. Class imbalance is handled with `class_weight='balanced'` for the first three; `GradientBoostingClassifier` has no `class_weight` parameter, so it is fitted with per-sample balanced weights via `compute_sample_weight('balanced')`. All other hyperparameters are scikit-learn library defaults (`LogisticRegression` `max_iter` raised to 1000 for convergence); no grid search (see Limitations). Seed 42 everywhere.

Metrics: accuracy, macro F1 (per-class F1 averaged with equal class weight), Cohen's kappa (chance-corrected agreement, appropriate for ordinal MOS), and the train-test gap as an overfitting diagnostic.

---

## 3. Full Results

Source: [`results/metrics/model_comparison.csv`](../results/metrics/model_comparison.csv), written by [`scripts/run_leakage_experiment.py`](../scripts/run_leakage_experiment.py).

| Model | Features | Train Acc | Test Acc | F1 (macro) | Kappa | Train-Test Gap |
| ------- | ---------- | ----------- | ---------- | ----------- | ------- | ---------------- |
| Baseline (majority) | — | 50.8% | 50.8% | 0.135 | 0.000 | 0.0 |
| Logistic Regression | Objective | 47.4% | 45.0% | 0.430 | 0.234 | 2.4 |
| Decision Tree | Objective | 100.0% | 46.6% | 0.400 | 0.197 | 53.4 |
| Random Forest | Objective | 98.5% | 46.9% | 0.429 | 0.241 | 51.5 |
| **Gradient Boosting** | **Objective** | 81.4% | **48.2%** | **0.442** | **0.261** | 33.1 |
| Logistic Regression | Full (leakage) | 77.8% | 77.7% | 0.727 | 0.679 | 0.1 |
| Decision Tree | Full (leakage) | 100.0% | 75.7% | 0.722 | 0.638 | 24.3 |
| **Random Forest** | **Full (leakage)** | 96.8% | **81.6%** | **0.793** | **0.735** | 15.2 |
| Gradient Boosting | Full (leakage) | 93.4% | 79.6% | 0.763 | 0.705 | 13.7 |

### Leakage evidence

The gap replicates across all four model families, so it is a property of the features, not of one model:

- Logistic Regression: 45.0% (objective) → 77.7% (full) = **+32.7 points**
- Decision Tree: 46.6% (objective) → 75.7% (full) = **+29.1 points**
- Random Forest: 46.9% (objective) → 81.6% (full) = **+34.6 points**
- Gradient Boosting: 48.2% (objective) → 79.6% (full) = **+31.4 points**
- Best objective (48.2%) vs best full (81.6%) = **33.3 points**

The full models generalize well (train-test gaps of 0.1-24.3, and even Logistic Regression reaches 77.7%) — but they generalize from subjective ratings to subjective ratings, which is exactly why the 81.6% number is misleading as a deployment estimate.

### Why the honest accuracy sits below the baseline

Every objective model lands **below** the 50.8% majority baseline on raw accuracy (45.0-48.2%). This is the direct consequence of balanced class weighting: the models are pushed to detect the four minority classes instead of defaulting to "Good", so they give up majority-class hits. The chance-corrected metrics tell the real story — best objective macro F1 0.442 vs the baseline's 0.135, kappa 0.261 vs 0.000. A majority-class predictor "wins" on accuracy while detecting nothing. Model choice matters little on objective features (macro F1 spread 0.400-0.442 across four families); the feature set is the binding constraint.

### Overfitting

At library defaults the trees memorize the 1,234 training samples: Decision Tree 100% train / 46.6% test (53.4-point gap), Random Forest 98.5% / 46.9% (51.5), Gradient Boosting 81.4% / 48.2% (33.1). The model ranking is based on test-set metrics only, so it is not affected, but the absolute 48.2% comes from an under-regularized, untuned model — hyperparameter search and stronger regularization were deliberately left out of scope (see Limitations).

---

## 4. Feature Importance (Random Forest, objective)

Source: `scripts/run_leakage_experiment.py` run output; figure: `results/figures/04_modeling_and_evaluation/6_feature_importance_rf.png`.

| Rank | Feature | Importance |
| ------ | --------- | ------------ |
| 1 | QoA_BUFFERINGtime | 17.8% |
| 2 | Buffering_Severity | 13.0% |
| 3 | QoA_VLCframerate | 10.5% |
| 4 | Video_Quality_Index | 10.4% |
| 5 | QoA_VLCbitrate | 8.6% |
| 6 | QoU_age | 7.3% |
| 7 | QoA_VLCaudiorate | 6.8% |
| 8 | Audio_Quality | 6.5% |
| 9 | QoA_BUFFERINGcount | 3.1% |
| 10 | QoD_api-level | 3.1% |

- **Buffering dominates**: the three buffering features (raw time 17.8% + severity 13.0% + count 3.1%) sum to **33.8%** of total importance.
- **Feature engineering paid off**: 3 of the top 10 are engineered features (Buffering_Severity, Video_Quality_Index, Audio_Quality).
- **Network type is indirect**: despite ANOVA significance, QoS_type (2.4%) and Network_Generation are not in the top 10 — poor networks cause buffering, and the model learns from buffering directly.

---

## 5. Per-Class Performance (Gradient Boosting, objective — the best objective model)

Confusion matrix (test set, actual x predicted):

```text
           Bad  Poor  Fair  Good  Excellent
Bad         16     3     0     0      0
Poor         9     4     5     3      3
Fair         1     5    19    16      8
Good         0     6    23    84     44
Excellent    1     0     8    25     26
```

| Actual MOS | Recall |
| ------------ | -------- |
| 1 (Bad) | 84.2% (16/19) |
| 2 (Poor) | 16.7% (4/24) |
| 3 (Fair) | 38.8% (19/49) |
| 4 (Good) | 53.5% (84/157) |
| 5 (Excellent) | 43.3% (26/60) |

Balanced class weighting changes the failure profile: instead of collapsing onto the majority class, the model spreads errors across **adjacent classes**. Bad sessions are caught reliably (84.2% — buffering makes them technically distinctive), while Poor is the weakest class (16.7%, mostly confused with Bad and Fair). The Good/Excellent boundary blurs in both directions: 44 of 157 true Good sessions are predicted Excellent and 25 of 60 true Excellent sessions are predicted Good — technically, a Good and an Excellent session look nearly identical (both stream smoothly with no buffering); the difference lives in subjective perception. The full model (Random Forest with QoF_* features) lifts MOS=5 recall to 93.3% and MOS=2 recall to 62.5% — further confirmation that the QoF features carry the subjective signal.

---

## 6. Critical Assessment

### What holds up

- Stratified split, scaler fitted on train only, seed fixed
- The two-variant design turns "is there leakage?" into a measured number instead of a suspicion
- Results are reported with train-test gaps and per-class recalls, not just headline accuracy

### Limitations

1. **48.2% is modest — and below the majority baseline on raw accuracy.** Balanced class weights trade majority-class accuracy for minority-class recall (kappa 0.261 and macro F1 0.442 clearly beat the baseline, accuracy does not); a ~52% raw error rate rules out high-stakes automation either way.
2. **Severe overfitting at defaults.** 33.1-53.4-point train-test gaps on the objective trees; no GridSearchCV or regularization tuning was performed (scope decision).
3. **The crowded middle is not predictable from objective metrics.** Poor recall is 16.7% and Good 53.5%; adjacent-class confusion (Poor/Fair, Good/Excellent) is likely a ceiling of the feature set, not just the model.
4. **Class imbalance only partially addressed.** Balanced class/sample weights were used; SMOTE and per-class threshold optimization were not attempted.
5. **No interaction terms** (e.g. buffering x network generation) or polynomial features were explored.

### Dataset-level caveats

- **Age**: 2015 data — pre-5G networks, 95% of sessions at 360p, older codecs
- **Geography**: Paris region only, 4 French operators
- **Devices**: Android only, 9 models
- **Population**: mostly researchers/students, ages 19-38

### Deployment recommendation

Suitable as a **soft decision-support signal**: trend monitoring, A/B comparison of network configurations, early-warning flags for human review (Bad-session recall is 84.2%). Not suitable for automated SLA enforcement, complaint prediction, or any decision where a ~52% raw error rate is unacceptable.

### Possible extensions

SMOTE or other resampling for minority classes, hyperparameter search on Gradient Boosting, per-class decision thresholds, interaction features, XGBoost/stacking, and — most importantly — more and newer data (the 1,543-sample size is the binding constraint).

---

## 7. References

1. Amour, L., Souihi, S., Hoceini, S., & Mellouk, A. (2015). "Building a Large Dataset for Model-based QoE Prediction in the Mobile Environment." *MSWiM '15*, 313-317.
2. Moteau, S., Guillemin, F., & Houdoin, T. (2017). "Correlation between QoS and QoE for HTTP YouTube content in Orange cellular networks." *SAR-SSI 2017*.

Dataset: PoQeMoN project, LiSSi laboratory, Paris Est Creteil University (UPEC), France. Contact: lamine.amour@u-pec.fr

## Appendix: Metric Notes

**Cohen's kappa** = (p_o - p_e) / (1 - p_e), agreement corrected for chance. 0.21-0.40 is conventionally "fair agreement"; the best objective model scores 0.261. **Macro F1** averages per-class F1 scores, weighting all five MOS classes equally — more informative than accuracy under this imbalance (the majority-class baseline gets 50.8% accuracy but only 0.135 macro F1).
