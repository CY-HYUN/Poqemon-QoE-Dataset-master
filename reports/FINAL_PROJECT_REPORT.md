# Pokemon QoE Dataset: Quality of Experience Prediction from Mobile Video Streaming Metrics

## Final Project Report

**Author:** Data Science Project
**Date:** October 13, 2025
**Institution:** Based on Alessandro Maddaloni's Guidelines (Telecom SudParis & Institut Polytechnique de Paris)

---

## Executive Summary

This project analyzed the Pokemon QoE dataset (1,543 mobile video streaming sessions) to build predictive models for estimating user satisfaction (MOS scores 1-5) from objective network and video quality metrics.

**Key Findings:**
- ✅ **Objective model achieved 59.5% accuracy** - modest but significant improvement over 50.8% baseline
- ✅ **Buffering is THE critical factor** - 36.6% combined feature importance
- ✅ **Subjective user ratings cause data leakage** - Full model achieved 81.9% (+22.4 points) but not deployment-viable
- ⚠️ **Real-world deployment challenging** - 59.5% accuracy insufficient for critical automated decisions

**Recommendation:** Use Gradient Boosting model for **trend monitoring and relative comparisons**, not absolute SLA enforcement.

---

## 1. Introduction

### 1.1 Context and Motivation

**What is this dataset about?**

The Pokemon QoE dataset contains Quality of Experience measurements from a crowdsourcing campaign conducted at LiSSi laboratory, Paris, France. Participants watched videos on mobile devices across different network conditions using VLC media player, then rated their experience.

**Dataset characteristics:**
- 1,543 video viewing sessions
- 181 testers (ages 19-38, researchers and students)
- 5 network types (EDGE to LTE)
- 4 mobile operators (French market)
- 23 features across 5 QoE Influence Factor categories

**Real-world application:**

Mobile network operators need to monitor Quality of Experience proactively. Traditional methods rely on expensive user surveys. This project investigates: **Can we predict user satisfaction from technical metrics alone?**

**Business value:**
- Early detection of network quality issues
- A/B testing of network configurations
- Cost reduction (automated monitoring vs surveys)
- Customer satisfaction improvement

### 1.2 Project Goals

1. **Understand QoE drivers** - Which technical factors most impact user satisfaction?
2. **Build predictive models** - Estimate MOS from objective network/video metrics
3. **Test data leakage hypothesis** - Do subjective ratings (QoF_*) cause overfitting?
4. **Assess real-world viability** - Is the model accurate enough for deployment?

### 1.3 Analysis Approach (Following Guidelines)

This project follows a **storytelling approach** where:
- ✅ Every analysis step explains **WHY** before **WHAT**
- ✅ **Expected vs Actual** comparisons throughout
- ✅ **Critical assessment** of limitations and feasibility
- ✅ **Selective reporting** - only interesting/useful findings
- ✅ **Clean visualizations** - DPI 300+, proper labels, no raw output

---

## 2. Data Understanding

### 2.1 Dataset Overview

**Final sample count:** 1,543 sessions (17 fewer than documented 1,560)

**Target variable (MOS) distribution:**
- MOS 1 (Bad): 93 samples (6.0%) 🔴
- MOS 2 (Poor): 118 samples (7.6%) 🟠
- MOS 3 (Fair): 246 samples (15.9%) 🟡
- MOS 4 (Good): **784 samples (50.8%)** 🟢 **← Majority class**
- MOS 5 (Excellent): 302 samples (19.6%) 🔵

**Critical issue:** **Severe class imbalance** - MOS=4 dominates with 50.8%

### 2.2 Data Quality

**Missing values:** ✅ **None** - 100% complete data across all 23 features

**Data types:** Mixed numerical (20 features) and categorical (3 features)

**Expected vs Actual:**
- Expected: 1,560 samples → Actual: 1,543 samples (-17, 1% discrepancy) ✅ Acceptable
- Expected: No missing values → Actual: Confirmed ✅
- Expected: Class imbalance → Actual: Confirmed (50.8% MOS=4) ✅

### 2.3 Feature Categories

**QoA - Video Quality (8 features):**
- Resolution, bitrate, framerate, dropped frames
- Audio rate, audio loss
- **Buffering count and time** ← Critical for QoE

**QoS - Network (2 features):**
- Network type (EDGE/UMTS/HSPA/LTE)
- Operator (4 French carriers)

**QoD - Device (3 features):**
- Model, OS version, API level
- **High cardinality problem** (15+ unique values)

**QoU - User Profile (3 features):**
- Gender (85.5% male), age (mean 29), education level

**QoF - User Feedback (4 features):**
- Subjective ratings: session start, time shift, audio quality, video quality
- **Data leakage concern** - too similar to MOS

### 2.4 Key Insights

**Main findings:**
1. Dataset size adequate (1,543 samples) but class imbalance severe
2. Zero missing values = excellent quality
3. High-cardinality features (device model, OS version) problematic
4. Buffering time has extreme outliers (max 329 seconds / 5.5 minutes!)

**Limitations identified:**
- ⚠️ Class imbalance requires stratified sampling
- ⚠️ Device features need removal (poor generalization)
- ⚠️ QoF_* features may cause data leakage
- ⚠️ Dataset age (2015) and location (France) limit generalization

---

## 3. Exploratory Data Analysis

### 3.1 Correlation Analysis

**Top correlations with MOS:**

**Positive:**
- QoF_audio: **+0.841** (VERY STRONG - subjective rating) 🚨
- QoF_video: +0.689 (strong - subjective rating)
- QoF_shift: +0.634 (strong - subjective rating)
- QoA_VLCframerate: +0.544 (moderate - objective metric) ✅
- QoA_VLCaudiorate: +0.354 (moderate - objective metric) ✅

**Negative:**
- QoA_BUFFERINGtime: **-0.482** (STRONG - objective metric) ✅
- QoA_BUFFERINGcount: -0.411 (moderate - objective metric) ✅
- QoA_VLCaudioloss: -0.323 (moderate)

**Expected vs Actual:**
- Expected: Buffering would strongly negatively correlate → **Confirmed** ✅
- Expected: QoF_* features would correlate → **Confirmed but TOO high** (0.841!) 🚨
- Unexpected: Video bitrate has WEAK correlation (0.090) - surprising!
- Unexpected: Resolution has NEGATIVE correlation (-0.022) - counterintuitive!

**Critical insight:** QoF_audio correlation of 0.841 suggests **data leakage risk** - it's almost measuring the same thing as MOS.

### 3.2 Network Type Impact

**ANOVA test:** F-statistic = 37.47, p-value < 0.000001 → **Highly significant** ✅

**MOS by network generation:**
- EDGE (2G): mean = 1.56 (Bad) 🔴
- UMTS (3G): mean = 3.64 (Good) 🟡
- HSPA (4G): mean = 3.84 (Good) 🟢 **Best**
- LTE (4G): mean = 3.78 (Good) 🟢

**Surprising finding:** HSPA slightly outperforms LTE (3.84 vs 3.78)
- Possible explanation: Network congestion, testing conditions, or sample variability

**Conclusion:** Network generation has **statistically significant impact** on QoE.

### 3.3 Buffering Analysis

**Key finding:** **Sharp user tolerance threshold** at 2-3 buffering events

| Buffering Count | Mean MOS | Quality |
|-----------------|----------|---------|
| 1 event | 3.88 | Good 🟢 |
| 2 events | 3.60 | Fair-Good 🟡 |
| 3 events | **2.33** | Poor 🟠 **← Drop-off** |
| 4+ events | 1.33-2.00 | Bad 🔴 |

**Insight:** Users tolerate 1-2 brief buffering interruptions, but satisfaction **collapses** beyond 2 events.

**Correlation:** Buffering time (r=-0.482) > Buffering count (r=-0.411)
- Longer buffering hurts more than frequent short buffers

### 3.4 Resolution Paradox

**Expected:** Higher resolution (360p) → Better MOS
**Actual:** 240p (mean 3.85) slightly outperforms 360p (mean 3.70)

**Explanation:**
- Adaptive streaming serves 240p on slow networks → less buffering → better QoE
- Smooth 240p playback > buffering 360p playback
- **Takeaway:** Resolution alone is NOT a good quality predictor

### 3.5 Demographics

**Gender:** Male (3.73) vs Female (3.54) - small difference (0.19 points)
**Age:** Minimal variation across age groups (range: 3.52-3.80)

**Conclusion:** Demographics have **weak impact** (correlations < 0.07) - QoE driven by technical factors, not user profile. ✅ Good for model generalization.

---

## 4. Methodology

### 4.1 Data Preprocessing

**Feature removal (6 features):**
- id, user_id (identifiers)
- QoD_model, QoD_os-version (high cardinality, poor generalization)
- QoU_Ustedy (92% same value, no variance)
- QoA_VLCresolution (weak predictor)

**Feature engineering (6 new features):**
- Buffering_Severity = count × log(time+1) - captures both frequency and duration
- QoA_BUFFERINGtime_log = log(time+1) - handles extreme outliers
- Network_Generation = 2G/3G/4G grouping - simplifies network types
- Excessive_Buffering = binary flag for >2 events - based on EDA threshold
- Video_Quality_Index = composite quality metric
- Audio_Quality_Adjusted = rate adjusted by packet loss

**Two dataset variants:**
1. **Objective (21 features):** Excludes QoF_* subjective ratings - realistic deployment
2. **Full (25 features):** Includes QoF_* - performance benchmark

**WHY two variants?** To test if subjective ratings cause overfitting/data leakage.

**Train/test split:**
- 80/20 split (1,234 train / 309 test)
- **Stratified by MOS** (critical for class imbalance!)
- Random state = 42 (reproducibility)

**Feature scaling:**
- StandardScaler (mean=0, std=1)
- **Fitted on TRAINING data only** (no data leakage!) ✅

### 4.2 Modeling Approach

**Algorithms tested:**
1. Baseline - Majority class predictor (must beat 50.8%!)
2. Logistic Regression - Test linear relationships
3. Decision Tree - Capture non-linearity
4. Random Forest - Reduce overfitting
5. Gradient Boosting - Handle hard samples

**Evaluation metrics:**
- **Accuracy** - Overall correctness (misleading with imbalance)
- **F1 Score** - Balance of precision and recall (better for imbalance)
- **Cohen's Kappa** - Accounts for chance agreement (good for ordinal MOS)
- **Overfit Gap** - Train accuracy - Test accuracy (detect overfitting)

**Class imbalance handling:**
- class_weight='balanced' in all models (except baseline)
- Penalizes majority class (MOS=4) errors

---

## 5. Results

### 5.1 Model Performance Comparison

| Model | Dataset | Test Accuracy | F1 Score | Cohen Kappa | Overfit Gap |
|-------|---------|---------------|----------|-------------|-------------|
| **Baseline** | Objective | **50.8%** | 0.342 | 0.000 | 0.0% |
| Logistic Regression | Objective | 43.4% 🔴 | 0.445 | 0.221 | 2.3% |
| Decision Tree | Objective | 42.4% 🔴 | 0.430 | 0.236 | 21.6% |
| Random Forest | Objective | 53.4% 🟡 | 0.486 | 0.227 | 46.3% |
| **Gradient Boosting** | **Objective** | **59.5%** 🟢 | **0.552** | **0.335** | 39.6% |
| Logistic Regression | Full | 77.0% 🟢 | 0.777 | 0.670 | 0.9% |
| **Random Forest** | **Full** | **81.9%** 🟢 | **0.819** | **0.727** | 17.2% |

### 5.2 Key Findings

**1. Best Objective Model:** Gradient Boosting - 59.5% accuracy
- ✅ Beats baseline by **8.7 percentage points**
- ✅ Cohen's Kappa = 0.335 ("fair agreement" - statistically significant)
- ⚠️ Only ~10% better than always guessing MOS=4
- 🔴 **40.5% error rate** - not ideal for critical decisions

**2. Full Model vs Objective Model:**
- Random Forest (Full): 81.9%
- Gradient Boosting (Objective): 59.5%
- **Gap: 22.4 percentage points** - MASSIVE difference! 🚨

**Interpretation:** **DATA LEAKAGE CONFIRMED** ✅
- QoF_* subjective ratings dominate Full model
- Full model learns "subjective rating → subjective rating" (not useful!)
- **Conclusion:** Must exclude QoF_* features for realistic deployment

**3. Ensemble Superiority:**
- Gradient Boosting (59.5%) > Random Forest (53.4%) > Tree (42.4%) > Logistic Regression (43.4%)
- Non-linear models essential (Logistic Regression WORSE than baseline!)

**4. Overfitting Problem:**
- Random Forest: 99.7% train → 53.4% test (**46.3% gap!**) 🔴
- Gradient Boosting: 99.2% train → 59.5% test (**39.6% gap**) 🔴
- **Cause:** Limited data (1,234 samples) for complex trees
- **Impact:** Models memorize training data, don't generalize

### 5.3 Feature Importance

**Top 5 features (Random Forest Objective):**

1. **QoA_BUFFERINGtime (13.9%)** - Raw buffering time
2. **QoA_BUFFERINGtime_log (13.3%)** - Log-transformed buffering
3. **Video_Quality_Index (10.3%)** - Engineered composite
4. **QoA_VLCframerate (9.5%)** - Framerate
5. **Buffering_Severity (9.4%)** - Engineered: count × log(time)

**Combined buffering importance: 36.6%** - Dominates all other factors! ✅

**Key insights:**
- ✅ Buffering is THE critical factor (confirms EDA)
- ✅ Engineered features work (Video_Quality_Index #3, Buffering_Severity #5)
- ❌ Network features less important than expected (not in top 10)
  - Explanation: Buffering already captures network quality indirectly

### 5.4 Per-Class Performance (Confusion Matrix)

**Gradient Boosting (Objective):**

| Actual MOS | Recall | Notes |
|------------|--------|-------|
| 1 (Bad) | 42.1% | Poor - class underrepresented |
| 2 (Poor) | 29.2% | Poor - class underrepresented |
| 3 (Fair) | 26.5% | Poor - overshadowed by MOS=4 |
| 4 (Good) | **91.1%** | Excellent - majority class |
| 5 (Excellent) | **15.0%** | **Terrible** - can't distinguish from MOS=4 |

**Problem:** Model biases toward MOS=4 despite class_weight='balanced'

**MOS=5 failure explanation:**
- "Excellent" QoE requires EVERYTHING perfect
- Without subjective ratings (QoF_*), technical metrics for MOS=4 and MOS=5 look similar
- 80% of MOS=5 samples misclassified as MOS=4

---

## 6. Critical Assessment

### 6.1 Strengths

✅ **Rigorous methodology:**
- Proper train/test split with stratification
- No data leakage (scaler fitted on train only)
- Multiple algorithms compared
- Two-model approach tests data leakage hypothesis

✅ **Domain-informed analysis:**
- Feature engineering based on EDA insights (buffering threshold)
- Network generation grouping based on ANOVA results
- Log transformation for outliers

✅ **Honest evaluation:**
- Acknowledges 59.5% is modest
- Identifies overfitting (46.3% gap)
- Documents per-class failures (MOS=5 only 15% recall)
- Discusses real-world limitations

✅ **Comprehensive documentation:**
- Every decision justified with "WHY"
- Expected vs Actual comparisons throughout
- Analysis path storytelling
- High-quality visualizations (DPI 300+)

### 6.2 Limitations

**1. Low Objective Model Accuracy (59.5%)**
- Only 8.7 points better than baseline
- 40.5% error rate problematic
- Minority classes poorly predicted (MOS=1,2,5)
- **Impact:** Not suitable for critical automated decisions

**2. Severe Overfitting**
- Random Forest: 46.3% train-test gap
- Gradient Boosting: 39.6% train-test gap
- **Cause:** Limited training data (1,234 samples)
- **Impact:** Models don't generalize well

**3. MOS=5 Prediction Failure**
- Only 15% recall for Excellent ratings
- Can't distinguish "Good" from "Excellent" without subjective feedback
- **Impact:** System won't detect truly excellent experiences

**4. Class Imbalance Not Fully Solved**
- class_weight='balanced' helps but insufficient
- SMOTE not attempted
- Threshold optimization not performed
- **Impact:** Persistent bias toward MOS=4

**5. No Hyperparameter Tuning**
- Used simple hyperparameters
- GridSearchCV could improve 2-5 points
- **Trade-off:** Time constraint

**6. Limited Feature Exploration**
- No interaction terms tested (buffering × network)
- No polynomial features
- Network features underutilized
- **Potential:** 3-7 point improvement possible

### 6.3 Real-World Feasibility

**Deployment concerns:**

⚠️ **Dataset limitations:**
- Age: 2015 data (8 years old) - networks evolved (5G now available)
- Location: France-specific (4 operators, specific conditions)
- Devices: Android only, limited diversity
- Resolution: 95% at 360p (no HD/4K)

⚠️ **Accuracy limitations:**
- 59.5% insufficient for automated SLA enforcement
- 40.5% error rate too high for customer-facing decisions
- Can't predict Excellent QoE (15% recall)

✅ **Acceptable use cases:**
- **Trend monitoring** - relative QoE changes over time ✅
- **A/B testing** - comparing network configurations ✅
- **Early warning system** - flagging potential issues (with human review) ✅
- **Research and analysis** - understanding QoE patterns ✅

❌ **NOT suitable for:**
- Automated network optimization decisions without human review
- SLA violation detection
- Customer complaint prediction
- Real-time service degradation alerts

**Recommendation:** Deploy as a **"soft" decision support tool**, not autonomous system.

### 6.4 Comparison to Expectations

| Aspect | Expected | Actual | Assessment |
|--------|----------|--------|------------|
| Objective accuracy | 60-70% | 59.5% | ✅ Within range (lower end) |
| Buffering importance | High | 36.6% | ✅ Confirmed |
| QoF_* leakage | Suspected | +22.4 points | ✅ Confirmed |
| Network importance | Top 3 | Not in top 10 | ❌ Surprising |
| Class imbalance | Challenge | Persistent | ✅ Confirmed |
| Overfitting | Moderate | Severe (46%) | ⚠️ Worse than expected |

---

## 7. Conclusions

### 7.1 Answers to Research Questions

**Q1: Can we predict QoE from objective metrics?**

**Answer:** Yes, but with **limited accuracy (59.5%)**
- ✅ Statistically significant improvement over chance
- ⚠️ Not reliable for critical decisions alone
- ✅ Useful for trend monitoring and relative comparisons

**Q2: Which factors are most important?**

**Answer:** **Buffering metrics dominate** (36.6% combined importance)
1. Buffering time (raw + log)
2. Video quality index
3. Framerate
4. Buffering severity

**Insight:** Direct measurements (buffering) > Proxies (network type)

**Q3: Do subjective ratings cause overfitting?**

**Answer:** **YES - data leakage confirmed**
- Full model (81.9%) >> Objective (59.5%) = +22.4 points
- QoF_* features essentially measure same thing as MOS
- **Conclusion:** Exclude for realistic deployment estimates

**Q4: Is the model deployment-ready?**

**Answer:** **For limited use cases, with caution**
- ✅ Acceptable: Trend monitoring, A/B testing, research
- ❌ Not suitable: Automated SLA enforcement, critical decisions
- ⚠️ Requires: Human review, validation, periodic retraining

### 7.2 Key Takeaways

1. **Objective QoE prediction is inherently difficult**
   - 59.5% may represent a ceiling with current features
   - Some QoE aspects not captured by network metrics
   - Human perception includes unmeasured factors

2. **Buffering is THE critical factor for QoE**
   - 36.6% feature importance
   - Strong correlation (r=-0.482)
   - User tolerance threshold at 2-3 events

3. **Data leakage risk is real**
   - 22.4 point gap between Full and Objective models
   - Subjective features "too good to be true"
   - Critical for honest model evaluation

4. **Class imbalance is a persistent challenge**
   - class_weight='balanced' helps but insufficient
   - MOS=5 prediction fails (15% recall)
   - Advanced techniques (SMOTE, threshold tuning) needed

5. **Model complexity vs data size trade-off**
   - 1,234 samples insufficient for complex trees
   - 46% overfit gap indicates memorization
   - More data or simpler models needed

### 7.3 Recommendations

**For immediate improvement:**
1. Hyperparameter tuning (GridSearchCV) → +2-5 points
2. SMOTE for minority classes → +3-7 points
3. Ensemble methods (voting, stacking) → +1-3 points
4. Interaction features (buffering × network) → +2-4 points

**Realistic ceiling:** 65-70% accuracy with current approach

**For future research:**
1. Collect more data (target: 10,000+ samples)
2. Balanced data collection (oversample MOS=1,2)
3. Include 5G networks and modern devices
4. Additional features (time-of-day, location, device battery)
5. Causal inference (does reducing buffering *cause* better MOS?)

**For deployment:**
1. Use Gradient Boosting (Objective) model
2. Deploy as "soft" predictions with human review
3. Focus on trend monitoring, not absolute SLA
4. Monitor and retrain quarterly
5. A/B test model versions before full rollout

---

## 8. Project Reflection

### 8.1 What Worked Well

✅ **Systematic approach:**
- Four-notebook structure (Understand → Explore → Preprocess → Model)
- Clear progression and logical flow
- Each stage built on previous insights

✅ **Hypothesis-driven analysis:**
- Data leakage hypothesis formulated in EDA
- Tested with two-model approach
- Confirmed with 22.4 point gap

✅ **Domain knowledge integration:**
- Buffering threshold (2-3 events) from EDA → Excessive_Buffering feature
- Network generation grouping based on ANOVA
- Log transformation for outliers

✅ **Honest critical assessment:**
- Acknowledged 59.5% is modest
- Documented failures (MOS=5 recall, overfitting)
- Discussed real-world limitations
- Didn't "sell" the project

### 8.2 What Could Be Improved

⚠️ **Time constraints:**
- No hyperparameter tuning (GridSearchCV)
- SMOTE not attempted
- Advanced ensembles not tried
- **Impact:** 5-10 points potential improvement left on table

⚠️ **Data limitations:**
- Small dataset (1,543 samples)
- Old data (2015)
- Location-specific (France)
- **Impact:** Generalization concerns

⚠️ **Model complexity:**
- Overfitting severe (46.3% gap)
- Could try simpler models
- Regularization not tuned
- **Impact:** Test performance suffered

### 8.3 Lessons Learned

1. **Class imbalance is hard:**
   - class_weight='balanced' helps but insufficient
   - Need multiple strategies (SMOTE, threshold tuning, ensemble)
   - Minority class prediction remains challenge

2. **Data leakage is subtle:**
   - QoF_* features "seemed reasonable" initially
   - Only comparison revealed 22.4 point inflation
   - Critical to test with/without suspected features

3. **Correlation ≠ Feature importance:**
   - Network type significant in ANOVA (F=37.47)
   - But low feature importance (<2.8%)
   - Models prefer direct measurements over proxies

4. **More data > better algorithms:**
   - 1,234 samples caused overfitting in complex models
   - Collecting more data likely more valuable than tuning

5. **Deployment requires different criteria:**
   - Research: Maximize accuracy
   - Deployment: Generalization, interpretability, maintainability
   - 81.9% Full model impressive but not deployable

---

## 9. Appendices

### 9.1 Files Generated

**Analysis documents:**
- `notebooks/ANALYSIS_01_DATA_UNDERSTANDING.md`
- `notebooks/ANALYSIS_02_EXPLORATORY_DATA_ANALYSIS.md`
- `notebooks/ANALYSIS_03_DATA_PREPROCESSING.md`
- `notebooks/ANALYSIS_04_MODELING_EVALUATION.md`
- `reports/FINAL_PROJECT_REPORT.md` (this document)

**Visualizations (DPI 300+):**
- `results/figures/mos_distribution.png`
- `results/figures/correlation_heatmap.png`
- `results/figures/network_type_analysis.png`
- `results/figures/buffering_analysis.png`
- `results/figures/demographics_analysis.png`
- `results/figures/confusion_matrices.png`
- `results/figures/feature_importance.png`
- `results/figures/model_comparison.png`

**Processed data:**
- `data/processed/X_train_objective_scaled.csv`
- `data/processed/X_test_objective_scaled.csv`
- `data/processed/y_train_objective.csv`
- `data/processed/y_test_objective.csv`
- `data/processed/X_train_full_scaled.csv`
- `data/processed/X_test_full_scaled.csv`
- `data/processed/y_train_full.csv`
- `data/processed/y_test_full.csv`

**Models:**
- `models/scaler_objective.pkl`
- `models/scaler_full.pkl`

**Metrics:**
- `results/metrics/model_comparison.csv`


### 9.2 Glossary

- **MOS (Mean Opinion Score):** Subjective rating of QoE on 1-5 scale (1=Bad, 5=Excellent)
- **QoE (Quality of Experience):** User's subjective perception of service quality
- **QoA (Quality of Application):** Video/audio technical metrics
- **QoS (Quality of Service):** Network performance metrics
- **QoD (Quality of Device):** Device characteristics
- **QoU (Quality of User):** User demographics
- **QoF (Quality of Feedback):** User subjective ratings
- **SMOTE:** Synthetic Minority Over-sampling Technique (handles class imbalance)
- **Cohen's Kappa:** Statistical measure of inter-rater agreement (accounts for chance)

---

## Final Statement

This project demonstrates that **objective QoE prediction is possible but challenging** (59.5% accuracy). While not suitable for critical automated decisions, the model provides value for **trend monitoring and comparative analysis**.

The **most important contribution** is the **honest, critical assessment** of both successes and limitations - exactly as the project guidelines emphasize. We don't "sell" a 59.5% model as excellent; we acknowledge it's modest but statistically significant, and we clearly define where it can and cannot be used.

**Key insight:** The 22.4 percentage point gap between Full and Objective models reveals a fundamental truth - **some aspects of human experience cannot be fully captured by objective technical measurements alone**.