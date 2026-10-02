# Pokemon QoE Dataset - Analysis Project

> ⚠️ **Superseded numbers.** This guide predates [`scripts/run_leakage_experiment.py`](../../scripts/run_leakage_experiment.py), which is now the canonical source for all model numbers (balanced class weights, library defaults, macro F1). Quote the README / [`results/metrics/model_comparison.csv`](../../results/metrics/model_comparison.csv), not the figures below. Kept as-is as a historical record.

## Project Overview

This project analyzes the Pokemon Quality of Experience (QoE) dataset to predict user satisfaction (MOS - Mean Opinion Score) from objective network and video quality metrics. The dataset was collected during a crowdsourcing campaign where users watched videos on mobile devices across different network conditions.

**Goal:** Build predictive models to estimate QoE without requiring subjective user surveys, enabling proactive network optimization.

---

## Dataset Information

### Context
- **Source:** Pokemon Project (Platform Quality Evaluation of Mobile Networks)
- **Samples:** 1,560 video viewing sessions
- **Testers:** 181 participants (researchers and students, ages 19-38)
- **Location:** LiSSi laboratory, Paris, France
- **Collection Method:** Android application with VLC media player

### Features (23 total)

#### QoA - Video Quality Metrics (8 features)
- `QoA_VLCresolution`: Video resolution (240p, 360p)
- `QoA_VLCbitrate`: Video bitrate
- `QoA_VLCframerate`: Frame rate
- `QoA_VLCdropped`: Dropped frames
- `QoA_VLCaudiorate`: Audio bitrate
- `QoA_VLCaudioloss`: Audio packet loss
- `QoA_BUFFERINGcount`: Number of buffering events
- `QoA_BUFFERINGtime`: Total buffering time (ms)

#### QoS - Network Information (2 features)
- `QoS_type`: Network type (EDGE, UMTS, HSPA, HSPAP, LTE)
- `QoS_operator`: Mobile operator (SFR, BOUYEGUES, ORANGE, FREE)

#### QoD - Device Characteristics (3 features)
- `QoD_model`: Device model
- `QoD_os-version`: Android OS version
- `QoD_api-level`: Android API level

#### QoU - User Profile (3 features)
- `QoU_sex`: Gender (0=Female, 1=Male)
- `QoU_age`: Age
- `QoU_Ustedy`: Education level (1-5)

#### QoF - User Feedback (4 features)
- `QoF_begin`: Session start time
- `QoF_shift`: Time shift
- `QoF_audio`: Audio quality feedback
- `QoF_video`: Video quality feedback

#### Target Variable
- **MOS**: Mean Opinion Score (1=Bad, 2=Poor, 3=Fair, 4=Good, 5=Excellent)

### Class Distribution
- MOS 1 (Bad): 92 samples (5.9%)
- MOS 2 (Poor): 119 samples (7.6%)
- MOS 3 (Fair): 244 samples (15.6%)
- MOS 4 (Good): 787 samples (50.4%) ⚠️ **Majority class**
- MOS 5 (Excellent): 300 samples (19.2%)

---

## Project Structure

```
Poqemon-QoE-Dataset-master/
├── config/                     # Configuration files
│   └── config.py              # Central project configuration
│
├── data/
│   ├── raw/                   # Original dataset files
│   │   ├── pokemon.csv
│   │   ├── pokemon.arff
│   │   ├── pokemon.data
│   │   └── pokemon.names
│   ├── processed/             # Preprocessed data (train/test splits)
│   │   ├── X_train_full_scaled.csv
│   │   ├── X_test_full_scaled.csv
│   │   ├── X_train_objective_scaled.csv
│   │   ├── X_test_objective_scaled.csv
│   │   └── y_*.csv files
│   └── external/              # External data (if any)
│
├── docs/                       # Documentation
│   ├── README.md              # Documentation navigation
│   ├── PROJECT_GUIDE.md       # This file
│   ├── SETUP_GUIDE.md         # Installation instructions
│   ├── COMPLETION_SUMMARY.md  # Project completion report
│   ├── DATASET_DESCRIPTION.md # Original dataset information
│   ├── GUIDELINES_KR.md       # Korean guidelines
│   └── project-guidelines.pdf # Original guidelines (PDF)
│
├── models/                     # Saved trained models
│   ├── scaler_full.pkl        # Feature scaler (Full dataset)
│   └── scaler_objective.pkl   # Feature scaler (Objective dataset)
│
├── notebooks/                  # Jupyter notebooks (analysis workflow)
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_modeling_and_evaluation.ipynb
│   ├── ANALYSIS_01_DATA_UNDERSTANDING.md
│   ├── ANALYSIS_02_EXPLORATORY_DATA_ANALYSIS.md
│   ├── ANALYSIS_03_DATA_PREPROCESSING.md
│   └── ANALYSIS_04_MODELING_EVALUATION.md
│
├── reports/                    # Final reports and documentation
│   └── FINAL_PROJECT_REPORT.md
│
├── results/                    # Analysis results
│   ├── figures/               # Generated plots (DPI 300+)
│   │   ├── mos_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── buffering_analysis.png
│   │   ├── network_type_analysis.png
│   │   ├── demographics_analysis.png
│   │   ├── confusion_matrices.png
│   │   ├── feature_importance.png
│   │   └── model_comparison.png
│   ├── metrics/               # Performance metrics
│   │   └── model_comparison.csv
│   └── tables/                # Result tables
│
├── src/                        # Source code (reusable modules)
│   ├── data/
│   │   └── data_loader.py     # Data loading utilities
│   ├── models/
│   │   └── model_utils.py     # Model training/evaluation utilities
│   ├── visualization/
│   │   └── plot_utils.py      # Visualization utilities
│   └── utils/                 # General utilities
│
├── .gitignore                  # Git ignore file
├── README.md                   # Main project overview
└── requirements.txt            # Python dependencies
```

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone or navigate to the project directory**
```bash
# Example (adjust path to your local directory):
cd "path/to/Poqemon-QoE-Dataset-master"
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Verify data files**
Check that data files exist in `data/raw/`:
- pokemon.csv
- pokemon.arff
- pokemon.data
- pokemon.names

---

## Usage

### Running the Analysis

Follow the notebooks in order:

1. **Data Understanding** (`01_data_understanding.ipynb`)
   - Load and inspect dataset
   - Understand feature categories
   - Analyze target distribution
   - Identify data quality issues

2. **Exploratory Data Analysis** (`02_exploratory_data_analysis.ipynb`)
   - Feature correlations
   - Network type impact
   - Buffering analysis
   - User demographics patterns

3. **Data Preprocessing** (`03_data_preprocessing.ipynb`)
   - Feature engineering
   - Categorical encoding
   - Train/test split
   - Feature scaling

4. **Modeling and Evaluation** (`04_modeling_and_evaluation.ipynb`)
   - Baseline model
   - Multiple ML algorithms
   - Model comparison
   - Feature importance analysis

### Using Utility Scripts

```python
# Load data
from src.data.data_loader import load_pokemon_data
df = load_pokemon_data()

# Evaluate model
from src.models.model_utils import ModelEvaluator
evaluator = ModelEvaluator(model, 'RandomForest')
results = evaluator.evaluate(X_train, X_test, y_train, y_test)

# Create visualizations
from src.visualization.plot_utils import plot_mos_distribution
plot_mos_distribution(df['MOS'], save_path='results/figures/mos_dist.png')
```

---

## Methodology

### Analysis Approach (Following Guidelines)

This project follows a **storytelling approach** where:

1. **WHY before WHAT**: Every analysis step explains its motivation
2. **Expected vs Actual**: Compare predictions with actual results
3. **Critical Assessment**: Identify limitations and feasibility concerns
4. **Selective Reporting**: Only interesting/useful findings included

### Data Science Protocol

1. **Data Understanding**
   - Dataset characteristics
   - Feature meanings and distributions
   - Class imbalance identification

2. **Exploratory Analysis**
   - Correlation analysis
   - Pattern discovery
   - Hypothesis testing

3. **Preprocessing**
   - Feature selection (remove high-cardinality features)
   - Encoding (one-hot for nominal, keep ordinal)
   - Engineering (buffering severity, network generation, video quality index)
   - Stratified train/test split (80/20)
   - Standard scaling (fit on train only!)

4. **Modeling**
   - Baseline: Majority class predictor
   - Linear: Logistic Regression
   - Non-linear: Decision Tree, Random Forest, Gradient Boosting
   - Evaluation: Accuracy, Precision, Recall, F1, Cohen's Kappa

5. **Critical Evaluation**
   - Real-world feasibility
   - Computational requirements
   - Model limitations
   - Deployment considerations

---

## Key Findings

**Dataset Insights:**
- ✅ Severe class imbalance (MOS=4 represents 50.8% of samples)
- ✅ Data leakage detected in subjective feedback features (QoF_* correlation r=0.84 with MOS)
- ✅ LTE networks show significantly better QoE than EDGE/UMTS
- ✅ Buffering events have strong negative correlation with user satisfaction

**Model Performance:**
- **Best model (Objective)**: Gradient Boosting with 59.5% accuracy
  - Baseline: 50.8% (majority class predictor)
  - Improvement: +8.7 percentage points
- **Best model (Full with data leakage)**: Random Forest with 81.9% accuracy
- **Most important features**: QoF_video, QoF_audio, QoA_BUFFERINGcount, QoS_type, QoA_VLCbitrate

**Limitations:**
- ✅ Model struggles with minority classes (MOS=1,2,3) due to class imbalance
- ✅ Overfitting concerns (Random Forest shows 46.3% overfit gap on Objective features)
- ✅ Real-world deployment limited to objective features only (no user feedback)
- ✅ Modest performance improvement over baseline suggests inherent prediction difficulty

**Recommendations:**
- ✅ Address class imbalance using SMOTE or class weights
- ✅ Collect more samples for minority classes (MOS=1,2,3)
- ✅ Deploy only Objective models in production (no data leakage)
- ✅ Consider ensemble methods or deep learning for improvement
- ✅ Implement real-time QoE monitoring using network metrics

---

## Evaluation Metrics

### Why These Metrics?

- **Accuracy**: Overall correctness (but misleading with imbalance)
- **Precision/Recall/F1**: Better for imbalanced datasets
- **Cohen's Kappa**: Accounts for chance agreement (good for ordinal MOS)
- **MAE/RMSE**: Regression-style metrics (MOS is ordinal)
- **Confusion Matrix**: Shows where model makes mistakes

### Visualization Standards (Following Guidelines)

- **Resolution:** DPI 300+ (publication quality)
- **Fonts:** Minimum 12pt, titles 14pt
- **Colors:**
  - 🟢 Green: Excellent (>80%)
  - 🟡 Yellow: Good (70-80%)
  - 🟠 Orange: Fair (60-70%)
  - 🔴 Red: Poor (<60%)
- **Clarity:** Clean labels, no raw output, rounded decimals

---

## References

### Papers
1. Lamine Amour, Sami Souihi, Said Hoceini, Abdelhamid Mellouk (2015).
   "Building a Large Dataset for Model-based QoE Prediction in the Mobile Environment."
   ACM MSWiM 2015.

2. Stéphanie Moteau, Fabrice Guillemin, Thierry Houdoin (2017).
   "Correlation between QoS and QoE for HTTP YouTube content in Orange cellular networks."

### Dataset Source
- Pokemon Project: Platform Quality Evaluation of Mobile Networks
- Paris Est Créteil University, France
- Contact: lamine.amour@u-pec.fr

---

## Project Guidelines Compliance

This project follows the guidelines provided by Alessandro Maddaloni (Telecom SudParis):

✅ **Context**: Dataset and goals clearly explained
✅ **Data Understanding**: EDA with selective reporting
✅ **Protocol**: Preprocessing steps justified
✅ **Analysis Path**: Storytelling approach with "WHY"
✅ **Expected vs Actual**: Comparisons throughout
✅ **Critical Assessment**: Limitations acknowledged
✅ **Clean Reporting**: No raw output, formatted results
✅ **High-Quality Visuals**: DPI 300+, large fonts, color coding

---

## Next Steps

1. **Run Complete Analysis**
   - Execute all notebooks in order
   - Fill in analysis findings
   - Generate all visualizations

2. **Model Improvement**
   - Address class imbalance (SMOTE, class weights)
   - Hyperparameter tuning (GridSearchCV)
   - Try advanced models (XGBoost, Neural Networks)
   - Feature selection

3. **Report Writing**
   - Compile findings
   - Write comprehensive report
   - Include critical assessment
   - Prepare presentation

4. **Deployment Considerations**
   - Real-time prediction feasibility
   - Model serving architecture
   - Monitoring and retraining strategy

---

## Project Information

**Course**: Data Science - Theory to Practice
**Institution**: Telecom SudParis
**Project Completed**: October 2024
**Last Updated**: November 2024

---

## License

This project is for educational purposes. Dataset source and original authors should be cited in any publication.
