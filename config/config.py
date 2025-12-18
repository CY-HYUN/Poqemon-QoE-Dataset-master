"""
Project Configuration
====================

Central configuration file for the Pokemon QoE prediction project.

Author: [Your Name]
Date: 2025-10-01
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'

# Results directories
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
METRICS_DIR = RESULTS_DIR / 'metrics'

# Reports directory
REPORTS_DIR = PROJECT_ROOT / 'reports'

# Dataset configuration
DATASET_CONFIG = {
    'filename': 'pokemon.csv',
    'target_column': 'MOS',
    'id_columns': ['id', 'user_id'],
    'test_size': 0.2,
    'random_state': 42
}

# Feature categories
FEATURE_CATEGORIES = {
    'identifiers': ['id', 'user_id'],
    'qoa_video': [
        'QoA_VLCresolution', 'QoA_VLCbitrate', 'QoA_VLCframerate',
        'QoA_VLCdropped', 'QoA_VLCaudiorate', 'QoA_VLCaudioloss',
        'QoA_BUFFERINGcount', 'QoA_BUFFERINGtime'
    ],
    'qos_network': ['QoS_type', 'QoS_operator'],
    'qod_device': ['QoD_model', 'QoD_os-version', 'QoD_api-level'],
    'qou_user': ['QoU_sex', 'QoU_age', 'QoU_Ustedy'],
    'qof_feedback': ['QoF_begin', 'QoF_shift', 'QoF_audio', 'QoF_video'],
    'target': ['MOS']
}

# Categorical mappings
CATEGORICAL_MAPPINGS = {
    'QoS_type': {
        1: 'EDGE',
        2: 'UMTS',
        3: 'HSPA',
        4: 'HSPAP',
        5: 'LTE'
    },
    'QoS_operator': {
        1: 'SFR',
        2: 'BOUYEGUES',
        3: 'ORANGE',
        4: 'FREE'
    },
    'QoU_sex': {
        0: 'Female',
        1: 'Male'
    },
    'QoU_Ustedy': {
        1: 'Other',
        2: 'Primary school',
        3: 'College',
        4: 'Secondary school',
        5: 'University'
    },
    'MOS': {
        1: 'Bad',
        2: 'Poor',
        3: 'Fair',
        4: 'Good',
        5: 'Excellent'
    }
}

# Network generation mapping
NETWORK_GENERATION_MAP = {
    1: 2,  # EDGE -> 2G
    2: 3,  # UMTS -> 3G
    3: 3,  # HSPA -> 3G
    4: 3,  # HSPAP -> 3G
    5: 4   # LTE -> 4G
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'features_to_remove': ['id', 'user_id', 'QoD_model', 'QoD_os-version'],
    'features_to_encode': ['QoS_operator'],
    'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
    'handle_missing': 'drop',      # 'drop', 'impute'
}

# Model configuration
MODEL_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'multi_class': 'multinomial',
        'random_state': 42
    },
    'decision_tree': {
        'max_depth': 10,
        'min_samples_split': 20,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'random_state': 42
    }
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision_weighted',
    'recall_weighted',
    'f1_weighted',
    'cohen_kappa'
]

# Visualization configuration (following guidelines)
PLOT_CONFIG = {
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'figure_size': (12, 8),
    'title_fontsize': 14,
    'label_fontsize': 12,
    'tick_fontsize': 10,
    'save_format': 'png',
    'bbox_inches': 'tight'
}

# Color scheme (following guidelines)
COLORS = {
    'performance': {
        'excellent': '#2ca02c',  # Green (>80%)
        'good': '#ffdd57',        # Yellow (70-80%)
        'fair': '#ff7f0e',        # Orange (60-70%)
        'poor': '#d62728'         # Red (<60%)
    },
    'mos': {
        1: '#d62728',  # Bad - Red
        2: '#ff7f0e',  # Poor - Orange
        3: '#ffdd57',  # Fair - Yellow
        4: '#2ca02c',  # Good - Green
        5: '#1f77b4'   # Excellent - Blue
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation configuration
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': RANDOM_SEED
}

# Grid search configuration
GRID_SEARCH_CONFIG = {
    'cv': 5,
    'n_jobs': -1,
    'verbose': 1,
    'scoring': 'accuracy'
}


def create_directories():
    """
    Create all necessary project directories if they don't exist.
    """
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR,
        REPORTS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✓ All project directories created/verified")


if __name__ == "__main__":
    print("Pokemon QoE Project Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("\nCreating directories...")
    create_directories()
