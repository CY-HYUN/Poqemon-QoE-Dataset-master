"""
Data Loading Utilities
======================

This module provides utilities for loading and basic validation of the Pokemon QoE dataset.

Author: [Your Name]
Date: 2025-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PokemonDataLoader:
    """
    Load and perform basic validation on Pokemon QoE dataset.

    Supports multiple file formats: CSV, ARFF, DATA
    """

    def __init__(self, data_dir='../../data/raw'):
        """
        Initialize data loader.

        Args:
            data_dir (str): Path to directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.df = None

    def load_csv(self, filename='pokemon.csv'):
        """
        Load data from CSV file.

        Args:
            filename (str): Name of CSV file

        Returns:
            pd.DataFrame: Loaded dataset
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath)
        logger.info(f"Loaded {self.df.shape[0]} samples with {self.df.shape[1]} features")

        return self.df

    def validate_data(self):
        """
        Perform basic validation checks on loaded data.

        Returns:
            dict: Validation results
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        validation = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'mos_distribution': self.df['MOS'].value_counts().to_dict() if 'MOS' in self.df.columns else None,
            'column_types': self.df.dtypes.to_dict()
        }

        # Log validation results
        logger.info("Data Validation Results:")
        logger.info(f"  Shape: {validation['shape']}")
        logger.info(f"  Missing values: {validation['missing_values']}")
        logger.info(f"  Duplicate rows: {validation['duplicate_rows']}")

        if validation['mos_distribution']:
            logger.info(f"  MOS distribution: {validation['mos_distribution']}")

        return validation

    def get_feature_categories(self):
        """
        Return features grouped by QoE influence factor categories.

        Returns:
            dict: Feature categories
        """
        return {
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

    def get_categorical_mappings(self):
        """
        Return mappings for categorical variables.

        Returns:
            dict: Category mappings
        """
        return {
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


def load_pokemon_data(data_path='../../data/raw/pokemon.csv', validate=True):
    """
    Convenience function to load and optionally validate Pokemon QoE dataset.

    Args:
        data_path (str): Path to dataset file
        validate (bool): Whether to perform validation

    Returns:
        pd.DataFrame: Loaded dataset
    """
    loader = PokemonDataLoader()
    df = loader.load_csv(Path(data_path).name)

    if validate:
        loader.validate_data()

    return df


if __name__ == "__main__":
    # Example usage
    print("Pokemon QoE Dataset Loader")
    print("=" * 60)

    loader = PokemonDataLoader()
    df = loader.load_csv()
    validation = loader.validate_data()

    print("\nFeature Categories:")
    for category, features in loader.get_feature_categories().items():
        print(f"  {category}: {len(features)} features")

    print("\nData loaded successfully!")
