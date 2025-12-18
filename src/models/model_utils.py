"""
Model Utilities
===============

Utilities for training, evaluating, and saving machine learning models
for QoE prediction.

Author: [Your Name]
Date: 2025-10-01
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score,
    mean_absolute_error, mean_squared_error
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for classification tasks.
    """

    def __init__(self, model, model_name='Model'):
        """
        Initialize evaluator.

        Args:
            model: Trained sklearn model
            model_name (str): Name of the model for reporting
        """
        self.model = model
        self.model_name = model_name
        self.results = {}

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Perform comprehensive evaluation.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels

        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating {self.model_name}...")

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        self.results = {
            'model_name': self.model_name,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
            'test_kappa': cohen_kappa_score(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }

        # Check for overfitting
        train_test_gap = self.results['train_accuracy'] - self.results['test_accuracy']
        if train_test_gap > 0.1:
            logger.warning(f"⚠️ Potential overfitting detected! Train-Test gap: {train_test_gap:.3f}")
            self.results['overfitting_warning'] = True
        else:
            self.results['overfitting_warning'] = False

        # Store predictions
        self.results['y_test_pred'] = y_test_pred
        self.results['y_train_pred'] = y_train_pred

        return self.results

    def get_classification_report(self, y_test):
        """
        Get detailed classification report.

        Args:
            y_test: True test labels

        Returns:
            str: Classification report
        """
        if 'y_test_pred' not in self.results:
            raise ValueError("Must run evaluate() first")

        target_names = ['Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)']
        return classification_report(y_test, self.results['y_test_pred'],
                                    target_names=target_names,
                                    zero_division=0)

    def get_confusion_matrix(self, y_test):
        """
        Get confusion matrix.

        Args:
            y_test: True test labels

        Returns:
            np.ndarray: Confusion matrix
        """
        if 'y_test_pred' not in self.results:
            raise ValueError("Must run evaluate() first")

        return confusion_matrix(y_test, self.results['y_test_pred'])

    def print_summary(self):
        """
        Print formatted evaluation summary.
        """
        if not self.results:
            raise ValueError("Must run evaluate() first")

        print(f"\n{'='*70}")
        print(f"{self.model_name} - Evaluation Summary")
        print(f"{'='*70}")
        print(f"Train Accuracy:     {self.results['train_accuracy']:.3f}")
        print(f"Test Accuracy:      {self.results['test_accuracy']:.3f}")
        print(f"Test Precision:     {self.results['test_precision']:.3f}")
        print(f"Test Recall:        {self.results['test_recall']:.3f}")
        print(f"Test F1-Score:      {self.results['test_f1']:.3f}")
        print(f"Test Cohen's Kappa: {self.results['test_kappa']:.3f}")
        print(f"Test MAE:           {self.results['test_mae']:.3f}")
        print(f"Test RMSE:          {self.results['test_rmse']:.3f}")

        if self.results['overfitting_warning']:
            gap = self.results['train_accuracy'] - self.results['test_accuracy']
            print(f"\n⚠️ Overfitting Warning: Train-Test gap = {gap:.3f}")

        print(f"{'='*70}\n")


class ModelManager:
    """
    Manage model saving, loading, and persistence.
    """

    def __init__(self, models_dir='../../models'):
        """
        Initialize model manager.

        Args:
            models_dir (str): Directory to save/load models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name, metadata=None):
        """
        Save model and optional metadata.

        Args:
            model: Trained model to save
            model_name (str): Name for the saved model
            metadata (dict): Optional metadata to save with model
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.pkl"

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model saved: {model_path}")

        # Save metadata if provided
        if metadata:
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"✓ Metadata saved: {metadata_path}")

    def load_model(self, model_name):
        """
        Load saved model.

        Args:
            model_name (str): Name of model to load

        Returns:
            Loaded model
        """
        model_path = self.models_dir / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"✓ Model loaded: {model_path}")
        return model

    def load_metadata(self, model_name):
        """
        Load model metadata if exists.

        Args:
            model_name (str): Name of model

        Returns:
            dict or None: Metadata if exists
        """
        metadata_path = self.models_dir / f"{model_name}_metadata.pkl"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        return metadata

    def list_models(self):
        """
        List all saved models.

        Returns:
            list: List of model names
        """
        models = [f.stem for f in self.models_dir.glob("*.pkl")
                  if not f.stem.endswith('_metadata')]
        return models


def compare_models(results_list):
    """
    Compare multiple model results.

    Args:
        results_list (list): List of result dictionaries from ModelEvaluator

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_df = pd.DataFrame(results_list)

    # Select relevant columns
    cols = ['model_name', 'train_accuracy', 'test_accuracy',
            'test_precision', 'test_recall', 'test_f1', 'test_kappa',
            'test_mae', 'test_rmse']

    comparison_df = comparison_df[[c for c in cols if c in comparison_df.columns]]

    # Format numeric columns
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(3)

    # Sort by test accuracy
    comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)

    return comparison_df


def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract feature importance from tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to return

    Returns:
        pd.DataFrame: Feature importance table
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

    return importance_df


if __name__ == "__main__":
    print("Model utilities loaded successfully")
    print("\nAvailable classes:")
    print("  - ModelEvaluator: Comprehensive model evaluation")
    print("  - ModelManager: Save/load models and metadata")
    print("\nAvailable functions:")
    print("  - compare_models(): Compare multiple model results")
    print("  - get_feature_importance(): Extract feature importance")
