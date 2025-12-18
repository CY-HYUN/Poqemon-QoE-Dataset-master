"""
Visualization Utilities
=======================

Reusable plotting functions following project guidelines:
- High resolution (DPI 300+)
- Large, readable fonts (12pt minimum)
- Proper color coding
- Clean, professional appearance

Author: [Your Name]
Date: 2025-10-01
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Default figure parameters (following guidelines)
DEFAULT_DPI = 300
DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
DEFAULT_TICK_SIZE = 10

# Color scheme for performance (following guidelines)
PERFORMANCE_COLORS = {
    'excellent': '#2ca02c',  # Green (>80%)
    'good': '#ffdd57',        # Yellow (70-80%)
    'fair': '#ff7f0e',        # Orange (60-70%)
    'poor': '#d62728'         # Red (<60%)
}

MOS_COLORS = {
    1: '#d62728',  # Bad - Red
    2: '#ff7f0e',  # Poor - Orange
    3: '#ffdd57',  # Fair - Yellow
    4: '#2ca02c',  # Good - Green
    5: '#1f77b4'   # Excellent - Blue
}


def setup_plot_style():
    """
    Configure matplotlib/seaborn style for high-quality plots.
    """
    plt.rcParams.update({
        'figure.dpi': DEFAULT_DPI,
        'savefig.dpi': DEFAULT_DPI,
        'font.size': DEFAULT_TICK_SIZE,
        'axes.titlesize': DEFAULT_TITLE_SIZE,
        'axes.labelsize': DEFAULT_LABEL_SIZE,
        'xtick.labelsize': DEFAULT_TICK_SIZE,
        'ytick.labelsize': DEFAULT_TICK_SIZE,
        'legend.fontsize': DEFAULT_TICK_SIZE,
        'figure.titlesize': DEFAULT_TITLE_SIZE + 2
    })


def get_performance_color(value):
    """
    Get color based on performance value (for metrics like accuracy).

    Args:
        value (float): Performance metric (0-1 or 0-100)

    Returns:
        str: Hex color code
    """
    if value > 100:
        value = value / 100  # Convert percentage to fraction

    if value >= 0.8:
        return PERFORMANCE_COLORS['excellent']
    elif value >= 0.7:
        return PERFORMANCE_COLORS['good']
    elif value >= 0.6:
        return PERFORMANCE_COLORS['fair']
    else:
        return PERFORMANCE_COLORS['poor']


def plot_mos_distribution(mos_series, save_path=None, title='MOS Distribution'):
    """
    Plot MOS distribution with proper labels and colors.

    Args:
        mos_series (pd.Series): Series containing MOS values
        save_path (str): Path to save figure (optional)
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    setup_plot_style()

    mos_counts = mos_series.value_counts().sort_index()
    labels = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent']
    colors = [MOS_COLORS[i] for i in mos_counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    axes[0].bar(labels[:len(mos_counts)], mos_counts.values, color=colors)
    axes[0].set_xlabel('MOS Rating', fontsize=DEFAULT_LABEL_SIZE, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=DEFAULT_LABEL_SIZE, fontweight='bold')
    axes[0].set_title(title, fontsize=DEFAULT_TITLE_SIZE, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Add count labels
    for i, (label, count) in enumerate(zip(labels[:len(mos_counts)], mos_counts.values)):
        axes[0].text(i, count + max(mos_counts) * 0.02, str(count),
                     ha='center', fontsize=DEFAULT_TICK_SIZE)

    # Pie chart
    axes[1].pie(mos_counts.values, labels=labels[:len(mos_counts)],
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('MOS Proportion', fontsize=DEFAULT_TITLE_SIZE, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")

    return fig


def plot_feature_importance(feature_names, importances, top_n=15, save_path=None):
    """
    Plot feature importance from tree-based models.

    Args:
        feature_names (list): List of feature names
        importances (array): Feature importance values
        top_n (int): Number of top features to display
        save_path (str): Path to save figure (optional)

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    setup_plot_style()

    # Sort and select top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Horizontal bar plot
    y_pos = np.arange(len(top_features))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

    ax.barh(y_pos, top_importances, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=DEFAULT_LABEL_SIZE, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances',
                 fontsize=DEFAULT_TITLE_SIZE, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(top_importances):
        ax.text(v + max(top_importances) * 0.01, i, f'{v:.4f}',
                va='center', fontsize=DEFAULT_TICK_SIZE - 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix with proper formatting.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): Class names for labels
        save_path (str): Path to save figure (optional)
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    from sklearn.metrics import confusion_matrix

    setup_plot_style()

    if class_names is None:
        class_names = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent']

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted MOS', fontsize=DEFAULT_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('True MOS', fontsize=DEFAULT_LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")

    return fig


def plot_model_comparison(results_df, metrics=['Test_Accuracy', 'Test_F1', 'Test_Kappa'],
                          save_path=None):
    """
    Compare multiple models across different metrics.

    Args:
        results_df (pd.DataFrame): DataFrame with columns 'Model' and metric columns
        metrics (list): List of metrics to compare
        save_path (str): Path to save figure (optional)

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))

    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        data = results_df[metric]
        models = results_df['Model']

        # Color code by performance
        colors = [get_performance_color(val) for val in data]

        axes[idx].bar(range(len(models)), data, color=colors)
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(models, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.replace('_', ' '),
                            fontsize=DEFAULT_LABEL_SIZE, fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ")} Comparison',
                           fontsize=DEFAULT_TITLE_SIZE, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([0, 1.0])

        # Add value labels
        for i, v in enumerate(data):
            axes[idx].text(i, v + 0.02, f'{v:.3f}',
                          ha='center', fontsize=DEFAULT_TICK_SIZE)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")

    return fig


def plot_correlation_matrix(corr_matrix, save_path=None, title='Feature Correlation Matrix'):
    """
    Plot correlation matrix heatmap.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix
        save_path (str): Path to save figure (optional)
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE + 2, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")

    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
    print(f"Default DPI: {DEFAULT_DPI}")
    print(f"Performance colors: {PERFORMANCE_COLORS}")
