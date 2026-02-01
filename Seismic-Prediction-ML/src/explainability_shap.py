"""
SHAP Explainability Analysis for Random Forest
==============================================
This script performs feature importance analysis using SHAP (SHapley Additive exPlanations).
It trains a Random Forest model and visualizes:
1. Beeswarm Plot: Showing how each feature's value affects the prediction.
2. Bar Plot: Ranking features by their average impact magnitude.

Publication-ready visualizations with "Box Style" formatting.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings


# ==========================================
# 1. Styling Configuration (The "Box" Look)
# ==========================================
def set_publication_style():
    """Sets matplotlib parameters for a boxed, academic look."""
    plt.rcdefaults()
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    except:
        pass

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'axes.linewidth': 1.5,  # Thicker border lines
        'axes.edgecolor': 'black',  # Black borders
        'axes.facecolor': 'white',  # White background
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def add_publication_border(ax):
    """Manually forces the 'Box' style on SHAP plots."""
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)

    ax.grid(True, axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax.tick_params(direction='out', length=6, width=1.5, colors='black', top=False, right=False)


# ==========================================
# 2. Data Loading
# ==========================================
def load_data(file_path, threshold):
    """Loads feature matrix and generates binary labels."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Assuming strict format: Cols 1 to -1 are features, last col is target
    X = df.iloc[:, 1:-1]
    y_raw = df.iloc[:, -1]

    # Generate Binary Labels
    y = (y_raw >= threshold).astype(int)

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}, Positive Samples (M>={threshold}): {y.sum()}")
    return X, y


# ==========================================
# 3. Model Training
# ==========================================
def train_rf_model(X, y, seed=42):
    print("\nTraining Random Forest for explanation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model, X_test


# ==========================================
# 4. SHAP Analysis
# ==========================================
def run_shap_analysis(model, X_test, output_dir):
    print("\nCalculating SHAP values (this may take a moment)...")

    # Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    # --- Robust SHAP Value Extraction ---
    # Handles different return types of shap_values (list vs array)
    shap_values_target = None
    if isinstance(shap_values, list):
        # For binary classification, index 1 usually corresponds to class "1" (Earthquake)
        shap_values_target = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            shap_values_target = shap_values[:, :, 1]  # [Samples, Features, Class]
        else:
            shap_values_target = shap_values

    # Dimension Check
    if shap_values_target.shape != X_test.shape:
        # Try Transpose if mismatch
        if shap_values_target.shape == (X_test.shape[1], X_test.shape[0]):
            shap_values_target = shap_values_target.T

    # ---------------------------
    # Plot 1: Beeswarm Summary
    # ---------------------------
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_target,
        X_test,
        plot_type="dot",
        show=False,
        max_display=20
    )

    ax = plt.gca()
    add_publication_border(ax)
    plt.title("SHAP Feature Importance (Beeswarm)", pad=20, fontweight='bold')

    out_path_1 = os.path.join(output_dir, 'Fig3_SHAP_Summary.png')
    plt.tight_layout()
    plt.savefig(out_path_1, dpi=300, bbox_inches='tight')
    # Optional PDF save (comment out if fontTools issue persists)
    # plt.savefig(out_path_1.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {out_path_1}")
    plt.close()

    # ---------------------------
    # Plot 2: Bar Summary
    # ---------------------------
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_target,
        X_test,
        plot_type="bar",
        show=False,
        max_display=20,
        color='#1f77b4'
    )

    ax = plt.gca()
    add_publication_border(ax)
    plt.title("Mean Feature Importance Ranking", pad=20, fontweight='bold')

    out_path_2 = os.path.join(output_dir, 'Fig4_SHAP_Bar.png')
    plt.tight_layout()
    plt.savefig(out_path_2, dpi=300, bbox_inches='tight')
    # Optional PDF save
    # plt.savefig(out_path_2.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {out_path_2}")
    plt.close()


# ==========================================
# Main
# ==========================================
def main():
    warnings.filterwarnings("ignore")
    set_publication_style()

    # Dynamic Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DEFAULT_DATA = os.path.join(PROJECT_ROOT, 'data', 'California_features.csv')
    DEFAULT_OUT = os.path.join(PROJECT_ROOT, 'results')

    parser = argparse.ArgumentParser(description="SHAP Explainability Analysis")
    parser.add_argument('--data', type=str, default=DEFAULT_DATA, help="Path to input features CSV")
    parser.add_argument('--output', type=str, default=DEFAULT_OUT, help="Directory to save plots")
    parser.add_argument('--threshold', type=float, default=6.0,
                        help="Magnitude threshold for classification (default: 6.0)")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Workflow
    X, y = load_data(args.data, args.threshold)
    model, X_test = train_rf_model(X, y)
    run_shap_analysis(model, X_test, args.output)

    print("\n>>> Analysis Complete.")


if __name__ == '__main__':
    main()