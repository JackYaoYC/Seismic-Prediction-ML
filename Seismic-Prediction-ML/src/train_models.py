"""
Earthquake Prediction Model Benchmarking Tool
=============================================
This script evaluates 8 different machine learning models for earthquake prediction.
It features:
1. Automated evaluation across multiple magnitude thresholds.
2. Publication-quality visualization (Metrics Comparison + ROC Curves).
3. Strict separation of training/testing data to avoid data leakage.

Original Logic: Optimized for [Bridging Synthetic and Observed Seismicity: An Interpretable Machine Learning Framework for Revealing Distinct Precursor Mechanisms]
Refactored: [2025-12-26]

Usage:
    python train_models.py --data ./data/California_features.csv --output ./results
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgb

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Data Settings
    'random_seed': 42,
    'test_size': 0.2,

    # Thresholds for analysis (Magnitude)
    # Adjust these ranges based on your region (e.g., Japan vs California)
    'mag_thresholds': np.arange(5.0, 7.1, 0.1).round(1).tolist(),
    'roc_target_threshold': 6.0,  # The specific threshold to plot ROC for

    # Feature Selection logic:
    # "1:-1" means: drop column 0 (index/time), keep middle as features, last column as target.

    # Visualization Colors
    'colors': {
        'rf': '#D62728',  # Red for Random Forest
        'xgb': '#1F77B4',  # Blue for XGBoost
        'other': '#7F7F7F'  # Gray for others
    }
}


def get_models():
    """
    Returns a list of models to evaluate.
    Format: (Name, Model_Instance, Needs_Scaling_Flag)
    Note: Tree-based models (RF, XGB, DT) often don't strictly need scaling,
    but scaling usually doesn't hurt. Linear/Distance models (SVM, MLP, LR) MUST be scaled.
    """
    seed = CONFIG['random_seed']
    return [
        ('Logistic Regression', LogisticRegression(random_state=seed), True),
        ('SVM', svm.SVC(kernel='rbf', C=1, probability=True, random_state=seed), True),
        ('Decision Tree', tree.DecisionTreeClassifier(criterion='gini', random_state=seed), False),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=seed), False),
        ('AdaBoost', AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=seed, algorithm='SAMME'),
         False),
        ('LightGBM',
         lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, verbose=-1, random_state=seed), False),
        ('MLP', MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=seed), True),
        ('XGBoost', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed, verbosity=0),
         False)
    ]


# =====================================
# Data Processing
# =====================================
def load_data(file_path):
    """
    Loads data using Pandas.
    Assumes:
      - Column 0: Metadata/Time (Ignored)
      - Columns 1 to N-1: Features
      - Column N: Target (Magnitude)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading data from: {file_path}")
    # Using read_csv. If your data has no header, add header=None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    # Drop NaNs if any
    df = df.dropna()

    # Feature Slicing logic based on user description:
    # X = Col 1 to -1 (Features)
    # y_raw = Col -1 (Magnitude, continuous)
    X = df.iloc[:, 1:-1].values
    y_raw = df.iloc[:, -1].values

    print(f"Features shape: {X.shape}, Target shape: {y_raw.shape}")
    return X, y_raw


# =====================================
# Visualization
# =====================================
def set_publication_style():
    """Sets matplotlib style for academic publishing."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'mathtext.fontset': 'stix',
    })


def plot_metrics_comparison(thresholds, results, output_dir):
    """Plots the 2x2 metrics grid."""
    set_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle('Performance Comparison of Earthquake Prediction Models', fontweight='bold', y=1.02)

    metrics_map = [
        ('acc', 'Accuracy', axes[0, 0], '(a)'),
        ('pre', 'Precision', axes[0, 1], '(b)'),
        ('rec', 'Recall', axes[1, 0], '(c)'),
        ('f1', 'F1 Score', axes[1, 1], '(d)')
    ]

    # Helper for style
    def get_style(name):
        if name == 'Random Forest':
            return {'color': CONFIG['colors']['rf'], 'lw': 2.5, 'alpha': 1.0, 'zorder': 10}
        elif name == 'XGBoost':
            return {'color': CONFIG['colors']['xgb'], 'lw': 2.5, 'alpha': 1.0, 'zorder': 9}
        else:
            return {'color': CONFIG['colors']['other'], 'lw': 1.0, 'alpha': 0.3, 'zorder': 1}

    # Plotting
    for key, label, ax, tag in metrics_map:
        for model_name, metrics in results.items():
            style = get_style(model_name)
            ax.plot(thresholds, metrics[key], label=model_name if key == 'acc' else "", **style)

        ax.set_ylabel(label, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.text(-0.08, 1.05, tag, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

        if label in ['Recall', 'F1 Score']:
            ax.set_xlabel("Magnitude Threshold (Mw)", fontweight='bold')

    # Custom Legend (Global)
    legend_elements = [
        plt.Line2D([0], [0], color=CONFIG['colors']['rf'], lw=2.5, label='Random Forest'),
        plt.Line2D([0], [0], color=CONFIG['colors']['xgb'], lw=2.5, label='XGBoost'),
        plt.Line2D([0], [0], color=CONFIG['colors']['other'], lw=1.5, label='Other Models')
    ]
    fig.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=(0.95, 0.05), frameon=True, fontsize=12)

    out_path = os.path.join(output_dir, 'Fig1_Model_Comparison.pdf')
    ##plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved Metrics Figure to {out_path}")
    # plt.show() # Uncomment if running locally with UI


def plot_roc(roc_data, threshold, output_dir):
    """Plots ROC curves for the specific threshold."""
    set_publication_style()
    plt.figure(figsize=(9, 8))

    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle=':', label='Random Guess')

    # 1. Plot background models
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        if name not in ['Random Forest', 'XGBoost']:
            plt.plot(fpr, tpr, color=CONFIG['colors']['other'], lw=1, alpha=0.3)

    # 2. Plot Highlight models
    for name in ['XGBoost', 'Random Forest']:
        if name in roc_data:
            fpr, tpr, roc_auc = roc_data[name]
            color = CONFIG['colors']['rf'] if name == 'Random Forest' else CONFIG['colors']['xgb']
            plt.plot(fpr, tpr, color=color, lw=2.5, alpha=1, label=f'{name} (AUC = {roc_auc:.3f})')
            plt.fill_between(fpr, tpr, color=color, alpha=0.1)

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'ROC Curves (M ≥ {threshold})', fontweight='bold')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color=CONFIG['colors']['other'], lw=1, label='Other Models'))
    plt.legend(handles=handles, loc="lower right", frameon=True, fontsize=12)

    out_path = os.path.join(output_dir, 'Fig2_ROC_Curve.pdf')
    ##plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved ROC Figure to {out_path}")


# =====================================
# Main Logic
# =====================================
def main():
    # 1. Parse Arguments (Good for Open Source)
    parser = argparse.ArgumentParser(description="Earthquake Prediction Model Trainer")
    parser.add_argument('--data', type=str, required=True, help="Path to the feature CSV file")
    parser.add_argument('--output', type=str, default='./results', help="Directory to save results")
    args = parser.parse_args()

    # 2. Setup
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    X, y_raw = load_data(args.data)
    if X is None: return

    # Initialize results containers
    model_list = get_models()
    results = {name: {'acc': [], 'pre': [], 'rec': [], 'f1': []} for name, _, _ in model_list}
    roc_results = {}  # To store ROC data for the target threshold
    detailed_stats = []

    print(f"\n>>> Starting Training across {len(CONFIG['mag_thresholds'])} thresholds...")

    # 3. Training Loop
    for threshold in CONFIG['mag_thresholds']:
        # Generate Labels for this threshold
        # 1 if Magnitude >= Threshold, else 0
        y_binary = (y_raw >= threshold).astype(int)

        # Split Data (Stratified to maintain class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=CONFIG['test_size'], random_state=CONFIG['random_seed'], stratify=y_binary
        )

        # --- PREPROCESSING (StandardScaler) ---
        # Proper way: Fit on Train, Transform on Test to avoid leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model, needs_scaling in model_list:
            # Select appropriate data
            X_tr = X_train_scaled if needs_scaling else X_train
            X_te = X_test_scaled if needs_scaling else X_test

            # Train
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results[name]['acc'].append(acc)
            results[name]['pre'].append(pre)
            results[name]['rec'].append(rec)
            results[name]['f1'].append(f1)

            # ROC Capture (Only for target threshold)
            if threshold == CONFIG['roc_target_threshold']:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_te)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    roc_results[name] = (fpr, tpr, roc_auc)

            # Detailed Logging for Highlights
            if name in ['Random Forest', 'XGBoost']:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                detailed_stats.append({
                    'Threshold': threshold,
                    'Model': name,
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'F1': f1
                })

    # 4. Output Text Report
    print("\n>>> Detailed Statistics (Selected Thresholds):")
    stats_df = pd.DataFrame(detailed_stats)
    print(stats_df.groupby(['Threshold', 'Model'])['F1'].mean().unstack())

    # 5. Visualization
    print("\n>>> Generating Plots...")
    plot_metrics_comparison(CONFIG['mag_thresholds'], results, args.output)

    if roc_results:
        plot_roc(roc_results, CONFIG['roc_target_threshold'], args.output)
    else:
        print(f"Warning: No ROC data captured. Check if threshold {CONFIG['roc_target_threshold']} exists in loop.")

    print("\n>>> Analysis Complete.")


if __name__ == "__main__":
    main()
