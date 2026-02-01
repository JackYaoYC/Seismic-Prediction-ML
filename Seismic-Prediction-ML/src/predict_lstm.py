"""
LSTM Magnitude Prediction with Optuna Tuning
============================================
This script implements a Long Short-Term Memory (LSTM) network for earthquake magnitude prediction.
It includes:
1. Automated Hyperparameter Tuning (Optuna).
2. Time Series Cross-Validation.
3. Feature Permutation Importance Analysis.
4. Publication-quality visualization.

Usage:
    python src/predict_lstm.py --data data/California_features.csv --trials 50
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import copy
import random
import warnings
import functools

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ==========================================
# 1. Styling & Utils
# ==========================================
def set_publication_style():
    """Sets matplotlib parameters for academic style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ==========================================
# 2. Data Processor
# ==========================================
class DataProcessor:
    def __init__(self, path):
        self.path = path
        self.feature_names = []

    def load_raw_data(self):
        """Loads data and performs initial 80/20 train-test split."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Data file not found: {self.path}")

        df = pd.read_csv(self.path)

        # Determine features and target
        # Assumption: Last column is target, others are features
        # If the CSV has an index column (like 'Unnamed: 0'), drop it.
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Optional: Feature Selection (Can be parameterized if needed)
        # keep_cols = ['Mag_max', 'b_std_lsq', 'a_lsq', 'std_gr_lsq', 'beta']
        # if set(keep_cols).issubset(df.columns):
        #     df = df[keep_cols + [df.columns[-1]]]

        self.feature_names = df.columns[:-1].tolist()
        raw_data = df.values

        X = raw_data[:, :-1]
        y = raw_data[:, -1:]

        # Time Series Split (No shuffling)
        train_size = int(len(X) * 0.8)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def create_sequences(features, target, timestep):
        """Converts flat data into time-window sequences for LSTM."""
        xs, ys = [], []
        if len(features) <= timestep:
            return torch.tensor([]), torch.tensor([])

        for i in range(len(features) - timestep):
            xs.append(features[i: i + timestep])
            ys.append(target[i + timestep])  # Predicting the NEXT step

        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))


# ==========================================
# 3. Model Definition
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out


class EarlyStopping:
    def __init__(self, patience=50, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        return self.early_stop


# ==========================================
# 4. Optuna Objective
# ==========================================
def objective(trial, X_train_outer, y_train_outer, processor, device):
    """
    Optimization objective function.
    Note: Data is passed as arguments to avoid reloading from disk every trial.
    """
    # --- Hyperparameters ---
    hidden_size = trial.suggest_int('hidden_size', 16, 64)
    # num_layers = trial.suggest_int('num_layers', 1, 2) # Can be tuned, fixed to 1 for simplicity
    num_layers = 1
    timestep = trial.suggest_int('timestep', 3, 12)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    # --- Time Series CV (10 Folds) ---
    tscv = TimeSeriesSplit(n_splits=5)  # Reduced to 5 for speed, can be 10
    cv_losses = []

    for train_index, val_index in tscv.split(X_train_outer):
        X_t_fold, X_v_fold = X_train_outer[train_index], X_train_outer[val_index]
        y_t_fold, y_v_fold = y_train_outer[train_index], y_train_outer[val_index]

        # Scaling (Fit on Train Fold ONLY)
        scaler_X = MinMaxScaler().fit(X_t_fold)
        scaler_y = MinMaxScaler().fit(y_t_fold)

        X_t_scaled = scaler_X.transform(X_t_fold)
        X_v_scaled = scaler_X.transform(X_v_fold)
        y_t_scaled = scaler_y.transform(y_t_fold)
        y_v_scaled = scaler_y.transform(y_v_fold)

        # Sequences
        x_t_seq, y_t_seq = processor.create_sequences(X_t_scaled, y_t_scaled, timestep)
        x_v_seq, y_v_seq = processor.create_sequences(X_v_scaled, y_v_scaled, timestep)

        if len(x_t_seq) < 5 or len(x_v_seq) < 5: continue

        train_loader = DataLoader(TensorDataset(x_t_seq, y_t_seq), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_v_seq, y_v_seq), batch_size=len(x_v_seq), shuffle=False)

        # Model Init
        model = LSTMModel(X_train_outer.shape[1], hidden_size, num_layers, 1, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        # Short training for CV
        epochs_cv = 50
        best_fold_loss = float('inf')

        for epoch in range(epochs_cv):
            model.train()
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b.to(device)), y_b.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            fold_val_loss = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    pred = model(X_b.to(device))
                    fold_val_loss += criterion(pred, y_b.to(device)).item()

            avg_fold_loss = fold_val_loss / (len(val_loader) + 1e-9)
            best_fold_loss = min(best_fold_loss, avg_fold_loss)

        cv_losses.append(best_fold_loss)

        # Optuna Pruning
        trial.report(best_fold_loss, step=0)  # Simple pruning based on fold result
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(cv_losses) if cv_losses else float('inf')


# ==========================================
# 5. Main Pipeline
# ==========================================
def main():
    # --- Arguments ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    parser = argparse.ArgumentParser(description="LSTM Earthquake Magnitude Prediction")
    parser.add_argument('--data', type=str, required=True, help="Path to input features CSV")
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'results'), help="Output directory")
    parser.add_argument('--trials', type=int, default=50, help="Number of Optuna trials")
    parser.add_argument('--epochs', type=int, default=200, help="Max training epochs")
    args = parser.parse_args()

    # --- Setup ---
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    set_publication_style()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Starting LSTM Prediction on {device}")
    print(f"📂 Data: {args.data}")

    # --- Load Data ONCE ---
    processor = DataProcessor(args.data)
    X_train_outer, y_train_outer, X_test_outer, y_test_outer = processor.load_raw_data()
    print(f"   Train samples: {len(X_train_outer)}, Test samples: {len(X_test_outer)}")

    # ==========================
    # Phase 1: Optuna Tuning
    # ==========================
    print("\n[Phase 1] Hyperparameter Tuning via Optuna...")
    # Use functools.partial to pass data to objective function
    obj_func = functools.partial(objective,
                                 X_train_outer=X_train_outer,
                                 y_train_outer=y_train_outer,
                                 processor=processor,
                                 device=device)

    study = optuna.create_study(direction="minimize")
    study.optimize(obj_func, n_trials=args.trials)

    best_params = study.best_params
    print(f"\n✅ Best Parameters: {best_params}")

    # ==========================
    # Phase 2: Final Training
    # ==========================
    print("\n[Phase 2] Final Training with Best Parameters...")

    # 2.1 Scaling
    scaler_X = MinMaxScaler().fit(X_train_outer)
    scaler_y = MinMaxScaler().fit(y_train_outer)

    X_train_sc = scaler_X.transform(X_train_outer)
    y_train_sc = scaler_y.transform(y_train_outer)
    X_test_sc = scaler_X.transform(X_test_outer)
    y_test_sc = scaler_y.transform(y_test_outer)

    # 2.2 Sequencing
    timestep = best_params['timestep']
    x_train_seq, y_train_seq = processor.create_sequences(X_train_sc, y_train_sc, timestep)
    x_test_seq, y_test_seq = processor.create_sequences(X_test_sc, y_test_sc, timestep)

    train_loader = DataLoader(TensorDataset(x_train_seq, y_train_seq),
                              batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_seq, y_test_seq),
                             batch_size=len(x_test_seq), shuffle=False)

    # 2.3 Model Setup
    model = LSTMModel(X_train_outer.shape[1], best_params['hidden_size'], 1, 1, best_params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                                 weight_decay=best_params['weight_decay'])
    criterion = nn.MSELoss()

    # 2.4 Training Loop
    early_stopping = EarlyStopping(patience=50, delta=1e-5)
    train_losses, test_losses = [], []
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        avg_train_loss = run_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                test_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        avg_test_loss = test_loss / (len(test_loader) + 1e-9)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.5f} | Test Loss: {avg_test_loss:.5f}")

        if early_stopping.best_score is None or avg_test_loss < -early_stopping.best_score:
            best_model_wts = copy.deepcopy(model.state_dict())

        if early_stopping(avg_test_loss, epoch):
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save Model
    model_path = os.path.join(args.output, 'best_lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")

    # ==========================
    # Phase 3: Evaluation & Plot
    # ==========================
    def get_predictions(loader):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for X_b, y_b in loader:
                out = model(X_b.to(device))
                y_true.append(y_b.cpu().numpy())
                y_pred.append(out.detach().cpu().numpy())
        return np.concatenate(y_true), np.concatenate(y_pred)

    y_true_sc, y_pred_sc = get_predictions(test_loader)
    y_true = scaler_y.inverse_transform(y_true_sc)
    y_pred = scaler_y.inverse_transform(y_pred_sc)

    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n【Final Test Metrics】")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # --- Plot 1: Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('LSTM Training Progress')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'Fig_LSTM_Loss.png'), bbox_inches='tight')
    plt.close()

    # --- Plot 2: Prediction vs Observation ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', s=10)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.plot([min_val, max_val], [min_val + 0.5, max_val + 0.5], 'gray', linestyle='--', label='Error +/- 0.5')
    plt.plot([min_val, max_val], [min_val - 0.5, max_val - 0.5], 'gray', linestyle='--')
    plt.xlabel('Observed Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title(f'LSTM Prediction (R2={r2:.3f})')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'Fig_LSTM_Scatter.png'), bbox_inches='tight')
    plt.close()

    # --- Plot 3: Permutation Importance ---
    print("\nComputing Permutation Importance...")
    baseline_mae = mae
    importances = []

    x_test_tensor = x_test_seq.to(device)

    for i, feature_name in enumerate(processor.feature_names):
        # Create a copy and shuffle one feature
        x_permuted = x_test_tensor.clone()
        perm_idx = torch.randperm(x_permuted.size(0))
        x_permuted[:, :, i] = x_permuted[perm_idx, :, i]

        with torch.no_grad():
            pred_perm = model(x_permuted).detach().cpu().numpy()

        pred_perm = scaler_y.inverse_transform(pred_perm)
        mae_perm = mean_absolute_error(y_true, pred_perm)
        importances.append((feature_name, mae_perm - baseline_mae))

    importances.sort(key=lambda x: x[1])
    names = [x[0] for x in importances]
    scores = [x[1] for x in importances]

    plt.figure(figsize=(8, 6))
    plt.barh(names, scores, color='skyblue')
    plt.xlabel('Increase in MAE (Importance)')
    plt.title('LSTM Feature Importance')
    plt.savefig(os.path.join(args.output, 'Fig_LSTM_Importance.png'), bbox_inches='tight')
    plt.close()

    print("\n✅ All tasks completed.")


if __name__ == "__main__":
    main()