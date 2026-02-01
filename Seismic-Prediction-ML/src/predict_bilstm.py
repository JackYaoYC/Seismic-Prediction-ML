"""
Bi-LSTM Magnitude Prediction with Optuna Tuning
===============================================
This script implements a Bidirectional LSTM (Bi-LSTM) network for earthquake magnitude prediction.
Key features:
1. Bidirectional context learning (Past & Future dependencies in sequence).
2. Automated Hyperparameter Tuning (Optuna).
3. Robust Permutation Feature Importance.

Usage:
    python src/predict_bilstm.py --data data/California_features.csv --trials 50
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

# Suppress warnings
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

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

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
        xs, ys = [], []
        if len(features) <= timestep:
            return torch.tensor([]), torch.tensor([])

        for i in range(len(features) - timestep):
            xs.append(features[i: i + timestep])
            ys.append(target[i + timestep])

        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))


# ==========================================
# 3. Bi-LSTM Model Definition
# ==========================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional = True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        # Output layer must handle 2x hidden size (Forward + Backward)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Initialize hidden states for bidirectional (num_layers * 2)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Take the last time step output.
        # Note: 'out' shape is (batch, seq, hidden*2)
        out = out[:, -1, :]

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
    # Hyperparameters
    # Note: Reduced max hidden_size slightly because Bi-LSTM doubles parameters
    hidden_size = trial.suggest_int('hidden_size', 16, 64)
    num_layers = 1
    timestep = trial.suggest_int('timestep', 3, 10)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_losses = []

    for train_index, val_index in tscv.split(X_train_outer):
        X_t_fold, X_v_fold = X_train_outer[train_index], X_train_outer[val_index]
        y_t_fold, y_v_fold = y_train_outer[train_index], y_train_outer[val_index]

        scaler_X = MinMaxScaler().fit(X_t_fold)
        scaler_y = MinMaxScaler().fit(y_t_fold)

        X_t_scaled = scaler_X.transform(X_t_fold)
        X_v_scaled = scaler_X.transform(X_v_fold)
        y_t_scaled = scaler_y.transform(y_t_fold)
        y_v_scaled = scaler_y.transform(y_v_fold)

        x_t_seq, y_t_seq = processor.create_sequences(X_t_scaled, y_t_scaled, timestep)
        x_v_seq, y_v_seq = processor.create_sequences(X_v_scaled, y_v_scaled, timestep)

        if len(x_t_seq) < 5 or len(x_v_seq) < 5: continue

        train_loader = DataLoader(TensorDataset(x_t_seq, y_t_seq), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_v_seq, y_v_seq), batch_size=len(x_v_seq), shuffle=False)

        model = BiLSTMModel(X_train_outer.shape[1], hidden_size, num_layers, 1, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

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

        trial.report(best_fold_loss, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(cv_losses) if cv_losses else float('inf')


# ==========================================
# 5. Main Pipeline
# ==========================================
def main():
    # Arguments
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    parser = argparse.ArgumentParser(description="Bi-LSTM Earthquake Magnitude Prediction")
    parser.add_argument('--data', type=str, required=True, help="Path to input features CSV")
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'results'), help="Output directory")
    parser.add_argument('--trials', type=int, default=50, help="Number of Optuna trials")
    parser.add_argument('--epochs', type=int, default=200, help="Max training epochs")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    set_publication_style()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Starting Bi-LSTM Prediction on {device}")
    print(f"📂 Data: {args.data}")

    processor = DataProcessor(args.data)
    X_train_outer, y_train_outer, X_test_outer, y_test_outer = processor.load_raw_data()
    print(f"   Train samples: {len(X_train_outer)}, Test samples: {len(X_test_outer)}")

    # Phase 1: Optuna
    print("\n[Phase 1] Hyperparameter Tuning via Optuna...")
    obj_func = functools.partial(objective,
                                 X_train_outer=X_train_outer,
                                 y_train_outer=y_train_outer,
                                 processor=processor,
                                 device=device)

    study = optuna.create_study(direction="minimize")
    study.optimize(obj_func, n_trials=args.trials)

    best_params = study.best_params
    print(f"\n✅ Best Parameters: {best_params}")

    # Phase 2: Final Training
    print("\n[Phase 2] Final Training with Best Parameters...")

    scaler_X = MinMaxScaler().fit(X_train_outer)
    scaler_y = MinMaxScaler().fit(y_train_outer)

    X_train_sc = scaler_X.transform(X_train_outer)
    y_train_sc = scaler_y.transform(y_train_outer)
    X_test_sc = scaler_X.transform(X_test_outer)
    y_test_sc = scaler_y.transform(y_test_outer)

    timestep = best_params['timestep']
    x_train_seq, y_train_seq = processor.create_sequences(X_train_sc, y_train_sc, timestep)
    x_test_seq, y_test_seq = processor.create_sequences(X_test_sc, y_test_sc, timestep)

    train_loader = DataLoader(TensorDataset(x_train_seq, y_train_seq),
                              batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_seq, y_test_seq),
                             batch_size=len(x_test_seq), shuffle=False)

    model = BiLSTMModel(X_train_outer.shape[1], best_params['hidden_size'], 1, 1, best_params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                                 weight_decay=best_params['weight_decay'])
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(patience=30, delta=1e-5)
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

    model.load_state_dict(best_model_wts)

    model_path = os.path.join(args.output, 'best_bilstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")

    # Phase 3: Evaluation & Plot
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

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n【Final Test Metrics】")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # Plot 1: Loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Bi-LSTM Training Progress')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'Fig_BiLSTM_Loss.png'), bbox_inches='tight')
    plt.close()

    # Plot 2: Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', s=10)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.plot([min_val, max_val], [min_val + 0.5, max_val + 0.5], 'gray', linestyle='--', label='Error +/- 0.5')
    plt.plot([min_val, max_val], [min_val - 0.5, max_val - 0.5], 'gray', linestyle='--')
    plt.xlabel('Observed Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title(f'Bi-LSTM Prediction (R2={r2:.3f})')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'Fig_BiLSTM_Scatter.png'), bbox_inches='tight')
    plt.close()

    # Plot 3: Importance
    print("\nComputing Permutation Importance...")
    baseline_mae = mae
    importances = []

    x_test_tensor = x_test_seq.to(device)

    for i, feature_name in enumerate(processor.feature_names):
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
    plt.title('Bi-LSTM Feature Importance')
    plt.savefig(os.path.join(args.output, 'Fig_BiLSTM_Importance.png'), bbox_inches='tight')
    plt.close()

    print("\n✅ Bi-LSTM Analysis Complete.")


if __name__ == "__main__":
    main()