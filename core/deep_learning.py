"""
Deep learning module: ANN, CNN1D, LSTM, GRU, TabTransformer.
"""
from __future__ import annotations

import copy
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna


# ─── Model Architectures ────────────────────────────────────────────────────

class ANN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Linear(128 * 8, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, F)
        x = self.conv(x)                 # (B, 128, 8)
        return self.fc(x.flatten(1))


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers,
            batch_first=True, bidirectional=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, F)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden, layers,
            batch_first=True, bidirectional=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TabTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # nhead must divide d_model; if input_dim < nhead fall back
        if d_model % nhead != 0:
            nhead = 1
        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)             # (B, F, 1)
        x = self.embed(x)               # (B, F, d_model)
        x = self.encoder(x)             # (B, F, d_model)
        return self.fc(x.flatten(1))


# ─── Public helpers ──────────────────────────────────────────────────────────

def get_available_architectures() -> List[str]:
    return ["ANN", "CNN1D", "LSTM", "GRU", "Transformer"]


def create_model(
    architecture: str,
    input_dim: int,
    output_dim: int,
    dropout: float = 0.3,
) -> nn.Module:
    if architecture == "ANN":
        return ANN(input_dim, output_dim, dropout)
    elif architecture == "CNN1D":
        return CNN1D(input_dim, output_dim)
    elif architecture == "LSTM":
        return LSTMNet(input_dim, output_dim)
    elif architecture == "GRU":
        return GRUNet(input_dim, output_dim)
    else:
        return TabTransformer(input_dim, output_dim)


# ─── Training ────────────────────────────────────────────────────────────────

def train_deep_learning(
    df: Any,
    target: Any,
    problem_type: Any,
    architecture: str = "ANN",
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 12,
    tune_hyperparameters: bool = False,
    dropout: float = 0.3,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a deep learning model.

    problem_type may be any of the full strings used in the app
    ('binary_classification', 'multiclass_classification', 'classification',
    'regression', …).  Normalised internally.
    """
    # ── Normalise problem type ─────────────────────────────────────────────
    pt_lower = str(problem_type).lower()
    if "regression" in pt_lower:
        is_classification = False
        problem_type_str = "regression"
    else:
        is_classification = True
        problem_type_str = "classification"

    # ── Prepare features ──────────────────────────────────────────────────
    import pandas as pd
    df_pd = pd.DataFrame(df) if not hasattr(df, "columns") else df
    X = df_pd.drop(columns=[target]).select_dtypes(include="number").values.astype(np.float32)
    y_raw = df_pd[target].values

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping target.")

    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(np.int64)
        output_dim = int(np.unique(y).shape[0])
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    else:
        y = y_raw.astype(np.float32)
        output_dim = 1
        loss_fn = nn.MSELoss()

    # ── Train/val split ───────────────────────────────────────────────────
    stratify = y if is_classification and len(np.unique(y)) > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long if is_classification else torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long if is_classification else torch.float32)

    # ── Optional Optuna sweep ─────────────────────────────────────────────
    if tune_hyperparameters:
        def _objective(trial: optuna.Trial) -> float:
            lr_ = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            bs_ = trial.suggest_categorical("batch_size", [16, 32, 64])
            dr_ = trial.suggest_float("dropout", 0.1, 0.5)
            m_ = create_model(architecture, X_tr.shape[1], output_dim, dr_)
            loader_ = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=int(bs_), shuffle=True)
            opt_ = optim.AdamW(m_.parameters(), lr=lr_)
            m_.train()
            for _ in range(10):
                for bx_, by_ in loader_:
                    opt_.zero_grad()
                    out_ = m_(bx_).squeeze()
                    loss_ = loss_fn(out_, by_)
                    loss_.backward()
                    opt_.step()
            m_.eval()
            with torch.no_grad():
                val_loss_ = loss_fn(m_(X_val_t).squeeze(), y_val_t).item()
            return val_loss_

        study = optuna.create_study(direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(_objective, n_trials=10)
        best = study.best_params
        learning_rate = float(best.get("lr", learning_rate))
        batch_size = int(best.get("batch_size", batch_size))
        dropout = float(best.get("dropout", dropout))

    # ── Build final model ─────────────────────────────────────────────────
    model = create_model(architecture, X_tr.shape[1], output_dim, dropout)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses: list = []
    val_losses: list = []

    for _epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx).squeeze()
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / max(len(loader), 1)
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t).squeeze(), y_val_t).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history: Dict[str, Any] = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "architecture": architecture,
        "input_dim": X_tr.shape[1],
        "output_dim": output_dim,
        "problem_type": problem_type_str,
    }
    return model, history