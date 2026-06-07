try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    import numpy as np
    import pandas as pd
    import copy
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    import optuna
    from typing import Tuple, Dict, List, Optional, Union
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None
    np = None
    pd = None
    copy = None
    train_test_split = None
    StratifiedKFold = None
    StandardScaler = None
    LabelEncoder = None
    accuracy_score = None
    mean_squared_error = None
    mean_absolute_error = None
    optuna = None
    Tuple = None
    Dict = None
    List = None
    Optional = None
    Union = None
    _TORCH_AVAILABLE = False


if not _TORCH_AVAILABLE:
    def get_available_architectures():
        return []

    def train_deep_learning(*args, **kwargs):
        raise RuntimeError('PyTorch is not installed. Install with `pip install torch` to use deep learning features')
else:
    # ---------------------------------------------------------------------------
    # RNN recurrent helper modules
    # ---------------------------------------------------------------------------
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1)]

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float = 0.1):
            super().__init__()
            self.layer_norm = nn.LayerNorm(dim)
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.dropout(self.block(self.layer_norm(x)))

    # ---------------------------------------------------------------------------
    # Architecture definitions
    # ---------------------------------------------------------------------------
    class ANN(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None, dropout: float = 0.3):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64]
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.LayerNorm(h))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class CNN1D(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
            super().__init__()
            self.conv_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
            ])
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.fc = nn.Sequential(
                nn.Linear(128 * 8, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            for conv in self.conv_layers:
                x = conv(x)
            x = self.pool(x)
            return self.fc(x.flatten(1))

    class LSTMNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if layers > 1 else 0.0,
            )
            fc_input = hidden * 2 if bidirectional else hidden
            self.fc = nn.Sequential(
                nn.LayerNorm(fc_input),
                nn.Linear(fc_input, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.fc(last)

    class GRUNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if layers > 1 else 0.0,
            )
            fc_input = hidden * 2 if bidirectional else hidden
            self.fc = nn.Sequential(
                nn.LayerNorm(fc_input),
                nn.Linear(fc_input, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.gru(x)
            last = out[:, -1, :]
            return self.fc(last)

    class TransformerNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=dropout,
                dim_feedforward=d_model * 4,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.pos_enc = PositionalEncoding(d_model, max_len=5000)
            self.fc = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.fc(x)

    # ---------------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------------
    def get_available_architectures() -> List[str]:
        return ['ANN', 'CNN1D', 'LSTM', 'GRU', 'Transformer']


    def _build_model(architecture: str, input_dim: int, output_dim: int, hidden: int = 128, layers: int = 2,
                     dropout: float = 0.3) -> nn.Module:
        if architecture == 'ANN':
            return ANN(input_dim, output_dim, hidden_dims=[hidden, max(32, hidden // 2)], dropout=dropout)
        elif architecture == 'CNN1D':
            return CNN1D(input_dim, output_dim, dropout=dropout)
        elif architecture == 'LSTM':
            return LSTMNet(input_dim, output_dim, hidden=hidden, layers=layers, dropout=dropout)
        elif architecture == 'GRU':
            return GRUNet(input_dim, output_dim, hidden=hidden, layers=layers, dropout=dropout)
        elif architecture == 'Transformer':
            return TransformerNet(input_dim, output_dim, d_model=hidden, num_layers=layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")


    def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
        classes, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(classes) * counts)
        return torch.tensor(weights, dtype=torch.float32)


    def _create_sampler(y: np.ndarray) -> WeightedRandomSampler:
        class_counts = np.bincount(y)
        weights = 1.0 / class_counts
        sample_weights = weights[y]
        return WeightedRandomSampler(torch.from_numpy(sample_weights).double(), num_samples=len(y), replacement=True)


    def _train_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn: nn.Module,
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, epochs: int,
                    patience: int, device: torch.device, problem_type: str) -> Dict:
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            all_preds, all_true = [], []
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                if problem_type == 'classification':
                    loss = loss_fn(outputs, batch_y.long())
                    preds = outputs.argmax(dim=1).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_true.extend(batch_y.detach().cpu().numpy())
                else:
                    outputs_squeezed = outputs.squeeze()
                    batch_y_squeezed = batch_y.squeeze().float()
                    loss = loss_fn(outputs_squeezed, batch_y_squeezed)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            train_losses.append(avg_train_loss)

            if problem_type == 'classification' and len(all_preds) > 0:
                train_metrics.append(accuracy_score(all_true, all_preds))
            elif problem_type == 'regression' and len(all_preds) > 0:
                train_metrics.append(mean_absolute_error(all_true, all_preds))

            model.eval()
            val_loss_epoch = 0.0
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    if problem_type == 'classification':
                        loss = loss_fn(outputs, batch_y.long())
                        preds = outputs.argmax(dim=1).detach().cpu().numpy()
                        all_preds.extend(preds)
                        all_true.extend(batch_y.detach().cpu().numpy())
                    else:
                        outputs_squeezed = outputs.squeeze()
                        batch_y_squeezed = batch_y.squeeze().float()
                        loss = loss_fn(outputs_squeezed, batch_y_squeezed)
                    val_loss_epoch += loss.item()
            avg_val_loss = val_loss_epoch / max(len(val_loader), 1)
            val_losses.append(avg_val_loss)

            if problem_type == 'classification' and len(all_preds) > 0:
                val_metrics.append(accuracy_score(all_true, all_preds))
            elif problem_type == 'regression' and len(all_preds) > 0:
                val_metrics.append(mean_absolute_error(all_true, all_preds))

            if scheduler is not None:
                scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss - 1e-6:
                best_loss = avg_val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
        }
        if problem_type == 'classification':
            history['train_acc'] = train_metrics
            history['val_acc'] = val_metrics
        else:
            history['train_mae'] = train_metrics
            history['val_mae'] = val_metrics
        return model, history


    def _optuna_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, input_dim: int, output_dim: int,
                          problem_type: str, epochs: int, batch_size: int, patience: int, device: torch.device,
                          use_sampler: bool) -> float:
        architecture = trial.suggest_categorical('architecture', get_available_architectures())
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        trial_batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if problem_type == 'classification' else None
        indices = np.arange(len(y))
        fold_scores = []

        if skf is not None:
            for train_idx, val_idx in skf.split(indices, y):
                model = _build_model(architecture, input_dim, output_dim, hidden=hidden_dim, layers=n_layers, dropout=dropout)
                model = model.to(device)
                if problem_type == 'classification':
                    loss_fn = nn.CrossEntropyLoss(weight=_compute_class_weights(y[train_idx]).to(device))
                else:
                    loss_fn = nn.MSELoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

                X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
                y_train_t = torch.tensor(y[train_idx], dtype=torch.long if problem_type == 'classification' else torch.float32)
                X_val_t = torch.tensor(X[val_idx], dtype=torch.float32)
                y_val_t = torch.tensor(y[val_idx], dtype=torch.long if problem_type == 'classification' else torch.float32)

                train_ds = TensorDataset(X_train_t, y_train_t)
                val_ds = TensorDataset(X_val_t, y_val_t)

                train_sampler = _create_sampler(y[train_idx]) if (use_sampler and problem_type == 'classification') else None
                train_loader = DataLoader(train_ds, batch_size=trial_batch_size, sampler=train_sampler,
                                          shuffle=train_sampler is None, num_workers=0)
                val_loader = DataLoader(val_ds, batch_size=trial_batch_size, shuffle=False, num_workers=0)

                _, history = _train_fold(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
                                         epochs=min(epochs, 30), patience=min(patience, 6), device=device,
                                         problem_type=problem_type)
                fold_scores.append(min(history['val_loss']))
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model = _build_model(architecture, input_dim, output_dim, hidden=hidden_dim, layers=n_layers, dropout=dropout)
            model = model.to(device)
            if problem_type == 'classification':
                loss_fn = nn.CrossEntropyLoss(weight=_compute_class_weights(y_train).to(device))
            else:
                loss_fn = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.long if problem_type == 'classification' else torch.float32)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long if problem_type == 'classification' else torch.float32)

            train_ds = TensorDataset(X_train_t, y_train_t)
            val_ds = TensorDataset(X_val_t, y_val_t)

            train_sampler = _create_sampler(y_train) if (use_sampler and problem_type == 'classification') else None
            train_loader = DataLoader(train_ds, batch_size=trial_batch_size, sampler=train_sampler,
                                      shuffle=train_sampler is None, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=trial_batch_size, shuffle=False, num_workers=0)

            _, history = _train_fold(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
                                     epochs=min(epochs, 30), patience=min(patience, 6), device=device,
                                     problem_type=problem_type)
            fold_scores.append(min(history['val_loss']))

        return float(np.mean(fold_scores))


    def _select_features(X: np.ndarray, y: np.ndarray, problem_type: str) -> np.ndarray:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        selector_fn = mutual_info_classif if problem_type == 'classification' else mutual_info_regression
        scores = selector_fn(X, y)
        mask = scores > np.median(scores)
        if mask.sum() == 0:
            mask = np.ones(X.shape[1], dtype=bool)
        return X[:, mask]


    def train_deep_learning(df: pd.DataFrame, target: str, problem_type: str, architecture: str = 'ANN',
                            epochs: int = 80, batch_size: int = 32, learning_rate: float = 1e-3,
                            patience: int = 12, tune_hyperparameters: bool = False, dropout: float = 0.3) -> Tuple[nn.Module, Dict]:
        if torch is None:
            raise RuntimeError('PyTorch is not installed. Install with `pip install torch` to use deep learning features')
        if problem_type not in ('classification', 'regression'):
            raise ValueError(f"problem_type must be 'classification' or 'regression', got '{problem_type}'")
        if architecture not in get_available_architectures():
            raise ValueError(f"Unknown architecture '{architecture}'. Available: {get_available_architectures()}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prepare data
        feature_df = df.drop(columns=[target]).select_dtypes(include='number')
        if feature_df.shape[1] == 0:
            raise ValueError("No numeric features found in the DataFrame.")
        X = feature_df.values.astype(np.float32)
        y = df[target].values

        # Feature selection - uses mutual_info internally
        X = _select_features(X, y, problem_type)

        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Encode labels
        label_encoder = None
        if problem_type == 'classification':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            output_dim = len(np.unique(y))
            loss_fn = nn.CrossEntropyLoss()
        else:
            y = y.astype(np.float32)
            output_dim = 1
            loss_fn = nn.MSELoss()

        input_dim = X.shape[1]

        # Hyperparameter tuning
        dropout_val = dropout
        if tune_hyperparameters and optuna is not None:
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: _optuna_objective(
                    trial, X, y, input_dim, output_dim, problem_type, epochs, batch_size, patience, device,
                    use_sampler=True
                ),
                n_trials=20,
                show_progress_bar=False,
            )
            params = study.best_trial.params
            architecture = params['architecture']
            hidden = params['hidden_dim']
            layers = params['n_layers']
            dropout_val = params['dropout']
            learning_rate = params['lr']
            batch_size = params['batch_size']
        elif tune_hyperparameters and optuna is None:
            raise RuntimeError("Optuna is required for hyperparameter tuning. Install with `pip install optuna`.")

        # Stratified train/val split
        test_size_val = 0.15
        if problem_type == 'classification':
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size_val, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size_val, random_state=42
            )

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long if problem_type == 'classification' else torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long if problem_type == 'classification' else torch.float32)

        # Datasets and loaders
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)

        class_weights = _compute_class_weights(y_train) if problem_type == 'classification' else None
        if class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

        train_sampler = _create_sampler(y_train) if problem_type == 'classification' else None
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            shuffle=train_sampler is None, num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        # Model
        model = _build_model(architecture, input_dim, output_dim, hidden=128, layers=2, dropout=dropout_val)
        model = model.to(device)

        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(patience // 3, 2), factor=0.5)

        # Train
        model, history = _train_fold(
            model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            epochs=epochs, patience=patience, device=device, problem_type=problem_type
        )

        # Attach metadata for downstream use
        history['label_encoder'] = label_encoder
        history['scaler'] = scaler
        history['problem_type'] = problem_type
        history['architecture'] = architecture
        history['input_dim'] = input_dim
        history['output_dim'] = output_dim
        history['class_weights'] = class_weights

        return model, history
