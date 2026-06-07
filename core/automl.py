try:
    from autogluon.tabular import TabularPredictor
    _AUTOGLUON_AVAILABLE = True
except Exception:
    TabularPredictor = None
    _AUTOGLUON_AVAILABLE = False

import pandas as pd
from typing import Optional, Dict, Any


def train_automl(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    time_limit: int = 300,
    preset: str = "medium_quality_faster_train",
    eval_metric: Optional[str] = None,
    calibrate: bool = True,
    num_bag_folds: int = 5,
    num_stack_levels: int = 1,
) -> TabularPredictor:
    if problem_type == 'classification':
        if df[target].nunique() == 2:
            problem = 'binary'
        else:
            problem = 'multiclass'
    else:
        problem = 'regression'

    if df is None:
        raise ValueError('Input DataFrame is None. Please provide a valid dataset.')
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input must be a pandas DataFrame')
    n_samples = int(df.shape[0])
    if n_samples == 0:
        raise ValueError('Input DataFrame is empty. Load a dataset with at least one row.')
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    if df[target].dropna().shape[0] == 0:
        raise ValueError('Target column contains only missing values.')

    if not _AUTOGLUON_AVAILABLE:
        raise RuntimeError('autogluon is not installed. Install with `pip install autogluon`')

    # Pick a robust default metric tuned for generalization
    if eval_metric is None:
        if problem == 'regression':
            eval_metric = 'root_mean_squared_error'
        else:
            eval_metric = 'roc_auc' if problem == 'binary' else 'log_loss'

    train_data = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    predictor = TabularPredictor(
        label=target,
        problem_type=problem,
        eval_metric=eval_metric,
        verbosity=0,
    )

    # Stronger regularization / robustness:
    # - stacking (controlled levels/folds)
    # - bagging with holdout (reduces over-reliance on train set)
    # - calibration improves probability reliability
    fit_args = {
        'time_limit': max(time_limit, 60),
        'calibrate': calibrate,
        'num_bag_folds': min(num_bag_folds, max(2, n_samples // 200)),
        'num_stack_levels': num_stack_levels,
        'holdout_frac': 0.1,
        'auto_stack': True,
        'verbosity': 0,
    }

    predictor.fit(train_data, **fit_args)

    # Persist lightweight diagnostics for UI / logging
    diagnostics: Dict[str, Any] = {
        'preset': preset,
        'eval_metric': eval_metric,
        'time_limit': int(time_limit),
        'num_bag_folds': int(fit_args['num_bag_folds']),
        'num_stack_levels': int(num_stack_levels),
    }

    try:
        leaderboard = predictor.leaderboard(silent=True)
        best = leaderboard.iloc[0]
        diagnostics['best_model'] = best['model']
        diagnostics['best_score'] = float(best['score_val']) if 'score_val' in best else None
        diagnostics['fit_time'] = float(best['fit_time']) if 'fit_time' in best else None
    except Exception:
        pass

    # Attach diagnostics to the predictor so app.py can read them
    try:
        setattr(predictor, '_fit_diagnostics', diagnostics)
    except Exception:
        pass

    return predictor


def get_feature_importance(predictor, df: pd.DataFrame):
    try:
        return predictor.feature_importance(df)
    except Exception:
        return None