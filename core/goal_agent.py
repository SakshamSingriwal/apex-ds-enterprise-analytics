import optuna
try:
    from autogluon.tabular import TabularPredictor
    _AUTOGLUON_AVAILABLE = True
except Exception:
    TabularPredictor = None
    _AUTOGLUON_AVAILABLE = False


def run_goal_agent(df, target, problem_type, target_score, n_trials=8):
    if problem_type == 'classification':
        if df[target].nunique() == 2:
            prob = 'binary'
        else:
            prob = 'multiclass'
    else:
        prob = 'regression'

    n_samples = int(df.shape[0])
    num_bag_folds = max(2, min(5, n_samples // 100))
    num_stack_levels = 2 if n_samples > 5000 else 1

    if prob == 'binary':
        eval_metric = 'roc_auc'
    elif prob == 'multiclass':
        eval_metric = 'accuracy'
    else:
        eval_metric = 'root_mean_squared_error'

    def objective(trial):
        time_limit = trial.suggest_int('time_limit', 30, 120)

        if not _AUTOGLUON_AVAILABLE:
            raise RuntimeError('autogluon not installed')

        predictor = TabularPredictor(
            label=target,
            problem_type=prob,
            eval_metric=eval_metric,
            presets='medium_quality_faster_train'
        )

        fit_kwargs = {
            'train_data': df,
            'time_limit': time_limit,
            'verbosity': 0,
            'num_bag_folds': num_bag_folds,
            'num_stack_levels': num_stack_levels,
        }
        if prob in ('binary', 'multiclass'):
            fit_kwargs['calibrate_pred_proba'] = True

        predictor.fit(**fit_kwargs)

        leaderboard = predictor.leaderboard(silent=True)
        score = leaderboard.iloc[0]['score_val']

        if eval_metric == 'root_mean_squared_error':
            return -score
        else:
            return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_score = study.best_value
    if eval_metric == 'root_mean_squared_error':
        best_score = -best_score
    goal_met = best_score >= target_score
    return {
        'best_score': best_score,
        'goal_met': goal_met,
        'best_params': study.best_params,
    }