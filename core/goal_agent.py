"""
Goal-driven AutoML agent using Optuna to find optimal time_limit.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

import pandas as pd
import optuna

logger = logging.getLogger("apex_ds.goal_agent")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_goal_agent(
    df: pd.DataFrame,
    target: Any,
    problem_type: Any,
    target_score: float = 0.85,
    n_trials: int = 8,
) -> Dict[str, Any]:
    """
    Search for best AutoGluon time_limit that meets target_score.
    Returns dict with best_score, goal_met, best_params.
    """
    from core.automl import train_automl

    best_score = 0.0
    best_params: Dict[str, Any] = {}

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score, best_params
        time_limit = trial.suggest_int("time_limit", 30, 180, step=30)
        preset = trial.suggest_categorical("preset", ["medium_quality_faster_train", "good_quality_faster_train"])

        predictor = train_automl(
            df, str(target), str(problem_type),
            time_limit=time_limit,
            preset=str(preset),
            calibrate=False,
        )
        if predictor is None:
            return 0.0

        try:
            lb = predictor.leaderboard(silent=True)
            if len(lb) == 0:
                return 0.0
            score = float(lb.iloc[0]["score_val"])
        except Exception:
            return 0.0

        # Optuna minimises; we want to maximise score
        if score > best_score:
            best_score = score
            best_params = {"time_limit": time_limit, "preset": preset}

        return -score  # negate for minimisation

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_score": best_score,
        "goal_met": best_score >= target_score,
        "best_params": best_params,
        "target_score": target_score,
    }