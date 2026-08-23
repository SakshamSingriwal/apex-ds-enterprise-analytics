"""
Autopilot: feed it ANY tabular dataset (CSV/XLSX/Parquet, up to ~100 GB) and it
runs the entire data-science workflow unattended:

    profile/EDA -> clean -> encode -> scale -> imbalance handling ->
    train/val/test split -> model leaderboard -> evaluation -> SHAP ->
    deployable model bundle -> batch prediction

Large files are profiled with DuckDB (streams from disk, never loads all rows)
and trained on a stratified sample; scoring streams in chunks.

Usage
-----
    python autopilot.py run data.csv                    # full pipeline, auto target
    python autopilot.py run data.csv --target Churn     # explicit target
    python autopilot.py run data.csv --time-budget 300  # cap training seconds
    python autopilot.py predict runs/<run_id> new.csv   # score new data
    python autopilot.py serve runs/<run_id>             # FastAPI endpoint on :8000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RUNS_DIR = Path(__file__).parent / "autopilot_runs"
SAMPLE_ROWS = 500_000          # max rows used for training
CHUNK_ROWS = 250_000           # streaming chunk size for prediction
LARGE_FILE_MB = 500            # above this, use DuckDB sampling instead of full load

TARGET_HINTS = [
    r"^is_?fraud", r"^fraud", r"^churn", r"^target$", r"^label$", r"^y$",
    r"fatal", r"outcome", r"^class$", r"^sentiment$", r"^quality$",
    r"^survived$", r"^default", r"^species$", r"^status$", r"^segmentation$",
]


# ── helpers ────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def load_sample(path: Path, target: Optional[str] = None) -> pd.DataFrame:
    """Load full file if small; DuckDB reservoir sample if large."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if file_size_mb(path) <= LARGE_FILE_MB:
        return pd.read_csv(path, low_memory=False) if suffix == ".csv" else pd.read_parquet(path)

    import duckdb
    src = f"read_csv_auto('{path.as_posix()}')" if suffix == ".csv" else f"read_parquet('{path.as_posix()}')"
    log(f"Large file ({file_size_mb(path):,.0f} MB) — DuckDB reservoir sample of {SAMPLE_ROWS:,} rows")
    return duckdb.sql(
        f"SELECT * FROM {src} USING SAMPLE reservoir({SAMPLE_ROWS} ROWS) REPEATABLE (42)"
    ).df()


def infer_target(df: pd.DataFrame) -> Optional[str]:
    for hint in TARGET_HINTS:
        for col in df.columns:
            if re.search(hint, str(col).strip().lower().replace(" ", "_")):
                return col
    last = df.columns[-1]
    if df[last].nunique(dropna=True) <= max(20, int(len(df) * 0.05)):
        return last
    return None


def infer_problem_type(y: pd.Series) -> str:
    n = y.nunique(dropna=True)
    if n == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(y) and n > 20:
        return "regression"
    return "multiclass"


# ── stage 1: profiling / EDA ───────────────────────────────────────────────

def profile(df: pd.DataFrame, target: str, out: Path) -> Dict[str, Any]:
    log("Stage 1/7: profiling & EDA")
    prof: Dict[str, Any] = {
        "rows_sampled": len(df),
        "columns": df.shape[1],
        "missing_pct": round(float(df.isnull().mean().mean() * 100), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_summary": {},
    }
    for col in df.columns:
        s = df[col]
        info: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isnull().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update(mean=round(float(s.mean()), 4), std=round(float(s.std()), 4),
                        min=float(s.min()), max=float(s.max()))
        prof["column_summary"][col] = info

    if target in df.columns:
        vc = df[target].value_counts(normalize=True)
        prof["target_distribution"] = {str(k): round(float(v), 4) for k, v in vc.head(20).items()}

    (out / "eda_profile.json").write_text(json.dumps(prof, indent=2, default=str))

    # Simple HTML EDA report
    rows = "".join(
        f"<tr><td>{c}</td><td>{i['dtype']}</td><td>{i['missing_pct']}%</td><td>{i['unique']}</td></tr>"
        for c, i in prof["column_summary"].items()
    )
    tgt = "".join(f"<li><b>{k}</b>: {v:.1%}</li>" for k, v in
                  (df[target].value_counts(normalize=True).head(10).items() if target in df else []))
    (out / "eda_report.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>EDA Report</title><style>body{{font-family:system-ui;max-width:900px;margin:2rem auto}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}</style>
<h1>EDA Report</h1>
<p>Rows sampled: {prof['rows_sampled']:,} | Columns: {prof['columns']} |
Missing: {prof['missing_pct']}% | Duplicates: {prof['duplicate_rows']:,}</p>
<h2>Target: {target}</h2><ul>{tgt}</ul>
<h2>Columns</h2><table><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr>{rows}</table>""")
    return prof


# ── stage 2-4: preprocessing, splits, imbalance ────────────────────────────

def build_pipeline(X: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def prepare(df: pd.DataFrame, target: str, problem: str):
    """Clean, drop IDs/leaky cols, split into train/val/test."""
    from sklearn.model_selection import train_test_split

    log("Stage 2/7: cleaning & feature preparation")
    df = df.dropna(subset=[target]).drop_duplicates()

    # Drop ID-like and constant columns
    drop: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        n = df[c].nunique(dropna=True)
        if n <= 1 or (n >= len(df) * 0.98 and not pd.api.types.is_float_dtype(df[c])):
            drop.append(c)
    if drop:
        log(f"  dropping {len(drop)} id/constant column(s): {drop[:8]}")
        df = df.drop(columns=drop)

    # Datetime feature extraction
    for c in df.select_dtypes(include=["object"]).columns:
        if c == target:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9:
            df[f"{c}_year"] = parsed.dt.year
            df[f"{c}_month"] = parsed.dt.month
            df[f"{c}_dow"] = parsed.dt.dayofweek
            df = df.drop(columns=[c])

    y_raw = df[target]
    X = df.drop(columns=[target])

    label_encoder = None
    if problem != "regression" and not pd.api.types.is_numeric_dtype(y_raw):
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y_raw.astype(str)), index=y_raw.index)
    else:
        y = y_raw

    log("Stage 3/7: train/val/test split (70/15/15)")
    strat = y if problem != "regression" else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=strat)
    strat2 = y_tmp if problem != "regression" else None
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=strat2)

    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te), label_encoder, drop


def imbalance_strategy(y: pd.Series, problem: str) -> Tuple[str, Optional[Dict[int, float]]]:
    if problem == "regression":
        return "none", None
    counts = y.value_counts()
    ratio = counts.max() / max(counts.min(), 1)
    if ratio < 3:
        return "none", None
    log(f"Stage 4/7: imbalance detected (ratio {ratio:.1f}:1) — using class weights")
    total = len(y)
    weights = {int(k): total / (len(counts) * v) for k, v in counts.items()}
    return "class_weight", weights


# ── stage 5: training leaderboard ──────────────────────────────────────────

def candidate_models(problem: str, weights: Optional[Dict[int, float]]):
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from xgboost import XGBClassifier, XGBRegressor

    if problem == "regression":
        return {
            "LightGBM": LGBMRegressor(n_estimators=400, verbose=-1, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=400, verbosity=0, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
            "Ridge": Ridge(),
        }
    cw = weights if weights else None
    spw = (max(weights.values()) / min(weights.values())) if weights else 1.0
    return {
        "LightGBM": LGBMClassifier(n_estimators=400, class_weight=cw, verbose=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, scale_pos_weight=spw, verbosity=0,
                                 eval_metric="logloss", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight=cw),
    }


def score(model, X, y, problem: str) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                                 r2_score, roc_auc_score)
    pred = model.predict(X)
    if problem == "regression":
        return {"r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred)}
    out = {
        "accuracy": accuracy_score(y, pred),
        "f1_weighted": f1_score(y, pred, average="weighted"),
    }
    if problem == "binary" and hasattr(model, "predict_proba"):
        out["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return out


def train_leaderboard(train, val, problem, weights, time_budget: int, out: Path):
    from sklearn.pipeline import Pipeline

    log(f"Stage 5/7: training leaderboard (budget {time_budget}s)")
    (X_tr, y_tr), (X_val, y_val) = train, val
    primary = "r2" if problem == "regression" else ("roc_auc" if problem == "binary" else "f1_weighted")

    rows, best_name, best_pipe, best_score = [], None, None, -np.inf
    start = time.time()
    for name, model in candidate_models(problem, weights).items():
        if time.time() - start > time_budget:
            log(f"  budget exhausted, skipping {name}")
            continue
        t0 = time.time()
        pipe = Pipeline([("prep", build_pipeline(X_tr)), ("model", model)])
        try:
            pipe.fit(X_tr, y_tr)
            metrics = score(pipe, X_val, y_val, problem)
        except Exception as exc:
            log(f"  {name} failed: {exc}")
            continue
        m = metrics.get(primary, metrics.get("f1_weighted", metrics.get("accuracy", 0)))
        rows.append({"model": name, **{k: round(v, 4) for k, v in metrics.items()},
                     "fit_s": round(time.time() - t0, 1)})
        log(f"  {name:<20} {primary}={m:.4f} ({rows[-1]['fit_s']}s)")
        if m > best_score:
            best_name, best_pipe, best_score = name, pipe, m

    lb = pd.DataFrame(rows).sort_values(rows[0] and primary or "model", ascending=False) \
        if rows else pd.DataFrame()
    lb.to_csv(out / "leaderboard.csv", index=False)
    if best_pipe is None:
        raise RuntimeError("All candidate models failed")
    return best_name, best_pipe, primary, best_score


# ── stage 6: evaluation + SHAP ─────────────────────────────────────────────

def evaluate(pipe, test, problem, label_encoder, out: Path) -> Dict[str, Any]:
    log("Stage 6/7: final evaluation on held-out test set")
    X_te, y_te = test
    metrics = {k: round(v, 4) for k, v in score(pipe, X_te, y_te, problem).items()}
    result: Dict[str, Any] = {"test_metrics": metrics}

    if problem != "regression":
        from sklearn.metrics import classification_report, confusion_matrix
        pred = pipe.predict(X_te)
        names = [str(c) for c in (label_encoder.classes_ if label_encoder is not None
                                  else sorted(pd.Series(y_te).unique()))]
        result["confusion_matrix"] = confusion_matrix(y_te, pred).tolist()
        result["classification_report"] = classification_report(
            y_te, pred, target_names=names, output_dict=True, zero_division=0)

    # SHAP on a small sample (tree models only; skip quietly otherwise)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        model = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        Xs = X_te.sample(min(500, len(X_te)), random_state=42)
        Xt = prep.transform(Xs)
        feat_names = prep.get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[-1]
        if getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, -1]
        shap.summary_plot(sv, Xt, feature_names=feat_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(out / "shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close("all")
        result["shap"] = "ok"
        log("  SHAP summary saved")
    except Exception as exc:
        result["shap"] = f"skipped: {exc}"

    (out / "evaluation.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# ── stage 7: bundle + prediction ───────────────────────────────────────────

def save_bundle(pipe, meta: Dict[str, Any], label_encoder, out: Path) -> None:
    log("Stage 7/7: saving deployable model bundle")
    joblib.dump({"pipeline": pipe, "label_encoder": label_encoder}, out / "model_bundle.joblib")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def predict_file(run_dir: Path, data_path: Path, out_path: Optional[Path] = None) -> Path:
    bundle = joblib.load(run_dir / "model_bundle.joblib")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    pipe, le = bundle["pipeline"], bundle["label_encoder"]
    target = meta["target"]
    out_path = out_path or (run_dir / f"predictions_{data_path.stem}.csv")

    reader = (pd.read_csv(data_path, chunksize=CHUNK_ROWS, low_memory=False)
              if data_path.suffix == ".csv" else [load_sample(data_path)])
    first = True
    for chunk in reader:
        chunk = chunk.drop(columns=[c for c in [target, *meta.get("dropped_columns", [])]
                                    if c in chunk.columns])
        preds = pipe.predict(chunk)
        if le is not None:
            preds = le.inverse_transform(preds.astype(int))
        res = chunk.copy()
        res["prediction"] = preds
        if meta["problem_type"] == "binary" and hasattr(pipe, "predict_proba"):
            res["probability"] = pipe.predict_proba(chunk)[:, 1]
        res.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
    log(f"Predictions written: {out_path}")
    return out_path


# ── forecasting mode ───────────────────────────────────────────────────────

def _detect_time_and_value(df: pd.DataFrame, date_col: Optional[str], value_col: Optional[str]):
    """Find the datetime column and the numeric series to forecast."""
    if date_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
            if df[c].dtype == object:
                parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.9:
                    date_col = c
                    break
    if date_col is None:
        raise SystemExit("No datetime column found — pass --date-col <column>")

    if value_col is None:
        numerics = [c for c in df.columns
                    if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numerics:
            raise SystemExit("No numeric column to forecast — pass --value-col <column>")
        # Pick the numeric column with the most variation (most likely the signal)
        value_col = max(numerics, key=lambda c: df[c].std() / (abs(df[c].mean()) + 1e-9))
    return date_col, value_col


def _lag_features(s: pd.Series, lags: List[int]) -> pd.DataFrame:
    X = pd.DataFrame(index=s.index)
    for l in lags:
        X[f"lag_{l}"] = s.shift(l)
    X["rolling_mean_7"] = s.shift(1).rolling(7, min_periods=1).mean()
    X["rolling_std_7"] = s.shift(1).rolling(7, min_periods=1).std()
    idx = s.index
    X["month"] = idx.month
    X["dow"] = idx.dayofweek
    return X


def forecast(data_path: Path, date_col: Optional[str], value_col: Optional[str],
             horizon: int) -> Path:
    """Time-series mode: detect datetime axis, time-split, race SARIMA vs
    LightGBM-with-lags, forecast `horizon` future periods with the winner."""
    from sklearn.metrics import mean_absolute_error

    started = time.time()
    run_dir = RUNS_DIR / f"forecast_{data_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Forecast run: {data_path.name} -> {run_dir}")

    df = load_sample(data_path)
    date_col, value_col = _detect_time_and_value(df, date_col, value_col)
    log(f"Time axis: {date_col!r} | forecasting: {value_col!r}")

    s = (df.assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce", format="mixed")})
           .dropna(subset=[date_col])
           .set_index(date_col)[value_col]
           .sort_index())
    # Aggregate duplicates and infer a regular frequency
    s = s.groupby(level=0).mean()
    freq = pd.infer_freq(s.index) or "D"
    s = s.asfreq(freq).interpolate(limit=5)
    s = s.dropna()
    if len(s) < 30:
        raise SystemExit(f"Only {len(s)} usable time points — need at least 30")
    log(f"Series: {len(s):,} points at freq {freq!r} "
        f"({s.index.min():%Y-%m-%d} -> {s.index.max():%Y-%m-%d})")

    n_test = max(int(len(s) * 0.2), horizon)
    train_s, test_s = s.iloc[:-n_test], s.iloc[-n_test:]
    results: Dict[str, Any] = {}

    # Candidate 1: SARIMA
    sarima_fit = None
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        seasonal = {"D": 7, "W": 52, "M": 12, "MS": 12, "H": 24}.get(freq[:2].rstrip("S"), 0)
        order, s_order = (1, 1, 1), ((1, 0, 1, seasonal) if seasonal and len(train_s) > 2 * seasonal
                                     else (0, 0, 0, 0))
        sarima_fit = SARIMAX(train_s, order=order, seasonal_order=s_order,
                             enforce_stationarity=False).fit(disp=False)
        pred = sarima_fit.forecast(len(test_s))
        results["SARIMA"] = mean_absolute_error(test_s, pred)
        log(f"  SARIMA               mae={results['SARIMA']:.4f}")
    except Exception as exc:
        log(f"  SARIMA failed: {exc}")

    # Candidate 2: LightGBM with lag features (recursive forecast)
    lgbm, lags = None, [1, 2, 3, 7, 14]
    try:
        from lightgbm import LGBMRegressor
        X_all = _lag_features(s, lags)
        X_tr, y_tr = X_all.iloc[:-n_test].dropna(), None
        y_tr = s.loc[X_tr.index]
        lgbm = LGBMRegressor(n_estimators=300, verbose=-1, random_state=42).fit(X_tr, y_tr)

        hist = train_s.copy()
        preds = []
        for ts in test_s.index:
            row = _lag_features(pd.concat([hist, pd.Series([np.nan], index=[ts])]), lags).iloc[[-1]]
            p = float(lgbm.predict(row)[0])
            preds.append(p)
            hist.loc[ts] = p
        results["LightGBM_lags"] = mean_absolute_error(test_s, preds)
        log(f"  LightGBM_lags        mae={results['LightGBM_lags']:.4f}")
    except Exception as exc:
        log(f"  LightGBM failed: {exc}")

    if not results:
        raise SystemExit("Both forecasting models failed")
    best = min(results, key=results.get)
    log(f"Best: {best} — refitting on full history, forecasting {horizon} periods")

    future_idx = pd.date_range(s.index.max(), periods=horizon + 1, freq=freq)[1:]
    if best == "SARIMA":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        full = SARIMAX(s, order=sarima_fit.specification["order"],
                       seasonal_order=sarima_fit.specification["seasonal_order"],
                       enforce_stationarity=False).fit(disp=False)
        fc = full.get_forecast(horizon)
        out_df = pd.DataFrame({"forecast": fc.predicted_mean.values}, index=future_idx)
        ci = fc.conf_int()
        out_df["lower"], out_df["upper"] = ci.iloc[:, 0].values, ci.iloc[:, 1].values
    else:
        X_all = _lag_features(s, lags).dropna()
        lgbm.fit(X_all, s.loc[X_all.index])
        hist, preds = s.copy(), []
        for ts in future_idx:
            row = _lag_features(pd.concat([hist, pd.Series([np.nan], index=[ts])]), lags).iloc[[-1]]
            p = float(lgbm.predict(row)[0])
            preds.append(p)
            hist.loc[ts] = p
        out_df = pd.DataFrame({"forecast": preds}, index=future_idx)

    out_df.index.name = date_col
    out_df.to_csv(run_dir / "forecast.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        s.iloc[-min(len(s), 5 * horizon):].plot(ax=ax, label="history")
        out_df["forecast"].plot(ax=ax, label=f"forecast ({best})", color="tab:red")
        if "lower" in out_df:
            ax.fill_between(out_df.index, out_df["lower"], out_df["upper"], alpha=0.2, color="tab:red")
        ax.legend(); ax.set_title(f"{value_col} — {horizon}-period forecast")
        fig.savefig(run_dir / "forecast.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        log(f"  plot skipped: {exc}")

    meta = {
        "mode": "forecast", "data": str(data_path), "date_col": date_col,
        "value_col": value_col, "freq": freq, "horizon": horizon,
        "backtest_mae": {k: round(v, 4) for k, v in results.items()},
        "best_model": best, "duration_s": round(time.time() - started, 1),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"DONE in {meta['duration_s']}s | best={best} | forecast.csv + forecast.png in {run_dir}")
    return run_dir


# ── orchestrator ───────────────────────────────────────────────────────────

def run(data_path: Path, target: Optional[str], time_budget: int) -> Path:
    started = time.time()
    run_dir = RUNS_DIR / f"{data_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Autopilot run: {data_path.name} -> {run_dir}")

    df = load_sample(data_path)
    target = target or infer_target(df)
    if target is None or target not in df.columns:
        raise SystemExit("Could not infer target column — pass --target <column>")
    problem = infer_problem_type(df[target])
    log(f"Target: {target!r} | problem: {problem}")

    profile(df, target, run_dir)
    train, val, test, le, dropped = prepare(df, target, problem)
    _, weights = imbalance_strategy(train[1], problem)
    best_name, pipe, primary, best = train_leaderboard(train, val, problem, weights, time_budget, run_dir)
    evaluation = evaluate(pipe, test, problem, le, run_dir)

    meta = {
        "data": str(data_path), "target": target, "problem_type": problem,
        "best_model": best_name, f"val_{primary}": round(float(best), 4),
        "test_metrics": evaluation["test_metrics"], "dropped_columns": dropped,
        "rows_trained": len(train[0]), "duration_s": round(time.time() - started, 1),
    }
    save_bundle(pipe, meta, le, run_dir)

    log(f"DONE in {meta['duration_s']}s | best={best_name} | test={evaluation['test_metrics']}")
    log(f"Artifacts: {run_dir}")
    log(f"Serve it:  python autopilot.py serve \"{run_dir}\"")
    return run_dir


def serve(run_dir: Path, port: int = 8000) -> None:
    import uvicorn
    from fastapi import FastAPI

    bundle = joblib.load(run_dir / "model_bundle.joblib")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    pipe, le = bundle["pipeline"], bundle["label_encoder"]

    app = FastAPI(title=f"Autopilot model: {meta['best_model']}")

    @app.get("/")
    def info():
        return meta

    @app.post("/predict")
    def predict(records: List[Dict[str, Any]]):
        df = pd.DataFrame(records)
        preds = pipe.predict(df)
        if le is not None:
            preds = le.inverse_transform(preds.astype(int))
        out = {"predictions": [str(p) for p in preds]}
        if meta["problem_type"] == "binary" and hasattr(pipe, "predict_proba"):
            out["probabilities"] = pipe.predict_proba(df)[:, 1].round(4).tolist()
        return out

    log(f"Serving {meta['best_model']} on http://127.0.0.1:{port}  (POST /predict)")
    uvicorn.run(app, host="127.0.0.1", port=port)


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end DS autopilot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline on a dataset")
    p_run.add_argument("data", type=Path)
    p_run.add_argument("--target")
    p_run.add_argument("--time-budget", type=int, default=600)

    p_pred = sub.add_parser("predict", help="score a new file with a finished run")
    p_pred.add_argument("run_dir", type=Path)
    p_pred.add_argument("data", type=Path)

    p_srv = sub.add_parser("serve", help="serve a finished run as a FastAPI endpoint")
    p_srv.add_argument("run_dir", type=Path)
    p_srv.add_argument("--port", type=int, default=8000)

    p_fc = sub.add_parser("forecast", help="time-series forecasting on a dataset")
    p_fc.add_argument("data", type=Path)
    p_fc.add_argument("--date-col")
    p_fc.add_argument("--value-col")
    p_fc.add_argument("--horizon", type=int, default=30)

    args = ap.parse_args()
    if args.cmd == "run":
        run(args.data, args.target, args.time_budget)
    elif args.cmd == "predict":
        predict_file(args.run_dir, args.data)
    elif args.cmd == "forecast":
        forecast(args.data, args.date_col, args.value_col, args.horizon)
    else:
        serve(args.run_dir, args.port)


if __name__ == "__main__":
    main()
