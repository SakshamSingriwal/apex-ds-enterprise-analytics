"""
Batch runner: discovers every dataset across the Desktop AI/ML/DL projects and
runs the full Apex DS pipeline (clean -> AutoML -> importance -> XAI -> insights
-> report) on each one, unattended.

Usage
-----
    python batch_runner.py --scan                 # discover datasets, write registry
    python batch_runner.py                        # run every enabled job in registry
    python batch_runner.py --only fraud_detection # run one job
    python batch_runner.py --time-limit 600       # per-job AutoML budget (seconds)

The registry (batch_registry.json) is written on --scan and is meant to be edited
by hand: fix a wrong target, disable a job, change the problem type.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core.automl import train_automl, get_feature_importance
from core.business_insights import generate_business_insights
from core.preprocessing import preview_clean
from core.reports import export_report
from core.xai import explain_model_shap

DESKTOP = Path(__file__).resolve().parent.parent
# All roots scanned by default: the Desktop projects plus the python_bia coursework folder.
DEFAULT_ROOTS = [DESKTOP, Path("C:/python_bia")]
REGISTRY = Path(__file__).parent / "batch_registry.json"
RUNS_DIR = Path(__file__).parent / "batch_runs"

# Directories that never contain project data worth training on.
SKIP_DIRS = {
    ".venv", "venv", "node_modules", ".git", "__pycache__", ".kilo",
    "AutogluonModels", "catboost_info", ".streamlit", ".vscode", "temp",
    # python_bia is a venv root — skip its interpreter/library dirs
    "Lib", "Scripts", "Include", "share", "etc", ".ipynb_checkpoints",
}
MIN_ROWS = 50
MAX_ROWS_LOADED = 200_000

# Files that are pipeline *outputs*, not source data. Training on these leaks the
# label (predictions.csv holds the model's own scores) or trains on a metrics table.
SKIP_FILE_PATTERNS = [
    r"^predictions", r"^powerbi_", r"^leaderboard", r"^metrics",
    r"^feature_importance", r"^experiment_log", r"^thresholds",
    r"^fairness", r"^robustness", r"^business_metrics", r"^feature_ablation",
]

logger = logging.getLogger("apex_ds.batch")


# ── discovery ──────────────────────────────────────────────────────────────

def _is_skipped(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    name = path.name.lower()
    return any(re.search(p, name) for p in SKIP_FILE_PATTERNS)


def discover_datasets(root: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in ("*.csv", "*.xlsx", "*.parquet"):
        for p in root.rglob(pattern):
            if not _is_skipped(p.relative_to(root)):
                found.append(p)
    return sorted(set(found))


def load_dataset(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if path.suffix == ".xlsx":
        return pd.read_excel(path, nrows=nrows)
    return pd.read_parquet(path)


# Column-name hints, strongest first. A dataset whose columns match one of these
# almost always has that column as the label.
TARGET_HINTS = [
    r"^is_?fraud", r"^fraud", r"^churn", r"^target$", r"^label$", r"^y$",
    r"fatal", r"death", r"murder", r"^default", r"^survived$", r"outcome",
    r"^class$", r"^status$",
]


def infer_target(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for hint in TARGET_HINTS:
        for col in cols:
            if re.search(hint, str(col).strip().lower().replace(" ", "_")):
                return col
    # Fallback: last column, if it looks like a label rather than an ID.
    last = cols[-1]
    if df[last].nunique(dropna=True) <= max(20, len(df) * 0.05):
        return last
    return None


def infer_problem_type(series: pd.Series) -> str:
    n_unique = series.nunique(dropna=True)
    if n_unique == 2:
        return "binary_classification"
    if pd.api.types.is_numeric_dtype(series) and n_unique > 20:
        return "regression"
    return "multiclass_classification"


def build_registry(roots: List[Path]) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for root in roots:
        if not root.exists():
            logger.warning("Root does not exist, skipping: %s", root)
            continue
        for path in discover_datasets(root):
            try:
                head = load_dataset(path, nrows=5000)
            except Exception as exc:
                logger.warning("Unreadable, skipping %s: %s", path, exc)
                continue
            if len(head) < MIN_ROWS or head.shape[1] < 2:
                continue

            target = infer_target(head)
            if target is None:
                logger.info("No target inferable, skipping %s", path.name)
                continue

            rel = path.relative_to(root)
            project = rel.parts[0] if len(rel.parts) > 1 else root.name
            job_id = re.sub(r"[^a-z0-9]+", "_", f"{project}_{path.stem}".lower()).strip("_")
            base_id = job_id
            n = 2
            while job_id in seen_ids:
                job_id = f"{base_id}_{n}"
                n += 1
            seen_ids.add(job_id)

            jobs.append({
                "id": job_id,
                "project": project,
                "dataset": str(path),
                "target": target,
                "problem_type": infer_problem_type(head[target]),
                "enabled": True,
            })

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "roots": [str(r) for r in roots],
        "jobs": jobs,
    }


# ── execution ──────────────────────────────────────────────────────────────

def run_job(job: Dict[str, Any], time_limit: int, run_dir: Path) -> Dict[str, Any]:
    """Run the full pipeline for one dataset. Never raises; returns a result dict."""
    started = time.time()
    out = run_dir / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"id": job["id"], "project": job["project"], "stages": {}}

    try:
        df = load_dataset(Path(job["dataset"]), nrows=MAX_ROWS_LOADED)
        target, problem_type = job["target"], job["problem_type"]
        if target not in df.columns:
            raise ValueError(f"target {target!r} not in {Path(job['dataset']).name}")

        df = df.dropna(subset=[target])
        df = preview_clean(df, target=target)
        result["stages"]["load"] = {"rows": len(df), "cols": df.shape[1]}

        predictor = train_automl(df, target, problem_type, time_limit=time_limit)
        if predictor is None:
            raise RuntimeError("AutoML returned no predictor (see log above)")
        result["stages"]["automl"] = "ok"

        try:
            lb = predictor.leaderboard(silent=True)
            lb.to_csv(out / "leaderboard.csv", index=False)
            result["best_model"] = str(lb.iloc[0]["model"])
            result["best_score"] = float(lb.iloc[0]["score_val"])
        except Exception as exc:
            logger.warning("[%s] leaderboard failed: %s", job["id"], exc)

        fi = get_feature_importance(predictor, df)
        if fi is not None:
            fi.to_csv(out / "feature_importance.csv")
            result["stages"]["importance"] = "ok"

        try:
            shap_out = explain_model_shap(predictor, df, target)
            result["stages"]["xai"] = "ok" if shap_out.get("success") else shap_out.get("error")
        except Exception as exc:
            result["stages"]["xai"] = f"failed: {exc}"

        insights = generate_business_insights(df, target, problem_type, predictor)
        (out / "insights.json").write_text(json.dumps(insights, indent=2, default=str))
        result["stages"]["insights"] = "ok"

        try:
            (out / "report.pdf").write_bytes(
                export_report(df, target, problem_type, predictor, format="pdf")
            )
            result["stages"]["report"] = "ok"
        except Exception as exc:
            result["stages"]["report"] = f"failed: {exc}"

        result["status"] = "success"

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        (out / "traceback.txt").write_text(traceback.format_exc())
        logger.error("[%s] %s", job["id"], exc)

    result["duration_s"] = round(time.time() - started, 1)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Apex DS pipeline across every Desktop project.")
    ap.add_argument("--scan", action="store_true", help="rebuild the registry and exit")
    ap.add_argument("--only", help="run a single job id")
    ap.add_argument("--time-limit", type=int, default=300, help="AutoML seconds per job")
    ap.add_argument(
        "--root", action="append", dest="roots",
        help="directory to scan (repeatable; default: Desktop + C:/python_bia)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.scan or not REGISTRY.exists():
        roots = [Path(r) for r in args.roots] if args.roots else DEFAULT_ROOTS
        registry = build_registry(roots)
        REGISTRY.write_text(json.dumps(registry, indent=2))
        print(f"Registry written: {REGISTRY} ({len(registry['jobs'])} jobs)")
        for job in registry["jobs"]:
            print(f"  {job['id']:<45} target={job['target']!r} ({job['problem_type']})")
        if args.scan:
            print("\nReview targets above, then run: python batch_runner.py")
            return 0
    else:
        registry = json.loads(REGISTRY.read_text())

    jobs = [j for j in registry["jobs"] if j.get("enabled", True)]
    if args.only:
        jobs = [j for j in jobs if j["id"] == args.only]
        if not jobs:
            print(f"No enabled job with id {args.only!r}", file=sys.stderr)
            return 1

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(jobs)} job(s) -> {run_dir}\n")

    results = [run_job(job, args.time_limit, run_dir) for job in jobs]
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))

    print("\n=== Summary ===")
    for r in results:
        mark = "OK  " if r["status"] == "success" else "FAIL"
        score = f" score={r['best_score']:.4f}" if "best_score" in r else ""
        detail = f" {r.get('error', '')}" if r["status"] == "failed" else score
        print(f"{mark} {r['id']:<45} {r['duration_s']:>7.1f}s{detail}")

    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{len(results) - failed} succeeded, {failed} failed. Artifacts in {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
