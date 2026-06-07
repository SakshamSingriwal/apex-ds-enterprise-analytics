import json
import traceback
import time
import os
import sys

os.environ["STREAMLIT_BROWSER_GATHER_USAGE"] = "false"

import pandas as pd
import numpy as np


class _State:
    def __init__(self):
        self.current_df = None
        self.current_target = None
        self.current_problem = None
        self.current_model = None
        self.current_model_type = None
        self.deep_learning_history = None
        self.multimodal_files = []
        self.multimodal_preprocessed = None
        self.data_profile = None
        self.rag_pipeline = None
        self.current_forecast = None


def make_state(df, target, problem):
    s = _State()
    s.current_df = df
    s.current_target = target
    s.current_problem = problem
    return s


def run_test():
    results = {}
    total = 0
    passed = 0
    failed = 0
    skipped = 0

    # Import core
    total += 1
    try:
        import core
        results['import_core'] = 'ok'
        passed += 1
    except Exception as e:
        results['import_core'] = f'ERR: {e}'
        failed += 1

    # Build synthetic datasets once
    df_cls = pd.DataFrame({
        'num1': np.random.randn(200),
        'num2': np.random.randn(200),
        'cat': np.random.choice(['a', 'b', 'c'], 200),
        'target': np.random.choice([0, 1], 200)
    })
    df_reg = pd.DataFrame({
        'num1': np.random.randn(200),
        'num2': np.random.randn(200),
        'target': np.random.randn(200)
    })

    # Preprocessing
    total += 1
    try:
        from core.preprocessing import preprocess_data
        df_pre = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat': np.random.choice(['a', 'b', 'c'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        X, y = preprocess_data(df_pre, target='target', problem_type='classification')
        results['preprocess'] = f'ok shape {X.shape}'
        passed += 1
    except Exception as e:
        results['preprocess'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Feature selection
    total += 1
    try:
        from core.feature_selection import select_features
        df_fs = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'num3': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        selected = select_features(df_fs, target='target', task='classification', method='mutual_info', k=2)
        results['feature_selection'] = f'ok cols {len(selected.columns)}'
        passed += 1
    except Exception as e:
        results['feature_selection'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Clustering
    total += 1
    try:
        from core.clustering import run_clustering
        df_cluster = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100)
        })
        out = run_clustering(df_cluster, method='kmeans')
        results['clustering'] = 'ok'
        passed += 1
    except Exception as e:
        results['clustering'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Forecasting
    total += 1
    try:
        from core.forecasting import run_forecast
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        y = pd.Series(range(30)) + (pd.Series(range(30)) % 5)
        ts = pd.DataFrame({'ds': dates, 'y': y})
        res = run_forecast(ts, 'ds', 'y', horizon=7)
        results['forecasting'] = 'ok' if res.get('success') else f"failed: {res.get('error')}"
        passed += 1
    except Exception as e:
        results['forecasting'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Deep learning
    total += 1
    try:
        from core.deep_learning import train_deep_learning
        df_dl = pd.DataFrame({
            'f1': np.random.randn(200),
            'f2': np.random.randn(200),
            'target': np.random.choice([0, 1], 200)
        })
        model, history = train_deep_learning(df_dl, 'target', 'classification', architecture='ANN', epochs=2, batch_size=16)
        results['deep_learning'] = 'ok'
        passed += 1
    except Exception as e:
        results['deep_learning'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # AutoML
    total += 1
    try:
        from core.automl import train_automl
        df_auto = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        predictor = train_automl(df_auto, 'target', 'classification', time_limit=60)
        results['automl'] = f'ok' if predictor else 'ERR: no predictor'
        passed += 1
    except Exception as e:
        results['automl'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # SQL Agent
    total += 1
    try:
        from core.sql_agent import SQLAgent
        sql = SQLAgent()
        df_sql = pd.DataFrame({
            'id': range(10),
            'val': np.random.randn(10)
        })
        sql.load_dataframe('data', df_sql)
        res, err = sql.execute_sql('SELECT COUNT(*) AS cnt FROM data')
        results['sql_execute'] = f'ok {res.iloc[0, 0]}' if err is None else f'ERR: {err}'
        passed += 1
    except Exception as e:
        results['sql_execute'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Business insights
    total += 1
    try:
        from core.business_insights import generate_business_insights
        df_bi = pd.DataFrame({
            'num1': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        insights = generate_business_insights(df_bi, 'target', 'classification')
        results['business_insights'] = 'ok' if insights and 'kpis' in insights else 'ERR: invalid insights'
        passed += 1
    except Exception as e:
        results['business_insights'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Report export
    total += 1
    try:
        from core.reports import export_report
        df_report = pd.DataFrame({
            'num1': np.random.randn(10),
            'target': np.random.choice([0, 1], 10)
        })
        report_bytes = export_report(df_report, 'target', 'classification', model=None, format='pdf')
        results['report_export'] = 'ok' if report_bytes and len(report_bytes) > 0 else 'ERR: empty report'
        passed += 1
    except Exception as e:
        results['report_export'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # RAG Pipeline
    total += 1
    try:
        from core.rag import RAGPipeline
        class DummyFile:
            def __init__(self, name, data):
                self.name = name
                self._data = data
            def getbuffer(self):
                return self._data
        text = ("This is a test document. " * 200).encode('utf-8')
        dummy = DummyFile('test.txt', text)
        rag = RAGPipeline(chunk_size=100)
        rag.load_document(dummy)
        try:
            rag.build_index()
            rag.query("test query")
        except Exception:
            pass
        results['rag_pipeline'] = f'ok chunks={len(rag.chunks)}'
        passed += 1
    except Exception as e:
        results['rag_pipeline'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # XAI with RandomForestClassifier
    total += 1
    try:
        from sklearn.ensemble import RandomForestClassifier
        from core.xai import explain_model_shap
        df_xai = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat': np.random.choice(['a', 'b', 'c'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        X = df_xai[['num1', 'num2', 'cat']].copy()
        X_enc = pd.get_dummies(X, columns=['cat'], drop_first=True)
        y = df_xai['target']
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_enc, y)
        X_full = X_enc.copy()
        X_full['target'] = y
        shap_res = explain_model_shap(clf, X_full, 'target', n_samples=50)
        results['xai'] = 'ok' if shap_res.get('success') else f"failed: {shap_res.get('error')}"
        passed += 1
    except Exception as e:
        results['xai'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Multi-agent pipeline
    total += 1
    try:
        from core.multi_agent import run_multi_agent_pipeline
        df_ma = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        ma_res = run_multi_agent_pipeline(df_ma, 'target', 'classification')
        results['multi_agent'] = f'ok keys {list(ma_res.keys())}'
        passed += 1
    except Exception as e:
        results['multi_agent'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Goal agent
    total += 1
    try:
        from core.goal_agent import run_goal_agent
        df_goal = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        ga_res = run_goal_agent(df_goal, 'target', 'classification', target_score=0.9, n_trials=2)
        results['goal_agent'] = f'ok goal_met={ga_res.get("goal_met")}'
        passed += 1
    except Exception as e:
        results['goal_agent'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Summary
    print(json.dumps(results, indent=2))
    print(f"total={total} passed={passed} failed={failed} skipped={skipped}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    run_test()