import os
os.environ["STREAMLIT_BROWSER_GATHER_USAGE"] = "false"

import json
import traceback
import time

def run_test():
    results = {}
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    start = time.time()

    # Core import
    total += 1
    try:
        import core
        results['import_core'] = 'ok'
        passed += 1
    except Exception as e:
        results['import_core'] = f'ERR: {e}'
        failed += 1

    import pandas as pd
    import numpy as np

    # Datasets
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
        X, y = preprocess_data(df_cls.copy(), target='target', problem_type='classification')
        results['preprocessing'] = f'ok shape {X.shape}'
        passed += 1
    except Exception as e:
        results['preprocessing'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Feature selection
    total += 1
    try:
        from core.feature_selection import select_features
        sel = select_features(df_cls, 'target', task='classification', method='mutual_info', k=2)
        assert 'target' in sel.columns
        results['feature_selection'] = f'ok cols {len(sel.columns)}'
        passed += 1
    except Exception as e:
        results['feature_selection'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Clustering
    total += 1
    try:
        from core.clustering import run_clustering
        run_clustering(df_cls[['num1', 'num2']].copy(), method='kmeans')
        results['clustering'] = 'ok'
        passed += 1
    except Exception as e:
        results['clustering'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Forecasting
    total += 1
    try:
        from core.forecasting import run_forecast
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        ts = pd.DataFrame({'ds': dates, 'y': np.linspace(0, 10, 60) + np.random.normal(0, 0.3, 60)})
        res = run_forecast(ts, 'ds', 'y', horizon=7)
        results['forecasting'] = 'ok' if res.get('success') else f"failed: {res.get('error')}"
        passed += 1 if res.get('success') else 0
        failed += 0 if res.get('success') else 1
    except Exception as e:
        results['forecasting'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Deep learning
    total += 1
    try:
        from core.deep_learning import train_deep_learning
        model, history = train_deep_learning(df_cls, 'target', 'classification', architecture='ANN', epochs=2, batch_size=16)
        results['deep_learning'] = 'ok'
        passed += 1
    except Exception as e:
        results['deep_learning'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # AutoML
    total += 1
    try:
        from core.automl import train_automl
        predictor = train_automl(df_cls.head(300), 'target', 'classification', time_limit=30)
        results['automl'] = 'ok'
        passed += 1
    except Exception as e:
        results['automl'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # SQL Agent
    total += 1
    try:
        from core.sql_agent import SQLAgent
        agent = SQLAgent()
        agent.load_dataframe('data', df_cls.head(20))
        r = agent.execute_sql('SELECT COUNT(*) AS cnt FROM data')
        results['sql_agent'] = 'ok' if r[1] is None else f'ERR: {r[1]}'
        passed += 1 if r[1] is None else 0
        failed += 0 if r[1] is None else 1
    except Exception as e:
        results['sql_agent'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Business insights
    total += 1
    try:
        from core.business_insights import generate_business_insights
        insights = generate_business_insights(df_cls, 'target', 'classification')
        results['business_insights'] = 'ok' if insights and 'kpis' in insights else 'ERR: invalid insights'
        passed += 1 if insights and 'kpis' in insights else 0
        failed += 0 if insights and 'kpis' in insights else 1
    except Exception as e:
        results['business_insights'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Report export
    total += 1
    try:
        from core.reports import export_report
        blob = export_report(df_cls.head(10), 'target', 'classification', model=None, format='pdf')
        results['report_export'] = 'ok' if blob and len(blob) > 0 else 'ERR: empty report'
        passed += 1 if blob and len(blob) > 0 else 0
        failed += 0 if blob and len(blob) > 0 else 1
    except Exception as e:
        results['report_export'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # RAG
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
        results['rag'] = f'ok chunks {len(rag.chunks)}'
        passed += 1
    except Exception as e:
        results['rag'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # XAI
    total += 1
    try:
        from sklearn.ensemble import RandomForestClassifier
        from core.xai import explain_model_shap
        X = df_cls[['num1', 'num2', 'cat']].copy()
        X_enc = pd.get_dummies(X, columns=['cat'], drop_first=True)
        y = df_cls['target']
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_enc, y)
        X_full = X_enc.copy()
        X_full['target'] = y
        shap_res = explain_model_shap(clf, X_full, 'target', n_samples=50)
        results['xai'] = 'ok' if shap_res.get('success') else f"failed: {shap_res.get('error')}"
        passed += 1 if shap_res.get('success') else 0
        failed += 0 if shap_res.get('success') else 1
    except Exception as e:
        results['xai'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Multi-agent
    total += 1
    try:
        from core.multi_agent import run_multi_agent_pipeline
        res = run_multi_agent_pipeline(df_cls, 'target', 'classification')
        results['multi_agent'] = 'ok' if res and isinstance(res, dict) else 'ERR: invalid result'
        passed += 1 if res and isinstance(res, dict) else 0
        failed += 0 if res and isinstance(res, dict) else 1
    except Exception as e:
        results['multi_agent'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    # Goal agent
    total += 1
    try:
        from core.goal_agent import run_goal_agent
        res = run_goal_agent(df_cls, 'target', 'classification', target_score=0.75, n_trials=2)
        results['goal_agent'] = f"ok score={res.get('best_score'):.4f}"
        passed += 1
    except Exception as e:
        results['goal_agent'] = 'ERR: ' + traceback.format_exc()
        failed += 1

    elapsed = time.time() - start
    print(json.dumps({
        'results': results,
        'total': total,
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'elapsed_sec': round(elapsed, 2)
    }, indent=2))

    if failed > 0:
        raise SystemExit(1)

if __name__ == '__main__':
    run_test()