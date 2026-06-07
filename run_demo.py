import os
import sys
import time

os.environ["STREAMLIT_BROWSER_GATHER_USAGE"] = "false"


def _make(df, target, problem):
    class _S:
        current_df = df
        current_target = target
        current_problem = problem
        current_model = None
        current_model_type = None
        deep_learning_history = None
        multimodal_files = []
        multimodal_preprocessed = {}
        data_profile = None
        rag_pipeline = None
        current_forecast = None
    return _S()


def main():
    print("[demo] Quick demo datasets")
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris, fetch_california_housing

    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris["target"] = iris.target

    n = 1000
    df_churn = pd.DataFrame(
        {
            "tenure": np.random.randint(1, 72, n),
            "monthly_charges": np.random.uniform(20, 120, n),
            "age": np.random.randint(18, 70, n),
            "contract": np.random.choice(["monthly", "yearly"], n),
            "churn": np.random.binomial(1, 0.25, n),
        }
    )

    housing = fetch_california_housing()
    df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
    df_housing["target"] = housing.target

    rng = pd.date_range("2024-01-01", periods=120, freq="D")
    df_ts = pd.DataFrame(
        {
            "date": rng,
            "sales": np.sin(np.linspace(0, 10, 120)) * 100 + 150 + np.random.normal(0, 5, 120),
        }
    )
    df_ts["target"] = df_ts["sales"]

    cases = [
        ("Iris (Classification)", df_iris, "target", "classification"),
        ("Churn (Classification)", df_churn, "churn", "classification"),
        ("Housing (Regression)", df_housing, "target", "regression"),
        ("Sales Time Series", df_ts, "sales", "regression"),
    ]

    for name, df, target, problem in cases:
        print(f"\n[dataset] {name}")
        state = _make(df, target, problem)

        try:
            import core.automl as automl
            predictor = automl.train_automl(df, target, problem, time_limit=30)
            state.current_model = predictor
            leaderboard = predictor.leaderboard(silent=True)
            print(f"[automl] leaderboard rows={len(leaderboard)}")
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[automl] skipped: {e}")

        try:
            from core.clustering import run_clustering
            run_clustering(df.select_dtypes(include="number"), method="kmeans")
            print("[clustering] ok")
        except Exception as e:
            print(f"[clustering] skipped: {e}")

        try:
            from core.forecasting import run_forecast
            res = run_forecast(df_ts, "date", "sales", horizon=7)
            print(f"[forecasting] success={res.get('success')}")
        except Exception as e:
            print(f"[forecasting] skipped: {e}")

        try:
            from core.deep_learning import train_deep_learning
            model, history = train_deep_learning(df, target, problem, architecture="ANN", epochs=2, batch_size=16)
            print(f"[deep_learning] ok history_keys={list(history.keys())}")
        except Exception as e:
            print(f"[deep_learning] skipped: {e}")

        try:
            from core.goal_agent import run_goal_agent
            res = run_goal_agent(df, target, problem, target_score=0.80, n_trials=3)
            print(f"[goal_agent] best_score={res.get('best_score')}")
        except Exception as e:
            print(f"[goal_agent] skipped: {e}")

        try:
            from core.multi_agent import run_multi_agent_pipeline
            res = run_multi_agent_pipeline(df, target, problem)
            print(f"[multi_agent] agents={list(res.keys())}")
        except Exception as e:
            print(f"[multi_agent] skipped: {e}")

        try:
            from core.business_insights import generate_business_insights
            insights = generate_business_insights(df, target, problem, state.current_model)
            print(f"[business] kpis={list(insights.get('kpis', {}).keys())[:3]}")
        except Exception as e:
            print(f"[business] skipped: {e}")

        try:
            from core.reports import export_report
            blob = export_report(df.head(20), target, problem, model=state.current_model, format="pdf")
            print(f"[reports] bytes={len(blob)}")
        except Exception as e:
            print(f"[reports] skipped: {e}")

        try:
            from core.sql_agent import SQLAgent
            agent = SQLAgent()
            agent.load_dataframe("data", df.head(50))
            r = agent.ask("SELECT COUNT(*) AS cnt FROM data")
            print(f"[sql] success={r.get('success')}")
        except Exception as e:
            print(f"[sql] skipped: {e}")

        try:
            from core.xai import explain_model_shap
            if state.current_model is not None and hasattr(state.current_model, "predict"):
                shap_res = explain_model_shap(state.current_model, df, target, n_samples=50)
                print(f"[xai] success={shap_res.get('success')}")
            else:
                print("[xai] skipped: no model")
        except Exception as e:
            print(f"[xai] skipped: {e}")

        try:
            from core.rag import RAGPipeline
            import io
            class DummyFile:
                def __init__(self, name, data):
                    self.name = name
                    self._data = data
                def getbuffer(self):
                    return self._data
            text = ("This is a test document. " * 200).encode("utf-8")
            dummy = DummyFile("test.txt", text)
            rag = RAGPipeline(chunk_size=100)
            rag.load_document(dummy)
            print(f"[rag] chunks={len(rag.chunks)}")
        except Exception as e:
            print(f"[rag] skipped: {e}")

    print("\n[demo] finished")


if __name__ == "__main__":
    main()