import pandas as pd
from core.automl import train_automl
from core.business_insights import generate_business_insights
def run_multi_agent_pipeline(df, target, problem_type):
    results = {}
    # Explorer Agent
    results['Explorer'] = {
        'shape': df.shape,
        'missing': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    # Data Quality Agent
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    results['Data Quality'] = {'missing_percentage': missing_pct}
    # Model Agent
    predictor = train_automl(df, target, problem_type, time_limit=120)
    leaderboard = predictor.leaderboard()
    results['Model'] = {
        'best_model': leaderboard.iloc[0]['model'],
        'best_score': leaderboard.iloc[0]['score_val']
    }
    # Critic Agent
    score = leaderboard.iloc[0]['score_val']
    if problem_type == 'classification':
        verdict = "OVERFIT" if score > 0.99 else "OK"
    else:
        verdict = "UNDERFIT" if score < 0.5 else "OK"
    results['Critic'] = {'verdict': verdict, 'score': score}
    # Business Agent
    try:
        fi = predictor.feature_importance(df)
        top_features = fi.head(3).index.tolist()
    except:
        top_features = []
    results['Business'] = {
        'recommendation': f"Focus on {', '.join(top_features)} to improve outcomes." if top_features else "No feature importance available."
    }
    return results     # core/goal_agent.py
