import pandas as pd
import numpy as np
def generate_business_insights(df, target, problem_type, model=None):
    insights = {
        'kpis': {},
        'summary': '',
        'recommendations': [],
        'risks': []
    }
    
    # KPIs
    insights['kpis']['rows'] = df.shape[0]
    insights['kpis']['features'] = df.shape[1]
    insights['kpis']['missing_pct'] = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    if problem_type == 'classification':
        target_dist = df[target].value_counts(normalize=True)
        insights['kpis']['target_distribution'] = target_dist.to_dict()
        if target_dist.min() < 0.1:
            insights['risks'].append("Severe class imbalance – models may be biased.")
    else:
        insights['kpis']['target_range'] = (df[target].min(), df[target].max())
    
    # Model-based insights
    if model is not None and hasattr(model, 'feature_importance'):
        try:
            fi = model.feature_importance(df)
            top_features = fi.head(3).index.tolist()
            insights['recommendations'].append(f"Focus on {', '.join(top_features)} – they are the strongest drivers.")
        except:
            pass
    
    # Executive summary
    insights['summary'] = f"Dataset contains {df.shape[0]} records with {df.shape[1]} features. Missing rate is {insights['kpis']['missing_pct']:.1f}%. "
    if problem_type == 'classification':
        insights['summary'] += f"Target variable '{target}' has {df[target].nunique()} classes."
    else:
        insights['summary'] += f"Target '{target}' ranges from {df[target].min():.2f} to {df[target].max():.2f}."
    
    return insights
