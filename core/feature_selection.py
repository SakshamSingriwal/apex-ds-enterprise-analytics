try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector as SklearnSequentialSelector, VarianceThreshold
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from typing import Optional
    _SKLEARN_AVAILABLE = True
except Exception:
    np = None
    pd = None
    SelectKBest = None
    SklearnSequentialSelector = None
    VarianceThreshold = None
    mutual_info_classif = None
    mutual_info_regression = None
    Optional = None
    _SKLEARN_AVAILABLE = False


if not _SKLEARN_AVAILABLE:
    def select_features(df, target, task='classification', method='mutual_info', k=50):
        raise RuntimeError('scikit-learn is not installed. Install with `pip install scikit-learn`')

    def select_features_sequential(df, target, task='classification', k=50):
        raise RuntimeError('scikit-learn is not installed. Install with `pip install scikit-learn`')
else:
    def select_features(df: pd.DataFrame, target: str, task: str = 'classification', method: str = 'mutual_info', k: int = 50) -> pd.DataFrame:
        feature_df = df.drop(columns=[target]).select_dtypes(include='number')
        if feature_df.shape[1] == 0:
            raise ValueError("No numeric features found in the DataFrame.")
        
        X = feature_df.values
        y = df[target].values
        
        if method == 'mutual_info':
            selector_fn = mutual_info_classif if task == 'classification' else mutual_info_regression
            selector = SelectKBest(score_func=selector_fn, k=min(k, X.shape[1]))
        elif method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
        elif method == 'f_classif' and task == 'classification':
            from sklearn.feature_selection import f_classif
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        elif method == 'f_regression':
            from sklearn.feature_selection import f_regression
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        else:
            selector_fn = mutual_info_classif if task == 'classification' else mutual_info_regression
            selector = SelectKBest(score_func=selector_fn, k=min(k, X.shape[1]))
        
        selector.fit(X, y)
        selected_mask = selector.get_support()
        selected_cols = feature_df.columns[selected_mask]
        return df[selected_cols.tolist() + [target]]


    def select_features_sequential(df: pd.DataFrame, target: str, task: str = 'classification', 
                                 k: int = 50, direction: str = 'forward',
                                 scoring: Optional[str] = None) -> pd.DataFrame:
        if SklearnSequentialSelector is None:
            raise RuntimeError('sklearn.SequentialFeatureSelector is not available.')
        
        feature_df = df.drop(columns=[target]).select_dtypes(include='number')
        if feature_df.shape[1] == 0:
            raise ValueError("No numeric features found in the DataFrame.")
        
        X = feature_df.values
        y = df[target].values
        
        if task == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            actual_scoring = scoring if scoring else 'accuracy'
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            actual_scoring = scoring if scoring else 'neg_mean_squared_error'
        
        selector = SklearnSequentialSelector(
            estimator=estimator,
            n_features_to_select=min(k, X.shape[1]),
            direction=direction,
            scoring=actual_scoring,
            n_jobs=-1
        )
        
        selector.fit(X, y)
        selected_mask = selector.get_support()
        selected_cols = feature_df.columns[selected_mask]
        return df[selected_cols.tolist() + [target]]