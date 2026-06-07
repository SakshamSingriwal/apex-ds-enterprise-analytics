import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, VarianceThreshold, SelectKBest, f_classif
from typing import Tuple, Dict, List, Optional


def preprocess_data(
    df: pd.DataFrame,
    target: str = None,
    problem_type: str = None,
    handle_outliers: bool = True,
    impute_strategy: str = 'median',
    scaling: str = 'robust',
    feature_selection: str = 'none',
    n_features: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None:
        raise ValueError("DataFrame cannot be None")
    df = df.copy()
    
    # Initialize result
    X = df.drop(columns=[target]) if target and target in df.columns else df.copy()
    
    # Target handling
    y = None
    if target and target in df.columns:
        y = df[[target]].copy()
    
    # Outlier handling
    if handle_outliers:
        num_cols = X.select_dtypes(include=np.number).columns
        for col in num_cols:
            q1, q3 = X[col].quantile([0.01, 0.99]) if False else X[col].quantile([0.01, 0.99])
            X[col] = X[col].clip(lower=q1, upper=q3)
    
    # Imputation
    num_cols = X.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        if impute_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=impute_strategy)
        X[num_cols] = imputer.fit_transform(X[num_cols])
    
    # Categorical encoding
    cat_cols = X.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        if X[col].nunique() <= 20:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Scaling
    if scaling == 'robust':
        scaler = RobustScaler()
    elif scaling == 'standard':
        scaler = StandardScaler()
    else:
        scaler = None
    
    if scaler is not None:
        num_cols_final = X.select_dtypes(include=np.number).columns
        if len(num_cols_final) > 0:
            X[num_cols_final] = scaler.fit_transform(X[num_cols_final])
    
    # Feature selection
    if feature_selection != 'none' and y is not None and problem_type in ('classification', 'regression'):
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_features) > n_features:
            if feature_selection == 'mutual_info':
                if problem_type == 'classification':
                    scores = mutual_info_classif(X[numeric_features], y.squeeze(), random_state=42)
                else:
                    scores = mutual_info_regression(X[numeric_features], y.squeeze(), random_state=42)
                importance = pd.Series(scores, index=numeric_features).sort_values(ascending=False)
                selected = importance.head(n_features).index.tolist()
            elif feature_selection == 'variance':
                selector = VarianceThreshold(threshold=0.01)
                X_sel = selector.fit_transform(X[numeric_features])
                selected = numeric_features[i] for i, v in enumerate(selector.get_support()) if v
                selected = list(set(selected) & set(numeric_features))
            elif feature_selection == 'l1':
                from sklearn.linear_model import LassoCV
                l1 = LassoCV(cv=5, max_iter=10000, random_state=42)
                l1.fit(X[numeric_features], y.squeeze())
                selected = [col for col, coef in zip(numeric_features, l1.coef_) if abs(coef) > 1e-5]
            else:
                selected = numeric_features
            
            # Keep selected features + non-numeric features
            non_numeric = [c for c in X.columns if c not in numeric_features]
            keep = selected + non_numeric
            keep = [c for c in keep if c in X.columns]
            X = X[keep]
    
    if y is not None and target in X.columns:
        X = X.drop(columns=[target])
    
    return X, y if y is not None else pd.DataFrame()


def get_preprocessing_summary(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame = None) -> Dict:
    return {
        'original_shape': df.shape,
        'original_features': list(df.columns),
        'target': y.columns[0] if y is not None and not y.empty else None,
        'preprocessed_shape': X.shape,
        'preprocessed_features': list(X.columns),
        'numeric_cols': X.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': X.select_dtypes(exclude=[np.number]).columns.tolist(),
        'missing_values_remaining': int(X.isnull().sum().sum()),
    }