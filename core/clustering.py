"""
Clustering module: KMeans (auto-K), DBSCAN, Gaussian Mixture, Agglomerative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any


def run_clustering(
    df: pd.DataFrame,
    method: str = "kmeans",
    max_k: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray, Any, Dict[str, Any]]:
    """
    Cluster numeric columns. Returns (cleaned_df, labels, model, metrics).
    """
    # Numeric only – avoid target leakage
    X = df.select_dtypes(include=[np.number]).copy()

    # Impute + scale
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    model: Any
    labels: np.ndarray
    metrics: Dict[str, Any] = {}

    method = method.lower()

    if method == "kmeans":
        best_k = 2
        best_score = -1.0
        for k in range(2, min(max_k + 1, len(X_scaled))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_scaled)
            if len(np.unique(lbl)) < 2:
                continue
            score = float(silhouette_score(X_scaled, lbl))
            if score > best_score:
                best_score = score
                best_k = k
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        metrics["best_k"] = best_k
        metrics["silhouette"] = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0.0

    elif method == "dbscan":
        # Auto-select eps via nearest neighbours heuristic
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        eps = float(np.percentile(distances[:, -1], 90))
        model = DBSCAN(eps=max(eps, 0.3), min_samples=5)
        labels = model.fit_predict(X_scaled)
        unique = np.unique(labels[labels >= 0])
        if len(unique) > 1:
            mask = labels >= 0
            metrics["silhouette"] = float(silhouette_score(X_scaled[mask], labels[mask]))
        else:
            metrics["silhouette"] = 0.0

    elif method == "gaussian":
        best_k = 2
        best_bic = np.inf
        for k in range(2, min(max_k + 1, len(X_scaled))):
            gm = GaussianMixture(n_components=k, random_state=42)
            gm.fit(X_scaled)
            bic = gm.bic(X_scaled)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        model = GaussianMixture(n_components=best_k, random_state=42)
        model.fit(X_scaled)
        labels = model.predict(X_scaled)
        metrics["best_k"] = best_k
        metrics["silhouette"] = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0.0

    elif method in ("agglomerative", "hierarchical"):
        best_k = 2
        best_score = -1.0
        for k in range(2, min(max_k + 1, len(X_scaled))):
            ag = AgglomerativeClustering(n_clusters=k)
            lbl = ag.fit_predict(X_scaled)
            if len(np.unique(lbl)) < 2:
                continue
            score = float(silhouette_score(X_scaled, lbl))
            if score > best_score:
                best_score = score
                best_k = k
        model = AgglomerativeClustering(n_clusters=best_k)
        labels = model.fit_predict(X_scaled)
        metrics["best_k"] = best_k
        metrics["silhouette"] = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0.0

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    result_df = pd.DataFrame(X_imp, columns=X.columns)
    return result_df, labels, model, metrics