import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
def run_clustering(df, method='kmeans', max_k=10):
    # Select numeric columns only (no target leakage)
    X = df.select_dtypes(include='number').copy()
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for clustering")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if X_scaled.shape[0] < 2:
        raise ValueError("Need at least 2 rows for clustering")

    if method == 'kmeans':
        best_k = 2
        best_score = -1
        max_k_search = min(max_k, X_scaled.shape[0] - 1)
        for k in range(2, max_k_search + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        metrics = {'silhouette': best_score, 'best_k': best_k}
    elif method == 'dbscan':
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = min(5, X_scaled.shape[0])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        eps = np.percentile(distances[:, -1], 90)
        min_samples = min(5, X_scaled.shape[0])
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = -1
        metrics = {'silhouette': sil, 'eps': eps}
    elif method == 'gaussian_mixture':
        n_components = min(5, X_scaled.shape[0])
        if n_components < 2:
            raise ValueError("Need at least 2 rows for Gaussian mixture clustering")
        model = GaussianMixture(n_components=n_components, random_state=42)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = -1
        metrics = {'silhouette': sil, 'components': n_components}
    else:  # agglomerative
        n_clusters = min(5, X_scaled.shape[0])
        if n_clusters < 2:
            raise ValueError("Need at least 2 rows for agglomerative clustering")
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = -1
        metrics = {'silhouette': sil, 'clusters': n_clusters}
    
    # Add labels to dataframe
    df_out = df.copy()
    df_out['cluster'] = labels
    return df_out, labels, model, metrics
