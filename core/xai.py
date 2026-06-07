try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    shap = None
    _SHAP_AVAILABLE = False
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def explain_model_shap(model, df, target, n_samples=100):
    try:
        if not _SHAP_AVAILABLE:
            return {'success': False, 'error': 'shap package not installed. Install with `pip install shap`'}
        X = df.drop(columns=[target])
        # Sample if too large
        if len(X) > n_samples:
            X = X.sample(n_samples, random_state=42)
        # Keep original columns so we can present the same features to the model
        orig_cols = X.columns.tolist()
        X_for_explainer = X.fillna(0)
        # Ensure explainer input is numeric float array (SHAP requires finite numeric inputs)
        try:
            X_proc = X_for_explainer.astype(float)
        except Exception:
            return {'success': False, 'error': 'Could not coerce features to float for SHAP explainer'}

        # Wrapper to reconstruct DataFrame with original columns before calling model
        def predict_wrapper(x_input):
            df_in = pd.DataFrame(x_input, columns=orig_cols)
            try:
                pred = model.predict(df_in)
            except Exception:
                # fallback: try passing numpy array directly
                pred = model.predict(x_input)
            arr = np.asarray(pred)
            try:
                return arr.astype(float)
            except Exception:
                raise RuntimeError('Model predictions could not be cast to float for SHAP')

        # quick check: ensure predict_wrapper returns float array
        try:
            _ = predict_wrapper(X_proc.iloc[:5].values)
        except Exception as e:
            return {'success': False, 'error': f'Predict wrapper failed: {e}'}

        explainer = shap.Explainer(predict_wrapper, X_proc)
        shap_values = explainer(X_proc)
        
        # Generate plots
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        ax1 = plt.gcf()
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        ax2 = plt.gcf()
        
        return {'success': True, 'summary_plot': ax1, 'bar_plot': ax2}
    except Exception as e:
        return {'success': False, 'error': str(e)}
