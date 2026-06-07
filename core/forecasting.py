import pandas as pd
import plotly.express as px
try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    _PROPHET_AVAILABLE = False
def run_forecast(df, date_col, value_col, horizon=30):
    try:
        if not _PROPHET_AVAILABLE:
            return {'success': False, 'error': 'prophet package not installed. Install with `pip install prophet`'}
        # Check date column
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Prepare data
        df_prophet = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
        df_prophet = df_prophet.dropna()
        
        if len(df_prophet) < 10:
            return {'success': False, 'error': 'Need at least 10 data points'}
        
        # Train model
        model = Prophet()
        model.fit(df_prophet)
        
        # Forecast
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        
        # Plot
        fig = px.line(forecast, x='ds', y='yhat', title='Forecast', labels={'ds': 'Date', 'yhat': 'Forecast'})
        fig.add_scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Actual')
        
        return {
            'success': True,
            'model': model,
            'forecast': forecast,
            'plot': fig
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
