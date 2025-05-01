from flask import Flask, render_template, request, jsonify, flash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

app = Flask(__name__)
app.secret_key = "timeseries_analysis_app"

# Global variables
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Helper functions
def create_plot(plt_obj):
    """Convert matplotlib plot to base64 encoded image"""
    buffer = BytesIO()
    plt_obj.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return plot_data


def calculate_errors(y_true, y_pred):
    """Calculate various error metrics"""
    metrics = {}
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)

    # Calculate MAPE and SMAPE handling zeros
    mask = y_true != 0
    if np.any(mask):
        metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['MAPE'] = np.nan

    # SMAPE calculation
    metrics['SMAPE'] = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)) * 100

    # R-squared (only for regression contexts)
    metrics['R2'] = r2_score(y_true, y_pred)

    return metrics


def perform_stationarity_test(series):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    result = adfuller(series.dropna())
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4]
    }
    return output


def prepare_timeseries_data(df, column_name, split_ratio=0.8):
    """Prepare time series data for modeling"""
    # Handle datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            # If no date column, create an artificial time index
            df.index = pd.date_range(start='2000-01-01', periods=len(df))

    # Handle missing values in the selected column
    if df[column_name].isnull().any():
        df[column_name] = df[column_name].interpolate(method='time')

    # Split the data
    train_size = int(len(df) * split_ratio)
    train_data = df[column_name][:train_size]
    test_data = df[column_name][train_size:]

    return train_data, test_data


def fit_arima_model(train_data, p, d, q):
    """Fit ARIMA model with given parameters"""
    try:
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print(f"Error fitting ARIMA({p},{d},{q}): {str(e)}")
        return None


def fit_sarima_model(train_data, p, d, q, P, D, Q, s):
    """Fit SARIMA model with given parameters"""
    try:
        model = SARIMAX(train_data,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        return model_fit
    except Exception as e:
        print(f"Error fitting SARIMA: {str(e)}")
        return None


def fit_lstm_model(train_data, test_data, look_back=1, epochs=50, batch_size=1):
    """Fit LSTM model for time series prediction"""
    if not TF_AVAILABLE:
        return None, "TensorFlow not available"

    try:
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))

        # Create dataset with lookback
        X_train, y_train = [], []
        for i in range(len(scaled_train) - look_back):
            X_train.append(scaled_train[i:(i + look_back), 0])
            y_train.append(scaled_train[i + look_back, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshape for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile and fit
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Prepare test data
        test_data_array = test_data.values.reshape(-1, 1)
        scaled_test = scaler.transform(test_data_array)

        # Use the model to make predictions
        X_test, y_test = [], []
        for i in range(len(scaled_test) - look_back):
            X_test.append(scaled_test[i:(i + look_back), 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Make predictions
        predicted_scaled = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted_scaled)

        return predicted.flatten(), None
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read and process the file
        try:
            df = pd.read_csv(file_path)
            # Store data in session or a temporary file
            temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_data.csv')
            df.to_csv(temp_file, index=False)

            # Return column names and basic info
            columns = df.columns.tolist()
            info = {
                "columns": columns,
                "rows": len(df),
                "preview": df.head(5).to_html(classes="table table-striped table-sm")
            }
            return jsonify(info)
        except Exception as e:
            return jsonify({"error": str(e)})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get form data
        column_name = request.form.get('column')
        model_type = request.form.get('model_type', 'arima')
        split_ratio = float(request.form.get('split_ratio', 0.8))

        # Load data
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_data.csv')
        df = pd.read_csv(temp_file)

        # Prepare time series data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        train_data, test_data = prepare_timeseries_data(df, column_name, split_ratio)

        # Create visualizations
        plots = {}

        # Original time series plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column_name], label='Original Series')
        plt.title(f'Time Series: {column_name}')
        plt.xlabel('Date')
        plt.ylabel(column_name)
        plt.grid(True)
        plt.legend()
        plots['original'] = create_plot(plt)

        # Stationarity test
        stationarity_result = perform_stationarity_test(df[column_name])

        # ACF and PACF plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(df[column_name].dropna(), lags=40, ax=ax1)
        plot_pacf(df[column_name].dropna(), lags=40, ax=ax2)
        plots['acf_pacf'] = create_plot(plt)

        # Seasonal decomposition
        # Replace the seasonal decomposition section with this code:

        # Seasonal decomposition
        try:
            # First check if we have a proper DatetimeIndex with frequency
            series = df[column_name].dropna()

            # If the index is a DatetimeIndex but frequency is not set
            if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
                # Try to infer frequency
                inferred_freq = pd.infer_freq(series.index)
                if inferred_freq:
                    # Create a new Series with the inferred frequency
                    series = series.asfreq(inferred_freq)
                else:
                    # If frequency cannot be inferred, use a default period
                    # For daily data, common choices are 7 (weekly), 30 (monthly), or 365 (yearly)
                    period = 12  # Default to 12 for monthly data, adjust based on your data
                    decomposition = seasonal_decompose(series, model='additive', period=period)
            else:
                # If index already has frequency or is not a DatetimeIndex
                # Specify a period explicitly based on the data frequency
                if isinstance(series.index, pd.DatetimeIndex):
                    # Try to determine an appropriate period based on the frequency
                    freq_str = series.index.freq.freqstr if series.index.freq else ''
                    if 'D' in freq_str:  # Daily data
                        period = 7  # Weekly seasonality
                    elif 'M' in freq_str:  # Monthly data
                        period = 12  # Yearly seasonality
                    elif 'Q' in freq_str:  # Quarterly data
                        period = 4  # Yearly seasonality
                    else:
                        period = 12  # Default
                else:
                    # For non-datetime indices, use a reasonable default
                    period = min(12, len(series) // 2)  # Default period

                decomposition = seasonal_decompose(series, model='additive', period=period)

            # Plot the decomposition
            fig, axes = plt.subplots(4, 1, figsize=(12, 16))
            decomposition.observed.plot(ax=axes[0])
            axes[0].set_title('Observed')
            decomposition.trend.plot(ax=axes[1])
            axes[1].set_title('Trend')
            decomposition.seasonal.plot(ax=axes[2])
            axes[2].set_title('Seasonality')
            decomposition.resid.plot(ax=axes[3])
            axes[3].set_title('Residuals')
            plt.tight_layout()
            plots['decomposition'] = create_plot(plt)
        except Exception as e:
            plots['decomposition'] = None
            print(f"Error in seasonal decomposition: {str(e)}")

        # Fit model and make predictions
        predictions = None
        error_metrics = None
        model_info = None

        if model_type == 'arima':
            # Get ARIMA parameters from form
            p = int(request.form.get('p', 5))
            d = int(request.form.get('d', 1))
            q = int(request.form.get('q', 0))

            model_fit = fit_arima_model(train_data, p, d, q)
            if model_fit:
                predictions = model_fit.forecast(steps=len(test_data))
                model_info = {
                    'type': 'ARIMA',
                    'params': f'({p},{d},{q})',
                    'aic': model_fit.aic,
                    'bic': model_fit.bic
                }

        elif model_type == 'sarima':
            # Get SARIMA parameters from form
            p = int(request.form.get('p', 1))
            d = int(request.form.get('d', 1))
            q = int(request.form.get('q', 1))
            P = int(request.form.get('P', 0))
            D = int(request.form.get('D', 0))
            Q = int(request.form.get('Q', 0))
            s = int(request.form.get('s', 12))  # Seasonal period

            model_fit = fit_sarima_model(train_data, p, d, q, P, D, Q, s)
            if model_fit:
                predictions = model_fit.forecast(steps=len(test_data))
                model_info = {
                    'type': 'SARIMA',
                    'params': f'({p},{d},{q})x({P},{D},{Q},{s})',
                    'aic': model_fit.aic,
                    'bic': model_fit.bic
                }

        elif model_type == 'lstm' and TF_AVAILABLE:
            look_back = int(request.form.get('look_back', 3))
            epochs = int(request.form.get('epochs', 50))
            batch_size = int(request.form.get('batch_size', 1))

            predictions, error = fit_lstm_model(
                train_data,
                test_data,
                look_back=look_back,
                epochs=epochs,
                batch_size=batch_size
            )

            if predictions is not None:
                # Adjust predictions to match test data length due to look_back
                if len(predictions) < len(test_data):
                    # Pad predictions to match test_data length
                    padding = np.array([np.nan] * (len(test_data) - len(predictions)))
                    predictions = np.append(padding, predictions)
                model_info = {
                    'type': 'LSTM',
                    'params': f'look_back={look_back}, epochs={epochs}, batch_size={batch_size}'
                }

        # Calculate errors if predictions are available
        if predictions is not None and len(predictions) == len(test_data):
            error_metrics = calculate_errors(test_data.values, predictions)

            # Plot predictions vs actual
            plt.figure(figsize=(12, 6))
            plt.plot(train_data.index, train_data, 'b-', label='Training Data')
            plt.plot(test_data.index, test_data, 'g-', label='Actual Values')
            plt.plot(test_data.index, predictions, 'r--', label='Predictions')
            plt.title(f'{model_type.upper()} Model Predictions vs Actual')
            plt.xlabel('Date')
            plt.ylabel(column_name)
            plt.legend()
            plt.grid(True)
            plots['predictions'] = create_plot(plt)

            # Plot residuals
            residuals = test_data.values - predictions
            plt.figure(figsize=(12, 6))
            plt.plot(test_data.index, residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Residuals')
            plt.xlabel('Date')
            plt.ylabel('Error')
            plt.grid(True)
            plots['residuals'] = create_plot(plt)

            # QQ Plot for residuals
            plt.figure(figsize=(10, 6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('QQ Plot of Residuals')
            plots['qq_plot'] = create_plot(plt)

            # Distribution of residuals
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.title('Distribution of Residuals')
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plots['residual_dist'] = create_plot(plt)

        # Prepare data for the template
        analysis_results = {
            'column_name': column_name,
            'model_type': model_type,
            'model_info': model_info,
            'stationarity': stationarity_result,
            'error_metrics': error_metrics,
            'plots': plots
        }

        return render_template('analysis.html', results=analysis_results)

    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get form data
        column_name = request.form.get('column')
        forecast_periods = int(request.form.get('forecast_periods', 30))
        model_type = request.form.get('model_type', 'arima')

        # Load data
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_data.csv')
        df = pd.read_csv(temp_file)

        # Prepare time series data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        series = df[column_name]

        # Fit model on full data
        forecast_values = None
        model_info = None

        if model_type == 'arima':
            p = int(request.form.get('p', 5))
            d = int(request.form.get('d', 1))
            q = int(request.form.get('q', 0))

            model_fit = fit_arima_model(series, p, d, q)
            if model_fit:
                forecast_values = model_fit.forecast(steps=forecast_periods)
                model_info = {
                    'type': 'ARIMA',
                    'params': f'({p},{d},{q})'
                }

        elif model_type == 'sarima':
            p = int(request.form.get('p', 1))
            d = int(request.form.get('d', 1))
            q = int(request.form.get('q', 1))
            P = int(request.form.get('P', 0))
            D = int(request.form.get('D', 0))
            Q = int(request.form.get('Q', 0))
            s = int(request.form.get('s', 12))

            model_fit = fit_sarima_model(series, p, d, q, P, D, Q, s)
            if model_fit:
                forecast_values = model_fit.forecast(steps=forecast_periods)
                model_info = {
                    'type': 'SARIMA',
                    'params': f'({p},{d},{q})x({P},{D},{Q},{s})'
                }

        # Generate future dates
        last_date = df.index[-1]
        if isinstance(last_date, pd.Timestamp):
            # For daily data
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods)
        else:
            # For numeric index
            future_dates = np.arange(len(df), len(df) + forecast_periods)

        # Plot the forecast
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, series, 'b-', label='Historical Data')
        plt.plot(future_dates, forecast_values, 'r--', label='Forecast')
        plt.title(f'{model_type.upper()} Forecast for {column_name}')
        plt.xlabel('Date')
        plt.ylabel(column_name)
        plt.legend()
        plt.grid(True)
        forecast_plot = create_plot(plt)

        # Prepare forecast results
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_values
        })

        forecast_results = {
            'column_name': column_name,
            'model_info': model_info,
            'forecast_plot': forecast_plot,
            'forecast_table': forecast_df.to_html(classes="table table-striped table-sm", index=False)
        }

        return render_template('forecast.html', results=forecast_results)

    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/model_results', methods=['POST'])
def api_model_results():
    """API endpoint for retrieving model results"""
    try:
        # Parse request data
        data = request.json
        column_name = data.get('column')
        model_type = data.get('model_type', 'arima')

        # Load data
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_data.csv')
        df = pd.read_csv(temp_file)

        # Prepare time series data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        train_data, test_data = prepare_timeseries_data(df, column_name)

        # Fit model based on type
        results = {
            'model_type': model_type,
            'column': column_name,
            'success': False,
            'error': None
        }

        if model_type == 'arima':
            # Try different combinations of p, d, q
            best_aic = float('inf')
            best_params = None
            best_model = None

            for p in range(0, 4):
                for d in range(0, 2):
                    for q in range(0, 4):
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            model_fit = model.fit()
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_params = (p, d, q)
                                best_model = model_fit
                        except:
                            continue

            if best_model:
                predictions = best_model.forecast(steps=len(test_data))
                error_metrics = calculate_errors(test_data.values, predictions)

                results['success'] = True
                results['best_params'] = best_params
                results['aic'] = best_aic
                results['error_metrics'] = error_metrics

        return jsonify(results)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)