from pandas import DataFrame, read_csv, to_numeric
from numpy import array, exp, nan, reshape, zeros
from prophet import Prophet
from lightgbm import LGBMRegressor
from os import listdir
from os.path import join
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from collections import defaultdict
from json import dump

# 10 ML Forecasts 
    # Linear Regression, ARIMA (AutoRegressive Integrated Moving Average)
    # LSTM (Long Short-Term Memory), Prophet, Random Forest, XGBoost
    # SVR (Support Vector Regression), SARIMA (Seasonal ARIMA)
    # GBM (Gradient Boosting Machine), and KNN (K-Nearest Neighbors)

# Map intervals to human-readable labels
interval_labels = {
    '1_days': 1,
    '3_days': 3,
    '5_days': 5,
    '7_days': 7,
    '14_days': 14,
    '30_days': 30,
    '91_days': 91,
    '183_days': 183,
    '365_days': 365
}

def load_data(filePath):
    data = read_csv(filePath, index_col='Date', parse_dates=True)
    return data

def linear_regression_forecast(data, intervals):
    data['Date'] = data.index
    data['Date'] = to_numeric(data['Date'])

    x = data[['Date']].values
    y = data['Close'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    poly = PolynomialFeatures(degree=3) # Using degree 3 for polynomial features
    model = make_pipeline(poly, LinearRegression())
    model.fit(x_train, y_train)
    
    predictions = {}
    for interval in intervals:
        future_dates = array([x_test[-1][0] + i for i in range(1, interval+2)]).reshape(-1, 1)
        predictions[f'{interval}_days'] = model.predict(future_dates)

    return predictions

def arima_forecast(data, intervals):
    data = data['Close']
    train_data, test_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
    
    model = ARIMA(train_data, order=(5, 1, 2))
    model_fit = model.fit()
    
    predictions = {}
    for interval in intervals:
        y_pred = model_fit.forecast(steps=interval+1)
        predictions[f'{interval}_days'] = y_pred.values
    
    return predictions

def create_lstm_model(data, feature_col='Close', n_steps=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[[feature_col]].values)

    X, y = [], []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i-n_steps:i, 0])
        y.append(data_scaled[i, 0])

    X, y = array(X), array(y)
    X = reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=25, batch_size=32, verbose=2)

    return model, scaler

def predict_lstm(model, scaler, data, n_steps=30, intervals=[1]):
    inputs = data['Close'][-n_steps:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    inputs = reshape(inputs, (1, n_steps, 1))

    predictions = {}
    for interval in intervals:
        future_inputs = zeros((1, n_steps + interval + 1, 1))
        future_inputs[:, :n_steps, :] = inputs
        for i in range(interval + 1):
            future_inputs[:, n_steps + i, :] = model.predict(future_inputs[:, i:i + n_steps, :])
        predicted_price = scaler.inverse_transform(future_inputs[:, n_steps:, :].reshape(-1, 1))
        predictions[f'{interval}_days'] = predicted_price.flatten()

    return predictions

def create_prophet_model(data):
    df = data.copy()
    df = df.rename_axis('ds').reset_index()
    df = df.rename(columns={'Close': 'y'})
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    
    return model

def predict_prophet(model, periods):
    future = model.make_future_dataframe(periods=periods[-1] + 1)
    forecast = model.predict(future)
    
    predictions = {}
    for period in periods:
        predictions[f'{period}_days'] = forecast['yhat'].values[-(period+1):]

    return predictions

def create_rf_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model

def predict_rf(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1]+1)
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-(interval+1):])
    
    return predictions

def create_xgb_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    return model

def predict_xgb(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1]+1)
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-(interval+1):])
    
    return predictions

# SVR with RBF Kernal
def create_svr_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    model = SVR(kernel='rbf', C=100, gamma=0.1) # Using RBF Kernal with adjusted parameters
    model.fit(X, y)

    return model

def predict_svr(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1]+1)
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-(interval+1):])
    
    return predictions

def create_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    return model_fit

def predict_sarima(model_fit, steps):
    predictions = {}
    for step in steps:
        forecast = model_fit.forecast(steps=step+1)
        predictions[f'{step}_days'] = forecast.values

    return predictions

def create_gbm_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = LGBMRegressor()
    model.fit(X, y)

    return model

def predict_gbm(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1]+1)
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-(interval+1):])
    
    return predictions

def create_knn_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)

    return model

def predict_knn(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1]+1)
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-(interval+1):])
    
    return predictions

def calculate_accuracy_percentage(predictions, actual_prices):
    percentage_differences = abs(predictions - actual_prices) / actual_prices * 100
    scaled_differences = exp(-percentage_differences / 5)  # exponential decay scaling
    weighted_accuracy = scaled_differences.mean() * 100
    return weighted_accuracy

def save_forecasts(predictions, actual_prices, data, company_name, accuracy_metrics):
    results = []
    for model_name, model_preds in predictions.items():
        for interval, preds in model_preds.items():
            interval_length = interval_labels[interval]
            actual_interval_prices = actual_prices[-(interval_length + 1):]
            accuracy = calculate_accuracy_percentage(preds, actual_interval_prices)
            accuracy_metrics[model_name][interval].append(accuracy)
            for i, pred in enumerate(preds):
                actual_price_index = len(data) - (interval_labels[interval]) + i - 1
                actual_price = actual_prices[actual_price_index] if actual_price_index >= 0 else nan
                results.append([company_name, model_name, interval, i, pred, actual_price, accuracy])
    df_results = DataFrame(results, columns=['Company', 'Model','Interval', 'Day', 'Prediction', 'Actual', 'Accuracy'])
    df_results.to_csv(f'forecast_data/{company_name}_forecasts.csv', index=False)

def main():
    # Load the data
    folder_path = 'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data'
    feature_cols = ['7_day_MA', '14_day_MA', 'Volume', 'Daily_Return', 'Volatility_7_day', 'High_Low_Diff']
    intervals = [1, 3, 5, 7, 14, 30, 91, 183, 365]  # 1 day, 3 days, 5 days, 1 week, 2 weeks, 1 month, 3 months, 6 months, 1 year

    accuracy_metrics = defaultdict(lambda: defaultdict(list))

    for _ in range(10):
        for filename in listdir(folder_path):
            if filename.endswith(".csv"):
                # Load and append data to all_data
                filePath = join(folder_path, filename)
                company_name = filename.split('_')[0]
                data = load_data(filePath)
                
                predictions = {}
                actual_prices = data['Close'].values

                # Linear Regression Forecast
                predictions['Linear Regression'] = linear_regression_forecast(data, intervals)

                # ARIMA Forecast
                predictions['ARIMA'] = arima_forecast(data, intervals)

                # LSTM Forecast
                lstm_model, lstm_scaler = create_lstm_model(data)
                predictions['LSTM'] = predict_lstm(lstm_model, lstm_scaler, data, intervals=intervals)

                # Prophet Forecast
                prophet_model = create_prophet_model(data)
                predictions['Prophet'] = predict_prophet(prophet_model, intervals)

                # Random Forest Forecast
                rf_model = create_rf_model(data, feature_cols)
                predictions['Random Forest'] = predict_rf(rf_model, data, feature_cols, intervals)

                # XGBoost Forecast
                xgb_model = create_xgb_model(data, feature_cols)
                predictions['XGBoost'] = predict_xgb(xgb_model, data, feature_cols, intervals)

                # SVR Forecast
                svr_model = create_svr_model(data, feature_cols)
                predictions['SVR'] = predict_svr(svr_model, data, feature_cols, intervals)

                # SARIMA Forecast
                sarima_model_fit = create_sarima_model(data)
                predictions['SARIMA'] = predict_sarima(sarima_model_fit, intervals)

                # GBM Forecast
                gbm_model = create_gbm_model(data, feature_cols)
                predictions['GBM'] = predict_gbm(gbm_model, data, feature_cols, intervals)

                # KNN Forecast
                knn_model = create_knn_model(data, feature_cols)
                predictions['KNN'] = predict_knn(knn_model, data, feature_cols, intervals)

                # Save Forecasts
                save_forecasts(predictions, actual_prices, data, company_name, accuracy_metrics)
    
    # Calculate average accuracy metrics
    average_accuracy_metrics = {model: {interval: sum(acc_list)/len(acc_list) for interval, acc_list in intervals_dict.items()} for model, intervals_dict in accuracy_metrics.items()}

    # Save average accuracy metrics
    with open('average_accuracy_metrics.json', 'w') as f:
        dump(average_accuracy_metrics, f, indent=4)
            

if __name__ == '__main__':
    main()