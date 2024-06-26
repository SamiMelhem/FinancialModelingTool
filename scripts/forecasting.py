from pandas import read_csv, to_numeric
from numpy import array, reshape, zeros
from prophet import Prophet
from lightgbm import LGBMRegressor
from os import listdir
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# 10 ML Forecasts 
    # Linear Regression, ARIMA (AutoRegressive Integrated Moving Average)
    # LSTM (Long Short-Term Memory), Prophet, Random Forest, XGBoost
    # SVR (Support Vector Regression), SARIMA (Seasonal ARIMA)
    # GBM (Gradient Boosting Machine), and KNN (K-Nearest Neighbors)

def load_data(filePath):
    data = read_csv(filePath, index_col='Date', parse_dates=True)
    return data

def linear_regression_forecast(data, intervals):
    data['Date'] = data.index
    data['Date'] = to_numeric(data['Date'])

    x = data[['Date']].values
    y = data['Close'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(x_train, y_train)
    
    predictions = {}
    for interval in intervals:
        future_dates = array([x_test[-1][0] + i for i in range(1, interval+1)]).reshape(-1, 1)
        predictions[f'{intervals}_days'] = model.predict(future_dates)

    return predictions

def arima_forecast(data, intervals):
    data = data['Close']
    train_data, test_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
    
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()
    
    predictions = {}
    for interval in intervals:
        y_pred = model_fit.forecast(steps=interval)
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
    inputs = scaler.trasnform(inputs)
    inputs = reshape(inputs, (1, n_steps, 1))

    predictions = {}
    for interval in intervals:
        future_inputs = zeros((1, n_steps + interval, 1))
        future_inputs[:, :n_steps, :] = inputs
        for i in range(interval):
            future_inputs[:, n_steps + i, :] = model.predict(future_inputs[:, i:i + n_steps, :])
        predicted_price = scaler.inverse_transform(future_inputs[:, n_steps:, :].reshape(-1, 1))
        predictions[f'{interval}_days'] = predicted_price.flatten()

    return predictions

def create_prophet_model(data):
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    return model

def predict_prophet(model, periods):
    future = model.make_future_dataframe(periods=periods[-1])
    forecast = model.predict(future)
    
    predictions = {}
    for period in periods:
        predictions[f'{period}_days'] = forecast['yhat'].values[-period:]

    return predictions

def create_rf_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model

def predict_rf(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1])
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-interval:])
    
    return predictions

def create_xgb_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    return model

def predict_xgb(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1])
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-interval:])
    
    return predictions

def create_svr_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = SVR(kernel='rbf')
    model.fit(X, y)

    return model

def predict_svr(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1])
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-interval:])
    
    return predictions

def create_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    return model_fit

def predict_sarima(model_fit, steps):
    predictions = {}
    for step in steps:
        forecast = model_fit.forecast(steps=step)
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
    X_future = data[feature_cols].tail(intervals[-1])
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-interval:])
    
    return predictions

def create_knn_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)

    return model

def predict_knn(model, data, feature_cols, intervals):
    predictions = {}
    X_future = data[feature_cols].tail(intervals[-1])
    for interval in intervals:
        predictions[f'{interval}_days'] = model.predict(X_future[-interval:])
    
    return predictions

def plot_forecasts(data, predictions, company_name):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Actual Close Price')

    for model_name, model_preds in predictions.items():
        for interval, preds in model_preds.items():
            plt.plot(data.index[-len(preds):], preds, label=f'{model_name} {interval}')

    plt.title(f'Model Forecasts Comparison for {company_name}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def main():
    # Load the data
    folder_path = 'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data'
    feature_cols = ['7_day_MA', '14_day_MA', '21_day_MA', 'Volume', 'Daily_Return', 'Volatility_7_day', 'High_Low_Diff']
    intervals = [1, 3, 5, 7, 14, 30, 91, 183, 365]  # 1 day, 3 days, 5 days, 1 week, 2 weeks, 1 month, 3 months, 6 months, 1 year

    for filename in listdir(folder_path):
        if filename.endswith(".csv"):
            # Load and append data to all_data
            filePath = join(folder_path, filename)
            company_name = filename.split('_')[0]
            data = load_data(filePath)
            
            predictions = {}

            # Linear Regression Forecast
            lr_predictions = linear_regression_forecast(data, intervals)
            predictions['Linear Regression'] = lr_predictions

            # ARIMA Forecast
            arima_predictions = arima_forecast(data, intervals)
            predictions['ARIMA'] = arima_predictions

            # LSTM Forecast
            lstm_model, lstm_scaler = create_lstm_model(data)
            lstm_predictions = predict_lstm(lstm_model, lstm_scaler, data, intervals=intervals)
            predictions['LSTM'] = lstm_predictions

            # Prophet Forecast
            prophet_model = create_prophet_model(data)
            prophet_predictions = predict_prophet(prophet_model, intervals)
            predictions['Prophet'] = prophet_predictions

            # Random Forest Forecast
            rf_model = create_rf_model(data, feature_cols)
            rf_predictions = predict_rf(rf_model, data, feature_cols, intervals)
            predictions['Random Forest'] = rf_predictions

            # XGBoost Forecast
            xgb_model = create_xgb_model(data, feature_cols)
            xgb_predictions = predict_xgb(xgb_model, data, feature_cols, intervals)
            predictions['XGBoost'] = xgb_predictions

            # SVR Forecast
            svr_model = create_svr_model(data, feature_cols)
            svr_predictions = predict_svr(svr_model, data, feature_cols, intervals)
            predictions['SVR'] = svr_predictions

            # SARIMA Forecast
            sarima_model_fit = create_sarima_model(data)
            sarima_predictions = predict_sarima(sarima_model_fit, intervals)
            predictions['SARIMA'] = sarima_predictions

            # GBM Forecast
            gbm_model = create_gbm_model(data, feature_cols)
            gbm_predictions = predict_gbm(gbm_model, data, feature_cols, intervals)
            predictions['GBM'] = gbm_predictions

            # KNN Forecast
            knn_model = create_knn_model(data, feature_cols)
            knn_predictions = predict_knn(knn_model, data, feature_cols, intervals)
            predictions['KNN'] = knn_predictions

            plot_forecasts(data, predictions, company_name)

if __name__ == '__main__':
    main()