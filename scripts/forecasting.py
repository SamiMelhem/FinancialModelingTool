from pandas import read_csv, to_datetime, to_numeric
from numpy import array, reshape
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    """
    Load the historical data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: DataFrame containing the loaded data
    """
    data = read_csv(filePath, index_col='Date', parse_dates=True)
    return data

def linear_regression_forecast(data):
    """
    Apply linear regression to forecast future trends

    :param data: DataFrame containing the historical data
    :return: Model, predictions, and metrics
    """
    data['Date'] = data.index
    data['Date'] = to_numeric(data['Date'])

    x = data[['Date']].values
    y = data['Close'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return model, y_pred, mae, mse, x_test, y_test

def plot_linear_regression(data, y_pred, x_test, y_test):
    plt.figure(figsize=(14,7))
    plt.plot(data.index, data['Close'], label='Actual Close Price')
    plt.plot(to_datetime(x_test.flatten()), y_pred, label='Predicted Close Price')
    plt.title('Linear Regression Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def arima_forecast(data):
    """
    Apply ARIMA to forecast future trends.
    
    :param data: DataFrame containing the historical data
    :return: Model, predictions, and metrics
    """
    data = data['Close']
    train_data, test_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
    
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()
    y_pred = model_fit.forecast(steps=len(test_data))
    
    mae = mean_absolute_error(test_data, y_pred)
    mse = mean_squared_error(test_data, y_pred)
    
    return model_fit, y_pred, mae, mse, test_data

def plot_arima(data, y_pred, test_data):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Actual Close Price')
    plt.plot(test_data.index, y_pred, label='Predicted Close Price', color='red')
    plt.title('ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

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

def predict_lstm(model, scaler, data, n_steps=30):
    inputs = data['Close'][-n_steps:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.trasnform(inputs)
    inputs = reshape(inputs, (1, n_steps, 1))

    predicted_price = model.predict(inputs)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price

def create_prophet_model(data):
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    return model

def predict_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def create_rf_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model

def predict_rf(model, data, feature_cols):
    return model.predict(data[feature_cols])

def create_xgb_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    return model

def predict_xgb(model, data, feature_cols):
    return model.predict(data[feature_cols])

def create_svr_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = SVR(kernel='rbf')
    model.fit(X, y)

    return model

def predict_svr(model, data, feature_cols):
    return model.predict(data[feature_cols])

def create_sarima_model(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    return model_fit

def predict_sarima(model_fit, steps=30):
    forecast = model_fit.forecast(steps=steps)
    return forecast

def create_gbm_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = LGBMRegressor()
    model.fit(X, y)

    return model

def predict_gbm(model, data, feature_cols):
    return model.predict(data[feature_cols])

def create_knn_model(data, feature_cols, target_col='Close'):
    X = data[feature_cols]
    y = data[target_col]

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)

    return model

def predict_knn(model, data, feature_cols):
    return model.predict(data[feature_cols])

def main():
    # Load the data
    filePath = 'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data\\AAPL_historical_data.csv'
    data = load_data(filePath)

    # Linear Regression Forecast
    lr_model, lr_y_pred, lr_mae, lr_mse, lr_x_test, lr_y_test = linear_regression_forecast(data)
    print(f"Linear Regression Mean Absolute Error: {lr_mae}")
    print(f"Linear Regression Mean Squared Error: {lr_mse}")
    plot_linear_regression(data, lr_y_pred, lr_x_test, lr_y_test)

    # ARIMA Forecast
    arima_model_fit, arima_y_pred, arima_mae, arima_mse, arima_test_data = arima_forecast(data)
    print(f"ARIMA Mean Absolute Error: {arima_mae}")
    print(f"ARIMA Mean Squared Error: {arima_mse}")
    plot_arima(data, arima_y_pred, arima_test_data)

    # LSTM Forecast
    lstm_model, lstm_scaler = create_lstm_model(data)
    lstm_y_pred = predict_lstm(lstm_model, lstm_scaler, data)
    print(f"LSTM Predicted Price: {lstm_y_pred}")

    # Prophet Forecast
    prophet_model = create_prophet_model(data)
    prophet_forecast = predict_prophet(prophet_model)
    print(prophet_forecast.tail())

    # Random Forest Forecast
    feature_cols = ['7_day_MA', '14_day_MA', '21_day_MA', 'Volume', 'Daily_Return', 'Volatility_7_day', 'High_Low_Diff']
    rf_model = create_rf_model(data, feature_cols)
    rf_y_pred = predict_rf(rf_model, data, feature_cols)
    print(f"Random Forest Predicted Prices: {rf_y_pred}")

    # XGBoost Forecast
    xgb_model = create_xgb_model(data, feature_cols)
    xgb_y_pred = predict_xgb(xgb_model, data, feature_cols)
    print(f"XGBoost Predicted Prices: {xgb_y_pred}")

    # SVR Forecast
    svr_model = create_svr_model(data, feature_cols)
    svr_y_pred = predict_svr(svr_model, data, feature_cols)
    print(f"SVR Predicted Prices: {svr_y_pred}")

    # SARIMA Forecast
    sarima_model_fit = create_sarima_model(data)
    sarima_forecast = predict_sarima(sarima_model_fit)
    print(f"SARIMA Predicted Prices: {sarima_forecast}")

    # GBM Forecast
    gbm_model = create_gbm_model(data, feature_cols)
    gbm_y_pred = predict_gbm(gbm_model, data, feature_cols)
    print(f"GBM Predicted Prices: {gbm_y_pred}")

    # KNN Forecast
    knn_model = create_knn_model(data, feature_cols)
    knn_y_pred = predict_knn(knn_model, data, feature_cols)
    print(f"KNN Predicted Prices: {knn_y_pred}")

if __name__ == '__main__':
    main()