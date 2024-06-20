import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def load_data(filePath):
    """
    Load the historical data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: DataFrame containing the loaded data
    """
    data = pd.read_csv(filePath, index_col='Date', parse_dates=True)
    return data

def linear_regression_forecast(data):
    """
    Apply linear regression to forecast future trends

    :param data: DataFrame containing the historical data
    :return: Model, predictions, and metrics
    """
    data['Date'] = data.index
    data['Date'] = pd.to_numeric(data['Date'])

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
    plt.plot(pd.to_datetime(x_test.flatten()), y_pred, label='Predicted Close Price')
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

def main():
    # Load the data
    filePath = f'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data\\AAPL_historical_data.csv'
    data = load_data(filePath)

    # Linear Regression and ARIMA Forecasts
    model, y_pred, mae, mse, x_test, y_test = linear_regression_forecast(data)
    model_fit, y_pred2, mae2, mse2, test_data = arima_forecast(data)

    print(f"Linear Regression Mean Absolute Error: {mae}")
    print(f"Linear Regression Mean Squared Error: {mse}")
    print(f"ARIMA Mean Absolute Error: {mae2}")
    print(f"ARIMA Mean Squared Error: {mse2}")

    # Plot the results
    plot_linear_regression(data, y_pred, x_test, y_test)
    plot_arima(data, y_pred, test_data)


if __name__ == '__main__':
    main()