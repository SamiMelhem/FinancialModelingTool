# Model Development
# Features:
    # Close: The closing price of the stock
    # Volume: The trading volume of the stock
    # Moving Averages: Moving 7-day average
        # Helps with identify the direction of the trend and potential support or resistance levels
    # Daily Returns: Percentage Change of a security fro one day to the next
        # Assess short-term performance of a security
        # Crucial for other metrics like volatility and for time-series analysis
    # Volatility: Degree of variation in the security price over a specific period
        # Gauges the riskinesss of a security
        # Helps in portfolio management by assessing the overall risk and making adjustments to balance the risk-return profile
    # High-Low Difference: The difference between the daily high and low prices
    # Lagged Features: Previous values of the above attributes (lags of 1, 3, 5, etc. days)
    # Rolling Statistics: Rolling mean, median, and standard deviation of the closing prices
    # Technical Indicators:
        # RSI (Relative Strength Index)
        # MACD (Moving Average Convergence Divergence)
    # Calendar Features: Day of the week, month, quarter


import pandas as pd
import os

def load_data(filePath):
    """
    Load the historical data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: DataFrame containing the loaded data
    """
    data = pd.read_csv(filePath, index_col='Date', parse_dates=True)
    return data

def inspect_data(data):
    """
    Inspect the data by displaying the first few rows and summary statistics.
    
    :param data: DataFrame containing the data
    """
    print(data.head())
    print(data.info())
    print(data.describe())

def feature_engineering(data):
    """
    Create new features for the data.

    :param data: DataFrame containing the data
    :return: DataFrame with new features
    """
    # Ensure the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)

    # Determine the minimum length of data to apply the largest window
    min_length = len(data)

    # Moving Averages
    windows = {
        '1_day_MA': 1, '3_day_MA': 3, '5_day_MA': 5, '7_day_MA': 7, '14_day_MA': 14,
        '1_month_MA': 30, '3_month_MA': 91, '6_month_MA': 183, '1_year_MA': 365
    }
    for key, window in windows.items():
        if min_length >= window:
            data[key] = data['Close'].rolling(window=window).mean()

    # Daily Returns
    data['Daily_Return'] = data['Close'].pct_change() * 100

    # Volatility
    vol_windows = {
        'Volatility_1_day': 1, 'Volatility_3_day': 3, 'Volatility_5_day': 5, 'Volatility_7_day': 7,
        'Volatility_14_day': 14, 'Volatility_1_month': 30, 'Volatility_3_month': 91,
        'Volatility_6_month': 183, 'Volatility_1_year': 365
    }
    for key, window in vol_windows.items():
        if min_length >= window:
            data[key] = data['Daily_Return'].rolling(window=window).std()

    # High-Low Difference
    data['High_Low_Diff'] = data['High'] - data['Low']

    # Lagged Features
    lags = [1, 3, 5]
    for lag in lags:
        data[f'Lag_{lag}'] = data['Close'].shift(lag)

    # Rolling Statistics
    roll_windows = {
        'Rolling_Std_1_day': 1, 'Rolling_Std_3_day': 3, 'Rolling_Std_5_day': 5, 'Rolling_Std_7_day': 7,
        'Rolling_Std_14_day': 14, 'Rolling_Std_1_month': 30, 'Rolling_Std_3_month': 91,
        'Rolling_Std_6_month': 183, 'Rolling_Std_1_year': 365
    }
    for key, window in roll_windows.items():
        if min_length >= window:
            data[key] = data['Close'].rolling(window=window).std()

    # Calendar Features
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter

    return data

def main():
    folder_path = 'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data'
    # Feature Engineering
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Load and inspect data
            filePath = os.path.join(folder_path, filename)
            data = load_data(filePath)
            inspect_data(data)

            data_with_features = feature_engineering(data)
            data_with_features.to_csv(filePath)
            inspect_data(data_with_features)

if __name__ == "__main__":
    main()

