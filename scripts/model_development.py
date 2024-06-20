# Model Development
# Features:
    # Moving Averages: Moving 7-day average
        # Helps with identify the direction of the trend and potential support or resistance levels
    # Daily Returns: Percentage Change of a security fro one day to the next
        # Assess short-term performance of a security
        # Crucial for other metrics like volatility and for time-series analysis
    # Volatility: Degree of variation in the security price over a specific period
        # Gauges the riskinesss of a security
        # Helps in portfolio management by assessing the overall risk and making adjustments to balance the risk-return profile

import pandas as pd

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
    # Create a 7-day moving average
    data['7_day_MA'] = data['Close'].rolling(window=7).mean()
    # Create a column for daily returns
    data['Daily_Return'] = data['Close'].pct_change() * 100
    # Create a column for volatility (standard deviation of returns over 7 days)
    data['Volatility'] = data['Daily_Return'].rolling(window=7).std()

    return data

def main():
    # Load and inspect data
    filePath = f'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data\\AAPL_historical_data.csv'
    data = load_data(filePath)
    
    inspect_data(data)

    # Feature Engineering
    data_with_features = feature_engineering(data)
    inspect_data(data_with_features)

if __name__ == "__main__":
    main()

