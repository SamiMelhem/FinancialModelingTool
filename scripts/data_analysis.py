# Cleaning and Preprocessing Data

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

def handle_missing_values(data):
    """
    Handle missing values in the data

    :param data: DataFrame containing the data
    :return: DataFrame with missing values handled
    """
    data_dropped = data.dropna()
    return data_dropped

def main():
    # Load and inspect data
    filePath = f'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data\\AAPL_historical_data.csv'
    data = load_data(filePath)
    
    inspect_data(data)

    # Handle missing values
    data_cleaned = handle_missing_values(data)
    inspect_data(data_cleaned)

if __name__ == "__main__":
    main()