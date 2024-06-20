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

def main():
    # Load the data
    filePath = f'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data\\AAPL_historical_data.csv'
    data = load_data(filePath)
    
    inspect_data(data)

if __name__ == "__main__":
    main()