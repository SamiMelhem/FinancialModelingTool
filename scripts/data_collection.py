# Data Sources
    # Yahoo Finance (We're going to start with this)
    # Alpha Vantage
    # Quandl
    # IEX Cloud
# Fetching Data
    # Ticker -> Company Abbreviation
    # start_date, end_date (YYYY-MM-DD)

import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical data for the given ticker symbol from Yahoo Finance

    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
    :param start_date: Start date for fetching data (format: 'YYYY-MM-DD')
    :param end_date: End date for fetching data (format: 'YYYY-MM-DD')
    :return: DataFrame containing the historical data
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def main():
    ticker = "AAPL"
    start_date = '2000-01-01'
    end_date = '2023-01-02'

    dataframe = fetch_data(ticker, start_date, end_date)
    filePath = f'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data\\{ticker}_historical_data.csv'
    dataframe.to_csv(filePath)
    print(f"Data saved to data/{ticker}_historical_data.csv")

if __name__ == '__main__':
    main()

