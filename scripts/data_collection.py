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
    # TODO: Create a system for grabbing the latest top 50 companies
    tickers = ["MSFT","AAPL","NVDA","GOOG","AMZN","2222.SR","META","TSM","BRK-B","LLY","AVGO","NVO","TSLA","V","JPM","WMT",
               "XOM","TCEHY","UNH","MA","ASML","PG","ORCL","MC.PA","005930.KS","COST","JNJ","HD","MRK","BAC","ABBV","NFLX",
               "CVX","NESN.SW","KO","TM","AMD","600519.SS","1398.HK","OR.PA","AZN","601857.SS","RMS.PA","QCOM",
               "CRM","ADBE","RELIANCE.NS","PEP","ROG.SW", "SAP"]
    start_date = '2022-01-01'
    end_date = '2024-01-01'
    for ticker in tickers:
        dataframe = fetch_data(ticker, start_date, end_date)
        filePath = f'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\data\\{ticker}_historical_data.csv'
        dataframe.to_csv(filePath)
        print(f"Data saved to data/{ticker}_historical_data.csv")

if __name__ == '__main__':
    main()

