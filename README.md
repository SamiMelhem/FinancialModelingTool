# Financial Modeling Tool

A Python tool for financial modeling and analysis. This tool performs data analysis, generates financial forecasts, and visualizes the results for the top 50 companies in the world (As of 6/21/2024). This tool includes a combination of machine learning models to predict stock prices and evaluates to accuracy of these predictions

## Table of Contents

- [Project Description](#project-description)
- [Technologies](#technologies)
- [Objective](#objective)
- [Setup Instructions](#setup-instructions)
- [Data Collection & Analysis](#data-collection-and-analysis)
- [Model Development](#model-development)
- [Forecasting](#forecasting)
- [Visualization](#visualization)
- [Findings](#findings)
- [Next Steps](#next-steps)

## Project Description

The Financial Modeling Tool that's designed to help financial analysts and investors analyze historical data and forecast future trends, aiding in better financial decision-making. The project uses various machine learning models to rpedict stock prices and evaluates the accuracy of these predictions over multiple iterations.

## Technologies

Python, Pandas, NumPy, Jupyter Notebooks, Scikit-learn, Tensorflow, XGBoost, LightGBM, Prophet, Statsmodels, Plotly, and Dash

## Objective

Use 10 ML model forecasts in certain time frames with the top 50 largest companies by market cap to determine which model is better at predicting the trend of the company. The tool runs 10 tests to determine the average performance of these ML forecasts across the Top 50 companies.

10 ML Forecasts:
- Linear Regression
- ARIMA (AutoRegressive Integrated Moving Average)
- LSTM (Long Short-Term Memory)
- Prophet
- Random Forest
- XGBoost
- SVR (Support Vector Regression)
- SARIMA (Seasonal ARIMA)
- GBM (Graident Boosting Machine)
- KNN (K-Nearest Neighbors)

Time Frames:
- 1 day, 3 days, 5 days, 1 week, 2 weeks, 1 month, 3 months, 6 months, 1 year

Top 50 companies by market cap (As of 6/21/2024):
- Microsoft, Apple, NVIDIA, Google, Amazon, Saudi Aramco, Meta, TSMC,Berkshire Hathaway, Eli Lilly, Broadcom, Novo Nordisk, Tesla, Visa, JPMorgan Chase, Walmart, Exxon Mobil, Tencent, United Health, Mastercard, ASML, Procter & Gamble, Oracle, LVMH, Samsung, Costco, Johnson & Johnson, Home Depot, Merck, Bank of America, AbbVie, Netflix, Chevron, Nestle, Coca-Cola, Toyota, AMD, Kweichow Moutai, ICBC, L'Oreal, AstraZeneca, PetroChina, Hermes, QUALCOMM, Salesforce, Adobe, Reliance Industries, Pepsico, Roche, SAP

## Setup Instructions
1. Clone Repository
```
git clone https://github.com/SamiMelhem/FinancialModelingTool.git
cd FinancialModelingTool
```
2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate # On Windows: \venv\Scripts\activate
```
3. Install the Required Packages
```
pip install -r requirements.txt
```
4. Collect historical stock data for all companies by market cap of the current day (Optional):
- Go to data_collection.py -> main() -> change 'tickers' to the desired company names by their ticker name for the stock market
5. Start going through the Jupyter Notebook that shows detailed steps of how to run the tool.

## Data Collection and Analysis
Used the 'yfinance' library to collect historical stock data and saved the data in the 'data' directory.
Then for data analysis the tool drops any na values that exist within the company's .csv file and uploads a clean version of the .csv back to the original .csv file.

## Model Development
The tool makes sure to perform feature engineering by preparing the .csv file of every company of needed metrics from the company stock data. Below are the metrics that were created as columns in the .csv files:

- Close: The closing price of the stock
- Volume: The trading volume of the stock
- Moving Averages: Moving 7-day average
  - Helps with identify the direction of the trend and potential support or resistance levels
- Daily Returns: Percentage Change of a security fro one day to the next
  - Assess short-term performance of a security
  - Crucial for other metrics like volatility and for time-series analysis
- Volatility: Degree of variation in the security price over a specific period
  - Gauges the riskinesss of a security
  - Helps in portfolio management by assessing the overall risk and making adjustments to balance the risk-return profile
- High-Low Difference: The difference between the daily high and low prices
- Lagged Features: Previous values of the above attributes (lags of 1, 3, 5, etc. days)
- Rolling Statistics: Rolling mean, median, and standard deviation of the closing prices
- Technical Indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
- Calendar Features: Day of the week, month, quarter

## Forecasting
The tool uses the following machine learning models for forecasting (along with detailed explanations on each ML model):
- Linear Regression
  - A simple yet powerful model that assumes a linear relationship between the input features and the target variable. We use polynomial features to improve the model's capability in capturing non-linear relationships
  ```
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.pipeline import make_pipeline
  
  poly = PolynomialFeatures(degree=3)  # Using degree 3 for polynomial features
  model = make_pipeline(poly, LinearRegression())
  model.fit(x_train, y_train)
  ``` 
- ARIMA
  - Widely used for time series forecasting. This model combines three components: AR (AutoRegressive), I (Integrated), and MA (Moving Average).
  ```
  from statsmodels.tsa.arima.model import ARIMA

  model = ARIMA(train_data, order=(5, 1, 2))
  model_fit = model.fit()
  ``` 
- LSTM
  - Type of recurrent neural network (RNN) capable of learning long-term dependencies, especially usefyul for time series forecasting.
  ```
  from tensorflow.keras.layers import LSTM, Dense
  from tensorflow.keras.models import Sequential
  
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
  model.add(LSTM(units=50))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(X, y, epochs=25, batch_size=32, verbose=2)
  ```
- Prophet
  - Facebook Forecasting tool that works well with time series data that has strong seasonal effects and several seasons of historical data.
  ```
  from prophet import Prophet

  df = data.copy()
  df = df.rename_axis('ds').reset_index()
  df = df.rename(columns={'Close': 'y'})
  
  model = Prophet(daily_seasonality=True, yearly_seasonality=True)
  model.fit(df)
  ``` 
- Random Forest
  - Ensemble learning method for regression that operates by constructing multiple decision trees and outputs the average predction of the indivdual trees.
  ```
  from sklearn.ensemble import RandomForestRegressor

  model = RandomForestRegressor(n_estimators=100)
  model.fit(X, y)
  ``` 
- XGBoost
  - An optimized distributed gradient boosting library designed for high efficiency, flexibility, and portability.
  ```
  from xgboost import XGBRegressor

  model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
  model.fit(X, y)
  ```
- SVR
  - Type of Support Vector Machine that supports linear and non-linear regression.
  ```
  from sklearn.svm import SVR
  
  model = SVR(kernel='rbf', C=100, gamma=0.1)
  model.fit(X, y)
  ``` 
- SARIMA
  - An extension of ARIMA that supports univariate time series data with a seasonal component.
  ```
  from statsmodels.tsa.statespace.sarimax import SARIMAX

  model = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
  model_fit = model.fit(disp=False)
  ``` 
- GBM
  - An ensembe machine learning technique for regression and classification problems that produces a prediction model in the form of an ensemble of decision trees.
  ```
  from lightgbm import LGBMRegressor
  
  model = LGBMRegressor()
  model.fit(X, y)
  ```
- KNN
  - A non-parametric method used for regression and classification in KNN regression, the output is the average of the values of its k nearest neighbors.
  ```
  from sklearn.neighbors import KNeighborsRegressor

  model = KNeighborsRegressor(n_neighbors=5)
  model.fit(X, y)
  ```

When going through the forecasting process of all 50 companies the tool collects the prediction accuracy metrics of these models and repeats the process 10 times.

### Prediction Accuracy Calculation
To calculate the prediction accuracy of these forecasts I developed my own way as opposed to using one's from the intraweb because I want to customize the metric to the tool itself. Here is the philosophy behind calculating this value:

1. Percentage Difference: The distance between the predicted and the actual price of the company stock is the most important objective to achieve with these ML forecasts. Therefore we would need to calculate this first.
2. Scaling the Difference: Since we want to put more value on the closer the forecast predicts the actual price, the metric uses exponential decay scaling to have a significant dip in the accuracy metric when it's not closer to the stock company price.
3. Weighted Accuracy: Finally I took the mean of the accuracy for every day predicted and multipled by 100 to get the percentage value of a ML forecast across a time frame.


## Visualization
The tool will put all of the results into a Dash application to visualize the results in a neat way. This application will include graphs of the prediction & actual prices of the company stock along with the accruacy metrics.

Below is an example of what the Dash Application should look like:


## Findings
After running this tool (~8 hours) the results of every ML forecast will dump into the 'average_accuracy_metrics.json' file. Here below are the results of each ML forecast with their respective time frames (rounded to the nearest 2 decimal places):

ML Model/Time Frame | Linear Regression | ARIMA  | LSTM   | Prophet | Random Forest | XGBoost | SVR    | SARIMA | GBM    | KNN
------------------- | ----------------- | ------ | ------ | ------- | ------------- | ------- | ------ | ------ | ------ | ---
1 Day               | 16.83%            | 26.43% | 81.25% | 7.69%   | 93.98%        | 99.77%  | 80.33% | 93.15% | 89.58% | 25.65%
3 Day               | 17.22%            | 26.97% | 80.93% | 8.27%   | 94.57%        | 99.80%  | 80.00% | 88.42% | 91.08% | 27.71%
5 Day               | 17.32%            | 27.59% | 77.13% | 8.30%   | 94.56%        | 99.78%  | 80.02% | 84.63% | 91.32% | 27.57%
1 Week              | 17.30%            | 27.98% | 72.54% | 8.14%   | 94.16%        | 99.77%  | 80.15% | 81.11% | 91.14% | 27.37%
2 Weeks             | 17.58%            | 30.08% | 57.75% | 7.99%   | 92.76%        | 99.76%  | 81.61% | 69.88% | 90.55% | 26.39%
1 Month             | 18.33%            | 34.78% | 44.90% | 7.79%   | 93.12%        | 99.76%  | 83.74% | 55.57% | 91.23% | 29.23%
3 Months            | 23.19%            | 39.23% | 30.75% | 6.80%   | 91.95%        | 99.70%  | 87.88% | 33.07% | 91.47% | 34.61%
6 Months            | 26.78%            | 40.41% | 24.32% | 7.15%   | 91.59%        | 99.68%  | 89.60% | 22.04% | 91.53% | 36.38%
1 Year              | 23.34%            | 28.79% | 20.43% | 9.07%   | 90.37%        | 99.64%  | 90.49% | 13.68% | 90.47% | 35.11%

### Trends
- LSTM & SARIMA: Great in short time frames, but trended downwards as the time frame increased.
- SVR: Not great in short time frames, but trended upwards as the time frame increased.

### Ranks
1. XGBoost (99.74%)
2. Random Forest (93.01%)
3. GBM (90.93%)
4. SVR (83.76%)
5. SARIMA (60.17%)
6. LSTM (54.44%)
7. ARIMA (31.36%)
8. KNN (30.00%)
9. Linear Regression (19.77%)
10. Prophet (7.91%)


## Next Steps
1. Create a web scraper that collects the top 50 companies on the current day to have up to date company analysis.
2. Fine-tune the ML forecasts towards financial data (Would include using different ML forecasts to compare/contrast).
3. Train the best forecasts on the top 50 company's complete historical stock data
4. Try different data sources to save time on model development like Alpha Vantage, Quandl, IEX Cloud, etc.
5. Create the ML model forecasts once and then reference the model for every company instead of having to create it every time for every company (This would reduce producing the results in a quick manner by a lot)
