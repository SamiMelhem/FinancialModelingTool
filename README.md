# Financial Modeling Tool

A Python tool for financial modeling and analysis. This tool performs data analysis, generates financial forecasts, and visualizes the results for the top 50 companies in the world (As of 6/21/2024). This tool includes a combination of machine learning models to predict stock prices and evaluates to accuracy of these predictions

## Table of Contents

- [Project Description](#project-description)
- [Technologies](#technologies)
- [Objective](#objective)
- [Setup Instructions](#setup-instructions)
- [Data Collection](#data-collection)
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
- 1 day
- 3 days
- 5 days
- 1 week
- 2 weeks
- 1 month
- 3 months
- 6 months
- 1 year

Top 50 companies by market cap (As of 6/21/2024):
- Microsoft, Apple, NVIDIA, Google, Amazon, Saudi Aramco, Meta, TSMC,Berkshire Hathaway, Eli Lilly, Broadcom, Novo Nordisk, Tesla, Visa, JPMorgan Chase, Walmart, Exxon Mobil, Tencent, United Health, Mastercard, ASML, Procter & Gamble, Oracle, LVMH, Samsung, Costco, Johnson & Johnson, Home Depot, Merck, Bank of America, AbbVie, Netflix, Chevron, Nestle, Coca-Cola, Toyota, AMD, Kweichow Moutai, ICBC, L'Oreal, AstraZeneca, PetroChina, Hermes, QUALCOMM, Salesforce, Adobe, Reliance Industries, Pepsico, Roche, SAP
