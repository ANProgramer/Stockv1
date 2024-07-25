# Stock Recommendation Engine V1

## Description
This Stock Recommendation Engine is a Python-based application that provides data-driven investment recommendations for both short-term and long-term horizons. By leveraging machine learning techniques and incorporating various financial and non-financial factors, it offers a comprehensive analysis of stock performance and potential.

## Before Using 
- Make sure to replace ALPHA_VANTAGE_API_KEY, and NEWS_API_KEY with your api keys to ensure the program works---> analysis.py
- In analysis.py, for more GPI countries you will need to update it manually
  


## Key Features
- Short-term recommendations (1 day and 1 month) based on technical analysis and machine learning predictions
- Long-term recommendations (1 year and 10 years) considering fundamental analysis, historical performance, and additional factors
- Integration of multiple data sources:
  - Stock price and volume data
  - Company fundamentals
  - News sentiment analysis
  - ESG (Environmental, Social, and Governance) scores
  - Economic indicators
  - Geopolitical risk assessment
- Custom scoring system that balances various aspects of stock performance and market conditions

## Technologies Used
- Python
- Pandas for data manipulation
- Scikit-learn for machine learning models
- yfinance for stock data retrieval
- NewsAPI for sentiment analysis
- Alpha Vantage API for additional financial data
- World Bank API for economic indicators

## How It Works
1. Fetches historical stock data and calculates technical indicators
2. Trains a machine learning model for short-term predictions
3. Retrieves fundamental company data and calculates financial ratios
4. Incorporates news sentiment, ESG scores, and macroeconomic factors
5. Applies a custom scoring algorithm to rank stocks
6. Generates top 5 recommendations for different time horizons

## Disclaimer
This tool is for educational and research purposes only. It should not be considered as financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## Future Improvements
- Incorporate more advanced machine learning models
- Add backtesting functionality to validate strategy performance
- Implement real-time data updates
- Expand the range of analyzed stocks and markets
- Better UI
- Make it Less Pricy To Use
