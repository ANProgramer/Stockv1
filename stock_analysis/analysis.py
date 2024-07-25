import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import logging
import requests
from textblob import TextBlob
import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

logging.basicConfig(level=logging.INFO)

# API Keys (replace with your own)
NEWS_API_KEY = '6ff50a150545415d8bf8e20293977370'
ALPHA_VANTAGE_API_KEY = 'C6OSZOVGM6S6CGDG'
WORLD_BANK_API_URL = "http://api.worldbank.org/v2"

import pandas as pd

# Global Peace Index data (you'll need to update this annually)
gpi_data = {
    'US': 2.44,
    'UK': 1.77,
    'JP': 1.26,
    'DE': 1.46,
    # Add more countries as needed
}

def get_geopolitical_risk(country='US'):
    try:
        gpi_score = gpi_data.get(country, 2.5)  
        risk_score = (gpi_score / 5) * 100  
        return risk_score
    except Exception as e:
        logging.error(f"Error calculating geopolitical risk: {str(e)}")
        return 50  

def get_economic_data(country='US'):
    indicators = {
        'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
        'FP.CPI.TOTL.ZG': 'inflation_rate',
        'SL.UEM.TOTL.ZS': 'unemployment_rate'
    }
    
    economic_data = {}
    
    for indicator, name in indicators.items():
        url = f"{WORLD_BANK_API_URL}/country/{country}/indicator/{indicator}?format=json&per_page=1"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()[1][0]
            economic_data[name] = data['value']
        except Exception as e:
            logging.error(f"Error fetching {name} data: {str(e)}")
            economic_data[name] = None  
    
    return economic_data

def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json()['articles']
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles[:10]]
    return np.mean(sentiments)



def get_esg_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        sustainability = stock.sustainability
        if sustainability is not None:
            if 'totalEsg' in sustainability:
                return float(sustainability['totalEsg'])
            else:
                scores = [float(v) for v in sustainability.values if isinstance(v, (int, float))]
                return sum(scores) / len(scores) if scores else 50  
        else:
            return 50  
    except Exception as e:
        logging.error(f"Error fetching ESG data for {ticker}: {str(e)}")
        return 50  

def get_stock_data(ticker, start_date, end_date):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='full')
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return data

def calculate_features(data):
    data['Returns'] = data['Close'].pct_change()
    data['MA50'] = data['Close'].rolling(window=50).mean() / data['Close'] - 1
    data['MA200'] = data['Close'].rolling(window=200).mean() / data['Close'] - 1
    data['RSI'] = calculate_rsi(data['Close'], 14)
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['Volume_MA'] = data['Volume'] / data['Volume'].rolling(window=20).mean() - 1
    return data.dropna()

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(data, forecast_period):
    features = ['Returns', 'MA50', 'MA200', 'RSI', 'MACD', 'Volume_MA']
    X = data[features]
    y = data['Close'].pct_change(periods=forecast_period).shift(-forecast_period)
    return X[:-forecast_period], y[:-forecast_period].dropna()

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def get_short_term_recommendations(tickers, time_horizon):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  
    
    recommendations = []
    
    for ticker in tickers:
        try:
            data = get_stock_data(ticker, start_date, end_date)
            data = calculate_features(data)
            
            X, y = prepare_data(data, time_horizon)
            
            if len(X) < 252:  
                logging.warning(f"Insufficient data for {ticker}. Skipping.")
                continue
            
            model = train_model(X, y)
            
            latest_data = X.iloc[-1:] 
            prediction = model.predict(latest_data)[0]
            
            
            sentiment_score = get_news_sentiment(ticker)
            esg_score = get_esg_score(ticker)
            geopolitical_risk = get_geopolitical_risk()
            economic_data = get_economic_data()
            
            
            prediction_adjustment = (
                sentiment_score * 0.1 +
                (esg_score / 100) * 0.1 +
                (1 - geopolitical_risk / 100) * 0.1 +
                (economic_data['gdp_growth'] / 10) * 0.1
            )
            
            adjusted_prediction = prediction * (1 + prediction_adjustment)
            
            recommendations.append({
                'ticker': ticker,
                'expected_return': float(adjusted_prediction)
            })
            logging.info(f"Successfully processed {ticker}")
        except Exception as e:
            logging.error(f"Error processing {ticker}: {str(e)}")
    
    recommendations.sort(key=lambda x: x['expected_return'], reverse=True)
    return recommendations[:5]

def get_long_term_recommendations(tickers):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  
    
    recommendations = []
    
    fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')    
    for ticker in tickers:
        try:
            data = get_stock_data(ticker, start_date, end_date)
            
            if len(data) < 252:  
                logging.warning(f"Insufficient data for {ticker}. Skipping.")
                continue
            
           
            overview, _ = fd.get_company_overview(ticker)
            
           
            pe_ratio = float(overview['PERatio'])
            pb_ratio = float(overview['PriceToBookRatio'])
            profit_margin = float(overview['ProfitMargin'])
            roe = float(overview['ReturnOnEquityTTM'])
            dividend_yield = float(overview['DividendYield'])
            debt_to_equity = float(overview['DebtToEquityRatio'])
            
            cagr = (data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252/len(data)) - 1
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            
            value_score = (1 / pe_ratio if pe_ratio > 0 else 0) + (1 / pb_ratio if pb_ratio > 0 else 0)
            growth_score = cagr + profit_margin
            quality_score = roe + (1 / debt_to_equity if debt_to_equity > 0 else 0)
            stability_score = dividend_yield - volatility
            
            sentiment_score = get_news_sentiment(ticker)
            esg_score = get_esg_score(ticker) / 100  
            geopolitical_risk = get_geopolitical_risk() / 100  
            economic_data = get_economic_data()       


            gdp_score = economic_data['gdp_growth'] / 5 if economic_data['gdp_growth'] is not None else 0
            inflation_score = (3 - economic_data['inflation_rate']) / 3 if economic_data['inflation_rate'] is not None else 0
            unemployment_score = (5 - economic_data['unemployment_rate']) / 5 if economic_data['unemployment_rate'] is not None else 0
            economic_score = (gdp_score + inflation_score + unemployment_score) / 3     

            score = (
                value_score * 0.2 +
                growth_score * 0.2 +
                quality_score * 0.15 +
                stability_score * 0.1 +
                sentiment_score * 0.1 +
                esg_score * 0.1 +
                geopolitical_risk * 0.05 +
                economic_score * 0.1
            )
            
            recommendations.append({
                'ticker': ticker,
                'score': float(score),
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'profit_margin': profit_margin,
                'roe': roe,
                'dividend_yield': dividend_yield,
                'cagr': float(cagr),
                'volatility': float(volatility),
                'sentiment_score': float(sentiment_score),
                'esg_score': float(esg_score * 100),
                'geopolitical_risk': float((1 - geopolitical_risk) * 100)
            })
            logging.info(f"Successfully processed long-term data for {ticker}")
        except Exception as e:
            logging.error(f"Error processing long-term data for {ticker}: {str(e)}")
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:5]

def get_all_recommendations():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 
               'WMT', 'PG', 'XOM', 'UNH', 'HD', 'BAC', 'PFE', 'DIS', 'CSCO', 'VZ']
    
    return {
        '1_day': get_short_term_recommendations(tickers, 1),
        '1_month': get_short_term_recommendations(tickers, 21),  
        '1_year': get_long_term_recommendations(tickers),
        '10_years': get_long_term_recommendations(tickers)
    }

def get_stock_recommendations():
    return get_all_recommendations()