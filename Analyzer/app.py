#!/usr/bin/env python3
"""
Flask Backend Server for Stock Signal Analyzer - Debug Version
Includes better error handling and data formatting
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import json
import os
import sys
from datetime import datetime
import logging
import yfinance as yf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
SIGNALS_PATH = "../Signals/signals.json"
GETYFINFO_PATH = "./getyfinfo.py"  # Adjust if needed

def load_signals_framework():
    """Load the signals framework from JSON file"""
    try:
        with open(SIGNALS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Signals file not found at {SIGNALS_PATH}")
        # Return a minimal framework for testing
        return {
            "meta": {"v": "4.1", "updated": "2025-06-16"},
            "caps": {
                "nano": {"min": 1000000, "max": 25000000, "mult": 5.0},
                "micro": {"min": 25000000, "max": 75000000, "mult": 3.5},
                "small": {"min": 75000000, "max": 500000000, "mult": 2.0},
                "mid": {"min": 500000000, "max": 2000000000, "mult": 1.2}
            }
        }
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in signals file: {e}")
        return None

def get_stock_data_yfinance(ticker):
    """Get stock data directly using yfinance"""
    try:
        logger.info(f"Fetching data for {ticker} using yfinance")
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data
        hist = stock.history(period="3mo")
        
        if hist.empty:
            logger.warning(f"No historical data for {ticker}")
            return None
        
        # Calculate current values
        current_price = float(hist['Close'].iloc[-1])
        current_volume = int(hist['Volume'].iloc[-1])
        avg_volume = float(hist['Volume'].rolling(window=20).mean().iloc[-1]) if len(hist) >= 20 else float(hist['Volume'].mean())
        
        # Calculate RSI
        rsi = calculate_rsi(hist['Close'])
        
        # Calculate volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = float(returns.std() * (252 ** 0.5) * 100)  # Annualized volatility
        
        # Get market cap (handle missing data)
        market_cap = info.get('marketCap', 0)
        if market_cap == 0 and info.get('sharesOutstanding', 0) > 0:
            market_cap = current_price * info.get('sharesOutstanding', 0)
        
        # Build response data
        data = {
            'symbol': ticker,
            'ticker': ticker,
            'shortName': info.get('shortName', ticker),
            'longName': info.get('longName', ticker),
            'currentPrice': current_price,
            'price': current_price,
            'volume': current_volume,
            'avgVolume': avg_volume,
            'averageVolume': avg_volume,
            'volumeRatio': current_volume / avg_volume if avg_volume > 0 else 1.0,
            'marketCap': market_cap or 0,
            'market_cap': market_cap or 0,
            'floatShares': info.get('floatShares', info.get('sharesOutstanding', 0)),
            'shares_float': info.get('floatShares', info.get('sharesOutstanding', 0)),
            'shortPercentOfFloat': float(info.get('shortPercentOfFloat', 0) * 100) if info.get('shortPercentOfFloat') else 0,
            'short_percent_of_float': float(info.get('shortPercentOfFloat', 0) * 100) if info.get('shortPercentOfFloat') else 0,
            'fiftyTwoWeekLow': float(info.get('fiftyTwoWeekLow', 0)),
            'fiftyTwoWeekHigh': float(info.get('fiftyTwoWeekHigh', 0)),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'beta': float(info.get('beta', 1.0)) if info.get('beta') else 1.0,
            'longBusinessSummary': info.get('longBusinessSummary', ''),
            'business_summary': info.get('longBusinessSummary', ''),
            'website': info.get('website', ''),
            'country': info.get('country', ''),
            'employees': info.get('fullTimeEmployees', 0),
            'rsi': rsi,
            'volatility': volatility
        }
        
        logger.info(f"Successfully fetched data for {ticker}: Price=${current_price}, Volume={current_volume}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_rsi(prices, periods=14):
    """Calculate RSI"""
    try:
        if len(prices) < periods + 1:
            return 50.0
        
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=periods).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=periods).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    except:
        return 50.0

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/signals', methods=['GET'])
def get_signals():
    """Get the signals framework"""
    signals = load_signals_framework()
    if signals:
        return jsonify(signals)
    else:
        return jsonify({'error': 'Failed to load signals framework'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    """Analyze a stock ticker"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        logger.info(f"Analyzing ticker: {ticker}")
        
        # Always use yfinance for now (can integrate getyfinfo.py later)
        stock_data = get_stock_data_yfinance(ticker)
        
        if not stock_data:
            # Return minimal data structure to prevent NaN errors
            return jsonify({
                'ticker': ticker,
                'symbol': ticker,
                'currentPrice': 0,
                'price': 0,
                'volume': 0,
                'avgVolume': 1,  # Prevent division by zero
                'volumeRatio': 0,
                'marketCap': 0,
                'floatShares': 0,
                'shortPercentOfFloat': 0,
                'fiftyTwoWeekLow': 0,
                'fiftyTwoWeekHigh': 0,
                'rsi': 50,
                'volatility': 0,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'error': f'No data found for {ticker}'
            })
        
        return jsonify(stock_data)
        
    except Exception as e:
        logger.error(f"Error analyzing stock: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create minimal signals.json if it doesn't exist
    if not os.path.exists(SIGNALS_PATH):
        os.makedirs(os.path.dirname(SIGNALS_PATH), exist_ok=True)
        minimal_signals = {
            "meta": {"v": "4.1", "updated": "2025-06-16"},
            "caps": {
                "nano": {"min": 1000000, "max": 25000000, "mult": 5.0},
                "micro": {"min": 25000000, "max": 75000000, "mult": 3.5},
                "small": {"min": 75000000, "max": 500000000, "mult": 2.0},
                "mid": {"min": 500000000, "max": 2000000000, "mult": 1.2}
            }
        }
        with open(SIGNALS_PATH, 'w') as f:
            json.dump(minimal_signals, f, indent=2)
        logger.info(f"Created minimal signals.json at {SIGNALS_PATH}")
    
    # Run the Flask app
    logger.info("Starting Flask backend server on http://localhost:5000")
    logger.info("Make sure to install: pip install flask flask-cors yfinance")
    app.run(debug=True, port=5000)