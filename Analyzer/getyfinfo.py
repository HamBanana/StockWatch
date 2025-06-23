#!/usr/bin/env python3
"""
Stock Info Retriever
A simple script to get stock information using yfinance
"""

import yfinance as yf
import sys

def get_stock_info(ticker):
    """
    Get stock information for a given ticker symbol
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        dict: Stock information dictionary
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get info
        info = stock.info
        
        if not info or len(info) == 0:
            print(f"No information found for ticker: {ticker}")
            return None
            
        return info
        
    except Exception as e:
        print(f"Error retrieving information for {ticker}: {e}")
        return None

def main():
    # Get ticker from command line argument or user input
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("Enter stock ticker symbol: ").upper()
    
    if not ticker:
        print("Please provide a valid ticker symbol")
        return
    
    print(f"Fetching information for {ticker}...")
    
    # Get stock info
    info = get_stock_info(ticker)
    
    if info:
        print(f"\nComplete Stock Information for {ticker}:")
        print("=" * 50)
        
        # Display all information in the info object
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print("=" * 50)
        print(f"Total fields: {len(info)}")

if __name__ == "__main__":
    main()