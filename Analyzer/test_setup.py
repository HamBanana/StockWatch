#!/usr/bin/env python3
"""
Test script to verify the setup and diagnose issues
"""

import os
import sys
import subprocess
import json

print("🔍 Stock Analyzer Setup Diagnostic")
print("=" * 50)

# Check Python version
print(f"✓ Python version: {sys.version}")

# Check required packages
packages = ['flask', 'flask_cors', 'yfinance', 'pandas', 'numpy']
missing_packages = []

for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print(f"Install with: pip install {' '.join(missing_packages)}")

# Check for signals.json
signals_path = "../Signals/signals.json"
if os.path.exists(signals_path):
    print(f"✓ signals.json found at {signals_path}")
else:
    print(f"✗ signals.json NOT found at {signals_path}")
    print("  Creating minimal signals.json...")
    os.makedirs(os.path.dirname(signals_path), exist_ok=True)
    minimal_signals = {
        "meta": {"v": "4.1", "updated": "2025-06-16"},
        "caps": {
            "nano": {"min": 1000000, "max": 25000000, "mult": 5.0},
            "micro": {"min": 25000000, "max": 75000000, "mult": 3.5},
            "small": {"min": 75000000, "max": 500000000, "mult": 2.0},
            "mid": {"min": 500000000, "max": 2000000000, "mult": 1.2}
        }
    }
    with open(signals_path, 'w') as f:
        json.dump(minimal_signals, f, indent=2)
    print("  ✓ Created minimal signals.json")

# Test yfinance with SRM
print("\n📊 Testing yfinance with SRM ticker...")
try:
    import yfinance as yf
    ticker = yf.Ticker("SRM")
    info = ticker.info
    hist = ticker.history(period="1mo")
    
    if not hist.empty:
        print(f"✓ SRM data found:")
        print(f"  Latest price: ${hist['Close'].iloc[-1]:.2f}")
        print(f"  Latest volume: {hist['Volume'].iloc[-1]:,}")
        print(f"  Company: {info.get('longName', 'Unknown')}")
    else:
        print("✗ No historical data for SRM")
        print("  This might be a delisted or invalid ticker")
        
        # Try alternative tickers
        print("\n  Testing alternative tickers...")
        for test_ticker in ['AAPL', 'MSFT', 'GOOGL']:
            test = yf.Ticker(test_ticker)
            test_hist = test.history(period="1d")
            if not test_hist.empty:
                print(f"  ✓ {test_ticker} works: ${test_hist['Close'].iloc[-1]:.2f}")
                break
        
except Exception as e:
    print(f"✗ Error testing yfinance: {e}")

# Check if Flask server is running
print("\n🌐 Checking Flask server...")
try:
    import requests
    response = requests.get("http://localhost:5000/health", timeout=2)
    if response.status_code == 200:
        print("✓ Flask server is running")
    else:
        print("✗ Flask server returned error")
except:
    print("✗ Flask server is NOT running")
    print("  Start it with: python app.py")

print("\n" + "=" * 50)
print("📝 Quick Start Guide:")
print("1. Install missing packages (if any)")
print("2. Run: python app.py")
print("3. Open index.html in your browser")
print("4. Try ticker symbols like AAPL, MSFT, or GOOGL")
print("\nNote: SRM might be delisted or unavailable in yfinance")
