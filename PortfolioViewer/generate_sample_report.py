#!/usr/bin/env python3
"""
Generate sample analysis reports for testing the visualizer
This creates mock data with realistic variations
"""

import json
import os
from datetime import datetime, timedelta
import random

def generate_sample_report(symbol, timestamp, base_score=50, variation=10):
    """Generate a sample analysis report with realistic data"""
    
    # Create score with some variation
    score = base_score + random.uniform(-variation, variation)
    score = max(0, min(100, score))  # Keep between 0-100
    
    # Generate triggered signals based on score
    triggered_signals = []
    signal_templates = [
        ("Enhanced Volume Intelligence", "Explosive Volume Breakout", 95),
        ("Corporate Transformation Signals", "Sector Transformation Catalyst", 85),
        ("Enhanced Market Microstructure", "Volatility Amplification Zone", 90),
        ("Enhanced Volume Intelligence", "Institutional Accumulation Pattern", 90),
        ("Advanced Technical Pattern Recognition", "Extreme Volatility Compression", 70),
        ("Comprehensive Catalyst Framework", "Biotech Clinical Catalysts", 75)
    ]
    
    # Trigger more signals for higher scores
    num_signals = int((score / 100) * len(signal_templates) * random.uniform(0.5, 1.2))
    selected_signals = random.sample(signal_templates, min(num_signals, len(signal_templates)))
    
    for category, signal, max_score in selected_signals:
        triggered_signals.append({
            "category": category,
            "signal": signal,
            "score": int(max_score * random.uniform(0.7, 1.0)),
            "description": f"Signal triggered for {symbol}"
        })
    
    # Generate category results
    categories = {
        "corporateTransformation": {
            "name": "Corporate Transformation Signals",
            "description": "Business model pivots, sector changes, and strategic repositioning",
            "weight": 0.2,
            "normalizedScore": score * random.uniform(0.8, 1.2),
            "signals": []
        },
        "volume": {
            "name": "Enhanced Volume Intelligence",
            "description": "Sophisticated volume pattern analysis with quality metrics",
            "weight": 0.3,
            "normalizedScore": score * random.uniform(0.9, 1.3),
            "signals": []
        },
        "technical": {
            "name": "Advanced Technical Pattern Recognition",
            "description": "Enhanced technical indicators with compression patterns",
            "weight": 0.25,
            "normalizedScore": score * random.uniform(0.7, 1.1),
            "signals": []
        },
        "microstructure": {
            "name": "Enhanced Market Microstructure",
            "description": "Refined float analysis with volatility amplification factors",
            "weight": 0.15,
            "normalizedScore": score * random.uniform(0.6, 1.0),
            "signals": []
        },
        "catalyst": {
            "name": "Comprehensive Catalyst Framework",
            "description": "Enhanced catalyst detection including commercial milestones",
            "weight": 0.07,
            "normalizedScore": score * random.uniform(0.5, 0.9),
            "signals": []
        },
        "sentiment": {
            "name": "Multi-Platform Sentiment Intelligence",
            "description": "Enhanced sentiment tracking with timing analysis",
            "weight": 0.03,
            "normalizedScore": score * random.uniform(0.4, 0.8),
            "signals": []
        }
    }
    
    # Normalize category scores
    for cat in categories.values():
        cat["normalizedScore"] = max(0, min(100, cat["normalizedScore"]))
    
    # Generate stock data with variations
    base_price = random.uniform(5, 50)
    price_change = random.uniform(-0.2, 0.3) if score < 50 else random.uniform(0, 0.5)
    current_price = base_price * (1 + price_change)
    
    report = {
        "exportInfo": {
            "timestamp": timestamp.isoformat(),
            "tool": "Stock Signals Analyzer",
            "version": "2.0"
        },
        "stockData": {
            "symbol": symbol,
            "longName": f"{symbol} Corporation",
            "currentPrice": round(current_price, 2),
            "regularMarketPrice": round(current_price, 2),
            "previousClose": round(base_price, 2),
            "regularMarketPreviousClose": round(base_price, 2),
            "marketCap": int(random.uniform(10e6, 500e6)),
            "floatShares": int(random.uniform(1e6, 50e6)),
            "volume": int(random.uniform(100000, 10000000)),
            "averageVolume": int(random.uniform(50000, 1000000)),
            "shortPercentOfFloat": random.uniform(0.01, 0.3),
            "beta": random.uniform(0.5, 3.0),
            "sector": random.choice(["Healthcare", "Technology", "Energy", "Financial"]),
            "industry": "Biotechnology",
            "fiftyTwoWeekLow": round(base_price * 0.5, 2),
            "fiftyTwoWeekHigh": round(base_price * 2.0, 2),
            "trailingPE": random.uniform(5, 50) if random.random() > 0.5 else None,
            "revenueGrowth": random.uniform(-0.5, 3.0)
        },
        "analysisResults": {
            "symbol": symbol,
            "totalScore": score * 10,
            "maxPossibleScore": 1000,
            "scorePercentage": score,
            "categoryResults": categories,
            "triggeredSignals": triggered_signals,
            "missingDataFields": []
        }
    }
    
    return report

def main():
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Define stocks to generate reports for
    stocks = ["KZIA", "ABCD", "ZETA"]
    
    # Generate multiple reports for each stock
    reports_per_stock = 5
    
    print("Generating sample reports...")
    
    for symbol in stocks:
        # Random starting score for each stock
        base_score = random.uniform(20, 70)
        score_trend = random.uniform(-2, 3)  # Trend over time
        
        for i in range(reports_per_stock):
            # Generate timestamp (going back in time)
            days_ago = (reports_per_stock - i - 1) * 2
            timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))
            
            # Adjust base score with trend
            current_score = base_score + (i * score_trend)
            
            # Generate report
            report = generate_sample_report(symbol, timestamp, current_score, variation=5)
            
            # Save report
            filename = f"{symbol}_analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("reports", filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"Created: {filename} (Score: {report['analysisResults']['scorePercentage']:.1f}%)")
    
    print(f"\nGenerated {len(stocks) * reports_per_stock} sample reports in the 'reports' directory")
    print("You can now run the visualizer to see the analysis!")

if __name__ == "__main__":
    main()