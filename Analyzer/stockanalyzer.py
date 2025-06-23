#!/usr/bin/env python3
"""
Stock Signal Analyzer - Python Application
Uses yfinance API to analyze stocks against signals framework
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
import requests
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class StockSignalAnalyzer:
    def __init__(self, signals_path: str = "../Signals/signals.json", 
                 results_path: str = "./Results/history.csv"):
        """
        Initialize the Stock Signal Analyzer
        
        Args:
            signals_path: Path to the signals.json file
            results_path: Path to the history.csv file
        """
        self.signals_path = signals_path
        self.results_path = results_path
        self.signals_framework = self.load_signals_framework()
        self.ensure_results_directory()
        
    def load_signals_framework(self) -> Dict[str, Any]:
        """Load the signals framework from JSON file"""
        try:
            with open(self.signals_path, 'r') as f:
                framework = json.load(f)
                print(f"✅ Loaded signals framework from {self.signals_path}")
                print(f"   Version: {framework.get('meta', {}).get('v', 'unknown')}")
                print(f"   Updated: {framework.get('meta', {}).get('updated', 'unknown')}")
                return framework
        except FileNotFoundError:
            print(f"❌ Signals file not found at {self.signals_path}")
            print("Please ensure signals.json exists at the specified path.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in signals file: {e}")
            sys.exit(1)

    def ensure_results_directory(self):
        """Ensure the Results directory exists"""
        results_dir = os.path.dirname(self.results_path)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"📁 Created directory: {results_dir}")

    def calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < periods + 1:
            return 50.0  # Default neutral RSI
        
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=periods).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=periods).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_volatility(self, prices: pd.Series, periods: int = 20) -> float:
        """Calculate volatility as percentage"""
        if len(prices) < periods:
            return 0.0
        
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=periods).std().iloc[-1]
        # Annualize and convert to percentage
        return volatility * np.sqrt(252) * 100

    def detect_catalysts(self, ticker: str, info: Dict[str, Any]) -> Dict[str, bool]:
        """Detect potential catalysts from stock info"""
        catalysts = {
            'hasCryptoPivot': False,
            'hasDefensePivot': False,
            'hasPartnership': False,
            'hasAcquisition': False,
            'hasRegulatoryApproval': False,
            'hasAdvisorNews': False,
            'hasBlockchainFocus': False,
            'cryptoSignalStrength': 0
        }
        
        # Check business description for keywords
        description = info.get('longBusinessSummary', '').lower()
        
        # Crypto/blockchain detection
        crypto_keywords = ['crypto', 'bitcoin', 'blockchain', 'defi', 'nft', 'web3', 'digital asset', 'treasury strategy']
        crypto_count = sum(1 for kw in crypto_keywords if kw in description)
        if crypto_count > 0:
            catalysts['hasCryptoPivot'] = True
            catalysts['cryptoSignalStrength'] = min(crypto_count * 20, 100)
            if 'blockchain' in description:
                catalysts['hasBlockchainFocus'] = True
        
        # Defense/government detection
        defense_keywords = ['defense', 'military', 'government contract', 'dod', 'pentagon', 'aerospace']
        if any(kw in description for kw in defense_keywords):
            catalysts['hasDefensePivot'] = True
        
        # Partnership detection
        partnership_keywords = ['partnership', 'collaboration', 'joint venture', 'strategic alliance']
        if any(kw in description for kw in partnership_keywords):
            catalysts['hasPartnership'] = True
        
        # Acquisition detection
        acquisition_keywords = ['acquisition', 'merger', 'buyout', 'takeover']
        if any(kw in description for kw in acquisition_keywords):
            catalysts['hasAcquisition'] = True
        
        # Regulatory detection
        regulatory_keywords = ['fda approval', 'regulatory approval', 'clinical trial', 'phase 3']
        if any(kw in description for kw in regulatory_keywords):
            catalysts['hasRegulatoryApproval'] = True
        
        return catalysts

    def get_market_cap_band(self, market_cap: float) -> str:
        """Determine market cap band based on framework"""
        caps = self.signals_framework.get('caps', {})
        
        for band_name, band_config in caps.items():
            if band_config.get('min', 0) <= market_cap < band_config.get('max', float('inf')):
                return band_name
        
        return 'large'  # Default for anything above defined ranges

    def fetch_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch and process stock data using yfinance"""
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Get stock info
            info = stock.info
            
            # Get historical data (60 days)
            hist = stock.history(period="2mo")
            
            if hist.empty:
                raise ValueError(f"No historical data available for {ticker}")
            
            # Get current price and basic metrics
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(hist['Close'])
            volatility = self.calculate_volatility(hist['Close'])
            
            # Get fundamental data
            market_cap = info.get('marketCap', current_price * info.get('sharesOutstanding', 0))
            float_shares = info.get('floatShares', info.get('sharesOutstanding', 0))
            short_percent = info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0
            
            # Detect catalysts
            catalysts = self.detect_catalysts(ticker, info)
            
            # Compile stock data
            stock_data = {
                'ticker': ticker.upper(),
                'price': current_price,
                'volume': current_volume,
                'avgVolume': avg_volume,
                'volumeRatio': current_volume / avg_volume if avg_volume > 0 else 1.0,
                'marketCap': market_cap,
                'marketCapBand': self.get_market_cap_band(market_cap),
                'floatShares': float_shares,
                'shortPercentOfFloat': short_percent,
                'rsi': rsi,
                'volatility': volatility,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                **catalysts  # Add all catalyst flags
            }
            
            return stock_data
            
        except Exception as e:
            raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")

    def apply_threshold_scoring(self, value: float, signal_config: Dict, signal_name: str) -> float:
        """
        Apply threshold-based scoring using the exact structure from signals.json
        
        Args:
            value: The measured value to score
            signal_config: The signal configuration from signals.json
            signal_name: Name of the specific signal (e.g., 'crypto', 'sustain')
        """
        if signal_name not in signal_config:
            return 0.0
        
        signal_def = signal_config[signal_name]
        thresholds = signal_def.get('t', [])
        weight = signal_def.get('w', 0.0)
        
        if not thresholds:
            return 0.0
        
        # Apply threshold scoring based on the arrays in signals.json
        if len(thresholds) == 3:  # [low, med, high] format
            if value >= thresholds[2]:    # Above high threshold
                score = 100
            elif value >= thresholds[1]:  # Above medium threshold  
                # Interpolate between medium and high
                range_size = thresholds[2] - thresholds[1]
                position = (value - thresholds[1]) / range_size if range_size > 0 else 0
                score = 85 + (position * 15)  # 85-100
            elif value >= thresholds[0]:  # Above low threshold
                # Interpolate between low and medium
                range_size = thresholds[1] - thresholds[0]
                position = (value - thresholds[0]) / range_size if range_size > 0 else 0
                score = 60 + (position * 25)  # 60-85
            else:
                # Below low threshold - minimal score
                score = min(value / thresholds[0] * 40, 40) if thresholds[0] > 0 else 0  # 0-40
        elif len(thresholds) == 2:  # [low, high] format
            if value >= thresholds[1]:
                score = 100
            elif value >= thresholds[0]:
                range_size = thresholds[1] - thresholds[0]
                position = (value - thresholds[0]) / range_size if range_size > 0 else 0
                score = 50 + (position * 50)  # 50-100
            else:
                score = min(value / thresholds[0] * 50, 50) if thresholds[0] > 0 else 0  # 0-50
        else:  # Single threshold
            score = 100 if value >= thresholds[0] else 0
        
        return score * weight

    def calculate_nuanced_treasury_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate nuanced treasury/crypto score"""
        signals = self.signals_framework.get('signals', {}).get('treasury', {})
        treasury_score = 0
        
        # Use the crypto signal strength if available
        crypto_strength = stock_data.get('cryptoSignalStrength', 0)
        crypto_weight = signals.get('crypto', {}).get('w', 0.50)
        
        # Crypto pivot scoring with granular levels
        if stock_data.get('hasCryptoPivot', False) or crypto_strength > 0:
            sector = stock_data.get('sector', '').lower()
            market_cap = stock_data.get('marketCap', 0)
            
            # Base crypto score
            if crypto_strength > 80:
                crypto_score = 95
            elif crypto_strength > 60:
                crypto_score = 85
            elif crypto_strength > 40:
                crypto_score = 75
            elif crypto_strength > 20:
                crypto_score = 60
            else:
                crypto_score = 45
            
            # Sector multipliers
            if 'entertainment' in sector:
                crypto_score *= 1.1  # 10% bonus for entertainment + crypto
            elif 'technology' in sector:
                crypto_score *= 1.05  # 5% bonus for tech + crypto
            
            # Market cap impact (smaller = more impact)
            if market_cap < 100000000:  # < $100M
                crypto_score *= 1.15
            elif market_cap < 500000000:  # < $500M
                crypto_score *= 1.05
            
            treasury_score += min(crypto_score, 100) * crypto_weight
        
        # Advisor/partnership scoring
        advisor_weight = signals.get('advisor', {}).get('w', 0.30)
        if stock_data.get('hasAdvisorNews', False):
            advisor_score = 85
            treasury_score += advisor_score * advisor_weight
        
        # Blockchain focus scoring
        blockchain_weight = signals.get('blockchain', {}).get('w', 0.20)
        if stock_data.get('hasBlockchainFocus', False):
            blockchain_score = 70
            treasury_score += blockchain_score * blockchain_weight
        
        # Apply overall treasury weight
        final_score = treasury_score * signals.get('w', 0.22) * 100
        return min(final_score, 100)

    def calculate_nuanced_corporate_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate nuanced corporate transformation score"""
        corp_score = 0
        signals = self.signals_framework.get('signals', {}).get('corp', {})
        
        if stock_data.get('hasDefensePivot', False):
            defense_weight = signals.get('defense', {}).get('w', 0.35)
            # Higher score for smaller companies in defense pivot
            market_cap = stock_data.get('marketCap', 0)
            if market_cap < 100000000:  # < $100M
                defense_score = 95
            elif market_cap < 500000000:  # < $500M
                defense_score = 85
            else:
                defense_score = 70
            corp_score += defense_score * defense_weight
        
        if stock_data.get('hasAcquisition', False):
            acquisition_weight = signals.get('acquisition', {}).get('w', 0.35)
            corp_score += 80 * acquisition_weight
        
        if stock_data.get('hasPartnership', False):
            pivot_weight = signals.get('pivot', {}).get('w', 0.30)
            corp_score += 75 * pivot_weight
        
        # Apply overall weight
        final_score = corp_score * signals.get('w', 0.20) * 100
        return min(final_score, 100)

    def calculate_nuanced_volume_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate nuanced volume score based on multiple factors"""
        vol_ratio = stock_data.get('volumeRatio', 1.0)
        
        signals = self.signals_framework.get('signals', {}).get('volume', {})
        vol_thresholds = self.signals_framework.get('vol', {})
        
        # Progressive volume scoring with more granular levels
        if vol_ratio >= 100:  # Legendary volume (100x+)
            explosive_score = 100
        elif vol_ratio >= 50:   # Extreme volume (50-100x)
            explosive_score = 90 + (vol_ratio - 50) * 0.2  # 90-100
        elif vol_ratio >= 25:   # Very high volume (25-50x)
            explosive_score = 80 + (vol_ratio - 25) * 0.4  # 80-90
        elif vol_ratio >= 15:   # High surge volume (15-25x)
            explosive_score = 70 + (vol_ratio - 15) * 1.0  # 70-80
        elif vol_ratio >= 10:   # Moderate surge (10-15x)
            explosive_score = 60 + (vol_ratio - 10) * 2.0  # 60-70
        elif vol_ratio >= 5:    # Building momentum (5-10x)
            explosive_score = 40 + (vol_ratio - 5) * 4.0   # 40-60
        elif vol_ratio >= 2.5:  # Sustained volume (2.5-5x)
            explosive_score = 20 + (vol_ratio - 2.5) * 8.0 # 20-40
        else:
            explosive_score = max(0, vol_ratio * 8)         # 0-20
        
        # Sustained volume component (consistency bonus)
        sustain_base = vol_thresholds.get('sustain', 2.5)
        if vol_ratio >= sustain_base * 3:  # 3x sustained threshold
            sustain_score = 85
        elif vol_ratio >= sustain_base * 2:  # 2x sustained threshold
            sustain_score = 70
        elif vol_ratio >= sustain_base:  # At sustained threshold
            sustain_score = 50
        else:
            sustain_score = max(0, (vol_ratio / sustain_base) * 30)
        
        # Compression score (for low volume setups)
        if vol_ratio < 0.3:  # Very low volume (compression)
            compression_score = (0.5 - vol_ratio) * 100
        elif vol_ratio < 0.5:  # Low volume
            compression_score = (0.5 - vol_ratio) * 50
        else:
            compression_score = 0
        
        # Combine scores with weights
        explosive_weight = signals.get('explosive', {}).get('w', 0.40)
        sustain_weight = signals.get('sustain', {}).get('w', 0.35)
        compression_weight = signals.get('compression', {}).get('w', 0.25)
        
        total_score = (explosive_score * explosive_weight + 
                      sustain_score * sustain_weight + 
                      compression_score * compression_weight)
        
        # Apply overall volume weight
        final_score = total_score * signals.get('w', 0.25) * 100
        return min(final_score, 100)

    def calculate_nuanced_technical_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate nuanced technical score"""
        rsi = stock_data.get('rsi', 50)
        volatility = stock_data.get('volatility', 0)
        
        signals = self.signals_framework.get('signals', {}).get('tech', {})
        
        # RSI Oversold scoring (granular)
        oversold_weight = signals.get('oversold', {}).get('w', 0.30)
        if rsi <= 15:  # Extremely oversold
            oversold_score = 100
        elif rsi <= 20:  # Very oversold
            oversold_score = 95
        elif rsi <= 25:  # Significantly oversold  
            oversold_score = 85 + (25 - rsi) * 2
        elif rsi <= 30:  # Oversold
            oversold_score = 70 + (30 - rsi) * 3
        elif rsi <= 35:  # Moderately oversold
            oversold_score = 50 + (35 - rsi) * 4
        elif rsi <= 40:  # Slightly oversold
            oversold_score = 25 + (40 - rsi) * 5
        elif rsi <= 45:  # Neutral-bearish
            oversold_score = max(0, 15 - (rsi - 40))
        else:
            oversold_score = 0
        
        # Momentum scoring (granular for upward momentum)
        momentum_weight = signals.get('momentum', {}).get('w', 0.35)
        if rsi >= 70:  # Strong momentum
            momentum_score = 80 + min((rsi - 70) * 0.5, 15)  # 80-95
        elif rsi >= 60:  # Good momentum
            momentum_score = 60 + (rsi - 60) * 2  # 60-80
        elif rsi >= 55:  # Building momentum
            momentum_score = 40 + (rsi - 55) * 4  # 40-60
        elif rsi >= 50:  # Slight momentum
            momentum_score = 20 + (rsi - 50) * 4  # 20-40
        else:
            momentum_score = 0
        
        # Volatility scoring (granular for extreme cases)
        volatility_weight = signals.get('volatility', {}).get('w', 0.35)
        if volatility >= 1000:  # Extreme volatility (like SRM)
            volatility_score = 95 + min((volatility - 1000) / 200, 5)  # 95-100
        elif volatility >= 500:   # Very high volatility
            volatility_score = 85 + (volatility - 500) / 50  # 85-95
        elif volatility >= 200:   # High volatility
            volatility_score = 70 + (volatility - 200) / 20  # 70-85
        elif volatility >= 100:   # Significant volatility
            volatility_score = 55 + (volatility - 100) / 6.7  # 55-70
        elif volatility >= 60:    # Above average volatility
            volatility_score = 40 + (volatility - 60) / 2.7  # 40-55
        elif volatility >= 40:    # Moderate volatility
            volatility_score = 25 + (volatility - 40) / 1.3  # 25-40
        elif volatility >= 20:    # Low volatility
            volatility_score = 10 + (volatility - 20) / 1.3  # 10-25
        else:
            volatility_score = volatility / 2  # 0-10
        
        # Combine all technical scores
        total_score = (oversold_score * oversold_weight + 
                      momentum_score * momentum_weight + 
                      volatility_score * volatility_weight)
        
        # Apply overall technical weight
        final_score = total_score * signals.get('w', 0.18) * 100
        return min(final_score, 100)

    def calculate_nuanced_catalyst_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate nuanced catalyst score"""
        catalyst_score = 0
        signals = self.signals_framework.get('signals', {}).get('catalyst', {})
        
        # Partnership scoring with size-based impact
        if stock_data.get('hasPartnership', False):
            partnership_weight = signals.get('partnership', {}).get('w', 0.35)
            market_cap = stock_data.get('marketCap', 0)
            if market_cap < 50000000:  # < $50M
                partnership_score = 90
            elif market_cap < 200000000:  # < $200M
                partnership_score = 80
            elif market_cap < 1000000000:  # < $1B
                partnership_score = 65
            else:
                partnership_score = 45
            catalyst_score += partnership_score * partnership_weight
        
        # Regulatory approval scoring
        if stock_data.get('hasRegulatoryApproval', False):
            regulatory_weight = signals.get('regulatory', {}).get('w', 0.25)
            catalyst_score += 85 * regulatory_weight
        
        # Social momentum based on volume surge
        social_weight = signals.get('social', {}).get('w', 0.40)
        volume_ratio = stock_data.get('volumeRatio', 1.0)
        
        # Progressive social momentum scoring
        if volume_ratio >= 50:  # Extreme viral volume
            social_score = 95
        elif volume_ratio >= 20:  # Very high volume
            social_score = 80
        elif volume_ratio >= 10:  # High volume
            social_score = 65
        elif volume_ratio >= 5:  # Moderate volume spike
            social_score = 45
        elif volume_ratio >= 3:  # Some momentum
            social_score = 25
        else:
            social_score = 0
        
        catalyst_score += social_score * social_weight
        
        # Apply overall weight
        final_score = catalyst_score * signals.get('w', 0.10) * 100
        return min(final_score, 100)

    def calculate_nuanced_microstructure_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate nuanced microstructure score"""
        float_shares = stock_data.get('floatShares', 0)
        short_ratio = stock_data.get('shortPercentOfFloat', 0)
        volume_ratio = stock_data.get('volumeRatio', 1.0)
        
        signals = self.signals_framework.get('signals', {}).get('micro', {})
        micro_score = 0
        
        # Low float scoring (granular based on optimal range)
        float_weight = signals.get('lowFloat', {}).get('w', 0.60)
        float_optimal = self.signals_framework.get('float', {}).get('optimal', [15000000, 50000000])
        
        if float_shares > 0:
            if float_shares <= 5000000:  # Ultra-low float
                float_score = 100
            elif float_shares <= float_optimal[0]:  # Very low float (5M-15M)
                float_score = 95 - (float_shares - 5000000) / (float_optimal[0] - 5000000) * 5
            elif float_shares <= float_optimal[1]:  # Optimal range (15M-50M)
                # Scale smoothly from 90 down to 75 within optimal range
                ratio = (float_shares - float_optimal[0]) / (float_optimal[1] - float_optimal[0])
                float_score = 90 - (ratio * 15)  # 90 down to 75
            elif float_shares <= 75000000:  # Moderately low float (50M-75M)
                excess = float_shares - float_optimal[1]
                float_score = 75 - (excess / 25000000) * 25  # 75 down to 50
            elif float_shares <= 100000000:  # Higher float (75M-100M)
                excess = float_shares - 75000000
                float_score = 50 - (excess / 25000000) * 25  # 50 down to 25
            elif float_shares <= 150000000:  # Large float (100M-150M)
                excess = float_shares - 100000000
                float_score = 25 - (excess / 50000000) * 20  # 25 down to 5
            else:  # Very large float
                float_score = max(0, 5 - (float_shares - 150000000) / 100000000 * 5)
        else:
            float_score = 0
        
        micro_score += float_score * float_weight
        
        # Short squeeze scoring (progressive)
        squeeze_weight = signals.get('shortSqueeze', {}).get('w', 0.40)
        
        if short_ratio >= 50:  # Extreme short interest
            squeeze_score = 95 + min((short_ratio - 50) / 10, 5)  # 95-100
        elif short_ratio >= 40:  # Very high short interest
            squeeze_score = 85 + (short_ratio - 40)  # 85-95
        elif short_ratio >= 30:  # High short interest
            squeeze_score = 70 + (short_ratio - 30) * 1.5  # 70-85
        elif short_ratio >= 20:  # Significant short interest
            squeeze_score = 50 + (short_ratio - 20) * 2  # 50-70
        elif short_ratio >= 15:  # Moderate short interest
            squeeze_score = 30 + (short_ratio - 15) * 4  # 30-50
        elif short_ratio >= 10:  # Some short interest
            squeeze_score = 15 + (short_ratio - 10) * 3  # 15-30
        elif short_ratio >= 5:   # Low short interest
            squeeze_score = 5 + (short_ratio - 5) * 2   # 5-15
        else:
            squeeze_score = short_ratio  # 0-5
        
        # Volume amplification bonus
        if short_ratio > 15 and volume_ratio > 10:
            squeeze_score *= 1.3  # 30% bonus for high volume + short interest
        elif short_ratio > 20 and volume_ratio > 5:
            squeeze_score *= 1.2  # 20% bonus
        elif short_ratio > 10 and volume_ratio > 3:
            squeeze_score *= 1.1  # 10% bonus
        
        micro_score += min(squeeze_score, 100) * squeeze_weight
        
        # Apply overall microstructure weight
        final_score = micro_score * signals.get('w', 0.05) * 100
        return min(final_score, 100)

    def calculate_signal_scores(self, stock_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate individual signal category scores with nuanced scoring
        """
        scores = {}
        
        # Treasury/Crypto Signals (nuanced)
        scores['treasury'] = self.calculate_nuanced_treasury_score(stock_data)
        
        # Corporate Transformation Signals (nuanced)
        scores['corp'] = self.calculate_nuanced_corporate_score(stock_data)
        
        # Volume Signals (nuanced)
        scores['volume'] = self.calculate_nuanced_volume_score(stock_data)
        
        # Technical Signals (nuanced)
        scores['tech'] = self.calculate_nuanced_technical_score(stock_data)
        
        # Catalyst Signals (nuanced)
        scores['catalyst'] = self.calculate_nuanced_catalyst_score(stock_data)
        
        # Microstructure Signals (nuanced)
        scores['micro'] = self.calculate_nuanced_microstructure_score(stock_data)
        
        return scores
    
    def calculate_total_score(self, stock_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total score with market cap multiplier
        """
        signal_scores = self.calculate_signal_scores(stock_data)
        
        # Apply market cap multiplier
        market_cap_band = stock_data.get('marketCapBand', 'large')
        caps = self.signals_framework.get('caps', {})
        multiplier = caps.get(market_cap_band, {}).get('mult', 1.0)
        
        # Calculate base score
        base_score = sum(signal_scores.values())
        total_score = base_score * multiplier
        
        return total_score, signal_scores
    
    def get_alert_level(self, score: float) -> str:
        """Determine alert level based on score"""
        alerts = self.signals_framework.get('alerts', {})
        
        if score >= alerts.get('legendary', {}).get('score', 350):
            return 'Legendary'
        elif score >= alerts.get('ultra', {}).get('score', 280):
            return 'Ultra-Critical'
        elif score >= alerts.get('critical', {}).get('score', 200):
            return 'Critical'
        else:
            return 'Low'
    
    def save_to_csv(self, stock_data: Dict[str, Any], total_score: float, 
                    signal_scores: Dict[str, float], alert_level: str, force_save: bool = False):
        """Save analysis results to CSV file (with 12-hour duplicate prevention)"""
        
        current_time = datetime.now()
        ticker = stock_data['ticker']
        
        # Check if CSV exists and has recent entry for this ticker (unless force_save is True)
        if not force_save and os.path.exists(self.results_path):
            try:
                df_existing = pd.read_csv(self.results_path)
                
                # Filter for this ticker's records
                ticker_records = df_existing[df_existing['Ticker'] == ticker]
                
                if not ticker_records.empty:
                    # Convert Date column to datetime
                    ticker_records['Date_parsed'] = pd.to_datetime(ticker_records['Date'])
                    
                    # Get the most recent entry
                    most_recent_entry = ticker_records.loc[ticker_records['Date_parsed'].idxmax()]
                    last_entry_time = most_recent_entry['Date_parsed']
                    
                    # Calculate time difference
                    time_diff = current_time - last_entry_time
                    hours_passed = time_diff.total_seconds() / 3600
                    
                    if hours_passed < 12:
                        print(f"⏳ Skipping save - Last entry for {ticker} was {hours_passed:.1f} hours ago")
                        print(f"   (Need to wait {12 - hours_passed:.1f} more hours)")
                        print(f"   Use --force-save to override this check")
                        return
                
            except Exception as e:
                print(f"⚠️  Warning: Error checking existing records: {e}")
                # Continue with save if there's an error reading existing data
        
        # Prepare data for CSV
        csv_data = {
            'Date': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Ticker': ticker,
            'Price': stock_data['price'],
            'Volume': stock_data['volume'],
            'Volume_Ratio': stock_data['volumeRatio'],
            'Market_Cap': stock_data['marketCap'],
            'Market_Cap_Band': stock_data['marketCapBand'],
            'Float_Shares': stock_data['floatShares'],
            'Short_Interest': stock_data['shortPercentOfFloat'],
            'RSI': stock_data['rsi'],
            'Volatility': stock_data['volatility'],
            'Sector': stock_data['sector'],
            'Industry': stock_data['industry'],
            'Total_Score': round(total_score, 2),
            'Alert_Level': alert_level,
            'Treasury_Score': round(signal_scores['treasury'], 2),
            'Corp_Score': round(signal_scores['corp'], 2),
            'Volume_Score': round(signal_scores['volume'], 2),
            'Tech_Score': round(signal_scores['tech'], 2),
            'Catalyst_Score': round(signal_scores['catalyst'], 2),
            'Micro_Score': round(signal_scores['micro'], 2),
            'Crypto_Pivot': stock_data.get('hasCryptoPivot', False),
            'Defense_Pivot': stock_data.get('hasDefensePivot', False),
            'Partnership': stock_data.get('hasPartnership', False),
            'Acquisition': stock_data.get('hasAcquisition', False)
        }
        
        # Create DataFrame
        df_new = pd.DataFrame([csv_data])
        
        # Append to existing CSV or create new one
        if os.path.exists(self.results_path):
            df_existing = pd.read_csv(self.results_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        # Save to CSV
        df_combined.to_csv(self.results_path, index=False)
        print(f"📊 Results saved to {self.results_path}")
    
    def display_results(self, stock_data: Dict[str, Any], total_score: float, 
                       signal_scores: Dict[str, float], alert_level: str):
        """Display analysis results"""
        
        print("\n" + "="*80)
        print(f"🎯 STOCK SIGNAL ANALYSIS: {stock_data['ticker']}")
        print("="*80)
        
        # Basic stock info
        print(f"\n📈 STOCK INFORMATION:")
        print(f"   Price: ${stock_data['price']:.2f}")
        print(f"   Market Cap: ${stock_data['marketCap']:,.0f} ({stock_data['marketCapBand'].upper()})")
        print(f"   Volume: {stock_data['volume']:,} ({stock_data['volumeRatio']:.2f}x avg)")
        print(f"   Float: {stock_data['floatShares']:,} shares")
        print(f"   Short Interest: {stock_data['shortPercentOfFloat']:.1f}%")
        print(f"   RSI: {stock_data['rsi']:.1f}")
        print(f"   Volatility: {stock_data['volatility']:.1f}%")
        print(f"   Sector: {stock_data['sector']}")
        
        # Score display
        print(f"\n🚀 SIGNAL ANALYSIS:")
        print(f"   TOTAL SCORE: {total_score:.1f}")
        print(f"   ALERT LEVEL: {alert_level}")
        
        # Signal breakdown
        print(f"\n📊 SIGNAL BREAKDOWN:")
        signal_names = {
            'treasury': 'Treasury/Crypto',
            'corp': 'Corporate Transform',
            'volume': 'Volume Intelligence', 
            'tech': 'Technical Setup',
            'catalyst': 'Catalyst Events',
            'micro': 'Microstructure'
        }
        
        for key, score in signal_scores.items():
            name = signal_names.get(key, key.title())
            print(f"   {name:.<25} {score:.1f}")
        
        # Catalyst indicators
        catalysts = []
        if stock_data.get('hasCryptoPivot'): catalysts.append("Crypto Pivot")
        if stock_data.get('hasDefensePivot'): catalysts.append("Defense Pivot") 
        if stock_data.get('hasPartnership'): catalysts.append("Partnership")
        if stock_data.get('hasAcquisition'): catalysts.append("Acquisition")
        if stock_data.get('hasRegulatoryApproval'): catalysts.append("Regulatory")
        
        if catalysts:
            print(f"\n✨ DETECTED CATALYSTS: {', '.join(catalysts)}")
        
        print("\n" + "="*80)
    
    def analyze_stock(self, ticker: str, force_save: bool = False) -> bool:
        """
        Analyze a single stock ticker
        
        Args:
            ticker: Stock ticker symbol
            force_save: Force save even if less than 12 hours since last entry
            
        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            print(f"\n🔍 Analyzing {ticker.upper()}...")
            
            # Fetch stock data
            stock_data = self.fetch_stock_data(ticker)
            
            # Calculate scores
            total_score, signal_scores = self.calculate_total_score(stock_data)
            alert_level = self.get_alert_level(total_score)
            
            # Display results
            self.display_results(stock_data, total_score, signal_scores, alert_level)
            
            # Save to CSV
            self.save_to_csv(stock_data, total_score, signal_scores, alert_level, force_save)
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {str(e)}")
            return False

def main():
    """Main application entry point"""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Stock Signal Analyzer v4.1 - Analyze stocks using signals framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s AAPL                    # Analyze single stock
  %(prog)s AAPL MSFT TSLA         # Analyze multiple stocks
  %(prog)s AAPL --signals custom.json  # Use custom signals file
  %(prog)s AAPL --output results.csv   # Custom output file
  %(prog)s AAPL --force-save           # Override 12-hour duplicate check

Note: By default, the analyzer prevents duplicate entries for the same ticker 
      within a 12-hour window. Use --force-save to override this behavior.
        '''
    )
    
    # Add arguments
    parser.add_argument('tickers', 
                        metavar='TICKER', 
                        type=str, 
                        nargs='+',
                        help='Stock ticker symbol(s) to analyze (e.g., AAPL, MSFT)')
    
    parser.add_argument('-s', '--signals',
                        default='../Signals/signals.json',
                        help='Path to signals.json file (default: ../Signals/signals.json)')
    
    parser.add_argument('-o', '--output',
                        default='./Results/history.csv',
                        help='Path to output CSV file (default: ./Results/history.csv)')
    
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--no-save',
                        action='store_true',
                        help='Do not save results to CSV')
    
    parser.add_argument('--force-save',
                        action='store_true',
                        help='Force save to CSV even if less than 12 hours since last entry')
    
    # Parse arguments
    args = parser.parse_args()
    
    print("🚀 Stock Signal Analyzer v4.1")
    print("=" * 50)
    
    # Initialize analyzer with specified paths
    try:
        analyzer = StockSignalAnalyzer(
            signals_path=args.signals,
            results_path=args.output
        )
        print(f"✅ Loaded signals framework v{analyzer.signals_framework.get('meta', {}).get('v', 'unknown')}")
        if args.verbose:
            print(f"   Signals file: {args.signals}")
            print(f"   Output file: {args.output}")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        sys.exit(1)
    
    # Track results
    success_count = 0
    failed_tickers = []
    
    # Analyze each ticker
    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            # Analyze the stock
            stock_data = analyzer.fetch_stock_data(ticker)
            total_score, signal_scores = analyzer.calculate_total_score(stock_data)
            alert_level = analyzer.get_alert_level(total_score)
            
            # Display results
            analyzer.display_results(stock_data, total_score, signal_scores, alert_level)
            
            # Save to CSV unless disabled
            if not args.no_save:
                analyzer.save_to_csv(stock_data, total_score, signal_scores, alert_level, args.force_save)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error analyzing {ticker}: {str(e)}")
            failed_tickers.append(ticker)
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully analyzed: {success_count} ticker(s)")
    if failed_tickers:
        print(f"❌ Failed to analyze: {', '.join(failed_tickers)}")
    
    # Exit with appropriate code
    sys.exit(0 if not failed_tickers else 1)

if __name__ == "__main__":
    main()