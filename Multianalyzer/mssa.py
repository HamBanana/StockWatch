#!/usr/bin/env python3
"""
Multi-Stock Signal Analyzer
Runs stock signal analysis on multiple tickers from symbols.json using signals.json framework
"""

import json
import os
import yfinance as yf
from datetime import datetime
import time
import logging
from typing import Dict, List, Any, Optional
import sys

# ============================================================================
# CONFIGURATION - Easily configurable paths and settings
# ============================================================================

CONFIG = {
    # File paths
    "symbols_file": "symbols.json",           # Path to symbols JSON file
    "signals_file": "../analyzer/signals.json",          # Path to signals framework JSON
    "output_dir": "reports",                 # Directory to save analysis reports
    
    # Analysis settings
    "delay_between_requests": 1.0,           # Seconds between yfinance requests
    "max_retries": 3,                        # Max retries for failed requests
    "timeout": 30,                           # Request timeout in seconds
    
    # Output settings
    "save_individual_reports": True,         # Save individual JSON reports
    "save_summary_report": True,             # Save combined summary
    "verbose_logging": True,                 # Enable detailed logging
    
    # Filtering options
    "min_market_cap": 1000000,              # Minimum market cap ($1M)
    "max_market_cap": None,                 # Maximum market cap (None = no limit)
    "skip_on_error": True,                  # Continue processing other stocks on error
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup logging configuration"""
    log_level = logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_analyzer.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# ============================================================================
# STOCK DATA FETCHER
# ============================================================================

class StockDataFetcher:
    """Handles fetching stock data from yfinance"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def fetch_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive stock data for a symbol"""
        for attempt in range(CONFIG["max_retries"]):
            try:
                self.logger.info(f"Fetching data for {symbol} (attempt {attempt + 1})")
                
                # Create yfinance ticker
                ticker = yf.Ticker(symbol)
                
                # Get stock info
                info = ticker.info
                
                if not info or 'symbol' not in info:
                    self.logger.warning(f"No data found for {symbol}")
                    return None
                
                # Add current market data
                hist = ticker.history(period="5d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    info['currentPrice'] = float(latest['Close'])
                    info['volume'] = int(latest['Volume'])
                    info['dayHigh'] = float(latest['High'])
                    info['dayLow'] = float(latest['Low'])
                
                # Validate minimum requirements
                market_cap = info.get('marketCap', 0)
                if market_cap < CONFIG["min_market_cap"]:
                    self.logger.info(f"Skipping {symbol}: Market cap ${market_cap:,} below minimum")
                    return None
                    
                if CONFIG["max_market_cap"] and market_cap > CONFIG["max_market_cap"]:
                    self.logger.info(f"Skipping {symbol}: Market cap ${market_cap:,} above maximum")
                    return None
                
                self.logger.info(f"Successfully fetched data for {symbol}")
                return info
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < CONFIG["max_retries"] - 1:
                    time.sleep(CONFIG["delay_between_requests"] * 2)
                    
        self.logger.error(f"Failed to fetch data for {symbol} after {CONFIG['max_retries']} attempts")
        return None

# ============================================================================
# SIGNAL ANALYSIS ENGINE
# ============================================================================

class SignalAnalysisEngine:
    """Performs signal analysis using the signals.json framework"""
    
    def __init__(self, signals_config: Dict[str, Any], logger):
        self.signals_config = signals_config
        self.logger = logger
        
    def transform_yfinance_data(self, yfinance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform yfinance data to our analysis format"""
        transformed = {
            'symbol': yfinance_data.get('symbol'),
            'price': yfinance_data.get('currentPrice') or yfinance_data.get('regularMarketPrice', 0),
            'marketCap': yfinance_data.get('marketCap', 0),
            'floatShares': yfinance_data.get('floatShares', 0),
            'sharesOutstanding': yfinance_data.get('sharesOutstanding', 0),
            'volume': yfinance_data.get('volume', 0),
            'avgVolume30d': yfinance_data.get('averageVolume', 0),
            'avgVolume10d': yfinance_data.get('averageVolume10days', 0),
            
            # Technical indicators (would need price history for real calculations)
            'rsi14': self.estimate_rsi(yfinance_data),
            'williamsR': None,
            'fiftyDayAvg': yfinance_data.get('fiftyDayAverage'),
            'twoHundredDayAvg': yfinance_data.get('twoHundredDayAverage'),
            
            # Ownership
            'insiderOwnership': yfinance_data.get('heldByInsiders'),
            'institutionalOwnership': yfinance_data.get('heldByInstitutions'),
            
            # Short data
            'shortInterest': yfinance_data.get('shortPercentOfFloat'),
            'shortRatio': yfinance_data.get('shortRatio'),
            
            # Company info
            'sector': yfinance_data.get('sector'),
            'industry': yfinance_data.get('industry'),
            
            # Financial metrics
            'trailingPE': yfinance_data.get('trailingPE'),
            'forwardPE': yfinance_data.get('forwardPE'),
            'priceToBook': yfinance_data.get('priceToBook'),
            'debtToEquity': yfinance_data.get('debtToEquity'),
            'revenueGrowth': yfinance_data.get('revenueGrowth'),
            
            # Calculated fields
            'volumeRatio': self.calculate_volume_ratio(yfinance_data),
            'floatTurnover': self.calculate_float_turnover(yfinance_data),
            'daysToCover': yfinance_data.get('shortRatio'),
            
            # Raw data for reference
            'rawData': yfinance_data
        }
        
        return transformed
    
    def estimate_rsi(self, yfinance_data: Dict[str, Any]) -> Optional[float]:
        """Estimate RSI based on price vs moving averages"""
        current_price = yfinance_data.get('currentPrice') or yfinance_data.get('regularMarketPrice')
        fifty_day_avg = yfinance_data.get('fiftyDayAverage')
        
        if not current_price or not fifty_day_avg:
            return None
            
        price_vs_ma = current_price / fifty_day_avg
        
        if price_vs_ma > 1.2:
            return 75  # Likely overbought
        elif price_vs_ma < 0.8:
            return 25  # Likely oversold
        else:
            return 50  # Neutral
    
    def calculate_volume_ratio(self, yfinance_data: Dict[str, Any]) -> Optional[float]:
        """Calculate volume vs average volume ratio"""
        volume = yfinance_data.get('volume', 0)
        avg_volume = yfinance_data.get('averageVolume', 0)
        
        if avg_volume > 0:
            return volume / avg_volume
        return None
    
    def calculate_float_turnover(self, yfinance_data: Dict[str, Any]) -> Optional[float]:
        """Calculate float turnover ratio"""
        volume = yfinance_data.get('volume', 0)
        float_shares = yfinance_data.get('floatShares', 0)
        
        if float_shares > 0:
            return volume / float_shares
        return None
    
    def analyze_signals(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive signal analysis"""
        results = {
            'symbol': stock_data['symbol'],
            'totalScore': 0,
            'maxPossibleScore': 0,
            'categoryResults': {},
            'triggeredSignals': [],
            'missingDataFields': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for missing data
        required_fields = ['marketCap', 'floatShares', 'volume', 'avgVolume30d']
        for field in required_fields:
            if stock_data.get(field) is None:
                results['missingDataFields'].append(field)
        
        # Analyze each signal category
        signal_categories = self.signals_config.get('signalCategories', {})
        
        for category_key, category in signal_categories.items():
            category_result = self.analyze_category(category, stock_data)
            results['categoryResults'][category_key] = category_result
            
            # Add to total scores
            weight = category.get('weight', 1)
            results['totalScore'] += category_result['score'] * weight
            results['maxPossibleScore'] += category_result['maxScore'] * weight
            
            # Collect triggered signals
            for signal in category_result['signals']:
                if signal['triggered']:
                    results['triggeredSignals'].append({
                        'category': category['name'],
                        'signal': signal['name'],
                        'score': signal['score'],
                        'description': signal['description']
                    })
        
        # Calculate overall score percentage
        if results['maxPossibleScore'] > 0:
            results['scorePercentage'] = (results['totalScore'] / results['maxPossibleScore']) * 100
        else:
            results['scorePercentage'] = 0
        
        return results
    
    def analyze_category(self, category: Dict[str, Any], stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single signal category"""
        result = {
            'name': category['name'],
            'description': category['description'],
            'weight': category.get('weight', 1),
            'score': 0,
            'maxScore': 0,
            'signals': []
        }
        
        signals = category.get('signals', {})
        
        for signal_key, signal in signals.items():
            signal_result = self.evaluate_signal(signal, stock_data)
            result['signals'].append(signal_result)
            
            signal_weight = signal.get('weight', 1)
            result['score'] += signal_result['score'] * signal_weight
            result['maxScore'] += signal_result['maxScore'] * signal_weight
        
        # Calculate normalized score
        if result['maxScore'] > 0:
            result['normalizedScore'] = (result['score'] / result['maxScore']) * 100
        else:
            result['normalizedScore'] = 0
        
        return result
    
    def evaluate_signal(self, signal: Dict[str, Any], stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single signal"""
        result = {
            'name': signal['name'],
            'description': signal['description'],
            'triggered': False,
            'score': 0,
            'maxScore': 0,
            'details': [],
            'conditions': []
        }
        
        thresholds = signal.get('thresholds', {})
        
        # Find maximum possible score
        for threshold in thresholds.values():
            result['maxScore'] = max(result['maxScore'], threshold.get('score', 0))
        
        # Evaluate each threshold
        for threshold_key, threshold in thresholds.items():
            condition_met = self.evaluate_condition(threshold.get('condition', ''), stock_data)
            
            condition_result = {
                'condition': threshold.get('condition', ''),
                'met': condition_met,
                'score': threshold.get('score', 0),
                'description': threshold.get('description', '')
            }
            result['conditions'].append(condition_result)
            
            if condition_met:
                result['triggered'] = True
                result['score'] = max(result['score'], threshold.get('score', 0))
                result['details'].append(threshold.get('description', ''))
        
        return result
    
    def evaluate_condition(self, condition: str, stock_data: Dict[str, Any]) -> bool:
        """Evaluate a signal condition (simplified implementation)"""
        if not condition or not isinstance(condition, str):
            return False
        
        try:
            # Market cap conditions
            market_cap = stock_data.get('marketCap', 0)
            if 'marketCap' in condition:
                if '50M-500M' in condition or ('50000000' in condition and '500000000' in condition):
                    return 50000000 <= market_cap <= 500000000
                elif '100M-2B' in condition:
                    return 100000000 <= market_cap <= 2000000000
                elif '2M-100M' in condition:
                    return 2000000 <= market_cap <= 100000000
            
            # Volume conditions
            volume_ratio = stock_data.get('volumeRatio', 0)
            if 'volume' in condition and volume_ratio:
                if '25x' in condition:
                    return volume_ratio >= 25
                elif '50x' in condition:
                    return volume_ratio >= 50
                elif '100x' in condition:
                    return volume_ratio >= 100
                elif '2.0' in condition and 'avgVolume' in condition:
                    return volume_ratio >= 2.0
                elif '3.0' in condition and 'avgVolume' in condition:
                    return volume_ratio >= 3.0
            
            # RSI conditions
            rsi = stock_data.get('rsi14', 50)
            if 'rsi14' in condition or 'RSI' in condition:
                if '< 30' in condition or '<= 30' in condition:
                    return rsi <= 30
                elif '< 25' in condition or '<= 25' in condition:
                    return rsi <= 25
                elif '< 20' in condition:
                    return rsi < 20
            
            # Short interest conditions
            short_interest = stock_data.get('shortInterest', 0)
            if 'shortInterest' in condition and short_interest:
                if '20%' in condition or '0.20' in condition:
                    return short_interest >= 0.20
                elif '30%' in condition or '0.30' in condition:
                    return short_interest >= 0.30
                elif '50%' in condition or '0.50' in condition:
                    return short_interest >= 0.50
            
            # Float turnover conditions
            float_turnover = stock_data.get('floatTurnover', 0)
            if 'floatTurnover' in condition and float_turnover:
                if '1.0' in condition:
                    return float_turnover >= 1.0
                elif '2.0' in condition:
                    return float_turnover >= 2.0
                elif '5.0' in condition:
                    return float_turnover >= 5.0
            
            # Sector conditions
            sector = stock_data.get('sector', '').lower()
            if 'biotech' in condition.lower():
                return 'biotechnology' in sector or 'healthcare' in sector
            if 'defense' in condition.lower():
                return 'defense' in sector or 'aerospace' in sector
            
            # Default: use simple heuristics based on stock characteristics
            if 'catalyst' in condition.lower():
                # Biotech companies more likely to have catalysts
                return sector in ['healthcare', 'biotechnology']
            
            if 'acquisition' in condition.lower():
                # Smaller companies more likely acquisition targets
                return market_cap < 1000000000
            
            # For demonstration, trigger some conditions based on stock characteristics
            if market_cap < 100000000:  # Small cap stocks
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates and saves analysis reports"""
    
    def __init__(self, output_dir: str, logger):
        self.output_dir = output_dir
        self.logger = logger
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def save_individual_report(self, symbol: str, stock_data: Dict[str, Any], 
                             analysis_results: Dict[str, Any], signals_config: Dict[str, Any]) -> str:
        """Save individual stock analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_analysis_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        report = {
            'exportInfo': {
                'timestamp': datetime.now().isoformat(),
                'tool': 'Multi-Stock Signal Analyzer',
                'version': '1.0',
                'symbol': symbol
            },
            'stockData': stock_data,
            'analysisResults': analysis_results,
            'signalsConfig': signals_config
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Saved individual report: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving individual report for {symbol}: {e}")
            return ""
    
    def save_summary_report(self, all_results: List[Dict[str, Any]]) -> str:
        """Save combined summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_summary_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Sort by score descending
        sorted_results = sorted(all_results, 
                              key=lambda x: x['analysis']['scorePercentage'], 
                              reverse=True)
        
        summary = {
            'exportInfo': {
                'timestamp': datetime.now().isoformat(),
                'tool': 'Multi-Stock Signal Analyzer',
                'version': '1.0',
                'totalStocks': len(all_results)
            },
            'summary': {
                'totalAnalyzed': len(all_results),
                'highSignalStocks': len([r for r in all_results if r['analysis']['scorePercentage'] >= 60]),
                'averageScore': sum(r['analysis']['scorePercentage'] for r in all_results) / len(all_results) if all_results else 0,
                'topPerformers': sorted_results[:5]
            },
            'allResults': sorted_results
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Saved summary report: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving summary report: {e}")
            return ""

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class MultiStockAnalyzer:
    """Main application class"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.fetcher = StockDataFetcher(self.logger)
        self.signals_config = None
        self.analyzer = None
        self.reporter = ReportGenerator(CONFIG["output_dir"], self.logger)
        
    def load_configuration(self) -> bool:
        """Load signals.json configuration"""
        try:
            with open(CONFIG["signals_file"], 'r') as f:
                self.signals_config = json.load(f)
            
            self.analyzer = SignalAnalysisEngine(self.signals_config, self.logger)
            self.logger.info(f"Loaded signals configuration from {CONFIG['signals_file']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading signals configuration: {e}")
            return False
    
    def load_symbols(self) -> List[str]:
        """Load symbols from symbols.json"""
        try:
            with open(CONFIG["symbols_file"], 'r') as f:
                symbols_data = json.load(f)
            
            # Handle different formats
            symbols = []
            if isinstance(symbols_data, dict):
                # Format: {"0": {"ticker": "MSFT", ...}, ...}
                for item in symbols_data.values():
                    if isinstance(item, dict) and 'ticker' in item:
                        symbols.append(item['ticker'])
                    elif isinstance(item, str):
                        symbols.append(item)
            elif isinstance(symbols_data, list):
                # Format: ["MSFT", "AAPL", ...]
                symbols = symbols_data
            
            self.logger.info(f"Loaded {len(symbols)} symbols: {', '.join(symbols)}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error loading symbols: {e}")
            return []
    
    def analyze_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock"""
        self.logger.info(f"Starting analysis for {symbol}")
        
        # Fetch stock data
        stock_data = self.fetcher.fetch_stock_data(symbol)
        if not stock_data:
            return None
        
        # Transform data
        transformed_data = self.analyzer.transform_yfinance_data(stock_data)
        
        # Run signal analysis
        analysis_results = self.analyzer.analyze_signals(transformed_data)
        
        # Save individual report if enabled
        report_path = ""
        if CONFIG["save_individual_reports"]:
            report_path = self.reporter.save_individual_report(
                symbol, stock_data, analysis_results, self.signals_config)
        
        result = {
            'symbol': symbol,
            'stockData': stock_data,
            'analysis': analysis_results,
            'reportPath': report_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log results
        score = analysis_results['scorePercentage']
        triggered_count = len(analysis_results['triggeredSignals'])
        self.logger.info(f"{symbol}: Score {score:.1f}%, {triggered_count} triggered signals")
        
        return result
    
    def run_analysis(self):
        """Run analysis on all symbols"""
        self.logger.info("Starting Multi-Stock Signal Analyzer")
        
        # Load configuration
        if not self.load_configuration():
            self.logger.error("Failed to load configuration. Exiting.")
            return
        
        # Load symbols
        symbols = self.load_symbols()
        if not symbols:
            self.logger.error("No symbols to analyze. Exiting.")
            return
        
        # Analyze each stock
        all_results = []
        successful_analyses = 0
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"Processing {symbol} ({i}/{len(symbols)})")
            
            try:
                result = self.analyze_stock(symbol)
                if result:
                    all_results.append(result)
                    successful_analyses += 1
                else:
                    self.logger.warning(f"Analysis failed for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                if not CONFIG["skip_on_error"]:
                    break
            
            # Add delay between requests
            if i < len(symbols):
                time.sleep(CONFIG["delay_between_requests"])
        
        # Generate summary report
        if all_results and CONFIG["save_summary_report"]:
            summary_path = self.reporter.save_summary_report(all_results)
            self.logger.info(f"Summary report saved to: {summary_path}")
        
        # Final summary
        self.logger.info(f"Analysis complete: {successful_analyses}/{len(symbols)} stocks analyzed")
        
        if all_results:
            # Show top performers
            sorted_results = sorted(all_results, 
                                  key=lambda x: x['analysis']['scorePercentage'], 
                                  reverse=True)
            
            self.logger.info("Top 5 signal scores:")
            for i, result in enumerate(sorted_results[:5], 1):
                symbol = result['symbol']
                score = result['analysis']['scorePercentage']
                triggered = len(result['analysis']['triggeredSignals'])
                self.logger.info(f"  {i}. {symbol}: {score:.1f}% ({triggered} signals)")

def main():
    """Main entry point"""
    print("Multi-Stock Signal Analyzer")
    print("=" * 50)
    print(f"Symbols file: {CONFIG['symbols_file']}")
    print(f"Signals file: {CONFIG['signals_file']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print()
    
    # Verify files exist
    for file_path in [CONFIG['symbols_file'], CONFIG['signals_file']]:
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            return
    
    # Run analysis
    analyzer = MultiStockAnalyzer()
    analyzer.run_analysis()
    
    print("\nAnalysis complete! Check the reports directory for results.")
    print("Use the portfolioviewer.py script to visualize the results.")

if __name__ == "__main__":
    main()