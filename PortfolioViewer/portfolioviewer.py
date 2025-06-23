#!/usr/bin/env python3
"""
Complete Enhanced Stock Analysis Portfolio Viewer
Professional visualization and analysis of stock signal reports with all implementations
"""

import json
import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import warnings
from pathlib import Path
import textwrap
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

class CompleteStockAnalysisVisualizer:
    def __init__(self, reports_dir="reports"):
        self.reports_dir = Path(reports_dir)
        self.data = defaultdict(list)
        self.stocks = set()
        self.analysis_dates = []
        
        # Enhanced styling
        plt.style.use('dark_background')
        self.setup_styling()
        
    def setup_styling(self):
        """Setup enhanced styling for all plots"""
        # Custom color palette
        self.colors = {
            'critical': '#a855f7',   # Purple
            'high': '#22c55e',       # Green  
            'medium': '#f59e0b',     # Orange
            'low': '#ef4444',        # Red
            'neutral': '#6b7280',    # Gray
            'accent': '#3b82f6',     # Blue
            'background': '#0f172a', # Dark blue
            'text': '#e2e8f0',       # Light gray
            'success': '#10b981',    # Emerald
            'warning': '#fbbf24',    # Amber
            'info': '#06b6d4'        # Cyan
        }
        
        # Set default parameters
        plt.rcParams.update({
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': '#1e293b',
            'text.color': self.colors['text'],
            'axes.labelcolor': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
            'axes.edgecolor': '#475569',
            'grid.color': '#374151',
            'grid.alpha': 0.3,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 16
        })
        
    def load_reports(self):
        """Load and process all JSON reports from the reports directory"""
        pattern = str(self.reports_dir / "*_analysis_*.json")
        files = glob.glob(pattern)
        
        if not files:
            print(f"❌ No analysis files found in {self.reports_dir}")
            return False
            
        print(f"📁 Found {len(files)} analysis files")
        
        failed_files = []
        total_reports = 0
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    report = json.load(f)
                    
                # Extract timestamp and symbol
                timestamp_str = report['exportInfo']['timestamp']
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                symbol = report['stockData']['symbol']
                
                self.stocks.add(symbol)
                self.analysis_dates.append(timestamp)
                
                # Store the data with enhanced metadata
                report_data = {
                    'timestamp': timestamp,
                    'file': Path(file_path).name,
                    'data': report,
                    'processed_at': datetime.now()
                }
                
                self.data[symbol].append(report_data)
                total_reports += 1
                    
            except Exception as e:
                failed_files.append((file_path, str(e)))
                
        # Sort data by timestamp for each stock
        for symbol in self.data:
            self.data[symbol].sort(key=lambda x: x['timestamp'])
            
        # Print summary
        print(f"✅ Successfully loaded {total_reports} reports for {len(self.stocks)} stocks")
        print(f"📊 Stocks: {', '.join(sorted(self.stocks))}")
        
        if failed_files:
            print(f"❌ Failed to load {len(failed_files)} files:")
            for file_path, error in failed_files:
                print(f"   {Path(file_path).name}: {error}")
                
        if self.analysis_dates:
            date_range = f"{min(self.analysis_dates).strftime('%Y-%m-%d')} to {max(self.analysis_dates).strftime('%Y-%m-%d')}"
            print(f"📅 Analysis period: {date_range}")
            
        return len(self.data) > 0

    def create_comprehensive_dashboard(self):
        """Create a comprehensive multi-page dashboard"""
        if not self.stocks:
            print("❌ No data to visualize")
            return
            
        print("🎨 Generating comprehensive dashboard...")
        
        # Page 1: Executive Summary
        self.create_executive_summary()
        
        # Page 2: Portfolio Performance Overview  
        self.create_portfolio_overview()
        
        # Page 3: Signal Analysis Deep Dive
        self.create_signal_analysis()
        
        # Page 4: Risk and Opportunity Assessment
        self.create_risk_assessment()
        
        # Individual stock reports
        for symbol in sorted(self.stocks):
            self.create_detailed_stock_report(symbol)

    def create_executive_summary(self):
        """Executive summary dashboard page"""
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('📈 STOCK SIGNALS PORTFOLIO - EXECUTIVE SUMMARY', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3,
                     left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # 1. Portfolio Health Score (Large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self.plot_portfolio_health_score(ax1)
        
        # 2. Top Performers
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_top_performers(ax2)
        
        # 3. Alert Summary
        ax3 = fig.add_subplot(gs[0, 3])
        self.plot_alert_summary(ax3)
        
        # 4. Recent Activity
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_recent_activity(ax4)
        
        # 5. Signal Strength Distribution
        ax5 = fig.add_subplot(gs[2, 0:2])
        self.plot_signal_distribution(ax5)
        
        # 6. Market Cap Distribution
        ax6 = fig.add_subplot(gs[2, 2])
        self.plot_market_cap_distribution(ax6)
        
        # 7. Sector Breakdown
        ax7 = fig.add_subplot(gs[2, 3])
        self.plot_sector_breakdown(ax7)
        
        # 8. Key Statistics Table
        ax8 = fig.add_subplot(gs[3, :])
        self.plot_key_statistics_table(ax8)
        
        plt.tight_layout()
        self.save_and_show(fig, "executive_summary")

    def create_portfolio_overview(self):
        """Portfolio performance overview"""
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('📊 PORTFOLIO PERFORMANCE OVERVIEW', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                     left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # 1. Signal Score Evolution (Full Width)
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_portfolio_evolution(ax1)
        
        # 2. Category Performance Heatmap
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_comprehensive_heatmap(ax2)
        
        # 3. Volume Analysis
        ax3 = fig.add_subplot(gs[2, 0])
        self.plot_volume_analysis(ax3)
        
        # 4. Price Performance
        ax4 = fig.add_subplot(gs[2, 1])
        self.plot_price_performance(ax4)
        
        # 5. Correlation Matrix
        ax5 = fig.add_subplot(gs[2, 2])
        self.plot_correlation_matrix(ax5)
        
        # 6. Performance Rankings
        ax6 = fig.add_subplot(gs[3, :])
        self.plot_performance_rankings(ax6)
        
        plt.tight_layout()
        self.save_and_show(fig, "portfolio_overview")

    def create_signal_analysis(self):
        """Detailed signal analysis dashboard"""
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('🔍 SIGNAL ANALYSIS DEEP DIVE', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                     left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # 1. Signal Trigger Timeline
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_signal_timeline(ax1)
        
        # 2. Category Deep Dive
        ax2 = fig.add_subplot(gs[1, :2])
        self.plot_category_deep_dive(ax2)
        
        # 3. Signal Strength Evolution
        ax3 = fig.add_subplot(gs[1, 2])
        self.plot_signal_strength_evolution(ax3)
        
        # 4. Triggered Signals Analysis
        ax4 = fig.add_subplot(gs[2, :])
        self.plot_triggered_signals_analysis(ax4)
        
        # 5. Signal Effectiveness
        ax5 = fig.add_subplot(gs[3, 0])
        self.plot_signal_effectiveness(ax5)
        
        # 6. Missing Data Analysis
        ax6 = fig.add_subplot(gs[3, 1])
        self.plot_missing_data_analysis(ax6)
        
        # 7. Signal Quality Metrics
        ax7 = fig.add_subplot(gs[3, 2])
        self.plot_signal_quality_metrics(ax7)
        
        plt.tight_layout()
        self.save_and_show(fig, "signal_analysis")

    def create_risk_assessment(self):
        """Risk and opportunity assessment dashboard"""
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('⚠️ RISK & OPPORTUNITY ASSESSMENT', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                     left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # 1. Risk-Return Scatter
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_risk_return_analysis(ax1)
        
        # 2. Risk Factors Summary
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_risk_factors(ax2)
        
        # 3. Market Cap vs Signal Strength
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_market_cap_vs_signals(ax3)
        
        # 4. Volatility Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_volatility_analysis(ax4)
        
        # 5. Liquidity Assessment
        ax5 = fig.add_subplot(gs[1, 2])
        self.plot_liquidity_assessment(ax5)
        
        # 6. Opportunity Matrix
        ax6 = fig.add_subplot(gs[2, :])
        self.plot_opportunity_matrix(ax6)
        
        # 7. Portfolio Recommendations
        ax7 = fig.add_subplot(gs[3, :])
        self.plot_recommendations(ax7)
        
        plt.tight_layout()
        self.save_and_show(fig, "risk_assessment")

    def create_detailed_stock_report(self, symbol):
        """Create detailed individual stock report"""
        if symbol not in self.stocks:
            print(f"❌ No data found for {symbol}")
            return
            
        reports = self.data[symbol]
        latest = reports[-1]['data']
        
        fig = plt.figure(figsize=(24, 18))
        stock_name = latest['stockData'].get('longName', symbol)
        fig.suptitle(f'📈 {symbol} - {stock_name} - DETAILED ANALYSIS', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3,
                     left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # 1. Stock Header Info
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_stock_header(ax1, symbol, latest)
        
        # 2. Signal Score Trend with Events
        ax2 = fig.add_subplot(gs[1, :3])
        self.plot_detailed_score_trend(ax2, reports)
        
        # 3. Current Score Breakdown
        ax3 = fig.add_subplot(gs[1, 3])
        self.plot_score_breakdown(ax3, latest)
        
        # 4. Category Analysis
        ax4 = fig.add_subplot(gs[2, :])
        self.plot_detailed_category_analysis(ax4, reports)
        
        # 5. Triggered Signals Detail
        ax5 = fig.add_subplot(gs[3, :2])
        self.plot_triggered_signals_detail(ax5, latest)
        
        # 6. Technical Indicators
        ax6 = fig.add_subplot(gs[3, 2])
        self.plot_technical_indicators(ax6, latest)
        
        # 7. Key Metrics
        ax7 = fig.add_subplot(gs[3, 3])
        self.plot_detailed_metrics(ax7, latest)
        
        # 8. Action Items & Insights
        ax8 = fig.add_subplot(gs[4, :])
        self.plot_action_items(ax8, symbol, latest)
        
        plt.tight_layout()
        self.save_and_show(fig, f"stock_detail_{symbol}")

    # ===== EXECUTIVE SUMMARY PLOTS =====
    
    def plot_portfolio_health_score(self, ax):
        """Large portfolio health score gauge"""
        ax.set_title('PORTFOLIO HEALTH SCORE', fontsize=16, fontweight='bold', pad=20)
        
        # Calculate overall portfolio score
        total_score = 0
        total_weight = 0
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                score = latest['data']['analysisResults']['scorePercentage']
                # Weight by market cap if available
                market_cap = latest['data']['stockData'].get('marketCap', 1)
                weight = np.log10(max(market_cap, 1))
                total_score += score * weight
                total_weight += weight
                
        portfolio_score = total_score / total_weight if total_weight > 0 else 0
        
        # Create gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = 0.8
        
        # Background arc
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'gray', linewidth=20, alpha=0.3)
        
        # Score arc
        score_theta = np.linspace(0, np.pi * (portfolio_score / 100), 100)
        if portfolio_score >= 80:
            color = self.colors['critical']
        elif portfolio_score >= 60:
            color = self.colors['high']
        elif portfolio_score >= 40:
            color = self.colors['medium']
        else:
            color = self.colors['low']
            
        ax.plot(r * np.cos(score_theta), r * np.sin(score_theta), color, linewidth=20)
        
        # Score text
        ax.text(0, -0.3, f'{portfolio_score:.1f}%', ha='center', va='center', 
                fontsize=36, fontweight='bold', color=color)
        ax.text(0, -0.5, 'Portfolio Score', ha='center', va='center', 
                fontsize=14, alpha=0.8)
        
        # Add score thresholds
        thresholds = [0, 40, 60, 80, 100]
        labels = ['0%', '40%', '60%', '80%', '100%']
        for i, (threshold, label) in enumerate(zip(thresholds, labels)):
            angle = np.pi * (threshold / 100)
            x = (r + 0.1) * np.cos(angle)
            y = (r + 0.1) * np.sin(angle)
            ax.text(x, y, label, ha='center', va='center', fontsize=10, alpha=0.7)
            
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

    def plot_top_performers(self, ax):
        """Show top performing stocks"""
        ax.set_title('🏆 TOP PERFORMERS', fontsize=14, fontweight='bold')
        
        scores = []
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                score = latest['data']['analysisResults']['scorePercentage']
                scores.append((symbol, score))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = scores[:3]
        
        for i, (symbol, score) in enumerate(top_3):
            color = self.colors['critical'] if score >= 80 else \
                   self.colors['high'] if score >= 60 else \
                   self.colors['medium'] if score >= 40 else \
                   self.colors['low']
                   
            # Medal emojis
            medals = ['🥇', '🥈', '🥉']
            ax.text(0.1, 0.8 - i*0.25, medals[i], fontsize=20, va='center')
            ax.text(0.25, 0.8 - i*0.25, symbol, fontsize=14, fontweight='bold', va='center')
            ax.text(0.7, 0.8 - i*0.25, f'{score:.1f}%', fontsize=12, color=color, va='center')
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def plot_alert_summary(self, ax):
        """Alert summary with counts"""
        ax.set_title('🚨 ALERT SUMMARY', fontsize=14, fontweight='bold')
        
        critical_count = 0
        high_count = 0
        triggered_signals = 0
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                score = latest['data']['analysisResults']['scorePercentage']
                triggers = len(latest['data']['analysisResults'].get('triggeredSignals', []))
                
                if score >= 80:
                    critical_count += 1
                elif score >= 60:
                    high_count += 1
                    
                triggered_signals += triggers
                
        alerts = [
            ('🔴 Critical Signals', critical_count, self.colors['critical']),
            ('🟡 High Signals', high_count, self.colors['high']),
            ('⚡ Active Triggers', triggered_signals, self.colors['accent'])
        ]
        
        for i, (label, count, color) in enumerate(alerts):
            y_pos = 0.8 - i * 0.25
            ax.text(0.1, y_pos, label, fontsize=11, va='center')
            ax.text(0.9, y_pos, str(count), fontsize=16, fontweight='bold', 
                   color=color, ha='right', va='center')
                   
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def plot_recent_activity(self, ax):
        """Show recent analysis activity"""
        ax.set_title('📅 RECENT ACTIVITY', fontsize=14, fontweight='bold')
        
        # Get recent activities
        recent_activities = []
        for symbol in self.stocks:
            for report in self.data[symbol]:
                recent_activities.append({
                    'timestamp': report['timestamp'],
                    'symbol': symbol,
                    'score': report['data']['analysisResults']['scorePercentage']
                })
                
        recent_activities.sort(key=lambda x: x['timestamp'], reverse=True)
        recent_activities = recent_activities[:10]  # Last 10 activities
        
        if recent_activities:
            timestamps = [activity['timestamp'] for activity in recent_activities]
            symbols = [activity['symbol'] for activity in recent_activities]
            scores = [activity['score'] for activity in recent_activities]
            
            # Create timeline
            y_positions = list(range(len(recent_activities)))
            colors = [self.get_score_color(score) for score in scores]
            
            ax.scatter(timestamps, y_positions, c=colors, s=100, alpha=0.8)
            
            for i, (ts, symbol, score) in enumerate(zip(timestamps, symbols, scores)):
                ax.text(ts, i, f'  {symbol} ({score:.0f}%)', 
                       va='center', fontsize=9, alpha=0.9)
                       
            ax.set_ylim(-0.5, len(recent_activities) - 0.5)
            ax.set_ylabel('Recent Analyses')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        ax.grid(True, alpha=0.3)

    def plot_signal_distribution(self, ax):
        """Signal strength distribution"""
        ax.set_title('📊 SIGNAL STRENGTH DISTRIBUTION', fontsize=14, fontweight='bold')
        
        scores = []
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                score = latest['data']['analysisResults']['scorePercentage']
                scores.append(score)
                
        if scores:
            # Create histogram with custom bins
            bins = [0, 20, 40, 60, 80, 100]
            counts, _, patches = ax.hist(scores, bins=bins, alpha=0.7, edgecolor='white')
            
            # Color the bars
            colors = [self.colors['low'], self.colors['low'], self.colors['medium'], 
                     self.colors['high'], self.colors['critical']]
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
                
            # Add percentage labels
            total = len(scores)
            for i, count in enumerate(counts):
                if count > 0:
                    percentage = (count / total) * 100
                    ax.text(bins[i] + 10, count + 0.1, f'{percentage:.0f}%', 
                           ha='center', va='bottom', fontweight='bold')
                           
            ax.set_xlabel('Signal Score (%)')
            ax.set_ylabel('Number of Stocks')
            
        ax.grid(True, alpha=0.3, axis='y')

    def plot_market_cap_distribution(self, ax):
        """Market cap distribution pie chart"""
        ax.set_title('💰 MARKET CAP DISTRIBUTION', fontsize=14, fontweight='bold')
        
        nano_cap = 0  # < $50M
        micro_cap = 0  # $50M - $300M
        small_cap = 0  # $300M - $2B
        mid_cap = 0   # $2B - $10B
        large_cap = 0 # > $10B
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                market_cap = latest['data']['stockData'].get('marketCap', 0)
                
                if market_cap < 50e6:
                    nano_cap += 1
                elif market_cap < 300e6:
                    micro_cap += 1
                elif market_cap < 2e9:
                    small_cap += 1
                elif market_cap < 10e9:
                    mid_cap += 1
                else:
                    large_cap += 1
                    
        categories = ['Nano\n(<$50M)', 'Micro\n($50M-$300M)', 'Small\n($300M-$2B)', 
                     'Mid\n($2B-$10B)', 'Large\n(>$10B)']
        values = [nano_cap, micro_cap, small_cap, mid_cap, large_cap]
        colors = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6']
        
        # Only include non-zero categories
        non_zero_data = [(cat, val, col) for cat, val, col in zip(categories, values, colors) if val > 0]
        
        if non_zero_data:
            categories, values, colors = zip(*non_zero_data)
            wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.0f%%',
                                              colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

    def plot_sector_breakdown(self, ax):
        """Sector breakdown pie chart"""
        ax.set_title('🏭 SECTOR BREAKDOWN', fontsize=14, fontweight='bold')
        
        sectors = Counter()
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                sector = latest['data']['stockData'].get('sector', 'Unknown')
                sectors[sector] += 1
                
        if sectors:
            labels = list(sectors.keys())
            values = list(sectors.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.0f%%',
                                              colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

    def plot_key_statistics_table(self, ax):
        """Key statistics table"""
        ax.set_title('📋 KEY PORTFOLIO STATISTICS', fontsize=16, fontweight='bold', pad=20)
        
        # Calculate statistics
        total_stocks = len(self.stocks)
        avg_score = np.mean([self.data[symbol][-1]['data']['analysisResults']['scorePercentage'] 
                            for symbol in self.stocks if self.data[symbol]])
        
        total_market_cap = sum([self.data[symbol][-1]['data']['stockData'].get('marketCap', 0)
                               for symbol in self.stocks if self.data[symbol]])
        
        total_triggered = sum([len(self.data[symbol][-1]['data']['analysisResults'].get('triggeredSignals', []))
                              for symbol in self.stocks if self.data[symbol]])
        
        high_conviction = sum([1 for symbol in self.stocks if self.data[symbol] and 
                              self.data[symbol][-1]['data']['analysisResults']['scorePercentage'] >= 70])
        
        stats = [
            ['Total Stocks Tracked', f'{total_stocks}'],
            ['Average Signal Score', f'{avg_score:.1f}%'],
            ['Total Market Cap', f'${total_market_cap/1e9:.1f}B'],
            ['Active Signal Triggers', f'{total_triggered}'],
            ['High Conviction Picks', f'{high_conviction} ({high_conviction/total_stocks*100:.0f}%)'],
            ['Analysis Period', f'{len(self.analysis_dates)} analyses'],
            ['Last Updated', max(self.analysis_dates).strftime('%Y-%m-%d %H:%M') if self.analysis_dates else 'N/A']
        ]
        
        # Create table
        table_data = []
        for i, (metric, value) in enumerate(stats):
            color = 'lightgray' if i % 2 == 0 else 'white'
            table_data.append([metric, value])
            
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(stats) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#1e40af')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#374151' if i % 2 == 0 else '#4b5563')
                    cell.set_text_props(color='white')
                    
        ax.axis('off')

    # ===== PORTFOLIO OVERVIEW PLOTS =====

    def plot_portfolio_evolution(self, ax):
        """Portfolio evolution over time"""
        ax.set_title('📈 PORTFOLIO SIGNAL EVOLUTION', fontsize=16, fontweight='bold')
        
        # Collect all timeline data
        timeline_data = []
        for symbol in self.stocks:
            for report in self.data[symbol]:
                timeline_data.append({
                    'timestamp': report['timestamp'],
                    'symbol': symbol,
                    'score': report['data']['analysisResults']['scorePercentage'],
                    'triggered_count': len(report['data']['analysisResults'].get('triggeredSignals', []))
                })
                
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            # Plot individual stock trends
            for symbol in self.stocks:
                symbol_data = df[df['symbol'] == symbol]
                if len(symbol_data) > 1:
                    ax.plot(symbol_data['timestamp'], symbol_data['score'], 
                           marker='o', linewidth=2, markersize=6, label=symbol, alpha=0.7)
                           
            # Add portfolio average trend
            portfolio_avg = df.groupby('timestamp')['score'].mean().reset_index()
            ax.plot(portfolio_avg['timestamp'], portfolio_avg['score'], 
                   marker='s', linewidth=4, markersize=8, color='white', 
                   label='Portfolio Average', alpha=0.9)
                   
            # Add score level bands
            ax.axhspan(80, 100, alpha=0.1, color=self.colors['critical'], label='Critical')
            ax.axhspan(60, 80, alpha=0.1, color=self.colors['high'], label='High')
            ax.axhspan(40, 60, alpha=0.1, color=self.colors['medium'], label='Medium')
            ax.axhspan(0, 40, alpha=0.1, color=self.colors['low'], label='Low')
            
            ax.set_ylabel('Signal Score (%)')
            ax.set_ylim(0, 100)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    def plot_comprehensive_heatmap(self, ax):
        """Category performance heatmap across all stocks"""
        ax.set_title('🔥 CATEGORY PERFORMANCE HEATMAP', fontsize=16, fontweight='bold')
        
        # Prepare data for heatmap
        category_data = {}
        
        for symbol in sorted(self.stocks):
            if not self.data[symbol]:
                continue
                
            latest = self.data[symbol][-1]
            categories = latest['data']['analysisResults']['categoryResults']
            
            for cat_key, cat_data in categories.items():
                cat_name = cat_data['name'].replace(' Signals', '').replace(' Intelligence', '')
                score = cat_data['normalizedScore']
                
                if cat_name not in category_data:
                    category_data[cat_name] = {}
                    
                category_data[cat_name][symbol] = score
                
        # Convert to DataFrame
        df = pd.DataFrame(category_data).T
        
        if not df.empty:
            # Create heatmap
            sns.heatmap(df, 
                       annot=True, 
                       fmt='.0f', 
                       cmap='RdYlGn', 
                       center=50,
                       ax=ax,
                       cbar_kws={'label': 'Score (%)'},
                       linewidths=0.5)
                       
            ax.set_xlabel('Stock Symbol')
            ax.set_ylabel('Signal Category')
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', rotation=0)
        else:
            ax.text(0.5, 0.5, 'No category data available', 
                   ha='center', va='center', transform=ax.transAxes)

    def plot_volume_analysis(self, ax):
        """Volume analysis across portfolio"""
        ax.set_title('📊 VOLUME ANALYSIS', fontsize=14, fontweight='bold')
        
        symbols = []
        volume_ratios = []
        float_turnovers = []
        
        for symbol in sorted(self.stocks):
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                
                volume = stock_data.get('volume', 0)
                avg_volume = stock_data.get('averageVolume', 1)
                float_shares = stock_data.get('floatShares', 1)
                
                if avg_volume > 0:
                    volume_ratio = volume / avg_volume
                    float_turnover = volume / float_shares if float_shares > 0 else 0
                    
                    symbols.append(symbol)
                    volume_ratios.append(volume_ratio)
                    float_turnovers.append(float_turnover * 100)  # Convert to percentage
                    
        if symbols:
            x = np.arange(len(symbols))
            width = 0.35
            
            ax.bar(x - width/2, volume_ratios, width, label='Volume Ratio', 
                  color=self.colors['accent'], alpha=0.7)
            ax.bar(x + width/2, float_turnovers, width, label='Float Turnover (%)', 
                  color=self.colors['success'], alpha=0.7)
            
            # Add horizontal line at 1.0 for volume ratio
            ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Stock Symbol')
            ax.set_ylabel('Multiple / Percentage')
            ax.set_xticks(x)
            ax.set_xticklabels(symbols)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    def plot_price_performance(self, ax):
        """Price performance analysis"""
        ax.set_title('💰 PRICE PERFORMANCE', fontsize=14, fontweight='bold')
        
        performance_data = []
        
        for symbol in self.stocks:
            reports = self.data[symbol]
            if len(reports) > 1:
                first_price = reports[0]['data']['stockData'].get('currentPrice') or \
                             reports[0]['data']['stockData'].get('regularMarketPrice', 0)
                latest_price = reports[-1]['data']['stockData'].get('currentPrice') or \
                              reports[-1]['data']['stockData'].get('regularMarketPrice', 0)
                
                if first_price > 0 and latest_price > 0:
                    performance = ((latest_price - first_price) / first_price) * 100
                    latest_score = reports[-1]['data']['analysisResults']['scorePercentage']
                    
                    performance_data.append({
                        'symbol': symbol,
                        'performance': performance,
                        'score': latest_score
                    })
                    
        if performance_data:
            symbols = [d['symbol'] for d in performance_data]
            performances = [d['performance'] for d in performance_data]
            scores = [d['score'] for d in performance_data]
            
            # Color based on performance
            colors = ['green' if p >= 0 else 'red' for p in performances]
            
            bars = ax.bar(symbols, performances, color=colors, alpha=0.7)
            
            # Add horizontal line at 0%
            ax.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            
            # Add value labels
            for bar, perf, score in zip(bars, performances, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (1 if height >= 0 else -3),
                       f'{perf:.1f}%\n(S:{score:.0f})', 
                       ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9)
                       
            ax.set_ylabel('Price Performance (%)')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Insufficient price data\nfor performance analysis', 
                   ha='center', va='center', transform=ax.transAxes)

    def plot_correlation_matrix(self, ax):
        """Correlation matrix of signal categories"""
        ax.set_title('🔗 SIGNAL CORRELATION', fontsize=14, fontweight='bold')
        
        # Collect category scores for correlation analysis
        category_scores = defaultdict(list)
        symbols_list = []
        
        for symbol in sorted(self.stocks):
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                categories = latest['data']['analysisResults']['categoryResults']
                symbols_list.append(symbol)
                
                for cat_key, cat_data in categories.items():
                    cat_name = cat_data['name'].replace(' Signals', '').replace(' Intelligence', '')
                    score = cat_data['normalizedScore']
                    category_scores[cat_name].append(score)
                    
        # Create correlation matrix
        if len(category_scores) > 1:
            df = pd.DataFrame(category_scores)
            correlation_matrix = df.corr()
            
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='RdBu_r', 
                       center=0,
                       ax=ax,
                       cbar_kws={'label': 'Correlation'},
                       linewidths=0.5)
                       
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes)

    def plot_performance_rankings(self, ax):
        """Comprehensive performance rankings"""
        ax.set_title('🏆 PERFORMANCE RANKINGS', fontsize=16, fontweight='bold')
        
        ranking_data = []
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                analysis = latest['data']['analysisResults']
                
                market_cap = stock_data.get('marketCap', 0)
                signal_score = analysis['scorePercentage']
                triggered_count = len(analysis.get('triggeredSignals', []))
                
                # Calculate composite score
                composite_score = signal_score * 0.6 + triggered_count * 10 * 0.3 + \
                                (np.log10(max(market_cap, 1)) / 10) * 0.1
                
                ranking_data.append({
                    'symbol': symbol,
                    'signal_score': signal_score,
                    'triggered_count': triggered_count,
                    'market_cap': market_cap,
                    'composite_score': composite_score
                })
                
        if ranking_data:
            # Sort by composite score
            ranking_data.sort(key=lambda x: x['composite_score'], reverse=True)
            
            symbols = [d['symbol'] for d in ranking_data]
            signal_scores = [d['signal_score'] for d in ranking_data]
            triggered_counts = [d['triggered_count'] for d in ranking_data]
            
            x = np.arange(len(symbols))
            width = 0.35
            
            # Create grouped bar chart
            bars1 = ax.bar(x - width/2, signal_scores, width, label='Signal Score', 
                          color=self.colors['accent'], alpha=0.7)
            bars2 = ax.bar(x + width/2, [t*10 for t in triggered_counts], width, 
                          label='Triggered Signals (x10)', color=self.colors['success'], alpha=0.7)
            
            # Add value labels
            for i, (bar1, bar2, score, count) in enumerate(zip(bars1, bars2, signal_scores, triggered_counts)):
                ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 1,
                       f'{score:.0f}%', ha='center', va='bottom', fontsize=9)
                ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 1,
                       f'{count}', ha='center', va='bottom', fontsize=9)
                       
            ax.set_xlabel('Stock Symbol (Ranked by Composite Score)')
            ax.set_ylabel('Score / Count')
            ax.set_xticks(x)
            ax.set_xticklabels(symbols)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    # ===== SIGNAL ANALYSIS PLOTS =====

    def plot_signal_timeline(self, ax):
        """Signal trigger timeline across all stocks"""
        ax.set_title('⚡ SIGNAL TRIGGER TIMELINE', fontsize=16, fontweight='bold')
        
        timeline_events = []
        
        for symbol in self.stocks:
            for report in self.data[symbol]:
                timestamp = report['timestamp']
                triggered = report['data']['analysisResults'].get('triggeredSignals', [])
                
                for signal in triggered:
                    timeline_events.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'signal': signal['signal'],
                        'category': signal['category'],
                        'score': signal['score']
                    })
                    
        if timeline_events:
            df = pd.DataFrame(timeline_events)
            
            # Create timeline scatter plot
            unique_signals = df['signal'].unique()
            signal_positions = {sig: i for i, sig in enumerate(unique_signals)}
            
            for _, event in df.iterrows():
                y_pos = signal_positions[event['signal']]
                color = self.get_score_color(event['score'])
                
                ax.scatter(event['timestamp'], y_pos, s=100, alpha=0.7, 
                          c=color, edgecolors='white', linewidth=1)
                          
                # Add symbol label
                ax.text(event['timestamp'], y_pos, f" {event['symbol']}", 
                       va='center', fontsize=8, alpha=0.8)
                       
            # Set y-axis labels
            ax.set_yticks(range(len(unique_signals)))
            ax.set_yticklabels([sig.replace(' ', '\n') for sig in unique_signals])
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        else:
            ax.text(0.5, 0.5, 'No triggered signals\nin portfolio', 
                   ha='center', va='center', transform=ax.transAxes)

    def plot_category_deep_dive(self, ax):
        """Deep dive into category performance"""
        ax.set_title('🔍 CATEGORY PERFORMANCE DEEP DIVE', fontsize=14, fontweight='bold')
        
        # Collect category performance data
        category_data = defaultdict(list)
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                categories = latest['data']['analysisResults']['categoryResults']
                
                for cat_key, cat_data in categories.items():
                    cat_name = cat_data['name'].replace(' Signals', '').replace(' Intelligence', '')
                    score = cat_data['normalizedScore']
                    category_data[cat_name].append(score)
                    
        if category_data:
            # Create box plot
            categories = list(category_data.keys())
            scores_list = [category_data[cat] for cat in categories]
            
            bp = ax.boxplot(scores_list, labels=categories, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
            ax.set_ylabel('Category Score (%)')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)

    def plot_signal_strength_evolution(self, ax):
        """Signal strength evolution over time"""
        ax.set_title('📈 SIGNAL STRENGTH EVOLUTION', fontsize=14, fontweight='bold')
        
        # Calculate portfolio-wide signal strength over time
        timeline_data = []
        
        for symbol in self.stocks:
            for report in self.data[symbol]:
                timeline_data.append({
                    'timestamp': report['timestamp'],
                    'score': report['data']['analysisResults']['scorePercentage'],
                    'triggered_count': len(report['data']['analysisResults'].get('triggeredSignals', []))
                })
                
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            # Group by timestamp and calculate metrics
            evolution_data = df.groupby('timestamp').agg({
                'score': ['mean', 'std'],
                'triggered_count': 'sum'
            }).reset_index()
            
            evolution_data.columns = ['timestamp', 'avg_score', 'score_std', 'total_triggered']
            
            # Plot average score with error bars
            ax.errorbar(evolution_data['timestamp'], evolution_data['avg_score'], 
                       yerr=evolution_data['score_std'], marker='o', 
                       linewidth=2, markersize=8, capsize=5, capthick=2,
                       color=self.colors['accent'], label='Avg Signal Score')
                       
            # Add triggered signals as secondary axis
            ax2 = ax.twinx()
            ax2.bar(evolution_data['timestamp'], evolution_data['total_triggered'], 
                   alpha=0.3, color=self.colors['warning'], width=0.02,
                   label='Total Triggered Signals')
            ax2.set_ylabel('Total Triggered Signals', color=self.colors['warning'])
            
            ax.set_ylabel('Average Signal Score (%)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def plot_triggered_signals_analysis(self, ax):
        """Analysis of triggered signals patterns"""
        ax.set_title('🎯 TRIGGERED SIGNALS ANALYSIS', fontsize=16, fontweight='bold')
        
        # Count signal triggers by category and signal
        signal_counts = Counter()
        category_counts = Counter()
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                triggered = latest['data']['analysisResults'].get('triggeredSignals', [])
                
                for signal in triggered:
                    signal_counts[signal['signal']] += 1
                    category_counts[signal['category']] += 1
                    
        if signal_counts:
            # Get top 10 most triggered signals
            top_signals = signal_counts.most_common(10)
            signals, counts = zip(*top_signals)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(signals))
            bars = ax.barh(y_pos, counts, color=self.colors['success'], alpha=0.7)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                       str(count), ha='left', va='center', fontweight='bold')
                       
            ax.set_yticks(y_pos)
            ax.set_yticklabels([textwrap.fill(sig, 20) for sig in signals])
            ax.set_xlabel('Number of Stocks with Signal')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No triggered signals\nto analyze', 
                   ha='center', va='center', transform=ax.transAxes)

    def plot_signal_effectiveness(self, ax):
        """Signal effectiveness metrics"""
        ax.set_title('⚡ SIGNAL EFFECTIVENESS', fontsize=14, fontweight='bold')
        
        # Calculate signal effectiveness by comparing signal strength to subsequent performance
        effectiveness_data = []
        
        for symbol in self.stocks:
            reports = self.data[symbol]
            for i in range(len(reports) - 1):
                current = reports[i]
                next_report = reports[i + 1]
                
                current_score = current['data']['analysisResults']['scorePercentage']
                next_score = next_report['data']['analysisResults']['scorePercentage']
                score_change = next_score - current_score
                
                effectiveness_data.append({
                    'current_score': current_score,
                    'score_change': score_change
                })
                
        if effectiveness_data:
            df = pd.DataFrame(effectiveness_data)
            
            # Create scatter plot
            ax.scatter(df['current_score'], df['score_change'], 
                      alpha=0.6, s=50, color=self.colors['accent'])
                      
            # Add trend line
            z = np.polyfit(df['current_score'], df['score_change'], 1)
            p = np.poly1d(z)
            ax.plot(df['current_score'], p(df['current_score']), 
                   color=self.colors['warning'], linewidth=2, linestyle='--')
                   
            ax.set_xlabel('Current Signal Score (%)')
            ax.set_ylabel('Score Change to Next Analysis')
            ax.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            correlation = df['current_score'].corr(df['score_change'])
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor effectiveness analysis', 
                   ha='center', va='center', transform=ax.transAxes)

    def plot_missing_data_analysis(self, ax):
        """Analysis of missing data across portfolio"""
        ax.set_title('📋 MISSING DATA ANALYSIS', fontsize=14, fontweight='bold')
        
        # Count missing fields across all stocks
        missing_fields = Counter()
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                missing = latest['data']['analysisResults'].get('missingDataFields', [])
                for field in missing:
                    missing_fields[field] += 1
                    
        if missing_fields:
            fields = list(missing_fields.keys())
            counts = list(missing_fields.values())
            
            # Create horizontal bar chart
            y_pos = np.arange(len(fields))
            bars = ax.barh(y_pos, counts, color=self.colors['warning'], alpha=0.7)
            
            # Add percentage labels
            total_stocks = len(self.stocks)
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                percentage = (count / total_stocks) * 100
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                       f'{count} ({percentage:.0f}%)', ha='left', va='center')
                       
            ax.set_yticks(y_pos)
            ax.set_yticklabels(fields)
            ax.set_xlabel('Number of Stocks Missing Field')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, '✅ No missing data fields\nacross portfolio', 
                   ha='center', va='center', transform=ax.transAxes, color=self.colors['success'])

    def plot_signal_quality_metrics(self, ax):
        """Signal quality metrics"""
        ax.set_title('🎯 SIGNAL QUALITY METRICS', fontsize=14, fontweight='bold')
        
        # Calculate various quality metrics
        quality_metrics = {
            'High Conviction Signals': 0,  # Score >= 80
            'Medium Conviction': 0,        # Score 60-80
            'Low Conviction': 0,           # Score 40-60
            'Weak Signals': 0              # Score < 40
        }
        
        total_signals = 0
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                score = latest['data']['analysisResults']['scorePercentage']
                total_signals += 1
                
                if score >= 80:
                    quality_metrics['High Conviction Signals'] += 1
                elif score >= 60:
                    quality_metrics['Medium Conviction'] += 1
                elif score >= 40:
                    quality_metrics['Low Conviction'] += 1
                else:
                    quality_metrics['Weak Signals'] += 1
                    
        if total_signals > 0:
            # Create pie chart
            labels = list(quality_metrics.keys())
            values = list(quality_metrics.values())
            colors = [self.colors['critical'], self.colors['high'], 
                     self.colors['medium'], self.colors['low']]
            
            # Only include non-zero categories
            non_zero_data = [(label, value, color) for label, value, color in zip(labels, values, colors) if value > 0]
            
            if non_zero_data:
                labels, values, colors = zip(*non_zero_data)
                wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.0f%%',
                                                  colors=colors, startangle=90)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

    # ===== RISK ASSESSMENT PLOTS =====

    def plot_risk_return_analysis(self, ax):
        """Risk-return scatter plot"""
        ax.set_title('⚖️ RISK-RETURN ANALYSIS', fontsize=16, fontweight='bold')
        
        risk_return_data = []
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                analysis = latest['data']['analysisResults']
                
                # Risk indicators
                beta = stock_data.get('beta', 1.0)
                market_cap = stock_data.get('marketCap', 0)
                short_interest = stock_data.get('shortPercentOfFloat', 0)
                
                # Return potential (signal score)
                signal_score = analysis['scorePercentage']
                
                # Calculate composite risk score
                risk_score = beta * 30 + (1 / np.log10(max(market_cap, 1)) * 10) + short_interest * 100
                
                risk_return_data.append({
                    'symbol': symbol,
                    'risk': risk_score,
                    'return_potential': signal_score,
                    'market_cap': market_cap
                })
                
        if risk_return_data:
            symbols = [d['symbol'] for d in risk_return_data]
            risks = [d['risk'] for d in risk_return_data]
            returns = [d['return_potential'] for d in risk_return_data]
            market_caps = [d['market_cap'] for d in risk_return_data]
            
            # Create scatter plot with size based on market cap
            sizes = [max(50, min(500, np.log10(max(mc, 1)) * 20)) for mc in market_caps]
            
            scatter = ax.scatter(risks, returns, s=sizes, alpha=0.6, 
                               c=returns, cmap='RdYlGn', vmin=0, vmax=100)
                               
            # Add labels
            for i, symbol in enumerate(symbols):
                ax.annotate(symbol, (risks[i], returns[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
                          
            # Add quadrant lines
            ax.axhline(y=60, color='white', linestyle='--', alpha=0.5)
            ax.axvline(x=np.median(risks), color='white', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax.text(0.05, 0.95, 'Low Risk\nHigh Return', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor=self.colors['high'], alpha=0.8))
            ax.text(0.75, 0.95, 'High Risk\nHigh Return', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor=self.colors['warning'], alpha=0.8))
            ax.text(0.05, 0.05, 'Low Risk\nLow Return', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor=self.colors['neutral'], alpha=0.8))
            ax.text(0.75, 0.05, 'High Risk\nLow Return', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor=self.colors['low'], alpha=0.8))
                   
            ax.set_xlabel('Risk Score (Higher = More Risky)')
            ax.set_ylabel('Return Potential (Signal Score %)')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Signal Score (%)')

    def plot_risk_factors(self, ax):
        """Risk factors summary"""
        ax.set_title('⚠️ RISK FACTORS', fontsize=14, fontweight='bold')
        
        risk_counts = {
            'High Beta (>2.0)': 0,
            'Low Market Cap (<$50M)': 0,
            'High Short Interest (>20%)': 0,
            'Poor Liquidity': 0,
            'Missing Data': 0
        }
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                analysis = latest['data']['analysisResults']
                
                beta = stock_data.get('beta', 1.0)
                market_cap = stock_data.get('marketCap', 0)
                short_interest = stock_data.get('shortPercentOfFloat', 0)
                volume = stock_data.get('volume', 0)
                avg_volume = stock_data.get('averageVolume', 1)
                missing_fields = analysis.get('missingDataFields', [])
                
                if beta > 2.0:
                    risk_counts['High Beta (>2.0)'] += 1
                if market_cap < 50e6:
                    risk_counts['Low Market Cap (<$50M)'] += 1
                if short_interest > 0.20:
                    risk_counts['High Short Interest (>20%)'] += 1
                if volume < avg_volume * 0.5:
                    risk_counts['Poor Liquidity'] += 1
                if missing_fields:
                    risk_counts['Missing Data'] += 1
                    
        # Create horizontal bar chart
        factors = list(risk_counts.keys())
        counts = list(risk_counts.values())
        
        y_pos = np.arange(len(factors))
        bars = ax.barh(y_pos, counts, color=self.colors['warning'], alpha=0.7)
        
        # Add percentage labels
        total_stocks = len(self.stocks)
        for bar, count in zip(bars, counts):
            if count > 0:
                width = bar.get_width()
                percentage = (count / total_stocks) * 100
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                       f'{count} ({percentage:.0f}%)', ha='left', va='center')
                       
        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors)
        ax.set_xlabel('Number of Stocks')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    def plot_market_cap_vs_signals(self, ax):
        """Market cap vs signal strength analysis"""
        ax.set_title('💰 MARKET CAP vs SIGNALS', fontsize=14, fontweight='bold')
        
        market_caps = []
        signal_scores = []
        symbols = []
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                market_cap = latest['data']['stockData'].get('marketCap', 0)
                score = latest['data']['analysisResults']['scorePercentage']
                
                if market_cap > 0:
                    market_caps.append(market_cap)
                    signal_scores.append(score)
                    symbols.append(symbol)
                    
        if market_caps:
            # Create scatter plot
            scatter = ax.scatter(market_caps, signal_scores, s=100, alpha=0.6, 
                               c=signal_scores, cmap='RdYlGn', vmin=0, vmax=100)
                               
            # Add labels
            for i, symbol in enumerate(symbols):
                ax.annotate(symbol, (market_caps[i], signal_scores[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
                          
            ax.set_xlabel('Market Cap ($)')
            ax.set_ylabel('Signal Score (%)')
            ax.set_xscale('log')
            
            # Add market cap category lines
            ax.axvline(x=50e6, color='red', linestyle='--', alpha=0.5, label='$50M')
            ax.axvline(x=300e6, color='orange', linestyle='--', alpha=0.5, label='$300M')
            ax.axvline(x=2e9, color='green', linestyle='--', alpha=0.5, label='$2B')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Signal Score (%)')

    def plot_volatility_analysis(self, ax):
        """Volatility analysis across portfolio"""
        ax.set_title('📈 VOLATILITY ANALYSIS', fontsize=14, fontweight='bold')
        
        volatility_data = []
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                
                beta = stock_data.get('beta', 1.0)
                fifty_two_week_low = stock_data.get('fiftyTwoWeekLow', 0)
                fifty_two_week_high = stock_data.get('fiftyTwoWeekHigh', 0)
                current_price = stock_data.get('currentPrice') or stock_data.get('regularMarketPrice', 0)
                
                if fifty_two_week_low > 0 and fifty_two_week_high > 0:
                    price_range = ((fifty_two_week_high - fifty_two_week_low) / fifty_two_week_low) * 100
                    volatility_data.append({
                        'symbol': symbol,
                        'beta': beta,
                        'price_range': price_range
                    })
                    
        if volatility_data:
            symbols = [d['symbol'] for d in volatility_data]
            betas = [d['beta'] for d in volatility_data]
            price_ranges = [d['price_range'] for d in volatility_data]
            
            x = np.arange(len(symbols))
            width = 0.35
            
            ax.bar(x - width/2, betas, width, label='Beta', 
                  color=self.colors['accent'], alpha=0.7)
            ax.bar(x + width/2, [pr/100 for pr in price_ranges], width, 
                  label='52W Range (scaled)', color=self.colors['warning'], alpha=0.7)
            
            ax.set_xlabel('Stock Symbol')
            ax.set_ylabel('Volatility Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(symbols)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at beta = 1.0
            ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)

    def plot_liquidity_assessment(self, ax):
        """Liquidity assessment"""
        ax.set_title('💧 LIQUIDITY ASSESSMENT', fontsize=14, fontweight='bold')
        
        liquidity_categories = {
            'High Liquidity': 0,      # Volume > 2x average
            'Medium Liquidity': 0,    # Volume 0.5x - 2x average
            'Low Liquidity': 0        # Volume < 0.5x average
        }
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                
                volume = stock_data.get('volume', 0)
                avg_volume = stock_data.get('averageVolume', 1)
                
                if avg_volume > 0:
                    volume_ratio = volume / avg_volume
                    
                    if volume_ratio > 2.0:
                        liquidity_categories['High Liquidity'] += 1
                    elif volume_ratio > 0.5:
                        liquidity_categories['Medium Liquidity'] += 1
                    else:
                        liquidity_categories['Low Liquidity'] += 1
                        
        # Create pie chart
        labels = list(liquidity_categories.keys())
        values = list(liquidity_categories.values())
        colors = [self.colors['high'], self.colors['medium'], self.colors['low']]
        
        # Only include non-zero categories
        non_zero_data = [(label, value, color) for label, value, color in zip(labels, values, colors) if value > 0]
        
        if non_zero_data:
            labels, values, colors = zip(*non_zero_data)
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.0f%%',
                                              colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

    def plot_opportunity_matrix(self, ax):
        """Opportunity matrix visualization"""
        ax.set_title('🎯 OPPORTUNITY MATRIX', fontsize=16, fontweight='bold')
        
        matrix_data = []
        
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                analysis = latest['data']['analysisResults']
                
                signal_score = analysis['scorePercentage']
                triggered_count = len(analysis.get('triggeredSignals', []))
                market_cap = stock_data.get('marketCap', 0)
                
                # Calculate opportunity score (combination of signals and size)
                opportunity_score = signal_score + (triggered_count * 10)
                
                # Calculate execution difficulty (based on liquidity and size)
                volume = stock_data.get('volume', 0)
                avg_volume = stock_data.get('averageVolume', 1)
                liquidity_ratio = volume / avg_volume if avg_volume > 0 else 0
                
                execution_difficulty = 100 - (liquidity_ratio * 20 + np.log10(max(market_cap, 1)) * 5)
                execution_difficulty = max(0, min(100, execution_difficulty))
                
                matrix_data.append({
                    'symbol': symbol,
                    'opportunity': opportunity_score,
                    'difficulty': execution_difficulty,
                    'signal_score': signal_score
                })
                
        if matrix_data:
            symbols = [d['symbol'] for d in matrix_data]
            opportunities = [d['opportunity'] for d in matrix_data]
            difficulties = [d['difficulty'] for d in matrix_data]
            signal_scores = [d['signal_score'] for d in matrix_data]
            
            # Create scatter plot
            scatter = ax.scatter(difficulties, opportunities, s=150, alpha=0.7,
                               c=signal_scores, cmap='RdYlGn', vmin=0, vmax=100)
                               
            # Add labels
            for i, symbol in enumerate(symbols):
                ax.annotate(symbol, (difficulties[i], opportunities[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
                          
            # Add quadrant lines
            ax.axhline(y=np.median(opportunities), color='white', linestyle='--', alpha=0.5)
            ax.axvline(x=np.median(difficulties), color='white', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax.text(0.05, 0.95, 'Easy Win\n(Low Difficulty, High Opportunity)', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=self.colors['high'], alpha=0.8))
            ax.text(0.55, 0.95, 'High Risk/Reward\n(High Difficulty, High Opportunity)', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=self.colors['warning'], alpha=0.8))
            ax.text(0.05, 0.05, 'Safe Plays\n(Low Difficulty, Low Opportunity)', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=self.colors['neutral'], alpha=0.8))
            ax.text(0.55, 0.05, 'Avoid\n(High Difficulty, Low Opportunity)', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=self.colors['low'], alpha=0.8))
                   
            ax.set_xlabel('Execution Difficulty')
            ax.set_ylabel('Opportunity Score')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Signal Score (%)')

    def plot_recommendations(self, ax):
        """Portfolio recommendations"""
        ax.set_title('💡 PORTFOLIO RECOMMENDATIONS', fontsize=16, fontweight='bold')
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # High conviction picks
        high_conviction = []
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                score = latest['data']['analysisResults']['scorePercentage']
                if score >= 75:
                    high_conviction.append(symbol)
                    
        if high_conviction:
            recommendations.append(f"🎯 HIGH CONVICTION: {', '.join(high_conviction[:3])}")
            
        # Risk warnings
        high_risk = []
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                beta = stock_data.get('beta', 1.0)
                market_cap = stock_data.get('marketCap', 0)
                
                if beta > 2.5 or market_cap < 20e6:
                    high_risk.append(symbol)
                    
        if high_risk:
            recommendations.append(f"⚠️ HIGH RISK: Monitor {', '.join(high_risk[:3])} closely")
            
        # Portfolio balance
        total_market_cap = sum([self.data[symbol][-1]['data']['stockData'].get('marketCap', 0)
                               for symbol in self.stocks if self.data[symbol]])
        if total_market_cap < 500e6:
            recommendations.append("💼 PORTFOLIO: Consider adding larger cap stocks for stability")
            
        # Data quality
        missing_data_stocks = []
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                missing = latest['data']['analysisResults'].get('missingDataFields', [])
                if len(missing) > 2:
                    missing_data_stocks.append(symbol)
                    
        if missing_data_stocks:
            recommendations.append(f"📊 DATA: Improve data quality for {', '.join(missing_data_stocks[:3])}")
            
        # Signal diversification
        triggered_by_category = defaultdict(int)
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                triggered = latest['data']['analysisResults'].get('triggeredSignals', [])
                for signal in triggered:
                    triggered_by_category[signal['category']] += 1
                    
        if len(triggered_by_category) < 3:
            recommendations.append("🔄 DIVERSIFICATION: Seek signals across more categories")
            
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations[:8]):  # Show max 8 recommendations
                y_pos = 0.9 - i * 0.11
                
                # Color coding
                if rec.startswith('🎯'):
                    color = self.colors['high']
                elif rec.startswith('⚠️'):
                    color = self.colors['warning']
                elif rec.startswith('💼'):
                    color = self.colors['accent']
                elif rec.startswith('📊'):
                    color = self.colors['info']
                else:
                    color = self.colors['text']
                    
                ax.text(0.05, y_pos, rec, fontsize=11, color=color, 
                       transform=ax.transAxes, wrap=True, va='top')
                       
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # ===== DETAILED STOCK REPORT PLOTS =====

    def plot_stock_header(self, ax, symbol, latest_data):
        """Stock header with key information"""
        stock_data = latest_data['stockData']
        analysis = latest_data['analysisResults']
        
        # Company information
        company_name = stock_data.get('longName', symbol)
        sector = stock_data.get('sector', 'Unknown')
        industry = stock_data.get('industry', 'Unknown')
        current_price = stock_data.get('currentPrice') or stock_data.get('regularMarketPrice', 0)
        market_cap = stock_data.get('marketCap', 0)
        signal_score = analysis['scorePercentage']
        
        # Create header layout
        header_text = f"""
        {company_name} ({symbol})
        Sector: {sector} | Industry: {industry}
        Current Price: ${current_price:.2f} | Market Cap: ${market_cap/1e6:.1f}M
        Signal Score: {signal_score:.1f}% | Triggered Signals: {len(analysis.get('triggeredSignals', []))}
        """
        
        ax.text(0.5, 0.5, header_text.strip(), ha='center', va='center', 
               fontsize=14, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=1', facecolor=self.colors['accent'], alpha=0.8))
        ax.axis('off')

    def plot_detailed_score_trend(self, ax, reports):
        """Detailed score trend with annotations"""
        ax.set_title('📈 SIGNAL SCORE EVOLUTION', fontsize=14, fontweight='bold')
        
        timestamps = []
        scores = []
        triggered_counts = []
        
        for report in reports:
            timestamps.append(report['timestamp'])
            scores.append(report['data']['analysisResults']['scorePercentage'])
            triggered_counts.append(len(report['data']['analysisResults'].get('triggeredSignals', [])))
            
        # Main score line
        ax.plot(timestamps, scores, marker='o', linewidth=3, markersize=10, 
               color=self.colors['accent'], label='Signal Score')
        
        # Add triggered signals as bars
        ax2 = ax.twinx()
        ax2.bar(timestamps, triggered_counts, alpha=0.3, color=self.colors['success'], 
               width=0.02, label='Triggered Signals')
        ax2.set_ylabel('Triggered Signals Count', color=self.colors['success'])
        
        # Add score level bands
        ax.axhspan(80, 100, alpha=0.1, color=self.colors['critical'])
        ax.axhspan(60, 80, alpha=0.1, color=self.colors['high'])
        ax.axhspan(40, 60, alpha=0.1, color=self.colors['medium'])
        ax.axhspan(0, 40, alpha=0.1, color=self.colors['low'])
        
        # Add score annotations
        for i, (ts, score, count) in enumerate(zip(timestamps, scores, triggered_counts)):
            if i == 0 or i == len(timestamps) - 1:  # First and last points
                ax.annotate(f'{score:.1f}%', xy=(ts, score), xytext=(0, 15),
                           textcoords='offset points', ha='center', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                           
        ax.set_ylabel('Signal Score (%)')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    def plot_score_breakdown(self, ax, latest_data):
        """Current score breakdown by category"""
        ax.set_title('🎯 SCORE BREAKDOWN', fontsize=14, fontweight='bold')
        
        categories = latest_data['analysisResults']['categoryResults']
        
        names = []
        scores = []
        weights = []
        
        for cat_data in categories.values():
            name = cat_data['name'].replace(' Signals', '').replace(' Intelligence', '')
            names.append(name)
            scores.append(cat_data['normalizedScore'])
            weights.append(cat_data['weight'])
            
        if names:
            # Create weighted donut chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            
            # Outer ring: scores
            wedges1, texts1 = ax.pie(scores, labels=names, colors=colors, radius=1.0,
                                    wedgeprops=dict(width=0.3))
            
            # Inner ring: weights
            wedges2, texts2 = ax.pie(weights, colors=colors, radius=0.7,
                                    wedgeprops=dict(width=0.3))
                                    
            # Add center text with overall score
            overall_score = latest_data['analysisResults']['scorePercentage']
            ax.text(0, 0, f'{overall_score:.1f}%', ha='center', va='center', 
                   fontsize=20, fontweight='bold')
            ax.text(0, -0.15, 'Overall Score', ha='center', va='center', fontsize=10)

    def plot_detailed_category_analysis(self, ax, reports):
        """Detailed category analysis over time"""
        ax.set_title('📊 CATEGORY EVOLUTION', fontsize=16, fontweight='bold')
        
        # Extract category data over time
        category_data = defaultdict(lambda: {'timestamps': [], 'scores': []})
        
        for report in reports:
            timestamp = report['timestamp']
            categories = report['data']['analysisResults']['categoryResults']
            
            for cat_key, cat_data in categories.items():
                cat_name = cat_data['name'].replace(' Signals', '').replace(' Intelligence', '')
                score = cat_data['normalizedScore']
                
                category_data[cat_name]['timestamps'].append(timestamp)
                category_data[cat_name]['scores'].append(score)
                
        # Plot each category
        colors = plt.cm.tab10(np.linspace(0, 1, len(category_data)))
        for i, (cat_name, data) in enumerate(category_data.items()):
            ax.plot(data['timestamps'], data['scores'], 
                   marker='o', linewidth=2, markersize=6, 
                   label=cat_name, color=colors[i])
                   
        ax.set_ylabel('Category Score (%)')
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    def plot_triggered_signals_detail(self, ax, latest_data):
        """Detailed triggered signals breakdown"""
        ax.set_title('⚡ TRIGGERED SIGNALS DETAIL', fontsize=14, fontweight='bold')
        
        triggered = latest_data['analysisResults'].get('triggeredSignals', [])
        
        if triggered:
            # Sort by score
            triggered.sort(key=lambda x: x['score'], reverse=True)
            
            signals = [signal['signal'] for signal in triggered]
            scores = [signal['score'] for signal in triggered]
            categories = [signal['category'] for signal in triggered]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(signals))
            bars = ax.barh(y_pos, scores, color=self.colors['success'], alpha=0.7)
            
            # Add score labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2., 
                       f'{score}', ha='left', va='center', fontweight='bold')
                       
            # Customize labels
            wrapped_signals = [textwrap.fill(sig, 25) for sig in signals]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(wrapped_signals)
            ax.set_xlabel('Signal Score')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            
            # Add category colors
            unique_categories = list(set(categories))
            category_colors = {cat: plt.cm.tab10(i) for i, cat in enumerate(unique_categories)}
            
            for i, (bar, category) in enumerate(zip(bars, categories)):
                bar.set_color(category_colors[category])
                
            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, color=category_colors[cat], 
                                           label=cat.replace(' Signals', '')) 
                             for cat in unique_categories]
            ax.legend(handles=legend_elements, loc='lower right')
        else:
            ax.text(0.5, 0.5, 'No triggered signals', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16, alpha=0.6)
            ax.axis('off')

    def plot_technical_indicators(self, ax, latest_data):
        """Technical indicators summary"""
        ax.set_title('📈 TECHNICAL INDICATORS', fontsize=14, fontweight='bold')
        
        stock_data = latest_data['stockData']
        
        # Extract technical data
        current_price = stock_data.get('currentPrice') or stock_data.get('regularMarketPrice', 0)
        fifty_day_avg = stock_data.get('fiftyDayAverage', 0)
        two_hundred_day_avg = stock_data.get('twoHundredDayAverage', 0)
        fifty_two_week_low = stock_data.get('fiftyTwoWeekLow', 0)
        fifty_two_week_high = stock_data.get('fiftyTwoWeekHigh', 0)
        
        # Calculate relative positions
        indicators = []
        if fifty_day_avg > 0:
            fifty_day_position = ((current_price - fifty_day_avg) / fifty_day_avg) * 100
            indicators.append(('50-Day MA', fifty_day_position))
            
        if two_hundred_day_avg > 0:
            two_hundred_day_position = ((current_price - two_hundred_day_avg) / two_hundred_day_avg) * 100
            indicators.append(('200-Day MA', two_hundred_day_position))
            
        if fifty_two_week_low > 0 and fifty_two_week_high > 0:
            week_52_position = ((current_price - fifty_two_week_low) / 
                               (fifty_two_week_high - fifty_two_week_low)) * 100
            indicators.append(('52W Range', week_52_position))
            
        if indicators:
            names = [ind[0] for ind in indicators]
            values = [ind[1] for ind in indicators]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(names))
            colors = ['green' if v > 0 else 'red' for v in values]
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(width + (1 if width >= 0 else -1), bar.get_y() + bar.get_height()/2., 
                       f'{value:.1f}%', ha='left' if width >= 0 else 'right', va='center')
                       
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel('Relative Position (%)')
            ax.axvline(x=0, color='white', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor technical analysis', 
                   ha='center', va='center', transform=ax.transAxes)

    def plot_detailed_metrics(self, ax, latest_data):
        """Detailed key metrics"""
        ax.set_title('📊 KEY METRICS', fontsize=14, fontweight='bold')
        
        stock_data = latest_data['stockData']
        
        metrics = [
            ('Market Cap', f"${stock_data.get('marketCap', 0) / 1e6:.1f}M"),
            ('Float Shares', f"{stock_data.get('floatShares', 0) / 1e6:.1f}M"),
            ('Short %', f"{stock_data.get('shortPercentOfFloat', 0) * 100:.1f}%"),
            ('Beta', f"{stock_data.get('beta', 0):.2f}"),
            ('P/E Ratio', f"{stock_data.get('trailingPE', 0):.1f}"),
            ('Volume Ratio', f"{stock_data.get('volume', 0) / max(stock_data.get('averageVolume', 1), 1):.1f}x"),
            ('Revenue Growth', f"{stock_data.get('revenueGrowth', 0) * 100:.1f}%" if stock_data.get('revenueGrowth') else "N/A")
        ]
        
        # Create formatted table
        for i, (metric, value) in enumerate(metrics):
            y_pos = 0.9 - i * 0.12
            
            # Color coding for certain metrics
            color = self.colors['text']
            if 'Growth' in metric and '%' in value and float(value.replace('%', '')) > 50:
                color = self.colors['success']
            elif 'Short %' in metric and '%' in value and float(value.replace('%', '')) > 20:
                color = self.colors['warning']
            elif 'Beta' in metric and value != 'N/A' and float(value) > 2:
                color = self.colors['warning']
                
            ax.text(0.1, y_pos, metric + ':', fontsize=11, fontweight='bold', 
                   transform=ax.transAxes, va='center')
            ax.text(0.7, y_pos, value, fontsize=11, color=color,
                   transform=ax.transAxes, va='center', ha='right')
                   
        ax.axis('off')

    def plot_action_items(self, ax, symbol, latest_data):
        """Action items and insights"""
        ax.set_title(f'💡 ACTION ITEMS & INSIGHTS - {symbol}', fontsize=16, fontweight='bold')
        
        stock_data = latest_data['stockData']
        analysis = latest_data['analysisResults']
        
        # Generate specific action items
        action_items = []
        
        # Signal-based actions
        score = analysis['scorePercentage']
        if score >= 80:
            action_items.append("🚀 STRONG BUY SIGNAL: Consider increasing position size")
        elif score >= 60:
            action_items.append("📈 BULLISH: Good entry opportunity, watch for confirmation")
        elif score < 40:
            action_items.append("⚠️ WEAK SIGNALS: Monitor closely, consider risk management")
            
        # Volume-based actions
        volume = stock_data.get('volume', 0)
        avg_volume = stock_data.get('averageVolume', 1)
        if volume > avg_volume * 3:
            action_items.append("🔥 HIGH VOLUME: Unusual activity detected, investigate news")
        elif volume < avg_volume * 0.5:
            action_items.append("💤 LOW VOLUME: Wait for volume confirmation before entry")
            
        # Risk-based actions
        market_cap = stock_data.get('marketCap', 0)
        beta = stock_data.get('beta', 1.0)
        
        if market_cap < 50e6:
            action_items.append("⚠️ MICRO-CAP: Use smaller position sizes due to volatility")
        if beta > 2.5:
            action_items.append("🎢 HIGH BETA: Expect significant price swings")
            
        # Technical actions
        short_interest = stock_data.get('shortPercentOfFloat', 0)
        if short_interest > 0.25:
            action_items.append("🔥 SHORT SQUEEZE: Monitor for covering activity")
            
        # Data quality actions
        missing_fields = analysis.get('missingDataFields', [])
        if missing_fields:
            action_items.append(f"📊 DATA: Missing {len(missing_fields)} fields, verify analysis accuracy")
            
        # Triggered signals actions
        triggered = analysis.get('triggeredSignals', [])
        if len(triggered) >= 3:
            action_items.append("⚡ MULTIPLE SIGNALS: Strong confirmation across categories")
        elif len(triggered) == 0:
            action_items.append("🔍 NO TRIGGERS: Wait for signal confirmation before action")
            
        # Display action items
        if action_items:
            for i, item in enumerate(action_items[:8]):  # Show max 8 items
                y_pos = 0.9 - i * 0.1
                
                # Color coding
                if item.startswith('🚀') or item.startswith('📈'):
                    color = self.colors['success']
                elif item.startswith('⚠️'):
                    color = self.colors['warning']
                elif item.startswith('🔥'):
                    color = self.colors['critical']
                else:
                    color = self.colors['text']
                    
                ax.text(0.05, y_pos, item, fontsize=11, color=color, 
                       transform=ax.transAxes, wrap=True, va='top')
                       
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # ===== UTILITY METHODS =====

    def get_score_color(self, score):
        """Get color based on score"""
        if score >= 80:
            return self.colors['critical']
        elif score >= 60:
            return self.colors['high']
        elif score >= 40:
            return self.colors['medium']
        else:
            return self.colors['low']

    def save_and_show(self, fig, name):
        """Save and show figure"""
        # Create output directory
        output_dir = Path("portfolio_reports")
        output_dir.mkdir(exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{name}_{timestamp}.png"
        
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        print(f"💾 Saved: {filename}")
        
        plt.show()

    def generate_summary_report(self):
        """Generate comprehensive text summary"""
        print("\n" + "="*80)
        print("📈 ENHANCED STOCK SIGNALS PORTFOLIO ANALYSIS")
        print("="*80)
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Stocks Analyzed: {len(self.stocks)}")
        print(f"📋 Total Reports: {sum(len(reports) for reports in self.data.values())}")
        
        if self.analysis_dates:
            print(f"📅 Analysis Period: {min(self.analysis_dates).strftime('%Y-%m-%d')} to {max(self.analysis_dates).strftime('%Y-%m-%d')}")
        
        # Portfolio health metrics
        avg_score = np.mean([self.data[symbol][-1]['data']['analysisResults']['scorePercentage'] 
                            for symbol in self.stocks if self.data[symbol]])
        total_market_cap = sum([self.data[symbol][-1]['data']['stockData'].get('marketCap', 0)
                               for symbol in self.stocks if self.data[symbol]])
        
        print(f"\n📊 PORTFOLIO HEALTH METRICS")
        print(f"   Average Signal Score: {avg_score:.1f}%")
        print(f"   Total Market Cap: ${total_market_cap/1e9:.1f}B")
        
        # Top performers
        scores = [(symbol, self.data[symbol][-1]['data']['analysisResults']['scorePercentage']) 
                 for symbol in self.stocks if self.data[symbol]]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP PERFORMERS")
        for i, (symbol, score) in enumerate(scores[:5]):
            print(f"   {i+1}. {symbol}: {score:.1f}%")
            
        # Risk analysis
        high_risk_stocks = []
        for symbol in self.stocks:
            if self.data[symbol]:
                latest = self.data[symbol][-1]
                stock_data = latest['data']['stockData']
                beta = stock_data.get('beta', 1.0)
                market_cap = stock_data.get('marketCap', 0)
                
                if beta > 2.5 or market_cap < 25e6:
                    high_risk_stocks.append(symbol)
                    
        if high_risk_stocks:
            print(f"\n⚠️ HIGH RISK STOCKS: {', '.join(high_risk_stocks)}")
            
        # Individual stock summaries
        print(f"\n📋 INDIVIDUAL STOCK ANALYSIS")
        print("-" * 80)
        
        for symbol in sorted(self.stocks):
            reports = self.data[symbol]
            if not reports:
                continue
                
            latest = reports[-1]
            stock_data = latest['data']['stockData']
            analysis = latest['data']['analysisResults']
            
            print(f"\n{symbol} - {stock_data.get('longName', 'Unknown Company')}")
            print(f"   Signal Score: {analysis['scorePercentage']:.1f}%")
            print(f"   Market Cap: ${stock_data.get('marketCap', 0)/1e6:.1f}M")
            print(f"   Triggered Signals: {len(analysis.get('triggeredSignals', []))}")
            print(f"   Sector: {stock_data.get('sector', 'Unknown')}")
            
            # Show top triggered signals
            triggered = analysis.get('triggeredSignals', [])
            if triggered:
                top_signals = sorted(triggered, key=lambda x: x['score'], reverse=True)[:3]
                print(f"   Top Signals: {', '.join([s['signal'][:25] + '...' if len(s['signal']) > 25 else s['signal'] for s in top_signals])}")
                
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("🚀 COMPLETE ENHANCED STOCK ANALYSIS PORTFOLIO VIEWER")
    print("=" * 70)
    
    # Create visualizer instance
    visualizer = CompleteStockAnalysisVisualizer("reports")
    
    # Load all reports
    if not visualizer.load_reports():
        print("❌ No valid reports found. Exiting...")
        return
        
    # Generate text summary
    visualizer.generate_summary_report()
    
    print("\n🎨 Generating comprehensive visual analysis...")
    
    try:
        # Create comprehensive dashboard
        visualizer.create_comprehensive_dashboard()
        
        print("\n✅ Complete portfolio analysis finished!")
        print("📁 Check the 'portfolio_reports' directory for all visualizations")
        print("🎯 Each chart provides detailed insights into your portfolio performance")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()