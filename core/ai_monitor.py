import threading
import time
from datetime import datetime
from typing import Dict, List

class AIMonitor:
    def __init__(self, ai_analyzer, symbols: List[str], update_interval: int = 300):
        self.ai_analyzer = ai_analyzer
        self.symbols = symbols
        self.update_interval = update_interval
        self.analysis_results = {}
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, data_provider):
        """Start continuous AI monitoring"""
        self.monitoring = True
        self.data_provider = data_provider
        
        def monitor_loop():
            while self.monitoring:
                try:
                    for symbol in self.symbols:
                        # Get latest data
                        data = data_provider(symbol)
                        if data is not None and not data.empty:
                            # Run AI analysis
                            analysis = self.ai_analyzer.analyze_crypto_data(symbol, data)
                            self.analysis_results[symbol] = analysis
                            
                            # Log significant changes
                            self._check_alerts(symbol, analysis)
                    
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    print(f"AI monitoring error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop AI monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _check_alerts(self, symbol: str, analysis: Dict):
        """Check for alert conditions"""
        recommendation = analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        confidence = recommendation.get('confidence', 50)
        
        # Alert on strong signals with high confidence
        if action in ['STRONG_BUY', 'STRONG_SELL'] and confidence > 70:
            print(f"🚨 ALERT: {symbol} - {action} (Confidence: {confidence}%)")
            
    def get_portfolio_recommendations(self) -> Dict:
        """Get AI recommendations for entire portfolio"""
        portfolio_analysis = {}
        
        for symbol, analysis in self.analysis_results.items():
            if 'error' not in analysis:
                portfolio_analysis[symbol] = {
                    'action': analysis.get('recommendation', {}).get('action', 'HOLD'),
                    'confidence': analysis.get('recommendation', {}).get('confidence', 50),
                    'current_price': analysis.get('current_price', 0),
                    'timestamp': analysis.get('timestamp')
                }
                
        return portfolio_analysis