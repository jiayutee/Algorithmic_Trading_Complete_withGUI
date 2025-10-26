import openai
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

class DeepSeekAnalyzer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"  # DeepSeek API endpoint
        )
        self.analysis_cache = {}
        
    def analyze_crypto_data(self, symbol: str, data: pd.DataFrame, 
                          additional_metrics: Dict = None) -> Dict:
        """
        Analyze crypto data using DeepSeek AI
        """
        try:
            # Prepare data summary for AI
            data_summary = self._prepare_data_summary(symbol, data, additional_metrics)
            
            # Create AI prompt
            prompt = self._create_analysis_prompt(symbol, data_summary)
            
            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # or "deepseek-coder" for more technical analysis
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency financial analyst. Provide detailed technical and fundamental analysis with specific trading insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=2000
            )
            
            analysis = response.choices[0].message.content
            
            # Parse the analysis into structured format
            structured_analysis = self._parse_ai_response(analysis, symbol, data_summary)
            
            return structured_analysis
            
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}", "recommendation": "HOLD"}
    
    def _prepare_data_summary(self, symbol: str, data: pd.DataFrame, additional_metrics: Dict) -> Dict:
        """Prepare comprehensive data summary for AI analysis"""
        if data.empty:
            return {}
            
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest
        
        # Calculate basic metrics
        price_change = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
        volume_change = ((latest['Volume'] - previous['Volume']) / previous['Volume']) * 100 if previous['Volume'] > 0 else 0
        
        # Calculate technical indicators if not present
        if 'RSI' not in data.columns:
            data['RSI'] = self._calculate_rsi(data['Close'])
        if 'MACD' not in data.columns:
            macd_data = self._calculate_macd(data['Close'])
            data['MACD'] = macd_data['macd']
            data['MACD_Signal'] = macd_data['signal']
        
        summary = {
            'symbol': symbol,
            'current_price': latest['Close'],
            'price_change_24h': price_change,
            'volume': latest['Volume'],
            'volume_change_24h': volume_change,
            'high_24h': latest['High'],
            'low_24h': latest['Low'],
            'rsi': data['RSI'].iloc[-1],
            'macd': data['MACD'].iloc[-1],
            'macd_signal': data['MACD_Signal'].iloc[-1],
            'trend': self._assess_trend(data),
            'support_levels': self._find_support_resistance(data, 'support'),
            'resistance_levels': self._find_support_resistance(data, 'resistance'),
            'volatility': data['Close'].pct_change().std() * 100,
            'data_points': len(data),
            'timeframe': f"{((data.index[-1] - data.index[0]).days)} days"
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            summary.update(additional_metrics)
            
        return summary
    
    def _create_analysis_prompt(self, symbol: str, data_summary: Dict) -> str:
        """Create detailed prompt for AI analysis"""
        return f"""
        Analyze this cryptocurrency data for {symbol} and provide a comprehensive trading analysis:

        DATA SUMMARY:
        {json.dumps(data_summary, indent=2, default=str)}

        Please provide analysis in this structured format:

        TECHNICAL ANALYSIS:
        - Trend Analysis: 
        - Key Support/Resistance Levels:
        - RSI Interpretation (current: {data_summary.get('rsi', 'N/A')}):
        - MACD Analysis (MACD: {data_summary.get('macd', 'N/A'):.4f}, Signal: {data_summary.get('macd_signal', 'N/A'):.4f}):
        - Volume Analysis:

        TRADING RECOMMENDATION:
        - Action: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
        - Confidence: [0-100%]
        - Price Targets:
          - Short-term (1-3 days):
          - Medium-term (1-2 weeks):
        - Stop-loss Level:

        RISK ASSESSMENT:
        - Risk Level: [LOW/MEDIUM/HIGH]
        - Key Risks:
        - Market Conditions:

        ADDITIONAL INSIGHTS:
        - Market Sentiment:
        - Key Factors to Watch:
        - Alternative Scenarios:

        Please be specific and data-driven in your analysis.
        """
    
    def _parse_ai_response(self, analysis: str, symbol: str, data_summary: Dict) -> Dict:
        """Parse AI response into structured data"""
        try:
            # Extract key sections using simple parsing
            sections = {
                'technical_analysis': self._extract_section(analysis, 'TECHNICAL ANALYSIS:'),
                'recommendation': self._extract_recommendation(analysis),
                'risk_assessment': self._extract_section(analysis, 'RISK ASSESSMENT:'),
                'additional_insights': self._extract_section(analysis, 'ADDITIONAL INSIGHTS:'),
                'raw_analysis': analysis
            }
            
            # Add metadata
            sections.update({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': data_summary.get('current_price'),
                'price_change': data_summary.get('price_change_24h')
            })
            
            return sections
            
        except Exception as e:
            return {
                'error': f"Failed to parse AI response: {str(e)}",
                'raw_analysis': analysis,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_section(self, text: str, section_header: str) -> str:
        """Extract a specific section from the AI response"""
        try:
            start_idx = text.find(section_header)
            if start_idx == -1:
                return "Section not found"
                
            start_idx += len(section_header)
            end_idx = text.find('\n\n', start_idx)
            if end_idx == -1:
                end_idx = len(text)
                
            return text[start_idx:end_idx].strip()
        except:
            return "Error extracting section"
    
    def _extract_recommendation(self, text: str) -> Dict:
        """Extract trading recommendation from AI response"""
        recommendation_keywords = {
            'STRONG_BUY': ['strong buy', 'strong_buy', 'strong buy'],
            'BUY': ['buy', 'long', 'bullish'],
            'HOLD': ['hold', 'neutral', 'consolidat'],
            'SELL': ['sell', 'short', 'bearish'],
            'STRONG_SELL': ['strong sell', 'strong_sell', 'strong sell']
        }
        
        default_rec = {'action': 'HOLD', 'confidence': 50, 'reason': 'No clear signal'}
        
        try:
            text_lower = text.lower()
            
            # Find action
            action = 'HOLD'
            for action_type, keywords in recommendation_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    action = action_type
                    break
            
            # Extract confidence (look for percentages)
            import re
            confidence_matches = re.findall(r'(\d+)%', text)
            confidence = int(confidence_matches[0]) if confidence_matches else 50
            
            return {
                'action': action,
                'confidence': min(confidence, 100),
                'price_targets': self._extract_price_targets(text),
                'stop_loss': self._extract_stop_loss(text)
            }
            
        except:
            return default_rec
    
    def _extract_price_targets(self, text: str) -> Dict:
        """Extract price targets from AI response"""
        # Simple pattern matching for price targets
        import re
        targets = {}
        
        # Look for short-term targets
        short_term_matches = re.findall(r'short.*?term.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if short_term_matches:
            targets['short_term'] = float(short_term_matches[0])
            
        # Look for medium-term targets  
        medium_term_matches = re.findall(r'medium.*?term.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if medium_term_matches:
            targets['medium_term'] = float(medium_term_matches[0])
            
        return targets
    
    def _extract_stop_loss(self, text: str) -> float:
        """Extract stop-loss level from AI response"""
        import re
        stop_loss_matches = re.findall(r'stop.*?loss.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        return float(stop_loss_matches[0]) if stop_loss_matches else None
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return {'macd': macd, 'signal': macd_signal}
    
    def _assess_trend(self, data: pd.DataFrame) -> str:
        """Assess overall trend"""
        if len(data) < 20:
            return "NEUTRAL"
        
        short_ma = data['Close'].tail(10).mean()
        long_ma = data['Close'].tail(20).mean()
        
        if short_ma > long_ma * 1.02:
            return "BULLISH"
        elif short_ma < long_ma * 0.98:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _find_support_resistance(self, data: pd.DataFrame, level_type: str) -> List[float]:
        """Find key support/resistance levels"""
        if len(data) < 20:
            return []
        
        # Simple implementation - use recent highs/lows
        if level_type == 'support':
            return [data['Low'].tail(20).min()]
        else:  # resistance
            return [data['High'].tail(20).max()]