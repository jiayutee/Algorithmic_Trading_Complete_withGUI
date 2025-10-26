from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import yfinance as yf
import pandas as pd
import ccxt
import websocket
import json
import threading
import time
from datetime import datetime, timedelta
from queue import Queue
import requests
from core.news_scraper import scrape_and_analyze_finviz_news

class DataLoader:
    def __init__(self, live_api_key=None, live_secret_key=None, kucoin_key=None, kucoin_secret=None, binance_key=None, binance_secret=None):
        # Crypto exchanges
        self.kucoin_connector = None
        self.binance_connector = None
        
        if kucoin_key and kucoin_secret:
            self.kucoin_connector = ccxt.kucoin({
                'apiKey': kucoin_key,
                'secret': kucoin_secret,
                'enableRateLimit': True,
            })
            
        if binance_key and binance_secret:
            self.binance_connector = ccxt.binance({
                'apiKey': binance_key,
                'secret': binance_secret,
                'enableRateLimit': True,
            })
        
        # Real-time streaming attributes
        self.realtime_queue = Queue()
        self.active_symbol = None
        self.ws_thread = None
        self.ws_connected = False
        self.ws = None
        self._callback = None

    def load_data(self, symbol, source="Historical", live=False, days=365, interval='1d'):
        """Load data from various sources"""
        if source == "FinRL-Yahoo":
            df = self._get_finrl_data(symbol, days, interval)
        elif live:
            df = self._get_live_data(symbol, days, interval)
        else:
            df = self._get_historical_data(symbol, days, interval)

        # News sentiment integration
        print(f"Fetching news sentiment for {symbol}...")
        news_df = scrape_and_analyze_finviz_news(symbol)
        if not news_df.empty:
            news_df['datetime'] = pd.to_datetime(news_df['datetime'])
            news_df['datetime'] = news_df['datetime'].dt.tz_localize('UTC')
            news_df.set_index('datetime', inplace=True)

            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            df = pd.merge_asof(df.sort_index(), news_df[['positive', 'negative', 'neutral']],
                              left_index=True, right_index=True, direction='backward')
            df[['positive', 'negative', 'neutral']] = df[['positive', 'negative', 'neutral']].ffill().fillna(0)
            print("News sentiment data merged successfully.")
        else:
            df['positive'] = 0
            df['negative'] = 0
            df['neutral'] = 0
            print("No news sentiment data found.")

        return df

    def _get_finrl_data(self, symbol, days=3650, interval='1d'):
        """Get data from FinRL's YahooDownloader"""
        if symbol == 'BTCUSDT':
            symbol = 'BTC-USD'
            
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        df = YahooDownloader(start_date=start_date,
                             end_date=end_date,
                             ticker_list=[symbol]).fetch_data()

        if 'tic' in df.columns:
            df.drop(columns=['tic'], inplace=True)
        df.rename(columns={'date': 'Datetime', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                           'volume': 'Volume'}, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _get_historical_data(self, symbol, days, interval='1d'):
        """Get historical data with crypto priority"""
        is_crypto = "USDT" in symbol.upper()

        if is_crypto:
            # Try Binance first (no API keys needed for public data)
            try:
                return self._get_binance_historical(symbol, days, interval)
            except Exception as e:
                print(f"Binance historical failed: {e}")
                # Fall back to KuCoin if configured
                if self.kucoin_connector:
                    return self._get_kucoin_historical(symbol, days, interval)
                else:
                    # Final fallback to Yahoo Finance for crypto-USD pairs
                    return self._get_yahoo_crypto_historical(symbol, days, interval)
        else:
            # Stocks - use Yahoo Finance
            return self._get_yahoo_historical(symbol, days, interval)

    def _get_binance_historical(self, symbol, days, interval):
        """Get historical data from Binance with batch loading to bypass 1000 limit"""
        import requests
        import time
        
        # Binance API endpoint
        url = "https://api.binance.com/api/v3/klines"
        
        # Convert interval to Binance format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '1d': '1d'
        }
        binance_interval = interval_map.get(interval, '1d')
        
        # Calculate total candles needed
        interval_to_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '1d': 1440
        }
        minutes_per_candle = interval_to_minutes.get(binance_interval, 1440)
        total_minutes = days * 24 * 60
        total_candles_needed = total_minutes // minutes_per_candle
        
        print(f"[Binance] Loading {total_candles_needed} candles for {symbol} ({interval})")
        
        # Binance limit per request
        BINANCE_LIMIT = 1000
        all_candles = []
        
        # Calculate batch parameters
        if total_candles_needed <= BINANCE_LIMIT:
            # Single request if under limit
            batches = [total_candles_needed]
        else:
            # Split into multiple batches
            num_batches = (total_candles_needed + BINANCE_LIMIT - 1) // BINANCE_LIMIT
            batches = [BINANCE_LIMIT] * (num_batches - 1)
            remaining = total_candles_needed - (BINANCE_LIMIT * (num_batches - 1))
            batches.append(remaining)
        
        print(f"[Binance] Splitting into {len(batches)} batches: {batches}")
        
        # Current time as end point
        end_time = int(time.time() * 1000)
        
        for i, batch_size in enumerate(batches):
            try:
                print(f"[Binance] Batch {i+1}/{len(batches)}: Loading {batch_size} candles...")
                
                # Calculate start time for this batch
                batch_minutes = batch_size * minutes_per_candle
                batch_ms = batch_minutes * 60 * 1000
                start_time = end_time - batch_ms
                
                params = {
                    'symbol': symbol,
                    'interval': binance_interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': batch_size
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    raise ValueError(f"Binance API error: {response.status_code} - {response.text}")
                
                batch_data = response.json()
                
                if not batch_data:
                    print(f"[Binance] No data returned for batch {i+1}")
                    break
                
                # Add to collection
                all_candles.extend(batch_data)
                print(f"[Binance] Batch {i+1} completed: {len(batch_data)} candles")
                
                # Update end_time for next batch (move backward in time)
                end_time = start_time - 1  # Move 1ms before this batch's start
                
                # Rate limiting - be nice to Binance API
                if i < len(batches) - 1:  # Don't sleep after last batch
                    time.sleep(0.2)  # 200ms delay between requests
                    
            except Exception as e:
                print(f"[Binance] Error in batch {i+1}: {e}")
                # Continue with whatever data we have
                break
        
        if not all_candles:
            raise ValueError(f"No data returned from Binance for {symbol}")
        
        print(f"[Binance] Total candles loaded: {len(all_candles)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        
        # Sort by timestamp (oldest first) since we loaded newest to oldest
        df = df.sort_values('Open time')
        
        # Convert columns
        df['Datetime'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Datetime', inplace=True)
        
        # Convert OHLCV to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        print(f"[Binance] Final DataFrame: {len(df)} candles for {symbol}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _get_kucoin_historical(self, symbol, days, interval):
        """Get historical data from KuCoin"""
        try:
            print(f"Fetching crypto historical data from KuCoin for {symbol}...")
            kucoin_symbol = symbol.replace("USDT", "/USDT")
            
            # Fetch OHLCV data
            ohlcv = self.kucoin_connector.fetch_ohlcv(kucoin_symbol, interval, limit=days)
            
            df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
            df.set_index('Datetime', inplace=True)
            df = df.astype(float)
            print(f"KuCoin historical data loaded: {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            print(f"KuCoin historical data failed: {e}")
            raise

    def _get_yahoo_crypto_historical(self, symbol, days, interval):
        """Fallback to Yahoo Finance for crypto (converts BTCUSDT to BTC-USD)"""
        if symbol == 'BTCUSDT':
            symbol = 'BTC-USD'
        elif symbol == 'ETHUSDT':
            symbol = 'ETH-USD'
        elif symbol == 'SOLUSDT':
            symbol = 'SOL-USD'
        elif symbol == 'ADAUSDT':
            symbol = 'ADA-USD'
            
        return self._get_yahoo_historical(symbol, days, interval)

    def _get_yahoo_historical(self, symbol, days, interval):
        """Get historical data from Yahoo Finance"""
        if interval in ['1m', '5m', '15m', '30m']:
            raise ValueError(f"Yahoo Finance doesn't support {interval} interval")

        valid_intervals = ['1h', '1d']
        if interval not in valid_intervals:
            interval = '1d'

        df = yf.download(symbol, period=f"{days}d", interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        
        if interval == '1d':
            df['Datetime'] = pd.to_datetime(df['Date'])
        elif interval == '1h':
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
        df.set_index('Datetime', inplace=True)
        print(f"Yahoo Finance historical data loaded: {len(df)} candles for {symbol}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _get_live_data(self, symbol, days, interval='1d'):
        """Get 'live' data - actually uses historical for display"""
        # For live display, we use recent historical data
        # Real-time updates come through WebSocket separately
        return self._get_historical_data(symbol, min(days, 7), interval)  # Limit to 7 days for performance

    # REAL-TIME STREAMING METHODS
    def start_realtime_stream(self, symbol, callback):
        """Start real-time WebSocket stream for crypto"""
        if self.ws_connected:
            self.stop_realtime_stream()

        self.active_symbol = symbol
        self._callback = callback
        
        print(f"[DataLoader] Starting real-time WebSocket for {symbol}")
        
        # Use Binance WebSocket (no API keys needed for public streams)
        self._start_binance_websocket(symbol)

    def _start_binance_websocket(self, symbol):
        """Start Binance WebSocket for real-time crypto data"""
        stream_name = f"{symbol.lower()}@trade"
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                trade_data = {
                    'symbol': data['s'],
                    'price': float(data['p']),
                    'quantity': float(data['q']),
                    'timestamp': datetime.fromtimestamp(data['T'] / 1000),
                    'is_buyer_maker': data['m'],
                    'exchange': 'binance',
                    'type': 'trade'
                }
                # print(f"[WebSocket] {symbol}: ${trade_data['price']}")  # Optional: comment out for less noise
                self.realtime_queue.put(trade_data)
                if self._callback:
                    self._callback(trade_data)
            except Exception as e:
                print(f"WebSocket message error: {e}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            self.ws_connected = False

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket connection closed for {self.active_symbol}")
            self.ws_connected = False

        def on_open(ws):
            print(f"WebSocket connected for {self.active_symbol}")
            self.ws_connected = True

        # Create and start WebSocket
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        def run_websocket():
            try:
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"WebSocket run error: {e}")
            
        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()
        
        # Wait for connection with timeout
        timeout = 10
        start_time = time.time()
        while not self.ws_connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.ws_connected:
            raise Exception(f"Failed to establish WebSocket connection for {symbol} within {timeout} seconds")

    def stop_realtime_stream(self):
        """Stop the real-time WebSocket stream"""
        print(f"[DataLoader] Stopping real-time stream for {self.active_symbol}")
        
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None
        
        if self.ws_thread:
            self.ws_thread.join(timeout=2)
            self.ws_thread = None
            
        self.ws_connected = False
        self.active_symbol = None
        self._callback = None
        
        # Clear the queue
        while not self.realtime_queue.empty():
            try:
                self.realtime_queue.get_nowait()
            except:
                break

    def get_realtime_updates(self):
        """Get all available real-time updates from the queue"""
        updates = []
        while not self.realtime_queue.empty():
            try:
                updates.append(self.realtime_queue.get_nowait())
            except:
                break
        return updates

    def get_connection_status(self):
        """Check WebSocket connection status"""
        return {
            'connected': self.ws_connected,
            'active_symbol': self.active_symbol,
            'queue_size': self.realtime_queue.qsize(),
            'thread_alive': self.ws_thread.is_alive() if self.ws_thread else False
        }

    def test_binance_connection(self):
        """Test if Binance API is reachable"""
        try:
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Binance connection test failed: {e}")
            return False