# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
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
from core.logger import logger

class DataLoader:
    """
    Manages data loading from various sources (Historical, Live, FinRL)
    and handles real-time data streaming via WebSockets.
    """
    def __init__(self, live_api_key=None, live_secret_key=None, kucoin_key=None, kucoin_secret=None, binance_key=None, binance_secret=None):
        """
        Initialize DataLoader with optional API keys.
        """
        # Crypto exchanges
        self.kucoin_connector = None
        self.binance_connector = None
        
        # Always initialize a public instance for historical data if keys not provided
        self.binance_public = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

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
        else:
            self.binance_connector = self.binance_public
        
        # Real-time streaming attributes
        self.realtime_queue = Queue()
        self.active_symbol = None
        self.ws_thread = None
        self.ws_connected = False
        self.ws = None
        self._callback = None

    def load_data(self, symbol, source="Historical", live=False, days=365, interval='1d'):
        """
        Load data from various sources.
        
        Args:
            symbol (str): Ticker symbol.
            source (str): Data source ("Historical", "FinRL-Yahoo").
            live (bool): If True, loads recent data (previously 'live').
            days (int): Number of days of history.
            interval (str): Candle interval ('1m', '1h', '1d').
            
        Returns:
            pd.DataFrame: OHLCV data.
        """
        logger.info(f"Loading data... Symbol: {symbol}, Source: {source}, Days: {days}, Interval: {interval}")
        
        if source == "FinRL-Yahoo":
            df = self._get_finrl_data(symbol, days, interval)
        elif live:
            df = self._get_recent_data(symbol, days, interval)
        else:
            df = self._get_historical_data(symbol, days, interval)

        # News sentiment integration
        logger.info(f"Fetching news sentiment for {symbol}...")
        try:
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
                logger.info("News sentiment data merged successfully.")
            else:
                df['positive'] = 0
                df['negative'] = 0
                df['neutral'] = 0
                logger.info("No news sentiment data found.")
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            df['positive'] = 0
            df['negative'] = 0
            df['neutral'] = 0

        return df

    def _get_finrl_data(self, symbol, days=3650, interval='1d'):
        """Get data simulating FinRL's YahooDownloader using internal method"""
        if symbol == 'BTCUSDT':
            symbol = 'BTC-USD'
        return self._get_yahoo_historical(symbol, days, interval)

    def _get_historical_data(self, symbol, days, interval='1d'):
        """Get historical data with crypto priority using ccxt for crypto"""
        is_crypto = "USDT" in symbol.upper()

        if is_crypto:
            try:
                return self._get_binance_historical(symbol, days, interval)
            except Exception as e:
                logger.error(f"Binance historical failed: {e}")
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
        """Get historical data from Binance using CCXT"""
        logger.info(f"Fetching {interval} data for {symbol} from Binance via CCXT for last {days} days")
        
        # Calculate start timestamp in milliseconds
        since = int(self.binance_public.milliseconds() - (days * 24 * 60 * 60 * 1000))
        
        all_ohlcv = []
        limit = 1000
        
        while True:
            try:
                ohlcv = self.binance_public.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=limit)
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Check if we reached current time
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1
                
                if len(ohlcv) < limit:
                    break
                    
                time.sleep(self.binance_public.rateLimit / 1000) # Respect rate limits
                
            except Exception as e:
                logger.warning(f"Failed to fetch batch from Binance (will retry/fallback): {e}")
                break
                
        if not all_ohlcv:
            raise ValueError(f"No data returned from Binance for {symbol}")
            
        df = pd.DataFrame(all_ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
        df.set_index('Datetime', inplace=True)
        
        # Ensure proper types
        df = df.astype(float)
        
        logger.info(f"Binance CCXT data loaded: {len(df)} candles for {symbol}")
        return df

    def _get_kucoin_historical(self, symbol, days, interval):
        """Get historical data from KuCoin"""
        try:
            logger.info(f"Fetching crypto historical data from KuCoin for {symbol}...")
            kucoin_symbol = symbol.replace("USDT", "/USDT")
            
            # Fetch OHLCV data
            ohlcv = self.kucoin_connector.fetch_ohlcv(kucoin_symbol, interval, limit=days) # CCXT handles some pagination
            
            if not ohlcv:
                raise ValueError(f"No data returned from KuCoin for {symbol}")

            df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
            df.set_index('Datetime', inplace=True)
            df = df.astype(float)
            logger.info(f"KuCoin historical data loaded: {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"KuCoin historical data failed: {e}")
            raise

    def _get_yahoo_crypto_historical(self, symbol, days, interval):
        """Fallback to Yahoo Finance for crypto (converts BTCUSDT to BTC-USD)"""
        symbol_map = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'SOLUSDT': 'SOL-USD',
            'ADAUSDT': 'ADA-USD'
        }
        yahoo_symbol = symbol_map.get(symbol, symbol)
        return self._get_yahoo_historical(yahoo_symbol, days, interval)

    def _get_yahoo_historical(self, symbol, days, interval):
        """Get historical data from Yahoo Finance"""
        if interval in ['1m', '5m', '15m', '30m']:
            # Yahoo might support some minute data effectively only for recent 7 or 60 days depending on interval
            logger.warning(f"Yahoo Finance support for {interval} is limited/unreliable.")

        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval not in valid_intervals:
             # Map some common formats if necessary
            if interval == '1h': interval = '60m'
            else: interval = '1d'

        logger.info(f"Downloading Yahoo Finance data: {symbol}, period={days}d, interval={interval}")
        df = yf.download(symbol, period=f"{days}d", interval=interval, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        
        # Yahoo Finance column names vary slightly often ('Date' vs 'Datetime')
        if 'Date' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'])
        elif 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
        df.set_index('Datetime', inplace=True)
        logger.info(f"Yahoo Finance historical data loaded: {len(df)} candles for {symbol}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _get_recent_data(self, symbol, days, interval='1d'):
        """
        Get recent data. Previously '_get_live_data'.
        Uses historical data for the last few days to simulate 'live' chart.
        """
        # For live display, we use recent historical data
        # Real-time updates come through WebSocket separately
        logger.info("Fetching recent data for display...")
        return self._get_historical_data(symbol, min(days, 7), interval)  # Limit to 7 days for performance

    # REAL-TIME STREAMING METHODS
    def start_realtime_stream(self, symbol, callback):
        """Start real-time WebSocket stream for crypto"""
        if self.ws_connected:
            self.stop_realtime_stream()

        self.active_symbol = symbol
        self._callback = callback
        
        logger.info(f"[DataLoader] Starting real-time WebSocket for {symbol}")
        
        # Use Binance WebSocket (no API keys needed for public streams)
        self._start_binance_websocket(symbol)

    def _start_binance_websocket(self, symbol):
        """Start Binance WebSocket for real-time crypto data"""
        stream_name = f"{symbol.lower()}@depth@100ms"
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # logger.debug(f"[WebSocket Raw] {message}") # Debug raw message
                
                # Check for order book depth update structure
                if 'b' in data and 'a' in data:
                    order_book_update = {
                        'symbol': data['s'],
                        'bids': data['b'], # [[price, quantity], ...]
                        'asks': data['a'], # [[price, quantity], ...]
                        'timestamp': datetime.fromtimestamp(data['E'] / 1000), # Event time
                        'exchange': 'binance',
                        'type': 'depthUpdate'
                    }
                    self.realtime_queue.put(order_book_update)
                    if self._callback:
                        self._callback(order_book_update)
                # Add handling for other message types if necessary
                else:
                    logger.warning(f"Unhandled WebSocket message type: {data.get('e', 'unknown_event')}")

            except Exception as e:
                logger.error(f"WebSocket message parsing error: {e}, Message: {message}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.ws_connected = False

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed for {self.active_symbol}")
            self.ws_connected = False

        def on_open(ws):
            logger.info(f"WebSocket connected for {self.active_symbol}")
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
                # Remove daemon=True for graceful shutdown
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
            
        self.ws_thread = threading.Thread(target=run_websocket) # Removed daemon=True
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
        logger.info(f"[DataLoader] Stopping real-time stream for {self.active_symbol}")
        
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None
        
        if self.ws_thread:
            # Explicitly join the thread for graceful termination
            self.ws_thread.join(timeout=2) 
            if self.ws_thread.is_alive():
                logger.warning("WebSocket thread did not terminate gracefully.")
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
            self.binance_public.fetch_time()
            return True
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            return False