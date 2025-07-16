from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import yfinance as yf
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import time
from queue import Queue
import threading
import asyncio


class DataLoader:
    def __init__(self, live_api_key=None, live_secret_key=None):
        self.live_mode = bool(live_api_key and live_secret_key)
        if self.live_mode:
            self.client = StockHistoricalDataClient(live_api_key, live_secret_key)
            self.stream_client = None
            self.realtime_queue = Queue()
            self.active_symbol = None
            self.stream_thread = None

    def load_data(self, symbol, source="Historical", live=False, days=365, interval='1d'):
        """Load data from either live, historical, or FinRL source"""
        if source == "FinRL-Yahoo":
            return self._get_finrl_data(symbol, days, interval)
        if live and self.live_mode:
            return self._get_live_data(symbol, days, interval)
        return self._get_historical_data(symbol, days, interval)

    

    

    def _get_finrl_data(self, symbol, days=3650, interval='1d'):
        """Get data from FinRL's YahooDownloader"""
        if symbol == 'BTCUSDT':
            symbol = 'BTC-USD'
        # FinRL's downloader uses a different date format
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        df = YahooDownloader(start_date=start_date,
                             end_date=end_date,
                             ticker_list=[symbol]).fetch_data()

        # The columns are lowercase and there is a 'tic' column
        if 'tic' in df.columns:
            df.drop(columns=['tic'], inplace=True)
        df.rename(columns={'date': 'Datetime', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                           'volume': 'Volume'}, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

        # Debug prints for FinRL-Yahoo data
        print(f"[DEBUG] FinRL-Yahoo data sample:\n{df.head()}")
        print(f"[DEBUG] FinRL-Yahoo index type: {type(df.index)}")
        print(f"[DEBUG] FinRL-Yahoo columns: {df.columns}")
        print(f"[DEBUG] FinRL-Yahoo OHLC dtypes:\n{df[['Open', 'High', 'Low', 'Close']].dtypes}")
        print(f"[DEBUG] FinRL-Yahoo NaN counts:\n{df.isna().sum()}")

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        

    def _get_historical_data(self, symbol, days, interval='1d'):
        """Get historical data from Yahoo Finance with interval support"""
        if symbol == 'BTCUSDT':
            symbol = 'BTC-USD'
        if interval in ['1m', '5m', '15m', '30m']:
            raise ValueError(f"Yahoo Finance doesn't support {interval} interval")

        valid_intervals = ['1h', '1d']  # Yahoo Finance supported intervals
        if interval not in valid_intervals:
            interval = '1d'  # Fallback to daily

        df = yf.download(symbol, period=f"{days}d", interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        if interval == '1d':
            df['Datetime'] = pd.to_datetime(df['Date'])
        elif interval == '1h':
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)


        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _get_live_data(self, symbol, days, interval='1d'):
        """Get live data from Alpaca with interval support"""
        if symbol == 'BTCUSDT':
            symbol = 'BTC/USD'
        interval_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '30m': TimeFrame(30, TimeFrameUnit.Minute),
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }
        if interval not in interval_map:
            interval = '1d'  # Fallback to daily

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=interval_map[interval],
            start=datetime.now() - timedelta(days=min(days, 30))  # Alpaca free tier limit
        )
        bars = self.client.get_stock_bars(request).df

        # Reset index and convert to single datetime index
        bars = bars.reset_index()
        bars['Datetime'] = pd.to_datetime(bars['timestamp'], utc=True)  # Ensure UTC
        bars['Datetime'] = bars['Datetime'].dt.tz_convert('America/New_York')  # Convert to desired timezone
        bars.set_index('Datetime', inplace=True)

        # return bars[['open', 'high', 'low', 'close', 'volume']].rename(
        #     columns={c: c.capitalize() for c in bars.columns}
        # )
        # Rename columns and select needed ones
        return bars.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })[['Open', 'High', 'Low', 'Close', 'Volume']]

    def start_realtime_stream(self, symbol, callback):
        """Start real-time data streaming for a symbol"""
        if not self.live_mode:
            raise Exception("Live mode not initialized with API keys")

        # Stop any existing stream
        if self.stream_client:
            self.stop_realtime_stream()

        print(f"[DataLoader] Starting real-time stream for {symbol}")  # Debug

        self.active_symbol = symbol
        self.stream_client = StockDataStream(
            self.client._api_key,
            self.client._secret_key
        )

        # Add connection state tracking
        self._stream_connected = False
        self._stream_error = None

        async def handle_trade(trade):
            try:
                data = {
                    'symbol': trade.symbol,
                    'price': trade.price,
                    'size': trade.size,
                    'timestamp': trade.timestamp.timestamp(),
                    'exchange': trade.exchange
                }
                print(f"[DataLoader] Received trade: {data}")  # Debug
                self.realtime_queue.put(data)
                callback(data)  # Direct callback in addition to queue
            except Exception as e:
                print(f"[DataLoader] Error in trade handler: {e}")

        async def run_stream():
            try:
                self.stream_client.subscribe_trades(handle_trade, symbol)
                print("[DataLoader] Subscription created")  # Debug
                self._stream_connected = True
                await self.stream_client._run_forever()
            except Exception as e:
                self._stream_error = str(e)
                print(f"[DataLoader] Stream error: {e}")  # Debug
            finally:
                self._stream_connected = False
        def run_async_stream():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print("[DataLoader] Starting event loop")  # Debug
                loop.run_until_complete(run_stream())
            except Exception as e:
                print(f"[DataLoader] Thread error: {e}")  # Debug

        # Start the thread with error handling
        self.stream_thread = threading.Thread(
            target=run_async_stream,
            daemon=True,
            name=f"AlpacaStreamThread-{symbol}"
        )
        self.stream_thread.start()

        # Verify connection within timeout
        timeout = 5  # seconds
        start_time = time.time()

        while not self._stream_connected and not self._stream_error:
            if time.time() - start_time > timeout:
                raise Exception("Connection timeout - failed to establish stream")
            time.sleep(0.1)

        if self._stream_error:
            raise Exception(f"Stream error: {self._stream_error}")

        print("[DataLoader] Stream successfully established")  # Debug

    def stop_realtime_stream(self):
        """Stop the real-time data stream"""
        if self.stream_client and self.active_symbol:
            try:
                self.stream_client.stop()
                if self.stream_thread:
                    self.stream_thread.join(timeout=1)
            except Exception as e:
                print(f"Error stopping stream: {e}")
            finally:
                self.stream_client = None
                self.active_symbol = None
                self.stream_thread = None

    def get_realtime_updates(self):
        """Get all available real-time updates from the queue"""
        updates = []
        while not self.realtime_queue.empty():
            updates.append(self.realtime_queue.get())
        return updates

    def get_connection_status(self):
        """Check if stream is connected"""
        return {
            'connected': self._connection_established.is_set(),
            'error': self._connection_error,
            'thread_alive': self.stream_thread.is_alive() if self.stream_thread else False,
            'queue_size': self.realtime_queue.qsize()
        }

    def test_connection(self):
        """Test if Alpaca API is reachable"""
        try:
            import requests
            response = requests.get(
                "https://paper-api.alpaca.markets/v2/clock",
                headers={
                    "APCA-API-KEY-ID": self.client._api_key,
                    "APCA-API-SECRET-KEY": self.client._secret_key
                },
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            return False