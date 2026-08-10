import yfinance as yf
import pandas as pd
import ccxt
import websocket
import json
import threading
import time
import os
from datetime import datetime, timedelta
from queue import Queue
import requests
from core.news_pipeline import get_default_news_pipeline
from core.logger import logger


def _to_float_or_none(value):
    """Convert a pandas cell to float, or None for NaN/missing (pd.isna(None) is False, so check both)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        # Reconnect / heartbeat state
        self._stream_active = False          # True while stream should remain alive
        self._stop_event = threading.Event() # wakes sleeping reconnect/heartbeat loops
        self._last_message_time = None       # epoch time of last received WS message
        self._reconnect_delay = 1.0          # current backoff delay (seconds)
        self._heartbeat_thread = None        # monitors message freshness
        # WebSocket tunables — exposed as instance attrs so tests can override them
        self._ws_reconnect_initial = 1.0     # first backoff delay (seconds)
        self._ws_reconnect_max = 30.0        # maximum backoff cap
        self._ws_heartbeat_interval = 5.0    # liveness check frequency (seconds)
        self._ws_heartbeat_staleness = 60.0  # seconds without a message → force reconnect
        self._ws_connect_timeout = 10        # seconds to wait for initial on_open
        self.news_pipeline = get_default_news_pipeline()

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

        # News and event feature integration
        logger.info(f"Fetching news and event features for {symbol}...")
        try:
            news_df = self.news_pipeline.fetch_news_dataframe(symbol)
            if not news_df.empty:
                df = self.news_pipeline.merge_features_into_prices(df, news_df, interval=interval)
                logger.info("News sentiment and event features merged successfully.")
            else:
                feature_columns = [
                    'positive', 'negative', 'neutral', 'sentiment_confidence', 'sentiment_balance',
                    'sentiment_magnitude', 'impact_score', 'source_reliability', 'news_count',
                    'headline_count', 'source_count', 'news_flow_ratio', 'event_earnings',
                    'event_guidance', 'event_mna', 'event_analyst', 'event_macro', 'event_regulatory',
                    'event_product', 'event_litigation', 'event_dividend', 'event_general'
                ]
                for column in feature_columns:
                    if column not in df.columns:
                        df[column] = 0
                logger.info("No news and event data found.")
        except Exception as e:
            logger.error(f"Error fetching news and event data: {e}")
            for column in [
                'positive', 'negative', 'neutral', 'sentiment_confidence', 'sentiment_balance',
                'sentiment_magnitude', 'impact_score', 'source_reliability', 'news_count',
                'headline_count', 'source_count', 'news_flow_ratio', 'event_earnings',
                'event_guidance', 'event_mna', 'event_analyst', 'event_macro', 'event_regulatory',
                'event_product', 'event_litigation', 'event_dividend', 'event_general'
            ]:
                df[column] = 0

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
            # Stocks — try OpenBB first (unified provider interface), fall back to yfinance
            # BACKUP NOTE: original _get_yahoo_historical call is preserved below as fallback.
            try:
                return self._get_openbb_historical(symbol, days, interval)
            except Exception as e:
                logger.warning("OpenBB data fetch failed (%s), falling back to Yahoo Finance: %s", symbol, e)
                # BACKUP (original): return self._get_yahoo_historical(symbol, days, interval)
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

    def _get_openbb_historical(self, symbol: str, days: int, interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data via OpenBB Platform (provider: yfinance by default).

        OpenBB normalises column names and handles provider switching transparently.
        Set OPENBB_EQUITY_PROVIDER in .env to switch provider (e.g. "polygon", "fmp").

        BACKUP NOTE: _get_yahoo_historical is still used as fallback if this fails.
        """
        from openbb import obb

        provider = os.getenv("OPENBB_EQUITY_PROVIDER", "yfinance").strip()

        # Map internal interval aliases to OpenBB/yfinance format
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "60m": "60m", "1d": "1d", "1wk": "1W",
        }
        obb_interval = interval_map.get(interval, "1d")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info("OpenBB: fetching %s %s data for %s (provider=%s)", interval, symbol, days, provider)

        result = obb.equity.price.historical(
            symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval=obb_interval,
            provider=provider,
        )

        df = result.to_df()
        if df.empty:
            raise ValueError(f"OpenBB returned empty DataFrame for {symbol}")

        # Normalise column names to match the rest of the app (Title Case OHLCV)
        col_map = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
            "adj_close": "Adj Close",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Normalise index name to "Datetime" to match the rest of the app
        df.index.name = "Datetime"

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"OpenBB response missing columns: {missing_cols}")

        logger.info("OpenBB data loaded: %d candles for %s", len(df), symbol)
        return df

    def _get_yahoo_historical(self, symbol, days, interval):
        """Get historical data from Yahoo Finance"""
        # Normalize interval aliases
        if interval == '1h':
            interval = '60m'

        # Minute-level intervals where Yahoo restricts historical range
        minute_intervals = {'1m', '2m', '5m', '15m', '30m', '60m'}

        # Cap days for intraday/minute intervals (Yahoo limits intraday history)
        capped_days = days
        if interval in minute_intervals:
            max_intraday_days = int(os.getenv('YAHOO_INTRADAY_MAX_DAYS', '7'))
            if days > max_intraday_days:
                logger.warning(
                    "Requested %sd of intraday %s data; capping to %sd because Yahoo Finance limits intraday history",
                    days,
                    interval,
                    max_intraday_days,
                )
                capped_days = max_intraday_days

        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo']
        if interval not in valid_intervals:
            interval = '1d'

        logger.info(f"Downloading Yahoo Finance data: {symbol}, period={capped_days}d, interval={interval}")
        df = yf.download(symbol, period=f"{capped_days}d", interval=interval, progress=False)

        # Drop the extra level when Yahoo returns a MultiIndex (Adj Close level)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # If download returned no rows, return an empty OHLCV-shaped DataFrame
        if df.empty:
            empty = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            empty.index.name = 'Datetime'
            logger.info(f"Yahoo Finance returned empty DataFrame for {symbol}")
            return empty

        # If the index is already a DatetimeIndex, normalize/coerce to datetime64[ns]
        if isinstance(df.index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                df.index = pd.to_datetime(df.index.astype(str), errors='coerce')

            df.index.name = 'Datetime'
            logger.info(f"Yahoo Finance historical data loaded: {len(df)} candles for {symbol} (index is DatetimeIndex)")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Otherwise, reset index and attempt to locate a datetime-like column
        df = df.reset_index()

        datetime_col = None

        # Prefer columns with explicit datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                datetime_col = col
                break

        # Next, look for likely name matches
        if datetime_col is None:
            for col in df.columns:
                if 'date' in str(col).lower() or 'time' in str(col).lower():
                    datetime_col = col
                    break

        # Try parsing the first column as a last resort
        if datetime_col is None and len(df.columns) > 0:
            first_col = df.columns[0]
            parsed = pd.to_datetime(df[first_col], errors='coerce')
            if not parsed.isna().all():
                datetime_col = first_col

        if datetime_col is None:
            raise KeyError(
                f"Could not determine a datetime column in Yahoo download for {symbol}. "
                f"Columns: {list(df.columns)}. Provide a column named like 'Date'/'Datetime' or ensure the first column is parseable as datetime."
            )

        # Coerce to datetime and drop NA datetimes
        df['Datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
        before = len(df)
        df = df.dropna(subset=['Datetime'])
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} rows with non-parseable Datetime for {symbol}")

        if df.empty:
            empty = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            empty.index.name = 'Datetime'
            logger.info(f"All rows dropped after parsing Datetime for {symbol}; returning empty DataFrame")
            return empty

        df.set_index('Datetime', inplace=True)
        df.index.name = 'Datetime'
        logger.info(f"Yahoo Finance historical data loaded: {len(df)} candles for {symbol} (datetime column: {datetime_col})")
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
        """Start real-time WebSocket stream for crypto.

        Public signature is unchanged — callers (ui/main_window.py etc.) are
        unaffected.  If a stream is already active (or mid-reconnect) it is
        cleanly stopped before the new one begins.
        """
        if self._stream_active:
            self.stop_realtime_stream()

        self.active_symbol = symbol
        self._callback = callback
        self._stop_event.clear()   # reset any lingering stop signal

        logger.info(f"[DataLoader] Starting real-time WebSocket for {symbol}")

        # Use Binance WebSocket (no API keys needed for public streams)
        self._start_binance_websocket(symbol)

    def _start_binance_websocket(self, symbol):
        """Start Binance WebSocket with automatic reconnect and heartbeat.

        Reconnect strategy
        ------------------
        * On ``on_close`` or ``on_error`` the reconnect loop (inside
          ``ws_thread``) waits ``_reconnect_delay`` seconds then opens a fresh
          connection.  The delay doubles on each attempt (exponential backoff)
          up to ``_ws_reconnect_max``.  A successful ``on_open`` + first
          message resets the delay to ``_ws_reconnect_initial``.

        Heartbeat liveness check
        ------------------------
        A separate ``_heartbeat_thread`` wakes every ``_ws_heartbeat_interval``
        seconds.  If ``ws_connected`` is True but the last message arrived more
        than ``_ws_heartbeat_staleness`` seconds ago it calls ``ws.close()``
        which unblocks ``run_forever()``, triggering the reconnect loop.

        Both loops honour ``_stop_event`` so ``stop_realtime_stream()``
        terminates everything promptly with no orphaned threads.
        """
        # ------------------------------------------------------------------ #
        #  Tunables (read once at start so tests can override per-instance)   #
        # ------------------------------------------------------------------ #
        _RECONNECT_INITIAL   = self._ws_reconnect_initial
        _RECONNECT_MAX       = self._ws_reconnect_max
        _HEARTBEAT_INTERVAL  = self._ws_heartbeat_interval
        _HEARTBEAT_STALENESS = self._ws_heartbeat_staleness
        _CONNECT_TIMEOUT     = self._ws_connect_timeout

        stream_name = f"{symbol.lower()}@depth@100ms"
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"

        # Reset per-stream state
        self._reconnect_delay = _RECONNECT_INITIAL
        self._last_message_time = time.time()
        self._stream_active = True

        # ------------------------------------------------------------------ #
        #  WebSocket callbacks                                                 #
        # ------------------------------------------------------------------ #
        def on_message(ws, message):
            try:
                self._last_message_time = time.time()
                self._reconnect_delay = _RECONNECT_INITIAL  # reset backoff on success
                data = json.loads(message)
                # logger.debug(f"[WebSocket Raw] {message}") # Debug raw message

                # Check for order book depth update structure
                if 'b' in data and 'a' in data:
                    order_book_update = {
                        'symbol': data['s'],
                        'bids': data['b'],   # [[price, quantity], ...]
                        'asks': data['a'],   # [[price, quantity], ...]
                        'timestamp': datetime.fromtimestamp(data['E'] / 1000),  # Event time
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
            self._last_message_time = time.time()
            self._reconnect_delay = _RECONNECT_INITIAL  # reset backoff on clean open

        def _make_ws():
            return websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )

        # ------------------------------------------------------------------ #
        #  Reconnect loop — runs inside ws_thread                             #
        # ------------------------------------------------------------------ #
        def _run_ws_with_reconnect():
            while self._stream_active and not self._stop_event.is_set():
                self.ws = _make_ws()
                try:
                    self.ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.error(f"WebSocket run error: {e}")

                # Exit immediately if stop was requested
                if not self._stream_active or self._stop_event.is_set():
                    break

                # Exponential backoff before next attempt
                delay = self._reconnect_delay
                self._reconnect_delay = min(self._reconnect_delay * 2, _RECONNECT_MAX)
                logger.info(
                    f"[DataLoader] WebSocket for {symbol} disconnected; "
                    f"reconnecting in {delay:.2f}s (backoff)"
                )
                # Block for `delay` seconds, but wake immediately on stop signal
                self._stop_event.wait(timeout=delay)

        # ------------------------------------------------------------------ #
        #  Heartbeat liveness monitor — runs in _heartbeat_thread             #
        # ------------------------------------------------------------------ #
        def _heartbeat_loop():
            # wait() returns True as soon as _stop_event is set, False on timeout
            while not self._stop_event.wait(timeout=_HEARTBEAT_INTERVAL):
                if not self._stream_active:
                    break
                if self.ws_connected and self._last_message_time is not None:
                    staleness = time.time() - self._last_message_time
                    if staleness > _HEARTBEAT_STALENESS:
                        logger.warning(
                            f"[DataLoader] No WS message for {staleness:.0f}s "
                            f"on {symbol}; forcing reconnect"
                        )
                        ws = self.ws
                        if ws:
                            try:
                                ws.close()
                            except Exception:
                                pass
                        # ws_thread's reconnect loop handles the new connection

        # ------------------------------------------------------------------ #
        #  Launch threads                                                      #
        # ------------------------------------------------------------------ #
        self.ws_thread = threading.Thread(target=_run_ws_with_reconnect, daemon=True)
        self.ws_thread.start()

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # Wait for the initial connection (not needed for subsequent reconnects)
        start_time = time.time()
        while not self.ws_connected and time.time() - start_time < _CONNECT_TIMEOUT:
            time.sleep(0.1)

        if not self.ws_connected:
            raise Exception(
                f"Failed to establish WebSocket connection for {symbol} "
                f"within {_CONNECT_TIMEOUT} seconds"
            )

    def stop_realtime_stream(self):
        """Stop the real-time WebSocket stream.

        Sets the stop flag and wakes any sleeping reconnect/heartbeat loops,
        then closes the current WebSocket and joins both threads.  No orphaned
        threads remain after this call returns.
        """
        logger.info(f"[DataLoader] Stopping real-time stream for {self.active_symbol}")

        # Signal all loops to exit before doing anything else
        self._stream_active = False
        self._stop_event.set()   # wake sleeping wait() calls immediately

        # Close the WebSocket to unblock run_forever()
        ws = self.ws
        if ws:
            try:
                ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        self.ws = None

        # Join WebSocket (reconnect) thread
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
            if self.ws_thread.is_alive():
                logger.warning("WebSocket thread did not terminate gracefully.")
            self.ws_thread = None

        # Join heartbeat thread
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
            if self._heartbeat_thread.is_alive():
                logger.warning("Heartbeat thread did not terminate gracefully.")
            self._heartbeat_thread = None

        self.ws_connected = False
        self.active_symbol = None
        self._callback = None

        # Clear the queue
        while not self.realtime_queue.empty():
            try:
                self.realtime_queue.get_nowait()
            except Exception:
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

    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent price for *symbol* as a float.

        Resolution order:
        1. Live websocket queue — if a stream is active and has order-book data
           for this symbol, derive the mid-price from the best bid/ask.
        2. Recent OHLCV via load_data() (last 2 days, 1-minute candles) —
           returns the last close price without triggering news merging.
        3. Binance CCXT spot ticker — direct fetch_ticker call on the public
           connector (no API keys required).

        Raises ValueError if all three sources fail.
        """
        # 1. Check live WebSocket queue for this symbol
        if self.ws_connected and self.active_symbol and self.active_symbol.upper() == symbol.upper():
            updates = self.get_realtime_updates()
            for update in reversed(updates):
                bids = update.get('bids', [])
                asks = update.get('asks', [])
                if bids and asks:
                    try:
                        best_bid = float(bids[0][0])
                        best_ask = float(asks[0][0])
                        if best_bid > 0 and best_ask > 0:
                            mid_price = (best_bid + best_ask) / 2.0
                            logger.debug(f"get_latest_price({symbol}): websocket mid-price = {mid_price}")
                            return mid_price
                    except (IndexError, ValueError, TypeError):
                        pass

        # 2. Fetch recent OHLCV and return last close (bypass news pipeline)
        try:
            is_crypto = "USDT" in symbol.upper()
            if is_crypto:
                df = self._get_binance_historical(symbol, days=2, interval='1m')
            else:
                try:
                    df = self._get_openbb_historical(symbol, days=2, interval='1m')
                except Exception:
                    df = self._get_yahoo_historical(symbol, days=2, interval='1m')

            if df is not None and not df.empty:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                price = float(df[close_col].iloc[-1])
                logger.debug(f"get_latest_price({symbol}): OHLCV last close = {price}")
                return price
        except Exception as e:
            logger.debug(f"get_latest_price({symbol}): OHLCV fetch failed: {e}")

        # 3. Binance CCXT spot ticker (no keys needed)
        try:
            ccxt_symbol = symbol if '/' in symbol else symbol.replace('USDT', '/USDT')
            ticker = self.binance_public.fetch_ticker(ccxt_symbol)
            price = ticker.get('last') or ticker.get('close')
            if price is not None:
                price = float(price)
                logger.debug(f"get_latest_price({symbol}): Binance ticker = {price}")
                return price
        except Exception as e:
            logger.debug(f"get_latest_price({symbol}): Binance ticker failed: {e}")

        raise ValueError(f"get_latest_price({symbol}): all price sources failed")

    def test_binance_connection(self):
        """Test if Binance API is reachable"""
        try:
            self.binance_public.fetch_time()
            return True
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            return False

    def get_earnings_calendar(self, symbol: str) -> list:
        """Fetch upcoming/past earnings dates for an equity symbol.

        BACKUP NOTE: originally planned via OpenBB's obb.equity.calendar.earnings(),
        but that endpoint only supports provider='fmp', and FMP restricts it to
        legacy accounts with subscriptions predating 2025-08-31 -- a fresh free-tier
        key gets UnauthorizedError regardless of a valid API key. Uses yfinance's
        Ticker.earnings_dates directly instead, which needs no separate credential
        and matches the fallback pattern already used elsewhere in this file.

        Returns a list of dicts: [{"date": "YYYY-MM-DD", "eps_estimate": float|None,
        "eps_actual": float|None, "revenue_estimate": float|None, "revenue_actual": float|None}, ...]
        Returns [] for crypto symbols (no earnings) and on any fetch failure —
        never raises, matching the graceful-degradation pattern used throughout
        this class.
        """
        is_crypto = "USDT" in symbol.upper()
        if is_crypto:
            logger.debug("get_earnings_calendar(%s): crypto symbol, no earnings data", symbol)
            return []

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.earnings_dates
            if df is None or df.empty:
                return []

            results = []
            for idx, row in df.iterrows():
                results.append({
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    "eps_estimate": _to_float_or_none(row.get("EPS Estimate")),
                    "eps_actual": _to_float_or_none(row.get("Reported EPS")),
                    "revenue_estimate": None,   # yfinance's earnings_dates doesn't carry revenue figures
                    "revenue_actual": None,
                })
            logger.info("get_earnings_calendar(%s): %d entries via yfinance", symbol, len(results))
            return results

        except Exception as e:
            logger.warning("get_earnings_calendar(%s) failed: %s", symbol, e)
            return []