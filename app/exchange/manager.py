"""
Exchange Manager

This module handles communication with cryptocurrency exchanges through the CCXT library.
"""

import logging
import asyncio
import ccxt.async_support as ccxt
import time
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple
import pandas as pd
from datetime import datetime, timedelta
import re
import json
import sys
import os
from enum import Enum
import numpy as np

# Assuming these imports are correct relative to your project structure
# from app.config.settings import Settings, AppMode
# from app.config.binance_config import BinanceConfig, BinanceMode
# from app.config.cryptocom_config import CryptocomConfig, CryptocomMode
# from app.exchange.binance_direct import BinanceDirectAPI

# Configure logging
logger = logging.getLogger(__name__)

class Exchange:
    """Exchange class that uses real market data even in simulation mode."""

    def __init__(self, name: str, config=None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.quote_currency = "USDC"  # Default quote currency
        if config and isinstance(config, dict):
            self.quote_currency = config.get('quote_currency', 'USDC')

        # Initialize mainnet CCXT client for market data
        self.mainnet_client = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })

        self.logger.info(f"Exchange {name} initialized with {self.quote_currency} as quote currency")

    async def fetch_markets(self) -> List[Dict[str, Any]]:
        """Fetch real markets from Binance mainnet."""
        self.logger.info(f"Fetching real markets from Binance mainnet")

        try:
            markets = await self.mainnet_client.fetch_markets()

            # Filter for active markets with our quote currency
            filtered_markets = []
            for market in markets:
                if (market['quote'] == self.quote_currency and
                    market['active'] and
                    market['spot']):  # Only include spot markets
                    filtered_markets.append(market)

            self.logger.info(f"Found {len(filtered_markets)} active {self.quote_currency} markets")
            return filtered_markets

        except Exception as e:
            self.logger.error(f"Error fetching markets from mainnet: {str(e)}")
            # Return empty markets list as fallback
            return []

    async def fetch_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch real ticker data from Binance mainnet."""
        self.logger.info(f"Fetching real tickers from Binance mainnet")

        try:
            # Get all tickers
            tickers = await self.mainnet_client.fetch_tickers() # Fixed: removed trailing 'a'

            # Filter for our quote currency pairs
            filtered_tickers = {}
            for symbol, ticker in tickers.items():
                if f"/{self.quote_currency}" in symbol:
                    filtered_tickers[symbol] = ticker

            self.logger.info(f"Fetched {len(filtered_tickers)} {self.quote_currency} tickers")
            return filtered_tickers

        except Exception as e:
            self.logger.error(f"Error fetching tickers from mainnet: {str(e)}")
            return {}

    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Simulate order status but use real market price."""
        self.logger.info(f"Simulating order {order_id} for {symbol}")

        try:
            # Get real current price
            ticker = await self.mainnet_client.fetch_ticker(symbol)
            price = ticker['last'] if ticker and 'last' in ticker else None

            if not price:
                raise ValueError(f"Could not get current price for {symbol}")

            # Return simulated order with real price
            return {
                'id': order_id,
                'symbol': symbol,
                'status': 'closed',
                'filled': 1.0,
                'remaining': 0.0,
                'price': price,
                'amount': 1.0,
                'cost': price * 1.0,
                'fee': {'cost': price * 1.0 * 0.001, 'currency': self.quote_currency},
                'info': {'simulatedOrder': True}
            }

        except Exception as e:
            self.logger.error(f"Error simulating order with real price: {str(e)}")
            return {}

    async def create_order(self, symbol: str, type: str, trade_type: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Simulate order creation but use real market price."""
        self.logger.info(f"Simulating {trade_type} {type} order for {amount} {symbol}")

        try:
            # Get real current price
            ticker = await self.mainnet_client.fetch_ticker(symbol)
            current_price = ticker['last'] if ticker and 'last' in ticker else None

            if not current_price:
                raise ValueError(f"Could not get current price for {symbol}")

            # Use provided price or current market price
            execution_price = price or current_price

            # Simulate some slippage based on order size and real market data
            slippage = 0.001  # 0.1% base slippage
            if amount * execution_price > 10000:  # Large orders get more slippage
                slippage = 0.002

            if trade_type == 'buy':
                execution_price *= (1 + slippage)
            else:
                execution_price *= (1 - slippage)

            # Return simulated order with realistic price
            return {
                'id': f"sim_{int(time.time())}",
                'symbol': symbol,
                'type': type,
                'trade_type': trade_type,
                'amount': amount,
                'price': execution_price,
                'status': 'closed',
                'filled': amount,
                'remaining': 0.0,
                'cost': amount * execution_price,
                'fee': {'cost': amount * execution_price * 0.001, 'currency': self.quote_currency},
                'info': {
                    'simulatedOrder': True,
                    'marketPrice': current_price,
                    'slippage': f"{slippage*100}%"
                }
            }

        except Exception as e:
            self.logger.error(f"Error simulating order with real price: {str(e)}")
            return {}

    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """Return simulated balance but validate against real trading pairs."""
        self.logger.info(f"Fetching simulated balance with real trading pairs")

        try:
            # Get real trading pairs first
            markets = await self.fetch_markets()
            available_assets = set([market['base'] for market in markets])
            available_assets.add(self.quote_currency)

            # Create simulated balance only for real trading pairs
            balance = {
                'free': {self.quote_currency: 10000.0},  # Start with quote currency
                'used': {self.quote_currency: 0.0},
                'total': {self.quote_currency: 10000.0}
            }

            # Add small balance for major trading pairs
            for asset in available_assets:
                if asset != self.quote_currency:
                    balance['free'][asset] = 0.1
                    balance['used'][asset] = 0.0
                    balance['total'][asset] = 0.1

            return balance

        except Exception as e:
            self.logger.error(f"Error creating simulated balance: {str(e)}")
            return {
                'free': {self.quote_currency: 10000.0},
                'used': {self.quote_currency: 0.0},
                'total': {self.quote_currency: 10000.0}
            }

    async def close(self) -> None:
        """Close the exchange connection."""
        try:
            await self.mainnet_client.close()
            self.logger.info(f"Closed mainnet client connection for {self.name}")
        except Exception as e:
            self.logger.error(f"Error closing mainnet client: {str(e)}")

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from real Binance mainnet."""
        self.logger.info(f"Fetching OHLCV data for {symbol} ({timeframe})")

        try:
            # Convert symbol format if needed
            formatted_symbol = symbol
            if '/' not in symbol and self.quote_currency in symbol:
                base = symbol[:-len(self.quote_currency)]
                formatted_symbol = f"{base}/{self.quote_currency}"

            # Fetch OHLCV data
            ohlcv_data = await self.mainnet_client.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)

            if ohlcv_data and len(ohlcv_data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                # Return empty DataFrame if no data
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data from real Binance mainnet."""
        self.logger.info(f"Fetching ticker for {symbol}")

        try:
            # Convert symbol format if needed
            formatted_symbol = symbol
            if '/' not in symbol and self.quote_currency in symbol:
                base = symbol[:-len(self.quote_currency)]
                formatted_symbol = f"{base}/{self.quote_currency}"

            # Fetch ticker data
            ticker = await self.mainnet_client.fetch_ticker(formatted_symbol)
            if ticker and isinstance(ticker, dict) and 'last' in ticker:
                return ticker
            else:
                # Return fallback ticker if we got invalid data
                return self._create_fallback_ticker(symbol)

        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            # Return simulated ticker as fallback
            return self._create_fallback_ticker(symbol)

    def _create_fallback_ticker(self, symbol: str) -> Dict[str, Any]:
        """Create a fallback ticker with simulated data."""
        return {
            'symbol': symbol,
            'last': 50000.0,
            'bid': 49990.0,
            'ask': 50010.0,
            'volume': 100.0,
            'quoteVolume': 5000000.0,
            'percentage': 0.0,
            'high': 50100.0,
            'low': 49900.0,
            'timestamp': int(time.time() * 1000)
        }

class ExchangeManager:
    """Manages interactions with cryptocurrency exchanges."""

    def __init__(self, settings, sandbox=True):
        """
        Initialize the exchange manager.

        Args:
            settings: Application settings with API credentials
            sandbox: Whether to use sandbox/testnet mode
        """
        self.settings = settings
        self.sandbox = sandbox
        self.exchanges = {}
        self.rate_limiters = {}
        self.logger = logging.getLogger(__name__)  # Initialize logger first

        # Initialize direct Binance API
        self.direct_apis = {}
        # Check if settings and binance_config exist before accessing
        if hasattr(settings, 'binance_config') and settings.binance_config:
             # Check if the import path is correct for your project
             # Assuming BinanceDirectAPI is in app.exchange.binance_direct
            from app.exchange.binance_direct import BinanceDirectAPI
            self.direct_apis["binance"] = BinanceDirectAPI(
                api_key=settings.binance_config.api_key,
                api_secret=settings.binance_config.secret_key,
                sandbox=sandbox
            )

        self.logger.info(f"Exchange manager initialized (sandbox: {sandbox})")

        # Check for credentials in settings
        self.exchange_credentials = getattr(settings, 'exchange_credentials', {})
        if not self.exchange_credentials:
            self.logger.warning("No exchange credentials found in settings. Using empty credentials.")
            self.exchange_credentials = {
                "binance": {"api_key": "", "secret_key": ""}
                # Add other exchanges if needed
            }

        # Initialize exchange configurations
        self.exchange_configs = {}
        if hasattr(settings, 'binance_config') and settings.binance_config:
            self.exchange_configs['binance'] = settings.binance_config
        if hasattr(settings, 'cryptocom_config') and settings.cryptocom_config:
            self.exchange_configs['cryptocom'] = settings.cryptocom_config

        # Initialize rate limits attribute to prevent attribute error
        self.rate_limits = {
            "binance": {
                "calls": 0,
                "limit": 1200,  # Default rate limit for Binance API
                "window": 60,   # 1 minute window
                "last_reset": time.time()
            },
            "cryptocom": {
                "calls": 0,
                "limit": 500,   # Default rate limit for Crypto.com API
                "window": 60,   # 1 minute window
                "last_reset": time.time()
            }
            # Add other exchanges if needed
        }

        # Initialize exchanges based on settings
        self._init_exchanges()

    def _init_exchanges(self):
        """Initialize exchange connections based on settings."""
        # For simulation mode, use our custom Exchange class
        if self.sandbox:
            # Get quote currency from config
            quote_currency = "USDC"  # Default
            if "binance" in self.exchange_configs:
                binance_config = self.exchange_configs["binance"]
                if hasattr(binance_config, 'default_quote_currency'):
                    quote_currency = binance_config.default_quote_currency

            # Initialize with proper config including quote currency
            self.exchanges["binance"] = Exchange("binance", {
                "simulated": True,
                "quote_currency": quote_currency
            })
            self.logger.info(f"Initialized Binance exchange in simulation mode with {quote_currency} as quote currency")
            # Add other simulated exchanges if needed
            return # Exit after setting up simulation mode

        # For live mode, use CCXT exchanges
        active_exchanges = getattr(self.settings, 'active_exchanges', [])

        if not active_exchanges:
            self.logger.warning("No active exchanges configured, using simulation mode as fallback")
            # Fallback to simulation if no active exchanges are listed
            quote_currency = "USDC"
            if "binance" in self.exchange_configs and hasattr(self.exchange_configs["binance"], 'default_quote_currency'):
                quote_currency = self.exchange_configs["binance"].default_quote_currency
            self.exchanges["binance"] = Exchange("binance", {
                "simulated": True,
                "quote_currency": quote_currency
            })
            return

        for exchange_name in active_exchanges:
            if exchange_name == "binance" and "binance" in self.exchange_configs:
                config = self.exchange_configs['binance']
                credentials = self.exchange_credentials.get("binance", {})
                if not credentials.get('api_key') or not credentials.get('secret_key'):
                    self.logger.warning(f"Missing API key or secret for Binance. Falling back to simulation.")
                    self.exchanges[exchange_name] = Exchange(exchange_name, {
                        "simulated": True,
                        "quote_currency": getattr(config, 'default_quote_currency', 'USDC')
                    })
                    continue # Skip to next exchange

                try:
                    # Initialize CCXT Binance exchange
                    exchange = ccxt.binance({
                        'apiKey': credentials['api_key'],
                        'secret': credentials['secret_key'], # Corrected key name 'secret_key' -> 'secret' for ccxt
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': getattr(config, 'trading_mode', 'spot'), # Use getattr for safety
                            'adjustForTimeDifference': True,
                            'recvWindow': 60000,
                            'createMarketBuyOrderRequiresPrice': False
                        },
                        'timeout': 30000,
                        # 'enableRateLimit': True, # Duplicate, removed
                        'rateLimit': 100 # Example rate limit setting
                    })

                    # Configure sandbox mode - CCXT handles this via set_sandbox_mode
                    # No separate sandbox URL setup needed here unless CCXT fails

                    self.exchanges[exchange_name] = exchange
                    self.logger.info(f"Initialized {exchange_name} exchange in live mode")

                except Exception as e:
                    self.logger.error(f"Failed to initialize {exchange_name} exchange: {str(e)}")
                    # Fall back to simulation
                    self.exchanges[exchange_name] = Exchange(exchange_name, {
                        "simulated": True,
                        "quote_currency": getattr(config, 'default_quote_currency', 'USDC')
                    })
                    self.logger.info(f"Falling back to simulated {exchange_name} exchange")

            elif exchange_name == "cryptocom" and "cryptocom" in self.exchange_configs:
                config = self.exchange_configs['cryptocom']
                credentials = self.exchange_credentials.get("cryptocom", {})
                if not credentials.get('api_key') or not credentials.get('secret_key'):
                    self.logger.warning(f"Missing API key or secret for Crypto.com. Falling back to simulation.")
                    self.exchanges[exchange_name] = Exchange(exchange_name, {"simulated": True})
                    continue # Skip to next exchange

                try:
                    # Initialize CCXT Crypto.com exchange
                    exchange = ccxt.cryptocom({
                        'apiKey': credentials['api_key'],
                        'secret': credentials['secret_key'], # Corrected key name
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': getattr(config, 'trading_mode', 'spot'), # Use getattr for safety
                            'adjustForTimeDifference': True
                        }
                    })

                    self.exchanges[exchange_name] = exchange # Use exchange_name here
                    api_key_suffix = credentials['api_key'][-4:] if credentials['api_key'] else 'N/A'
                    self.logger.info(f"CCXT {exchange_name} exchange initialized with API key ending with ...{api_key_suffix}")

                except Exception as e:
                    self.logger.error(f"Failed to initialize {exchange_name} exchange: {str(e)}")
                    # Fallback to simulation mode for this exchange
                    self.exchanges[exchange_name] = Exchange(exchange_name, {"simulated": True})
                    self.logger.warning(f"Falling back to simulation mode for {exchange_name}")
            else:
                self.logger.warning(f"Configuration or credentials missing for requested active exchange: {exchange_name}")


    def get_active_exchanges(self) -> List[Any]: # Return type might be Union[Exchange, ccxt.Exchange]
        """Get list of active exchange instances."""
        return list(self.exchanges.values())

    def get_exchange(self, name: str) -> Optional[Any]: # Return type might be Union[Exchange, ccxt.Exchange]
        """Get exchange instance by name."""
        return self.exchanges.get(name)

    async def fetch_market_data(self, symbol: str, exchange_id: str = "binance") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch both OHLCV and current market data.
        Uses direct API for Binance if available, with CCXT fallback.
        """
        ohlcv_df = pd.DataFrame()
        current_data = {}

        try:
            # Try direct API first if it exists and is for the requested exchange
            if exchange_id == "binance" and self.direct_apis.get("binance"):
                try:
                    direct_api = self.direct_apis["binance"]
                    # Get OHLCV data
                    # Note: BinanceDirectAPI needs to return DataFrame in expected format
                    ohlcv_df = await direct_api.get_klines(symbol, '1m', 100)

                    # Get 24h ticker data
                    ticker_24h = await direct_api.get_24h_ticker(symbol)
                    if ticker_24h:
                        current_data = {
                            'price': float(ticker_24h['lastPrice']),
                            'volume_24h': float(ticker_24h['quoteVolume']), # Use quoteVolume usually
                            'change_24h': float(ticker_24h['priceChangePercent']),
                            'high_24h': float(ticker_24h['highPrice']),
                            'low_24h': float(ticker_24h['lowPrice'])
                        }
                    # If we got both, return
                    if not ohlcv_df.empty and current_data:
                         self.logger.debug(f"Fetched market data via direct API for {symbol}")
                         return ohlcv_df, current_data
                    else:
                         self.logger.warning(f"Direct API fetch incomplete for {symbol}, proceeding to CCXT.")

                except Exception as e:
                    self.logger.warning(
                        f"Direct API failed for {symbol} on {exchange_id}, falling back to CCXT. "
                        f"Error: {str(e)}"
                    )

            # CCXT fallback or primary method if direct API not used/failed
            if exchange_id in self.exchanges:
                exchange = self.exchanges[exchange_id]
                is_simulated = isinstance(exchange, Exchange) # Check if it's our simulated class

                # Handle rate limits only for real CCXT exchanges
                if not is_simulated and hasattr(exchange, 'check_response_headers'):
                    await self._handle_rate_limit(exchange_id)

                try:
                    exchange_symbol = self.get_exchange_symbol_format(symbol, exchange_id)

                    # Fetch OHLCV data
                    if is_simulated:
                        # Our simulated class already returns a DataFrame
                        ohlcv_df = await exchange.fetch_ohlcv(exchange_symbol, '1m', limit=100)
                    else:
                        ohlcv = await exchange.fetch_ohlcv(exchange_symbol, '1m', limit=100)
                        if ohlcv:
                            ohlcv_df = pd.DataFrame(
                                ohlcv,
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            )
                            ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
                            ohlcv_df.set_index('timestamp', inplace=True)

                    # Fetch ticker data
                    ticker = await exchange.fetch_ticker(exchange_symbol)
                    if ticker:
                        # Make sure percentage is a valid number and handle potential None
                        percentage = ticker.get('percentage', 0.0) # Default to 0.0
                        percentage = percentage if percentage is not None else 0.0

                        current_data = {
                            'price': ticker.get('last', 0.0), # Default to 0.0
                            'volume_24h': ticker.get('quoteVolume', 0.0), # Default to 0.0
                            'change_24h': percentage, # Already defaulted
                            'high_24h': ticker.get('high', 0.0), # Default to 0.0
                            'low_24h': ticker.get('low', 0.0) # Default to 0.0
                        }

                    # Check if we got valid data before returning
                    if not ohlcv_df.empty and current_data.get('price', 0.0) > 0:
                         self.logger.debug(f"Fetched market data via {'Simulated' if is_simulated else 'CCXT'} API for {symbol}")
                         return ohlcv_df, current_data
                    else:
                        self.logger.warning(f"CCXT fetch incomplete for {symbol}.")

                except ccxt.NetworkError as e:
                    self.logger.error(f"CCXT NetworkError fetching data for {symbol}: {str(e)}")
                except ccxt.ExchangeError as e:
                     self.logger.error(f"CCXT ExchangeError fetching data for {symbol}: {str(e)}")
                except Exception as e:
                    self.logger.error(
                        f"Generic CCXT error fetching market data for {symbol}: {str(e)}"
                    )

            # If all fetches failed, return empty data
            self.logger.error(f"Failed to fetch market data for {symbol} from {exchange_id} via all methods.")
            return pd.DataFrame(), {} # Return empty structures

        except Exception as e:
            # Catch-all for unexpected errors in the function logic itself
            self.logger.error(f"Unexpected error in fetch_market_data for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame(), {}


    async def initialize(self):
        """Initialize connections and load markets for configured exchanges."""
        # Initialization logic moved to _init_exchanges called in __init__
        # This method now focuses on loading markets

        # Load markets for all initialized exchanges with retry mechanism
        all_loaded = True
        for exchange_id, exchange in self.exchanges.items():
            # Check if it's a CCXT exchange instance that needs loading
            if hasattr(exchange, 'load_markets') and not isinstance(exchange, Exchange):
                max_retries = 3
                loaded = False
                for retry in range(max_retries):
                    try:
                        self.logger.info(f"Attempting to load markets for {exchange_id} (attempt {retry+1}/{max_retries})")
                        # Try to load markets with a timeout
                        await asyncio.wait_for(exchange.load_markets(), timeout=20) # Increased timeout
                        self.logger.info(f"Successfully loaded markets for {exchange_id}")
                        loaded = True
                        break # Exit retry loop on success
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout loading markets for {exchange_id} (attempt {retry+1}/{max_retries})")
                        if retry == max_retries - 1:
                            self.logger.error(f"Failed to load markets for {exchange_id} after {max_retries} attempts due to timeout.")
                            all_loaded = False
                    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                        self.logger.warning(f"Network/Exchange error loading markets for {exchange_id} (attempt {retry+1}/{max_retries}): {str(e)}")
                        if retry < max_retries - 1:
                            await asyncio.sleep(2 * (retry + 1)) # Exponential backoff
                        else:
                             self.logger.error(f"Failed to load markets for {exchange_id} after {max_retries} attempts due to errors.")
                             all_loaded = False
                    except Exception as e:
                        self.logger.error(f"Unexpected error loading markets for {exchange_id}: {str(e)}\n{traceback.format_exc()}")
                        all_loaded = False
                        break # Exit retry loop on unexpected error
                if not loaded:
                     self.logger.warning(f"Continuing initialization, but {exchange_id} markets may not be fully loaded.")
            elif isinstance(exchange, Exchange):
                # For our simulated exchange, maybe call its own init if needed
                self.logger.info(f"Using simulated exchange {exchange_id}, skipping market load.")
            else:
                self.logger.warning(f"Exchange object for {exchange_id} doesn't seem to be a CCXT instance or simulated Exchange.")
                all_loaded = False

        # Fixed: Return statement moved outside the loop
        if all_loaded:
             self.logger.info("All exchange markets loaded successfully (or simulated).")
             return True
        else:
             self.logger.warning("Initialization complete, but some exchange markets failed to load.")
             return False # Indicate partial success/failure


    async def _handle_rate_limit(self, exchange_id: str):
        """Handle rate limiting for exchange API calls."""
        # This is a basic placeholder. Real CCXT rate limiting is often handled
        # internally if 'enableRateLimit': True is set.
        # This function could add custom logic if needed, e.g., checking
        # CCXT's internal rate limit counters if accessible, or enforcing
        # stricter limits based on custom rules.

        # Example: Accessing CCXT's built-in limiter (if available and needed)
        exchange = self.exchanges.get(exchange_id)
        if exchange and hasattr(exchange, 'rateLimit'):
             # CCXT's internal rate limit is usually in milliseconds
             delay = exchange.rateLimit / 1000.0 # Convert to seconds
             # You might sleep here based on last call time, but CCXT often does this.
             # logger.debug(f"CCXT internal rate limit for {exchange_id}: {delay}s. Relying on CCXT's handling.")
             pass # Rely on CCXT's built-in mechanism primarily

        # Custom logic based on self.rate_limits (less reliable than CCXT's own)
        # now = time.time()
        # rate_limit = self.rate_limits.get(exchange_id)
        # if not rate_limit:
        #     return
        # # Reset counter if window has passed
        # if now - rate_limit["last_reset"] > rate_limit["window"]:
        #     rate_limit["calls"] = 0
        #     rate_limit["last_reset"] = now
        # # Check if we're approaching the limit
        # if rate_limit["calls"] >= rate_limit["limit"] * 0.9:
        #     wait_time = rate_limit["window"] - (now - rate_limit["last_reset"])
        #     if wait_time > 0:
        #         logger.warning(f"Custom rate limit approaching for {exchange_id}, waiting {wait_time:.2f} seconds")
        #         await asyncio.sleep(wait_time)
        #         # Reset after waiting
        #         rate_limit["calls"] = 0
        #         rate_limit["last_reset"] = time.time()
        # # Increment the call counter
        # rate_limit["calls"] += 1
        await asyncio.sleep(0.05) # Small courtesy delay, adjust as needed


    async def fetch_tickers(self, exchange_id: str = None) -> Dict[str, Any]:
        """Fetch tickers for all symbols from a specific exchange or all active ones."""
        results = {}

        exchanges_to_query = []
        if exchange_id:
            if exchange_id in self.exchanges:
                exchanges_to_query = [(exchange_id, self.exchanges[exchange_id])]
            else:
                logger.error(f"Exchange {exchange_id} not initialized or not found.")
                return {}
        else:
            exchanges_to_query = self.exchanges.items()

        for ex_id, exchange in exchanges_to_query:
            try:
                # Check if it's a real CCXT exchange before applying rate limit logic
                if hasattr(exchange, 'check_response_headers'):
                     await self._handle_rate_limit(ex_id)

                tickers = await exchange.fetch_tickers()
                results[ex_id] = tickers
                logger.debug(f"Fetched {len(tickers)} tickers from {ex_id}")
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                 logger.error(f"Error fetching tickers from {ex_id}: {str(e)}")
                 results[ex_id] = {'error': str(e)} # Indicate error for this exchange
            except Exception as e:
                logger.error(f"Unexpected error fetching tickers from {ex_id}: {str(e)}\n{traceback.format_exc()}")
                results[ex_id] = {'error': str(e)}

        # If only one exchange was requested, return its tickers directly
        if exchange_id and exchange_id in results:
            return results[exchange_id]
        # Otherwise return the dictionary keyed by exchange ID
        return results


    async def fetch_ohlcv(self, symbol, timeframe='1h', limit=100, exchange_id="binance"):
        """
        Fetch OHLCV data for a symbol, ensuring proper symbol format.

        Args:
            symbol: Trading pair symbol (with or without slash)
            timeframe: Timeframe for candlesticks
            limit: Number of candlesticks to fetch
            exchange_id: Exchange to use

        Returns:
            DataFrame with OHLCV data or empty DataFrame on error.
        """
        try:
            # Ensure exchange is initialized
            if exchange_id not in self.exchanges:
                logger.error(f"Exchange {exchange_id} not initialized")
                return pd.DataFrame()

            exchange = self.exchanges[exchange_id]
            is_simulated = isinstance(exchange, Exchange)

            # Convert symbol to exchange format
            exchange_symbol = self.get_exchange_symbol_format(symbol, exchange_id)

            # Handle rate limits for real exchanges
            if not is_simulated and hasattr(exchange, 'check_response_headers'):
                await self._handle_rate_limit(exchange_id)

            # Fetch OHLCV data
            if is_simulated:
                # Our custom Exchange class already returns a DataFrame
                df = await exchange.fetch_ohlcv(exchange_symbol, timeframe=timeframe, limit=limit)
                if df is not None and not df.empty:
                    return df
                else:
                    logger.warning(f"Simulated exchange returned no OHLCV data for {symbol} ({timeframe})")
                    return pd.DataFrame()
            else:
                # For CCXT exchange that returns a list
                ohlcv_data = await exchange.fetch_ohlcv(exchange_symbol, timeframe=timeframe, limit=limit)

                # Convert to DataFrame
                if ohlcv_data and len(ohlcv_data) > 0:
                    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    # Fixed indentation for the following lines
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df
                else:
                    # Fixed indentation for this block
                    logger.warning(f"No OHLCV data returned from CCXT for {symbol} ({timeframe})")
                    return pd.DataFrame()

        except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
             logger.error(f"Exchange error fetching OHLCV for {symbol} on {exchange_id}: {str(e)}")
             return pd.DataFrame()
        except ccxt.NetworkError as e:
             logger.error(f"Network error fetching OHLCV for {symbol} on {exchange_id}: {str(e)}")
             return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching OHLCV for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()

    async def fetch_ticker(self, symbol, exchange_id="binance"):
        """
        Fetch ticker data for a symbol, ensuring proper symbol format.

        Args:
            symbol: Trading pair symbol (with or without slash)
            exchange_id: Exchange to use

        Returns:
            Dictionary with standardized ticker data or error info.
        """
        try:
            # Ensure exchange is initialized
            if exchange_id not in self.exchanges:
                logger.error(f"Exchange {exchange_id} not initialized")
                return {'symbol': symbol, 'error': f"Exchange {exchange_id} not found"}

            exchange = self.exchanges[exchange_id]
            is_simulated = isinstance(exchange, Exchange)

            # Convert symbol to exchange format
            exchange_symbol = self.get_exchange_symbol_format(symbol, exchange_id)

            # Handle rate limits for real exchanges
            if not is_simulated and hasattr(exchange, 'check_response_headers'):
                await self._handle_rate_limit(exchange_id)

            # Fetch ticker
            ticker = await exchange.fetch_ticker(exchange_symbol)

            # Check if we got a valid ticker object
            if ticker and isinstance(ticker, dict) and ticker.get('last') is not None:
                # Standardize the response
                return {
                    'symbol': symbol,  # Return the original symbol format for consistency
                    'last': ticker.get('last', 0.0),
                    'bid': ticker.get('bid', 0.0),
                    'ask': ticker.get('ask', 0.0),
                    'volume': ticker.get('volume', 0.0), # Base volume
                    'quoteVolume': ticker.get('quoteVolume', 0.0), # Quote volume
                    'percentage': ticker.get('percentage', 0.0), # 24h change %
                    'timestamp': ticker.get('timestamp', int(time.time() * 1000))
                }
            else:
                # Fixed indentation for logger
                logger.warning(f"No valid ticker data returned for {symbol} from {exchange_id}")
                # Fixed indentation for return
                return {
                    'symbol': symbol,
                    'error': "Invalid or incomplete ticker data received",
                    'last': 0.0,
                    'bid': 0.0,
                    'ask': 0.0,
                    'volume': 0.0,
                    'quoteVolume': 0.0,
                    'percentage': 0.0,
                    'timestamp': int(time.time() * 1000)
                }

        except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
            # Fixed indentation for logger and return
            logger.error(f"Exchange error fetching ticker for {symbol} on {exchange_id}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'last': 0.0,
                'timestamp': int(time.time() * 1000)
            }
        except ccxt.NetworkError as e:
             logger.error(f"Network error fetching ticker for {symbol} on {exchange_id}: {str(e)}")
             return {
                'symbol': symbol,
                'error': str(e),
                'last': 0.0,
                'timestamp': int(time.time() * 1000)
            }
        except Exception as e:
            logger.error(f"Unexpected error fetching ticker for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return {
                'symbol': symbol,
                'error': str(e),
                'last': 0.0,
                'timestamp': int(time.time() * 1000)
            }


    def get_exchange_symbol_format(self, symbol, exchange_id="binance"):
        """
        Converts the symbol to the correct format for the specified exchange.

        Args:
            symbol: Symbol to convert (can be with or without slash)
            exchange_id: The exchange to format for

        Returns:
            Symbol in the format required by the exchange, or original if unsure.
        """
        exchange_id_lower = exchange_id.lower()

        # Explicit formats known for specific exchanges
        if exchange_id_lower == 'binance':
            # Binance requires no slash: BTCUSDC
            return symbol.replace('/', '')
        elif exchange_id_lower == 'kucoin' or exchange_id_lower == 'kraken':
             # These often use dash: BTC-USDT
             return symbol.replace('/', '-')
        # Add other specific exchange formats here...
        # elif exchange_id_lower == 'some_other_exchange':
        #     return symbol.upper() # Example

        # Default/fallback: Assume slash format is common (CCXT standard)
        else:
            if '/' not in symbol:
                # Attempt to guess quote currency and add slash
                # This is fragile, better to rely on loaded markets if possible
                quote_currency = "USDC" # Default guess
                config = self.exchange_configs.get(exchange_id_lower)
                if config and hasattr(config, 'default_quote_currency'):
                    quote_currency = config.default_quote_currency

                # Check common quote currencies
                common_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'USD', 'EUR']
                found_quote = None
                for q in common_quotes:
                    if symbol.endswith(q):
                        found_quote = q
                        break
                # If a known quote is found, format it
                if found_quote:
                     base = symbol[:-len(found_quote)]
                     return f"{base}/{found_quote}"
                else:
                     # If ends with guessed default quote currency
                     if symbol.endswith(quote_currency):
                         base = symbol[:-len(quote_currency)]
                         return f"{base}/{quote_currency}"
                     else:
                         # Cannot reliably format, return as is
                         logger.warning(f"Could not determine slash format for '{symbol}' on {exchange_id}. Returning original.")
                         return symbol
            else:
                # Already has a slash, assume it's correct
                return symbol


    async def fetch_usdc_pairs(self, exchange_id="binance"):
        """Fetch all USDC (or configured quote currency) trading pairs from the exchange."""
        try:
            # Fixed indentation: Moved these checks out one level
            if exchange_id not in self.exchanges:
                logger.error(f"Exchange {exchange_id} not initialized")
                return []

            exchange = self.exchanges[exchange_id]
            if not exchange:
                logger.error(f"Exchange object for {exchange_id} is invalid.")
                return []

            # Get quote currency, default to USDC
            quote_currency = "USDC"
            config = self.exchange_configs.get(exchange_id)
            if config and hasattr(config, 'default_quote_currency'):
                quote_currency = config.default_quote_currency
            logger.info(f"Fetching pairs with quote currency: {quote_currency} on {exchange_id}")

            # Fetch all tickers for filtering
            # Use fetch_tickers method which includes rate limiting
            all_tickers = await self.fetch_tickers(exchange_id=exchange_id)
            if not all_tickers or 'error' in all_tickers:
                 logger.error(f"Failed to fetch tickers for {exchange_id} to find {quote_currency} pairs. Error: {all_tickers.get('error', 'Unknown')}")
                 return []

            logger.debug(f"Fetched {len(all_tickers)} tickers from {exchange_id} for filtering.")

            # Filter for quote currency pairs
            quote_pairs = []
            prefixed_pairs_info = {} # Store info about prefixed pairs found

            for symbol, ticker_data in all_tickers.items():
                # Handle different formats (with or without slash)
                base = None
                quote = None
                is_target_quote = False

                if '/' in symbol:
                    try:
                        base, quote = symbol.split('/')
                        if quote == quote_currency:
                            is_target_quote = True
                    except ValueError:
                        logger.warning(f"Could not parse symbol format: {symbol}")
                        continue
                elif symbol.endswith(quote_currency):
                     # Check it's not just the quote currency itself (e.g. 'USDC')
                     if len(symbol) > len(quote_currency):
                        base = symbol[:-len(quote_currency)]
                        quote = quote_currency
                        is_target_quote = True

                if is_target_quote and base:
                    # Add the symbol format found on the exchange
                    quote_pairs.append(symbol)

                    # Check for numeric prefixes (e.g., 1000PEPE/USDC, 1INCH/USDC is usually okay)
                    # Use regex to find numeric prefix, but exclude known valid ones like '1INCH'
                    match = re.match(r'^(\d+)(\D.*)', base)
                    if match and base.upper() != '1INCH':
                        prefix, actual_base = match.groups()
                        prefixed_pairs_info[symbol] = actual_base # Store relationship

            # Deduplication logic (optional but good):
            # Prefer non-prefixed versions if both exist (e.g., prefer PEPE/USDC over 1000PEPE/USDC)
            final_pairs = []
            bases_added = set() # Keep track of base assets added (without prefix)

            # First add non-prefixed pairs
            for pair in quote_pairs:
                 if pair not in prefixed_pairs_info:
                     base_asset = pair.split('/')[0] if '/' in pair else pair[:-len(quote_currency)]
                     final_pairs.append(pair)
                     bases_added.add(base_asset)

            # Then add prefixed pairs ONLY if the non-prefixed base wasn't already added
            for pair in quote_pairs:
                 if pair in prefixed_pairs_info:
                      non_prefixed_base = prefixed_pairs_info[pair]
                      if non_prefixed_base not in bases_added:
                           final_pairs.append(pair)
                           # Optionally log this decision:
                           # logger.debug(f"Adding prefixed pair {pair} as non-prefixed {non_prefixed_base} not found.")
                      # else:
                           # logger.debug(f"Skipping prefixed pair {pair} as non-prefixed {non_prefixed_base} already included.")

            logger.info(f"Found {len(final_pairs)} active {quote_currency} pairs on {exchange_id} after deduplication.")
            if prefixed_pairs_info:
                logger.debug(f"Identified {len(prefixed_pairs_info)} pairs with potential numeric prefixes.")

            return final_pairs

        except Exception as e:
            logger.error(f"Error fetching {quote_currency} pairs from {exchange_id}: {str(e)}\n{traceback.format_exc()}")
            return []

    async def fetch_usdc_trading_pairs(self, exchange_id="binance"):
        """Fetch all USDC (or quote currency) trading pairs, formatted with / for consistency."""
        try:
            # First get pairs in the format the exchange provided
            exchange_format_pairs = await self.fetch_usdc_pairs(exchange_id)
            if not exchange_format_pairs:
                return []

            # Convert to standard format with / for internal use
            formatted_pairs = []
            quote_currency = "USDC" # Get quote currency again for formatting
            config = self.exchange_configs.get(exchange_id)
            if config and hasattr(config, 'default_quote_currency'):
                 quote_currency = config.default_quote_currency

            for pair in exchange_format_pairs:
                if '/' not in pair:
                    # Ensure it ends with the quote currency before formatting
                    if pair.endswith(quote_currency) and len(pair) > len(quote_currency):
                        base = pair[:-len(quote_currency)]
                        formatted_pair = f"{base}/{quote_currency}"
                        formatted_pairs.append(formatted_pair)
                    else:
                         logger.warning(f"Skipping formatting for unexpected pair format: {pair}")
                else:
                    # Already has slash, assume correct format
                    formatted_pairs.append(pair)

            logger.info(f"Formatted {len(formatted_pairs)} {quote_currency} pairs with / separator for internal use.")
            return formatted_pairs

        except Exception as e:
            logger.error(f"Error formatting {quote_currency} trading pairs: {str(e)}\n{traceback.format_exc()}")
            return []

    async def fetch_current_prices(self, pairs: List[str], exchange_id="binance"):
        """
        Fetch current prices and 24h data for a list of trading pairs.

        Args:
            pairs: List of trading pairs (standard '/' format preferred)
            exchange_id: Exchange ID to use

        Returns:
            Dictionary keyed by standard pair format ('BASE/QUOTE'),
            with price, volume, change data or error info.
        """
        result = {}
        if not pairs:
            return result

        try:
            # Get quote currency for potential formatting help
            quote_currency = "USDC"
            config = self.exchange_configs.get(exchange_id)
            if config and hasattr(config, 'default_quote_currency'):
                quote_currency = config.default_quote_currency

            # Fixed indentation: Moved these checks out one level
            if exchange_id not in self.exchanges:
                logger.error(f"Exchange {exchange_id} not initialized")
                # Populate result with errors for all requested pairs
                for pair in pairs:
                     result[pair] = {'error': f"Exchange {exchange_id} not found"}
                return result

            exchange = self.exchanges[exchange_id]

            # Fetch all tickers in a single call for efficiency
            # Use the method that includes rate limiting and error handling
            all_tickers = await self.fetch_tickers(exchange_id=exchange_id)

            if not all_tickers or 'error' in all_tickers:
                logger.error(f"Failed to fetch tickers from {exchange_id}. Cannot get prices. Error: {all_tickers.get('error', 'Unknown')}")
                for pair in pairs:
                     std_pair_key = self.standardize_symbol_format(pair, quote_currency)
                     result[std_pair_key] = {'error': f"Failed to fetch tickers from {exchange_id}"}
                return result

            # Fixed indentation: Moved the loop out one level
            for pair in pairs:
                # Ensure we use the standard 'BASE/QUOTE' format for the result key
                standard_pair_key = self.standardize_symbol_format(pair, quote_currency)
                # Get the format required by the exchange for lookup
                exchange_pair_key = self.get_exchange_symbol_format(standard_pair_key, exchange_id)

                try:
                    ticker = None
                    # Try the expected exchange format first
                    if exchange_pair_key in all_tickers:
                        ticker = all_tickers[exchange_pair_key]
                    # Fallback: try the standard format (in case exchange returns it sometimes)
                    elif standard_pair_key in all_tickers:
                        ticker = all_tickers[standard_pair_key]
                        logger.debug(f"Found ticker for {pair} using standard key {standard_pair_key} instead of exchange key {exchange_pair_key}")
                    # Fallback: try original input format
                    elif pair in all_tickers and pair != standard_pair_key and pair != exchange_pair_key:
                         ticker = all_tickers[pair]
                         logger.debug(f"Found ticker for {pair} using original input key {pair}")


                    # Default error response
                    default_response = {
                        'price': 0.0, 'volume_24h': 0.0, 'change_24h': 0.0,
                        'high_24h': 0.0, 'low_24h': 0.0,
                        'error': 'Ticker data not found in fetched list'
                    }

                    if ticker is not None:
                        # Safely extract data, providing defaults for missing keys or None values
                        last_price = ticker.get('last')

                        if last_price is not None:
                            # Handle percentage which might be None
                            percentage = ticker.get('percentage', 0.0) # Default to 0.0
                            percentage = percentage if percentage is not None else 0.0

                            result[standard_pair_key] = {
                                'price': float(last_price),
                                'volume_24h': float(ticker.get('quoteVolume', 0.0)), # Prefer quoteVolume
                                'change_24h': float(percentage), # Use percentage directly
                                'high_24h': float(ticker.get('high', 0.0)),
                                'low_24h': float(ticker.get('low', 0.0))
                                # 'error': None # Indicate success implicitly
                            }
                        else:
                            logger.warning(f"Ticker found for {pair} ({exchange_pair_key}) but 'last' price is missing or None.")
                            result[standard_pair_key] = default_response
                            result[standard_pair_key]['error'] = "Ticker found but price is missing"
                    else:
                        logger.warning(f"No ticker data found for {pair} (tried keys: {exchange_pair_key}, {standard_pair_key}, {pair})")
                        result[standard_pair_key] = default_response

                except Exception as e:
                    logger.error(f"Error processing ticker data for {pair}: {str(e)}")
                    result[standard_pair_key] = {
                        'price': 0.0, 'volume_24h': 0.0, 'change_24h': 0.0,
                        'high_24h': 0.0, 'low_24h': 0.0,
                        'error': str(e)
                    }

            return result

        except Exception as e:
            logger.error(f"General error in fetch_current_prices: {str(e)}\n{traceback.format_exc()}")
            # Return errors for all requested pairs if a major failure occurs
            final_result = {}
            quote_currency = "USDC" # Guess quote currency for formatting
            config = self.exchange_configs.get(exchange_id if 'exchange_id' in locals() else 'binance')
            if config and hasattr(config, 'default_quote_currency'):
                 quote_currency = config.default_quote_currency
            for p in pairs if 'pairs' in locals() else []:
                 std_key = self.standardize_symbol_format(p, quote_currency)
                 final_result[std_key] = {'error': f"Failed due to: {str(e)}"}
            return final_result

    def standardize_symbol_format(self, symbol, quote_currency):
        """Ensures a symbol is in 'BASE/QUOTE' format."""
        if '/' in symbol:
            return symbol.upper() # Already correct format, ensure uppercase
        elif symbol.endswith(quote_currency) and len(symbol) > len(quote_currency):
            base = symbol[:-len(quote_currency)]
            return f"{base.upper()}/{quote_currency.upper()}"
        else:
            # Cannot reliably format, return uppercase original
            logger.warning(f"Could not standardize symbol format for '{symbol}' with quote '{quote_currency}'. Returning original.")
            return symbol.upper()


    async def close(self):
        """Close all connections gracefully."""
        self.logger.info("Closing Exchange Manager connections...")
        closed_exchanges = []
        closed_apis = []
        try:
            # Close CCXT and simulated exchange connections
            for name, exchange in self.exchanges.items():
                try:
                    # CCXT and our simulated class both have close methods
                    if hasattr(exchange, 'close') and asyncio.iscoroutinefunction(exchange.close):
                        await exchange.close()
                        self.logger.info(f"Closed connection for exchange: {name}")
                        closed_exchanges.append(name)
                    elif hasattr(exchange, 'close'): # Non-async close
                         exchange.close()
                         self.logger.info(f"Closed connection for exchange: {name}")
                         closed_exchanges.append(name)

                except Exception as e:
                    self.logger.error(f"Error closing exchange {name}: {str(e)}")

            # Close direct API connections if they exist
            if hasattr(self, 'direct_apis'):
                for name, api in self.direct_apis.items():
                    try:
                        if hasattr(api, 'close') and asyncio.iscoroutinefunction(api.close):
                            await api.close()
                            self.logger.info(f"Closed direct API connection: {name}")
                            closed_apis.append(name)
                        elif hasattr(api, 'close'): # Non-async close
                            api.close()
                            self.logger.info(f"Closed direct API connection: {name}")
                            closed_apis.append(name)
                    except Exception as e:
                        self.logger.error(f"Error closing direct API {name}: {str(e)}")

            self.logger.info(f"Finished closing connections. Exchanges closed: {closed_exchanges}. Direct APIs closed: {closed_apis}.")

        except Exception as e:
            # Catch errors during the closing loop itself
            self.logger.error(f"Error during exchange manager close process: {str(e)}\n{traceback.format_exc()}")


    async def get_exchange_credentials(self, exchange_id: str) -> Dict[str, str]:
        """Get credentials for a specific exchange from settings or environment."""
        # Prioritize credentials loaded during init
        if self.exchange_credentials and exchange_id in self.exchange_credentials:
            return self.exchange_credentials[exchange_id]

        # Fallback to environment variables (example format)
        api_key_env = f"{exchange_id.upper()}_API_KEY"
        secret_key_env = f"{exchange_id.upper()}_SECRET_KEY"
        password_env = f"{exchange_id.upper()}_PASSWORD" # If needed

        api_key = os.environ.get(api_key_env, '')
        secret_key = os.environ.get(secret_key_env, '')
        password = os.environ.get(password_env, '') # Often None/empty

        if api_key and secret_key:
             creds = {'api_key': api_key, 'secret_key': secret_key}
             if password:
                 creds['password'] = password
             # Optionally cache these if found via env vars
             if not self.exchange_credentials: self.exchange_credentials = {}
             self.exchange_credentials[exchange_id] = creds
             self.logger.info(f"Loaded credentials for {exchange_id} from environment variables.")
             return creds
        else:
            self.logger.warning(f"Credentials not found for {exchange_id} in settings or environment ({api_key_env}, {secret_key_env}).")
            return {}


    async def fetch_high_volume_usdc_pairs(self, limit=40, exchange_id="binance"):
        """
        Fetch trading pairs with configured quote currency, sorted by 24h volume.

        Args:
            limit: Maximum number of pairs to return.
            exchange_id: Exchange to query.

        Returns:
            List of high volume trading pairs in standard 'BASE/QUOTE' format.
        """
        try:
            # Get quote currency
            quote_currency = "USDC"
            config = self.exchange_configs.get(exchange_id)
            if config and hasattr(config, 'default_quote_currency'):
                quote_currency = config.default_quote_currency

            self.logger.info(f"Fetching top {limit} {quote_currency} trading pairs by 24h volume from {exchange_id}")

            # Fetch all tickers using our method (handles errors, rate limits)
            all_tickers = await self.fetch_tickers(exchange_id=exchange_id)

            if not all_tickers or 'error' in all_tickers:
                self.logger.error(f"Failed to fetch tickers for {exchange_id} to determine high volume pairs. Error: {all_tickers.get('error', 'Unknown')}")
                return []

            # Filter for quote currency pairs and extract volume data
            quote_pairs_with_volume = []
            for symbol, ticker in all_tickers.items():
                # Ensure ticker is a dict and has necessary keys
                if not isinstance(ticker, dict): continue

                standard_symbol = None
                is_target_quote = False

                # Check if it's a target quote pair and standardize format
                if '/' in symbol:
                    base, quote = symbol.split('/')
                    if quote == quote_currency:
                        is_target_quote = True
                        standard_symbol = symbol # Already standard format
                elif symbol.endswith(quote_currency) and len(symbol) > len(quote_currency):
                    base = symbol[:-len(quote_currency)]
                    quote = quote_currency
                    is_target_quote = True
                    standard_symbol = f"{base}/{quote}" # Create standard format

                if is_target_quote and standard_symbol:
                    # Safely get quoteVolume, fallback to calculating from baseVolume * last
                    volume = ticker.get('quoteVolume') # Prefer quote volume

                    if volume is None or volume <= 0: # If quoteVolume is missing or zero, try calculation
                        last_price = ticker.get('last')
                        base_volume = ticker.get('volume')
                        if last_price is not None and base_volume is not None and last_price > 0 and base_volume > 0:
                            volume = base_volume * last_price
                        else:
                            volume = 0 # Default to zero if calculation isn't possible

                    # Only add if volume is positive
                    if volume > 0:
                        quote_pairs_with_volume.append({
                            'symbol': standard_symbol, # Use standard format
                            'volume': volume
                        })
                    # else:
                         # logger.debug(f"Skipping {standard_symbol} due to zero or invalid volume.")


            # Sort by volume (highest first)
            quote_pairs_with_volume.sort(key=lambda x: x['volume'], reverse=True)

            # Apply limit and return only the symbols
            top_pairs = [pair['symbol'] for pair in quote_pairs_with_volume[:limit]]

            self.logger.info(f"Found {len(top_pairs)} high volume {quote_currency} pairs (out of {len(quote_pairs_with_volume)} total with volume > 0).")
            return top_pairs

        except Exception as e:
            self.logger.error(f"Error fetching high volume pairs from {exchange_id}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []


    async def fetch_balance(self, exchange_id="binance"):
        """
        Fetch balance information from the specified exchange.

        Args:
            exchange_id: Exchange ID to use (default is binance)

        Returns:
            Dictionary with balance information (free, used, total) or empty on error.
        """
        default_balance = {'free': {}, 'used': {}, 'total': {}}
        try:
            # Ensure exchange is initialized
            if exchange_id not in self.exchanges:
                logger.error(f"Exchange {exchange_id} not initialized")
                return default_balance

            exchange = self.exchanges[exchange_id]
            is_simulated = isinstance(exchange, Exchange)

            # Handle rate limits for real exchanges
            if not is_simulated and hasattr(exchange, 'check_response_headers'):
                await self._handle_rate_limit(exchange_id)

            # Call the exchange's fetch_balance method
            # Add timeout to prevent hanging
            balance = await asyncio.wait_for(exchange.fetch_balance(), timeout=30)

            # CCXT balance often includes 'info', 'timestamp', etc.
            # We only need free, used, total typically.
            # Also filter out zero balances if desired (optional)
            filtered_balance = {
                 'free': {k: v for k, v in balance.get('free', {}).items() if v > 0},
                 'used': {k: v for k, v in balance.get('used', {}).items() if v > 0},
                 'total': {k: v for k, v in balance.get('total', {}).items() if v > 0}
            }
            logger.info(f"Fetched balance from {exchange_id}. Total assets: {len(filtered_balance['total'])}")
            return filtered_balance

        except asyncio.TimeoutError:
             logger.error(f"Timeout fetching balance from {exchange_id}.")
             return default_balance
        except (ccxt.AuthenticationError, ccxt.ExchangeError) as e:
             logger.error(f"Authentication/Exchange error fetching balance from {exchange_id}: {str(e)}")
             return default_balance
        except ccxt.NetworkError as e:
             logger.error(f"Network error fetching balance from {exchange_id}: {str(e)}")
             return default_balance
        except Exception as e:
            logger.error(f"Unexpected error fetching balance from {exchange_id}: {str(e)}\n{traceback.format_exc()}")
            return default_balance

    async def fetch_positions(self, exchange_id="binance"):
        """
        Fetch position information from the specified exchange.
        
        Args:
            exchange_id: Exchange ID to use (default is binance)
            
        Returns:
            List or dict of positions or empty on error
        """
        try:
            # Ensure exchange is initialized
            if exchange_id not in self.exchanges:
                logger.error(f"Exchange {exchange_id} not initialized")
                return []

            exchange = self.exchanges[exchange_id]
            is_simulated = isinstance(exchange, Exchange)

            # Handle rate limits for real exchanges
            if not is_simulated and hasattr(exchange, 'check_response_headers'):
                await self._handle_rate_limit(exchange_id)

            # Check if the exchange's fetch_positions method exists and is callable
            if hasattr(exchange, 'fetch_positions') and callable(getattr(exchange, 'fetch_positions')):
                # Add timeout to prevent hanging
                positions = await asyncio.wait_for(exchange.fetch_positions(), timeout=30)
                logger.info(f"Fetched {len(positions)} positions from {exchange_id}")
                return positions
            elif is_simulated:
                # For simulated exchange, return empty positions
                logger.info(f"Simulated exchange {exchange_id} has no positions")
                return []
            else:
                # Method doesn't exist, try to fetch through private API if available
                logger.warning(f"Exchange {exchange_id} doesn't support fetch_positions directly")
                
                # Check for specific exchange implementations
                if exchange_id.lower() == 'binance':
                    try:
                        # Try to use fetchMyTrades or similar endpoints
                        recent_trades = await asyncio.wait_for(exchange.fetch_my_trades(), timeout=30)
                        logger.info(f"Fetched {len(recent_trades)} recent trades as position approximation")
                        return recent_trades
                    except Exception as trade_e:
                        logger.error(f"Failed to fetch trades as position fallback: {str(trade_e)}")
                
                return []

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching positions from {exchange_id}")
            return []
        except (ccxt.AuthenticationError, ccxt.ExchangeError) as e:
            logger.error(f"Authentication/Exchange error fetching positions from {exchange_id}: {str(e)}")
            return []
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching positions from {exchange_id}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching positions from {exchange_id}: {str(e)}\n{traceback.format_exc()}")
            return []
