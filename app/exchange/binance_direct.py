import logging
import aiohttp
import json
import hmac
import hashlib
import time
import websockets
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class BinanceDirectAPI:
    """Direct implementation of Binance API endpoints."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, sandbox: bool = True):
        """Initialize Binance API with optional authentication."""
        self.api_key = api_key
        self.secret_key = api_secret  # Store as secret_key internally
        self.sandbox = sandbox
        
        # Set base URLs based on sandbox mode
        if sandbox:
            self.base_url = "https://testnet.binance.vision/api"
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com/api"
            self.ws_url = "wss://stream.binance.com:9443/ws"
        
        # Initialize session
        self.session = None
        self.ws_connections = {}
        logger.info(f"Initialized BinanceDirectAPI in {'sandbox' if sandbox else 'production'} mode")
    
    async def _init_session(self):
        """Initialize aiohttp session if not exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for authenticated endpoints."""
        query_string = urlencode(params)
        return hmac.new(
            self.secret_key.encode('utf-8'),  # Use secret_key here
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_request(self, method: str, endpoint: str, 
                          params: Dict[str, Any] = None, 
                          auth: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Binance API."""
        await self._init_session()
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if auth:
            if not self.api_key or not self.secret_key:  # Check secret_key here
                raise ValueError("API key and secret required for authenticated endpoints")
            
            headers['X-MBX-APIKEY'] = self.api_key
            params = params or {}
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            if method == 'GET':
                if params:
                    url += '?' + urlencode(params)
                async with self.session.get(url, headers=headers) as response:
                    response_data = await response.json()
                    
                    # Log API call details if response status is not successful
                    if response.status != 200:
                        logger.error(
                            f"API Error - Endpoint: {endpoint}, Method: {method}, "
                            f"Status: {response.status}, Response: {response_data}, "
                            f"Params: {params if not auth else '[REDACTED]'}"
                        )
                    return response_data
            else:  # POST
                async with self.session.post(url, headers=headers, data=params) as response:
                    response_data = await response.json()
                    
                    # Log API call details if response status is not successful
                    if response.status != 200:
                        logger.error(
                            f"API Error - Endpoint: {endpoint}, Method: {method}, "
                            f"Status: {response.status}, Response: {response_data}, "
                            f"Params: {params if not auth else '[REDACTED]'}"
                        )
                    return response_data
        
        except Exception as e:
            logger.error(
                f"API Request Failed - Endpoint: {endpoint}, Method: {method}, "
                f"Error: {str(e)}, URL: {url}, "
                f"Params: {params if not auth else '[REDACTED]'}"
            )
            raise
    
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 1000,
                        start_time: Optional[int] = None, 
                        end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch klines/candlestick data using /api/v3/klines endpoint.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDC')
            interval: Timeframe (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles (max 1000)
            start_time: Optional start time in milliseconds
            end_time: Optional end time in milliseconds
        """
        try:
            # Convert symbol to Binance format
            binance_symbol = symbol.replace('/', '')
            
            # Validate symbol format
            if not self._is_valid_symbol_format(binance_symbol):
                logger.warning(f"Potentially invalid symbol format: {binance_symbol}")
                # Try alternative format (some symbols might need special handling)
                if 'USDC' in binance_symbol:
                    # Check if we need to add T for USD-settled futures
                    if not binance_symbol.endswith('USDC'):
                        logger.warning(f"Symbol doesn't end with USDC as expected: {binance_symbol}")
            
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            try:
                data = await self._make_request('GET', '/v3/klines', params)
            except Exception as e:
                if "Invalid symbol" in str(e):
                    logger.warning(f"Invalid symbol detected: {binance_symbol}, original: {symbol}")
                    # Let's try a couple of alternative formats
                    try:
                        if 'USDC' in binance_symbol:
                            # Some exchanges use USDT for everything
                            alt_symbol = binance_symbol.replace('USDC', 'USDT')
                            logger.info(f"Trying alternative symbol format: {alt_symbol}")
                            params['symbol'] = alt_symbol
                            data = await self._make_request('GET', '/v3/klines', params)
                        else:
                            raise ValueError(f"Unsupported symbol: {symbol}")
                    except Exception as alt_error:
                        logger.error(f"Alternative symbol also failed: {str(alt_error)}")
                        return pd.DataFrame()
                else:
                    # Re-raise other errors
                    raise e
            
            if not isinstance(data, list):
                logger.error(
                    f"Invalid klines response - Endpoint: /api/v3/klines, "
                    f"Symbol: {symbol}, Response: {data}"
                )
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(
                f"Klines request failed - Endpoint: /api/v3/klines, "
                f"Symbol: {symbol}, Interval: {interval}, Error: {str(e)}"
            )
            return pd.DataFrame()
    
    def _is_valid_symbol_format(self, symbol: str) -> bool:
        """Check if the symbol appears to be in a valid format for Binance API."""
        # Basic validation - at least 5 chars, typically contains the quote currency
        if len(symbol) < 5:
            return False
        
        # Most symbols on Binance end with common quote currencies
        common_quote_currencies = ['USDT', 'USDC', 'BTC', 'ETH', 'BNB', 'BUSD']
        for quote in common_quote_currencies:
            if symbol.endswith(quote):
                return True
        
        # If we're here, it doesn't match common patterns
        return False
    
    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price for a symbol using /api/v3/ticker/price endpoint.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDC')
        """
        try:
            binance_symbol = symbol.replace('/', '')
            
            # Validate symbol format
            if not self._is_valid_symbol_format(binance_symbol):
                logger.warning(f"Potentially invalid symbol format in ticker request: {binance_symbol}")
            
            try:
                response = await self._make_request('GET', '/v3/ticker/price', 
                                                 {'symbol': binance_symbol})
            except Exception as e:
                if "Invalid symbol" in str(e):
                    logger.warning(f"Invalid symbol detected in ticker request: {binance_symbol}, original: {symbol}")
                    # Try alternative format
                    try:
                        if 'USDC' in binance_symbol:
                            alt_symbol = binance_symbol.replace('USDC', 'USDT')
                            logger.info(f"Trying alternative symbol format for ticker: {alt_symbol}")
                            response = await self._make_request('GET', '/v3/ticker/price', 
                                                         {'symbol': alt_symbol})
                        else:
                            raise ValueError(f"Unsupported symbol: {symbol}")
                    except Exception as alt_error:
                        logger.error(f"Alternative symbol for ticker also failed: {str(alt_error)}")
                        return {}
                else:
                    # Re-raise other errors
                    raise e
            
            if 'price' not in response:
                logger.error(
                    f"Invalid ticker price response - Endpoint: /api/v3/ticker/price, "
                    f"Symbol: {symbol}, Response: {response}"
                )
            return response
            
        except Exception as e:
            logger.error(
                f"Ticker price request failed - Endpoint: /api/v3/ticker/price, "
                f"Symbol: {symbol}, Error: {str(e)}"
            )
            return {}
    
    async def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24-hour price statistics using /api/v3/ticker/24hr endpoint.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDC')
        """
        try:
            binance_symbol = symbol.replace('/', '')
            
            # Validate symbol format
            if not self._is_valid_symbol_format(binance_symbol):
                logger.warning(f"Potentially invalid symbol format in 24h ticker request: {binance_symbol}")
            
            try:
                response = await self._make_request('GET', '/v3/ticker/24hr', 
                                                 {'symbol': binance_symbol})
            except Exception as e:
                if "Invalid symbol" in str(e):
                    logger.warning(f"Invalid symbol detected in 24h ticker request: {binance_symbol}, original: {symbol}")
                    # Try alternative format
                    try:
                        if 'USDC' in binance_symbol:
                            alt_symbol = binance_symbol.replace('USDC', 'USDT')
                            logger.info(f"Trying alternative symbol format for 24h ticker: {alt_symbol}")
                            response = await self._make_request('GET', '/v3/ticker/24hr', 
                                                         {'symbol': alt_symbol})
                        else:
                            raise ValueError(f"Unsupported symbol: {symbol}")
                    except Exception as alt_error:
                        logger.error(f"Alternative symbol for 24h ticker also failed: {str(alt_error)}")
                        return {}
                else:
                    # Re-raise other errors
                    raise e
            
            if not isinstance(response, dict) or 'lastPrice' not in response:
                logger.error(
                    f"Invalid 24h ticker response - Endpoint: /api/v3/ticker/24hr, "
                    f"Symbol: {symbol}, Response: {response}"
                )
            return response
            
        except Exception as e:
            logger.error(
                f"24h ticker request failed - Endpoint: /api/v3/ticker/24hr, "
                f"Symbol: {symbol}, Error: {str(e)}"
            )
            return {}
    
    async def create_order(self, symbol: str, trade_type: str, order_type: str,
                         quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a new order using /api/v3/order endpoint.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDC')
            trade_type: 'buy' or 'sell' 
            order_type: 'MARKET' or 'LIMIT'
            quantity: Amount of base asset
            price: Required for LIMIT orders
        """
        try:
            binance_symbol = symbol.replace('/', '')
            
            # Validate symbol format
            if not self._is_valid_symbol_format(binance_symbol):
                logger.warning(f"Potentially invalid symbol format in order: {binance_symbol}")
            
            # Map our trade_type to Binance's 'side' parameter
            binance_side = trade_type.upper()
            
            params = {
                'symbol': binance_symbol,
                'side': binance_side,
                'type': order_type.upper(),
                'quantity': quantity
            }
            
            if order_type.upper() == 'LIMIT':
                if not price:
                    raise ValueError("Price required for LIMIT orders")
                params['price'] = price
                params['timeInForce'] = 'GTC'
            
            timestamp = int(time.time() * 1000)
            params['timestamp'] = timestamp
            
            # Add signature
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
            # Make the request
            endpoint = '/api/v3/order'
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            response = await self._make_request('POST', endpoint, params, auth=True)
            result = await response.json()
            
            # Map response to use trade_type
            if 'side' in result:
                result['trade_type'] = result['side'].lower()
            
            return result
        
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            return {'error': str(e)}
    
    async def start_websocket(self, symbol: str, callback: Callable) -> None:
        """
        Start WebSocket connection for real-time trade updates.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDC')
            callback: Async function to handle incoming messages
        """
        binance_symbol = symbol.lower().replace('/', '')
        ws_endpoint = f"{self.ws_url}/{binance_symbol}@trade"
        
        try:
            async with websockets.connect(ws_endpoint) as websocket:
                self.ws_connections[symbol] = websocket
                logger.info(f"WebSocket connected for {symbol}")
                
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        await callback({
                            'symbol': symbol,
                            'price': float(data['p']),
                            'quantity': float(data['q']),
                            'timestamp': data['T'],
                            'is_buyer_maker': data['m']
                        })
                    except Exception as e:
                        logger.error(f"WebSocket error for {symbol}: {str(e)}")
                        break
        
        except Exception as e:
            logger.error(f"Error establishing WebSocket connection for {symbol}: {str(e)}")
        finally:
            if symbol in self.ws_connections:
                del self.ws_connections[symbol]
    
    async def close(self):
        """Close all connections."""
        if self.session:
            await self.session.close()
            self.session = None
        
        # Close any open WebSocket connections
        for symbol, ws in self.ws_connections.items():
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket for {symbol}: {str(e)}")
        self.ws_connections.clear() 