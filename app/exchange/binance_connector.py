import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime, timedelta

from app.config.settings import Settings, AppMode
from app.config.binance_config import BinanceConfig, BinanceMode
from app.exchange.manager import ExchangeManager

logger = logging.getLogger(__name__)

class BinanceConnector:
    """
    Binance-specific connector that provides enhanced functionality for Binance exchange.
    Extends the base ExchangeManager with Binance-specific methods for market data, trading, and analysis.
    """
    
    def __init__(self, exchange_manager: ExchangeManager, config: BinanceConfig):
        """
        Initialize the Binance connector with an existing ExchangeManager.
        
        Args:
            exchange_manager: An initialized ExchangeManager with Binance connection
            config: Binance-specific configuration
        """
        self.exchange_manager = exchange_manager
        self.exchange_id = "binance"
        self.config = config
        
        # Initialize direct API if available
        self.direct_api = exchange_manager.binance_direct
        logger.info("Binance connector initialized")
    
    async def fetch_market_data(self, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch both OHLCV and current market data."""
        try:
            # Try direct API first
            if self.direct_api:
                try:
                    # Get OHLCV data
                    ohlcv_df = await self.direct_api.get_klines(symbol, '1m', 100)
                    
                    # Get 24h ticker data
                    ticker_24h = await self.direct_api.get_24h_ticker(symbol)
                    if ticker_24h:
                        current_data = {
                            'price': float(ticker_24h['lastPrice']),
                            'volume_24h': float(ticker_24h['volume']),
                            'change_24h': float(ticker_24h['priceChangePercent']),
                            'high_24h': float(ticker_24h['highPrice']),
                            'low_24h': float(ticker_24h['lowPrice'])
                        }
                        return ohlcv_df, current_data
                except Exception as e:
                    logger.warning(
                        f"Direct API failed for {symbol}, falling back to CCXT. "
                        f"Endpoints: [/api/v3/klines, /api/v3/ticker/24hr], Error: {str(e)}"
                    )
            
            # CCXT fallback
            return await self.exchange_manager.fetch_market_data(symbol)
            
        except Exception as e:
            logger.error(
                f"Market data fetch failed - Symbol: {symbol}, "
                f"Error: {str(e)}, Stack: {e.__traceback__}"
            )
            return pd.DataFrame(), {}
    
    async def create_order(self, symbol: str, trade_type: str, order_type: str,
                          quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Create a new order using direct API or CCXT fallback."""
        try:
            # Map our trade_type to Binance 'side' parameter
            side = trade_type.upper()
            
            # Try direct API first
            if self.direct_api:
                try:
                    response = await self.direct_api.create_order(
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        quantity=quantity,
                        price=price
                    )
                    
                    if not response or 'orderId' not in response:
                        logger.error(
                            f"Invalid order response - Endpoint: /api/v3/order, "
                            f"Symbol: {symbol}, Side: {side}, Type: {order_type}, "
                            f"Response: {response}"
                        )
                    
                    # Map Binance's response to use trade_type
                    if response and 'side' in response:
                        response['trade_type'] = response['side'].lower()
                    
                    return response
                    
                except Exception as e:
                    logger.warning(
                        f"Direct API order failed for {symbol}, falling back to CCXT. "
                        f"Endpoint: /api/v3/order, Error: {str(e)}"
                    )
            
            # CCXT fallback
            return await self.exchange_manager.create_order(
                symbol=symbol,
                order_type=order_type,
                trade_type=trade_type,
                amount=quantity,
                price=price,
                exchange_id=self.exchange_id
            )
            
        except Exception as e:
            logger.error(
                f"Order creation failed - Symbol: {symbol}, Side: {trade_type}, "
                f"Type: {order_type}, Quantity: {quantity}, Price: {price}, "
                f"Error: {str(e)}, Stack: {e.__traceback__}"
            )
            return {}
    
    async def start_price_stream(self, symbol: str, callback: Callable) -> None:
        """Start WebSocket price stream for real-time updates."""
        if self.direct_api:
            await self.direct_api.start_websocket(symbol, callback)
        else:
            logger.error("WebSocket streaming requires direct API implementation")
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get Binance exchange information using direct API or CCXT fallback."""
        try:
            if self.direct_api:
                try:
                    # Use direct API's _make_request method
                    return await self.direct_api._make_request('GET', '/v3/exchangeInfo')
                except Exception as e:
                    logger.warning(f"Direct API failed, falling back to CCXT: {str(e)}")
            
            # CCXT fallback
            exchange = self.exchange_manager.exchanges[self.exchange_id]
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            return await exchange.load_markets()
            
        except Exception as e:
            logger.error(f"Error fetching exchange info: {str(e)}")
            return {}
    
    async def get_usdc_trading_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get trading rules for USDC pairs using direct API or CCXT fallback."""
        try:
            exchange_info = await self.get_exchange_info()
            if not exchange_info or 'symbols' not in exchange_info:
                return {}
            
            # Use the quote currency from config
            quote_currency = self.config.default_quote_currency
            
            usdc_rules = {}
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info.get('symbol')
                
                # Check if this pair uses the configured quote currency
                if symbol and quote_currency in symbol:
                    # Convert to standard format
                    base, quote = None, None
                    
                    # Extract base and quote from Binance's format
                    for asset in symbol_info.get('baseAsset', ''), symbol_info.get('quoteAsset', ''):
                        if asset == quote_currency:
                            quote = asset
                        else:
                            base = asset
                    
                    if base and quote:
                        standard_symbol = f"{base}/{quote}"
                        
                        # Extract trading rules
                        usdc_rules[standard_symbol] = {
                            'min_notional': next((float(filter['minNotional']) for filter in symbol_info.get('filters', []) 
                                             if filter.get('filterType') == 'MIN_NOTIONAL'), None),
                            'min_qty': next((float(filter['minQty']) for filter in symbol_info.get('filters', []) 
                                         if filter.get('filterType') == 'LOT_SIZE'), None),
                            'max_qty': next((float(filter['maxQty']) for filter in symbol_info.get('filters', []) 
                                         if filter.get('filterType') == 'LOT_SIZE'), None),
                            'step_size': next((float(filter['stepSize']) for filter in symbol_info.get('filters', []) 
                                           if filter.get('filterType') == 'LOT_SIZE'), None),
                            'price_precision': symbol_info.get('pricePrecision', 8),
                            'qty_precision': symbol_info.get('quantityPrecision', 8),
                            'status': symbol_info.get('status', 'UNKNOWN')
                        }
            
            logger.info(f"Retrieved trading rules for {len(usdc_rules)} {quote_currency} pairs")
            return usdc_rules
            
        except Exception as e:
            logger.error(f"Error getting trading rules: {str(e)}")
            return {}
    
    async def fetch_24h_ticker_stats(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fetch 24-hour ticker statistics for specified symbols or all quote currency pairs.
        
        Args:
            symbols: Optional list of symbols to get stats for (e.g., ['BTC/USDC', 'ETH/USDC'])
                    If None, fetches all pairs with the configured quote currency
        
        Returns:
            Dictionary mapping symbols to their 24h stats
        """
        if not await self.is_available():
            logger.error("Binance exchange not initialized")
            return {}
        
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            
            # If symbols not provided, get all pairs with the configured quote currency
            if not symbols:
                symbols = await self.exchange_manager.fetch_usdc_trading_pairs(self.exchange_id)
            
            # Prepare for batch retrieval if possible
            if len(symbols) > 10:
                logger.info(f"Fetching 24h stats for {len(symbols)} symbols in batches")
                
                # Get all tickers at once (more efficient)
                all_tickers = await exchange.fetch_tickers()
                
                # Filter for requested symbols
                result = {symbol: all_tickers.get(symbol, {}) for symbol in symbols if symbol in all_tickers}
                return result
            else:
                # Fetch individual tickers for smaller sets
                result = {}
                for symbol in symbols:
                    await self.exchange_manager._handle_rate_limit(self.exchange_id)
                    ticker = await exchange.fetch_ticker(symbol)
                    result[symbol] = ticker
                
                return result
                
        except Exception as e:
            logger.error(f"Error fetching 24h ticker stats: {str(e)}")
            return {}
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch the order book for a specific symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDC')
            limit: Number of bids/asks to retrieve
            
        Returns:
            Order book data containing bids and asks
        """
        if not await self.is_available():
            logger.error("Binance exchange not initialized")
            return {}
        
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            order_book = await exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            return {}
    
    async def fetch_trading_fees(self) -> Dict[str, Any]:
        """Fetch trading fees for Binance account."""
        if not await self.is_available():
            logger.error("Binance exchange not initialized")
            return {}
        
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            # This requires authentication
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            fees = await exchange.fetch_trading_fees()
            return fees
        except Exception as e:
            logger.error(f"Error fetching trading fees: {str(e)}")
            return {}
    
    async def calculate_entry_exit_prices(self, symbol: str, 
                                        side: str, 
                                        amount: float, 
                                        slippage_percent: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate estimated entry and exit prices including slippage based on current order book.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDC')
            side: Order side ('buy' or 'sell')
            amount: Amount to buy/sell in base currency units
            slippage_percent: Expected slippage percentage (if None, uses default from config)
            
        Returns:
            Tuple of (entry_price, exit_price)
        """
        # Use config default if not specified
        if slippage_percent is None:
            slippage_percent = self.config.default_slippage_percent
            
        try:
            # Fetch order book
            order_book = await self.fetch_order_book(symbol, 100)
            
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                logger.error(f"Invalid order book data for {symbol}")
                return 0.0, 0.0
            
            bids = order_book['bids']  # Format: [price, amount]
            asks = order_book['asks']  # Format: [price, amount]
            
            # Calculate entry price with slippage
            if side == 'buy':
                # When buying, we'll take from the asks
                entry_price = self._calculate_avg_execution_price(asks, amount, 'buy', slippage_percent)
                # For exit, we'll be selling, so check the bids
                exit_price = self._calculate_avg_execution_price(bids, amount, 'sell', slippage_percent)
            else:  # sell
                # When selling, we'll take from the bids
                entry_price = self._calculate_avg_execution_price(bids, amount, 'sell', slippage_percent)
                # For exit, we'll be buying, so check the asks
                exit_price = self._calculate_avg_execution_price(asks, amount, 'buy', slippage_percent)
            
            return entry_price, exit_price
            
        except Exception as e:
            logger.error(f"Error calculating entry/exit prices for {symbol}: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_avg_execution_price(self, orders: List[List[float]], 
                                     amount: float, 
                                     side: str, 
                                     slippage_percent: float) -> float:
        """
        Calculate the average execution price for a given amount.
        
        Args:
            orders: List of [price, amount] from the order book
            amount: Amount to execute
            side: 'buy' or 'sell'
            slippage_percent: Expected slippage percentage
            
        Returns:
            Average execution price with slippage
        """
        remaining = amount
        total_cost = 0.0
        
        for price, order_amount in orders:
            if remaining <= 0:
                break
                
            executed = min(remaining, order_amount)
            total_cost += executed * price
            remaining -= executed
        
        if amount <= 0 or remaining == amount:
            # Couldn't fill the order from the order book
            return 0.0
        
        avg_price = total_cost / (amount - remaining)
        
        # Apply slippage
        if side == 'buy':
            # When buying, price goes up due to slippage
            avg_price *= (1 + slippage_percent / 100)
        else:
            # When selling, price goes down due to slippage
            avg_price *= (1 - slippage_percent / 100)
        
        return avg_price
    
    async def get_funding_rate(self, symbol: str) -> float:
        """
        Get current funding rate for a perpetual futures contract.
        (Only applicable for futures trading)
        
        Args:
            symbol: Futures trading pair symbol (e.g., 'BTC/USDC:USDC')
            
        Returns:
            Current funding rate as a percentage
        """
        if not await self.is_available():
            logger.error("Binance exchange not initialized")
            return 0.0
        
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            # Check if the exchange is in futures mode based on config
            if self.config.trading_mode not in [BinanceMode.FUTURES, BinanceMode.COIN_FUTURES]:
                logger.warning("Exchange is not in futures mode, funding rate not applicable")
                return 0.0
                
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            
            # Symbol may need conversion to Binance's format
            market = exchange.market(symbol)
            binance_symbol = market['id']
            
            # Fetch funding rate
            funding_rate_data = await exchange.fapiPublicGetFundingRate({'symbol': binance_symbol})
            
            if not funding_rate_data:
                return 0.0
                
            # Extract current funding rate
            current_rate = float(funding_rate_data[0]['fundingRate']) * 100  # Convert to percentage
            return current_rate
            
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {str(e)}")
            return 0.0
            
    async def set_leverage(self, symbol: str, leverage: Optional[int] = None) -> bool:
        """
        Set leverage for futures trading.
        
        Args:
            symbol: Trading pair symbol
            leverage: Leverage value (1-125). If None, uses default from config.
            
        Returns:
            Success status
        """
        if not await self.is_available():
            logger.error("Binance exchange not initialized")
            return False
            
        # Check if we're in futures mode
        if self.config.trading_mode not in [BinanceMode.FUTURES, BinanceMode.COIN_FUTURES]:
            logger.warning("Exchange is not in futures mode, leverage setting not applicable")
            return False
            
        # Use default from config if not specified
        if leverage is None:
            leverage = self.config.default_leverage
            
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            
            # Convert symbol to Binance format
            market = exchange.market(symbol)
            binance_symbol = market['id']
            
            # Set leverage
            response = await exchange.fapiPrivatePostLeverage({
                'symbol': binance_symbol,
                'leverage': leverage
            })
            
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
            
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {str(e)}")
            return False 