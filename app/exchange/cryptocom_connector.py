import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime, timedelta

from app.config.settings import Settings, AppMode
from app.config.cryptocom_config import CryptocomConfig, CryptocomMode
from app.exchange.manager import ExchangeManager

logger = logging.getLogger(__name__)

class CryptocomConnector:
    """
    Crypto.com-specific connector that provides enhanced functionality for Crypto.com exchange.
    Extends the base ExchangeManager with Crypto.com-specific methods for market data, trading, and analysis.
    """
    
    def __init__(self, exchange_manager: ExchangeManager, config: CryptocomConfig):
        """
        Initialize the Crypto.com connector with an existing ExchangeManager.
        
        Args:
            exchange_manager: An initialized ExchangeManager with Crypto.com connection
            config: Crypto.com-specific configuration
        """
        self.exchange_manager = exchange_manager
        self.exchange_id = "cryptocom"
        self.config = config
        logger.info("Crypto.com connector initialized")
    
    async def is_available(self) -> bool:
        """Check if Crypto.com exchange is available in the exchange manager."""
        return self.exchange_id in self.exchange_manager.exchanges
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get Crypto.com exchange information, including trading rules and limits."""
        if not await self.is_available():
            logger.error("Crypto.com exchange not initialized")
            return {}
        
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            # Use public API method to get exchange info
            exchange_info = {}
            
            # Get exchange status and info
            exchange_status = await exchange.fetch_status()
            if exchange_status:
                exchange_info["status"] = exchange_status
            
            # Get trading fees
            fees = await exchange.fetch_trading_fees()
            if fees:
                exchange_info["fees"] = fees
                
            return exchange_info
        except Exception as e:
            logger.error(f"Error fetching Crypto.com exchange info: {str(e)}")
            return {}
    
    async def get_usdc_trading_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Get trading rules specific to USDC pairs on Crypto.com.
        
        Returns:
            Dictionary mapping symbols to their trading rules
        """
        try:
            if not await self.is_available():
                logger.error("Crypto.com exchange not initialized")
                return {}
            
            exchange = self.exchange_manager.exchanges[self.exchange_id]
            
            # Use the quote currency from config
            quote_currency = self.config.default_quote_currency
            
            # Load markets if needed
            if not exchange.markets:
                await exchange.load_markets()
                
            markets = exchange.markets
            
            # Filter for USDC markets
            usdc_rules = {}
            for symbol, market in markets.items():
                # Check if this is a USDC pair
                if quote_currency in symbol:
                    # Create a dictionary of trading rules from market data
                    limits = market.get('limits', {})
                    precision = market.get('precision', {})
                    
                    usdc_rules[symbol] = {
                        'min_notional': limits.get('cost', {}).get('min', 0),
                        'min_qty': limits.get('amount', {}).get('min', 0),
                        'max_qty': limits.get('amount', {}).get('max', 0),
                        'step_size': precision.get('amount', 0),
                        'price_precision': precision.get('price', 0),
                        'qty_precision': precision.get('amount', 0),
                        'status': 'TRADING' if market.get('active', False) else 'INACTIVE'
                    }
                    
                    # Apply any trading rule overrides from config
                    if symbol in self.config.trading_rule_overrides:
                        usdc_rules[symbol].update(self.config.trading_rule_overrides[symbol])
            
            logger.info(f"Retrieved trading rules for {len(usdc_rules)} {quote_currency} pairs on Crypto.com")
            return usdc_rules
        except Exception as e:
            logger.error(f"Error getting {self.config.default_quote_currency} trading rules: {str(e)}")
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
            logger.error("Crypto.com exchange not initialized")
            return {}
        
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            
            # If symbols not provided, get all pairs with the configured quote currency
            if not symbols:
                symbols = await self.exchange_manager.fetch_usdc_trading_pairs(self.exchange_id)
            
            # Fetch all tickers at once (more efficient)
            all_tickers = await exchange.fetch_tickers()
            
            # Filter for requested symbols
            result = {symbol: all_tickers.get(symbol, {}) for symbol in symbols if symbol in all_tickers}
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
            logger.error("Crypto.com exchange not initialized")
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
        """Fetch trading fees for Crypto.com account."""
        if not await self.is_available():
            logger.error("Crypto.com exchange not initialized")
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
    
    async def set_leverage(self, symbol: str, leverage: Optional[int] = None) -> bool:
        """
        Set leverage for derivatives trading.
        
        Args:
            symbol: Trading pair symbol
            leverage: Leverage value. If None, uses default from config.
            
        Returns:
            Success status
        """
        if not await self.is_available():
            logger.error("Crypto.com exchange not initialized")
            return False
            
        # Check if we're in derivatives mode
        if self.config.trading_mode != CryptocomMode.DERIVATIVES:
            logger.warning("Exchange is not in derivatives mode, leverage setting not applicable")
            return False
            
        # Use default from config if not specified
        if leverage is None:
            leverage = self.config.default_leverage
            
        exchange = self.exchange_manager.exchanges[self.exchange_id]
        
        try:
            await self.exchange_manager._handle_rate_limit(self.exchange_id)
            
            # Different exchanges have different APIs for setting leverage
            # For Crypto.com, we'll use the appropriate method based on documentation
            response = await exchange.set_leverage(leverage, symbol=symbol)
            
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
            
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {str(e)}")
            return False 