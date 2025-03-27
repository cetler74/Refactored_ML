#!/usr/bin/env python3
"""
Trading Pair Selector Module

This module is responsible for selecting trading pairs based on various criteria including:
- Volume
- Liquidity
- Volatility
- Market cap
- Price trends

It fetches market data from exchanges and applies filters to select the best pairs for trading.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)

class TradingPairSelector:
    """
    Selects trading pairs for trading based on various market criteria.
    """
    
    def __init__(self, settings, max_pairs=10):
        """
        Initialize the TradingPairSelector.
        
        Args:
            settings: Application settings
            max_pairs: Maximum number of pairs to select
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.max_pairs = max_pairs
        self.selected_pairs = []
        self.pair_details = {}
        self.market_data = {}
        self.blacklisted_pairs = []
        self.mode = settings.APP_MODE.lower()
        
        # Configuration for pair selection
        self.min_volume_usd = 5_000_000  # Minimum 24h volume in USD
        self.min_volatility = 1.0        # Minimum volatility (as std dev %)
        self.max_volatility = 15.0       # Maximum volatility (as std dev %)
        self.min_liquidity_ratio = 0.05  # Minimum ratio of volume to market cap
        
        # Base currency is USDT for both simulation and live modes
        self.base_currency = "USDT"
        
        self.logger.info(f"TradingPairSelector initialized with max_pairs={max_pairs}")
    
    async def fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch market data from Binance for all available trading pairs with USDT.
        
        Returns:
            Dictionary mapping pair symbols to their market data
        """
        self.logger.info("Fetching market data from Binance...")
        
        try:
            # Binance API endpoints
            base_url = "https://api.binance.com/api/v3"
            exchange_info_url = f"{base_url}/exchangeInfo"
            
            # Store market data
            market_data = {}
            
            async with aiohttp.ClientSession() as session:
                # Get exchange info to identify all USDT pairs
                async with session.get(exchange_info_url) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch exchange info: {response.status}")
                        return {}
                    
                    exchange_info = await response.json()
                    
                    # Filter for USDT pairs that are trading
                    usdt_pairs = []
                    for symbol_info in exchange_info.get("symbols", []):
                        symbol = symbol_info.get("symbol", "")
                        quote_asset = symbol_info.get("quoteAsset", "")
                        status = symbol_info.get("status", "")
                        
                        if quote_asset == self.base_currency and status == "TRADING":
                            base_asset = symbol_info.get("baseAsset", "")
                            pair_notation = f"{base_asset}/{self.base_currency}"
                            binance_notation = symbol
                            
                            usdt_pairs.append({
                                "symbol": pair_notation,
                                "binance_symbol": binance_notation,
                                "base_asset": base_asset,
                                "quote_asset": quote_asset
                            })
                
                # Get 24hr ticker data for all USDT pairs
                self.logger.info(f"Fetching data for {len(usdt_pairs)} USDT pairs...")
                ticker_url = f"{base_url}/ticker/24hr"
                
                async with session.get(ticker_url) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch ticker data: {response.status}")
                        return {}
                    
                    all_tickers = await response.json()
                    
                    # Create a lookup dict for easy access
                    ticker_lookup = {t.get("symbol", ""): t for t in all_tickers}
                    
                    # Process data for each USDT pair
                    for pair in usdt_pairs:
                        binance_symbol = pair["binance_symbol"]
                        ticker_data = ticker_lookup.get(binance_symbol)
                        
                        if not ticker_data:
                            continue
                        
                        # Extract key metrics
                        try:
                            price = float(ticker_data.get("lastPrice", 0))
                            volume_24h = float(ticker_data.get("volume", 0)) * price  # Volume in USDT
                            price_change_pct = float(ticker_data.get("priceChangePercent", 0))
                            high_24h = float(ticker_data.get("highPrice", 0))
                            low_24h = float(ticker_data.get("lowPrice", 0))
                            
                            # Calculate volatility (simple high-low range as %)
                            if high_24h > 0 and low_24h > 0:
                                volatility = ((high_24h - low_24h) / low_24h) * 100
                            else:
                                volatility = 0
                            
                            # Calculate liquidity score (higher volume = higher liquidity)
                            # In a real implementation, we'd consider order book depth too
                            liquidity_score = volume_24h / 10_000_000  # Normalize to a 0-1 scale
                            if liquidity_score > 1:
                                liquidity_score = 1
                            
                            # Store processed data
                            market_data[pair["symbol"]] = {
                                "symbol": pair["symbol"],
                                "binance_symbol": binance_symbol,
                                "price": price,
                                "volume_24h": volume_24h,
                                "price_change_24h": price_change_pct,
                                "high_24h": high_24h,
                                "low_24h": low_24h,
                                "volatility": volatility,
                                "liquidity_score": liquidity_score,
                                "timestamp": datetime.now().isoformat()
                            }
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Error processing ticker data for {binance_symbol}: {e}")
                            continue
            
            self.logger.info(f"Fetched market data for {len(market_data)} pairs")
            return market_data
        
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return {}
    
    async def select_pairs(self) -> List[str]:
        """
        Select trading pairs based on market criteria.
        
        Returns:
            List of selected trading pair symbols
        """
        self.logger.info("Selecting trading pairs...")
        
        try:
            # Fetch latest market data
            market_data = await self.fetch_market_data()
            self.market_data = market_data
            
            if not market_data:
                self.logger.error("No market data available for pair selection")
                return self.selected_pairs  # Return previous selection if available
            
            # Try to get Redis client from orchestrator to check cooldowns
            redis_client = None
            try:
                from app.core.orchestrator import BotOrchestrator
                orchestrator = BotOrchestrator._instance if hasattr(BotOrchestrator, '_instance') else None
                if orchestrator and hasattr(orchestrator, 'redis_client'):
                    redis_client = orchestrator.redis_client
                    self.logger.info("Successfully obtained Redis client for cooldown checks")
            except Exception as e:
                self.logger.warning(f"Unable to get Redis client for cooldown checks: {str(e)}")
            
            # Apply filters
            filtered_pairs = {}
            
            for symbol, data in market_data.items():
                # Skip blacklisted pairs
                if symbol in self.blacklisted_pairs:
                    continue
                
                # Skip pairs in cooldown - NEW CHECK
                if redis_client:
                    symbol_cooldown_key = f"trading_bot:symbol_cooldown:{symbol}"
                    if redis_client.exists(symbol_cooldown_key):
                        ttl = redis_client.ttl(symbol_cooldown_key)
                        self.logger.info(f"Filtering out {symbol} from trading pairs - in cooldown period for {ttl} more seconds")
                        continue
                
                volume_24h = data.get("volume_24h", 0)
                volatility = data.get("volatility", 0)
                liquidity_score = data.get("liquidity_score", 0)
                
                # Apply minimum criteria
                if volume_24h < self.min_volume_usd:
                    continue
                
                if volatility < self.min_volatility or volatility > self.max_volatility:
                    continue
                
                if liquidity_score < self.min_liquidity_ratio:
                    continue
                
                # Pair passed all filters
                filtered_pairs[symbol] = data
            
            self.logger.info(f"{len(filtered_pairs)} pairs passed filtering criteria")
            
            # Calculate composite score for ranking
            # 50% volume, 30% volatility, 20% liquidity
            scored_pairs = []
            
            for symbol, data in filtered_pairs.items():
                # Normalize metrics for scoring
                volume_score = min(data.get("volume_24h", 0) / 100_000_000, 1.0)  # Cap at 100M
                volatility_score = data.get("volatility", 0) / self.max_volatility
                liquidity_score = data.get("liquidity_score", 0)
                
                # Weighted composite score
                composite_score = (
                    volume_score * 0.5 +
                    volatility_score * 0.3 +
                    liquidity_score * 0.2
                )
                
                scored_pairs.append({
                    "symbol": symbol,
                    "score": composite_score,
                    "data": data
                })
            
            # Sort by score (descending) and select top pairs
            scored_pairs.sort(key=lambda x: x["score"], reverse=True)
            selected = [p["symbol"] for p in scored_pairs[:self.max_pairs]]
            
            # Update selected pairs
            self.selected_pairs = selected
            
            # Update pair details
            for pair in scored_pairs[:self.max_pairs]:
                self.pair_details[pair["symbol"]] = pair["data"]
            
            self.logger.info(f"Selected {len(selected)} trading pairs: {', '.join(selected)}")
            return selected
            
        except Exception as e:
            self.logger.error(f"Error selecting trading pairs: {e}")
            if self.selected_pairs:
                return self.selected_pairs  # Return previous selection if available
            return []  # Return empty list if no previous selection
    
    async def get_pair_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dictionary with pair details
        """
        # Return cached details if available
        if symbol in self.pair_details:
            return self.pair_details[symbol]
        
        # If we have market data but not specific pair details
        if symbol in self.market_data:
            return self.market_data[symbol]
        
        # Try to fetch fresh data for this specific pair
        try:
            # Convert symbol format from "BTC/USDT" to "BTCUSDT" for Binance API
            binance_symbol = symbol.replace("/", "")
            base_url = "https://api.binance.com/api/v3"
            
            async with aiohttp.ClientSession() as session:
                # Fetch current price
                price_url = f"{base_url}/ticker/price?symbol={binance_symbol}"
                async with session.get(price_url) as response:
                    if response.status != 200:
                        self.logger.warning(f"Failed to fetch price for {symbol}: {response.status}")
                        price = 0
                    else:
                        price_data = await response.json()
                        price = float(price_data.get("price", 0))
                
                # Fetch 24h ticker stats
                stats_url = f"{base_url}/ticker/24hr?symbol={binance_symbol}"
                async with session.get(stats_url) as response:
                    if response.status != 200:
                        self.logger.warning(f"Failed to fetch 24h stats for {symbol}: {response.status}")
                        volume = 0
                        change_24h = 0
                        high_24h = 0
                        low_24h = 0
                    else:
                        stats_data = await response.json()
                        volume = float(stats_data.get("volume", 0)) * price
                        change_24h = float(stats_data.get("priceChangePercent", 0))
                        high_24h = float(stats_data.get("highPrice", 0))
                        low_24h = float(stats_data.get("lowPrice", 0))
                
                pair_details = {
                    "symbol": symbol,
                    "binance_symbol": binance_symbol,
                    "price": price,
                    "volume_24h": volume,
                    "price_change_24h": change_24h,
                    "high_24h": high_24h,
                    "low_24h": low_24h,
                    "volatility": ((high_24h - low_24h) / low_24h) * 100 if low_24h > 0 else 0,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the details
                self.pair_details[symbol] = pair_details
                return pair_details
                
        except Exception as e:
            self.logger.error(f"Error fetching pair details for {symbol}: {e}")
            # Return a default structure with minimal data
            return {
                "symbol": symbol,
                "price": 0,
                "volume_24h": 0,
                "price_change_24h": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def set_max_pairs(self, max_pairs: int) -> None:
        """
        Update the maximum number of pairs to select.
        
        Args:
            max_pairs: New maximum number of pairs
        """
        if max_pairs > 0:
            self.max_pairs = max_pairs
            self.logger.info(f"Updated max_pairs to {max_pairs}")
        else:
            self.logger.warning(f"Invalid max_pairs value: {max_pairs}")
    
    def add_blacklisted_pair(self, symbol: str) -> None:
        """
        Add a pair to the blacklist to exclude it from selection.
        
        Args:
            symbol: Trading pair symbol to blacklist
        """
        if symbol not in self.blacklisted_pairs:
            self.blacklisted_pairs.append(symbol)
            self.logger.info(f"Added {symbol} to blacklisted pairs")
    
    def remove_blacklisted_pair(self, symbol: str) -> None:
        """
        Remove a pair from the blacklist.
        
        Args:
            symbol: Trading pair symbol to remove from blacklist
        """
        if symbol in self.blacklisted_pairs:
            self.blacklisted_pairs.remove(symbol)
            self.logger.info(f"Removed {symbol} from blacklisted pairs") 