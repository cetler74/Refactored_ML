import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time

from app.config.settings import Settings, AppMode
from app.exchange.manager import ExchangeManager

logger = logging.getLogger(__name__)

class BalanceManager:
    """
    Manages and tracks balances for both real and simulated trading.
    Implements position sizing and risk management functions.
    """
    
    def __init__(self, settings: Settings, exchange_manager: ExchangeManager, mode: AppMode, db_manager=None):
        self.settings = settings
        self.exchange_manager = exchange_manager
        self.mode = mode
        self.db_manager = db_manager
        
        # Initial balances (will be updated later)
        self.balances = {
            "USDC": {
                "free": 0.0,
                "used": 0.0,
                "total": 0.0
            }
        }
        
        # Track open positions
        self.open_positions = {}
        
        # Track unrealized PnL
        self.unrealized_pnl = 0.0
        
        # Stats tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        logger.info(f"Balance manager initialized in {mode} mode")
    
    async def initialize(self):
        """Initialize balances from exchange or simulation settings."""
        try:
            if self.mode == AppMode.SIMULATION:
                # Use simulated balance from settings - use MAX_POSITION_SIZE_USD instead of max_position_size_usd
                initial_balance = self.settings.INITIAL_BALANCE
                self.balances["USDC"] = {
                    "free": initial_balance,
                    "used": 0.0,
                    "total": initial_balance
                }
                logger.info(f"Initialized simulated balance: {initial_balance} USDC")
            else:
                # Fetch real balance from exchange
                exchange_balances = await self.exchange_manager.fetch_balance("binance")
                
                if "USDC" in exchange_balances.get("free", {}):
                    self.balances["USDC"] = {
                        "free": exchange_balances["free"]["USDC"],
                        "used": exchange_balances["used"]["USDC"],
                        "total": exchange_balances["total"]["USDC"]
                    }
                    logger.info(f"Initialized real balance: {self.balances['USDC']['total']} USDC")
                else:
                    logger.warning("No USDC balance found on exchange")
            
            # Load open positions from database if available
            if self.db_manager:
                # This would be implemented to load open positions from the database
                pass
            
            return True
        except Exception as e:
            logger.exception(f"Error initializing balance manager: {str(e)}")
            return False
    
    async def update_balances(self):
        """Update balances from exchange or recalculate simulated balances."""
        try:
            if self.mode == AppMode.LIVE:
                # Fetch real balance from exchange
                exchange_balances = await self.exchange_manager.fetch_balance("binance")
                
                if "USDC" in exchange_balances.get("free", {}):
                    self.balances["USDC"] = {
                        "free": exchange_balances["free"]["USDC"],
                        "used": exchange_balances["used"]["USDC"],
                        "total": exchange_balances["total"]["USDC"]
                    }
                
                # Update other asset balances
                for symbol, position in self.open_positions.items():
                    asset = symbol.split('/')[0]
                    if asset in exchange_balances.get("free", {}):
                        if asset not in self.balances:
                            self.balances[asset] = {}
                        
                        self.balances[asset]["free"] = exchange_balances["free"][asset]
                        self.balances[asset]["used"] = exchange_balances["used"][asset]
                        self.balances[asset]["total"] = exchange_balances["total"][asset]
            else:
                # In simulation mode, USDC free balance is (total - reserved for positions)
                reserved_usdc = sum(position["cost"] for position in self.open_positions.values())
                self.balances["USDC"]["used"] = reserved_usdc
                self.balances["USDC"]["free"] = self.balances["USDC"]["total"] - reserved_usdc
            
            logger.debug(f"Updated balances: {self.balances}")
            return True
        except Exception as e:
            logger.error(f"Error updating balances: {str(e)}")
            return False
    
    def get_available_usdc(self) -> float:
        """Get available USDC balance for trading."""
        return self.balances.get("USDC", {}).get("free", 0.0)
    
    def calculate_position_size(self, signal: Dict[str, Any], market_data: pd.DataFrame = None) -> Tuple[float, float]:
        """
        Calculate the optimal position size based on risk parameters.
        Uses a modified Kelly criterion approach for dynamic position sizing.
        
        Args:
            signal: Trading signal dictionary with entry price, stop loss, etc.
            market_data: Recent market data for volatility calculation (optional)
        
        Returns:
            Tuple of (position size in USDC, amount of the asset)
        """
        available_usdc = self.get_available_usdc()
        max_position_usdc = min(self.settings.MAX_POSITION_SIZE_USD, available_usdc)
        
        # If we have neither price nor stop loss, use maximum allowed per trade
        if "price" not in signal or "stop_loss" not in signal:
            position_size_usdc = max_position_usdc * self.settings.RISK_PER_TRADE
            asset_amount = position_size_usdc / signal.get("price", 1.0)
            return position_size_usdc, asset_amount
        
        entry_price = signal["price"]
        stop_loss = signal["stop_loss"]
        
        # Calculate risk per share (distance to stop loss)
        if signal.get("type", "buy").lower() == "buy":
            risk_per_share = entry_price - stop_loss
            risk_percentage = risk_per_share / entry_price
        else:  # sell
            risk_per_share = stop_loss - entry_price
            risk_percentage = risk_per_share / entry_price
        
        # Avoid division by zero
        if risk_percentage <= 0:
            position_size_usdc = max_position_usdc * self.settings.RISK_PER_TRADE
            asset_amount = position_size_usdc / entry_price
            return position_size_usdc, asset_amount
        
        # Extract or calculate values needed for Modified Kelly formula
        
        # 1. Win rate estimate - use signal confidence or historical win rate
        win_rate = signal.get("confidence", 0.55)
        if self.total_trades > 10:  # Use historical win rate if we have enough trades
            historical_win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
            win_rate = (win_rate + historical_win_rate) / 2  # Blend signal confidence with historical data
        
        # 2. Calculate reward-to-risk ratio based on take profit and stop loss
        if "take_profit" in signal:
            take_profit = signal["take_profit"]
            if signal.get("type", "buy").lower() == "buy":
                potential_gain = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            else:  # sell
                potential_gain = entry_price - take_profit
                potential_loss = stop_loss - entry_price
            
            # Calculate reward-to-risk ratio, with a minimum of 1.0
            reward_to_risk = max(1.0, potential_gain / potential_loss if potential_loss > 0 else 2.0)
        else:
            # Default to signal's risk_reward_ratio or a sensible default
            reward_to_risk = signal.get("risk_reward_ratio", 2.0)
        
        # 3. Apply market volatility adjustment
        volatility_factor = 1.0
        if market_data is not None and not market_data.empty:
            # Calculate recent volatility (simple implementation)
            try:
                recent_data = market_data.iloc[-20:] if len(market_data) >= 20 else market_data
                close_prices = recent_data['close'].values
                daily_returns = np.diff(close_prices) / close_prices[:-1]
                volatility = np.std(daily_returns)
                
                # Adjust position size based on volatility
                # Lower size for high volatility, higher for low volatility
                base_volatility = 0.02  # Baseline expected volatility
                volatility_factor = base_volatility / max(volatility, 0.005)  # Avoid division by near-zero
                volatility_factor = max(0.5, min(1.5, volatility_factor))  # Limit adjustment range
                
                logger.debug(f"Volatility adjustment factor: {volatility_factor:.2f}")
            except Exception as e:
                logger.warning(f"Error calculating volatility adjustment: {str(e)}")
        
        # 4. Modified Kelly formula with safety factor
        # f* = (win_rate * reward_to_risk - (1 - win_rate)) / reward_to_risk
        kelly_percentage = (win_rate * reward_to_risk - (1 - win_rate)) / reward_to_risk
        
        # Apply safety factor (0.25-0.5 of Kelly is more conservative)
        # Lower values are more conservative, higher values more aggressive
        safety_factor = 0.3  # Using 30% of Kelly as a conservative approach
        adjusted_kelly = max(0.0, kelly_percentage * safety_factor)
        
        # Apply volatility adjustment
        adjusted_kelly *= volatility_factor
        
        # Ensure we respect the maximum risk per trade
        max_risk_percentage = self.settings.RISK_PER_TRADE
        final_risk_percentage = min(adjusted_kelly, max_risk_percentage)
        
        # Calculate position size based on risk and available capital
        risk_amount_usdc = available_usdc * final_risk_percentage
        
        # Calculate position size (risk amount / risk percentage)
        position_size_usdc = min(risk_amount_usdc / risk_percentage, max_position_usdc)
        
        # Calculate asset amount
        asset_amount = position_size_usdc / entry_price
        
        logger.info(f"Position size calculation - Kelly: {kelly_percentage:.4f}, Adjusted: {adjusted_kelly:.4f}, Final: {final_risk_percentage:.4f}")
        logger.info(f"Calculated position size: {position_size_usdc:.2f} USDC ({asset_amount:.6f} units)")
        return position_size_usdc, asset_amount
    
    async def open_position(self, signal: Dict[str, Any], amount: float = None) -> Dict[str, Any]:
        """
        Open a new position based on a signal.
        
        Args:
            signal: Trading signal dictionary
            amount: Optional override for position size
        
        Returns:
            Dictionary with order details
        """
        if not signal:
            return {"success": False, "error": "No signal provided"}
        
        symbol = signal.get("symbol")
        trade_type = signal.get("type")
        entry_price = signal.get("price")
        
        if not all([symbol, trade_type, entry_price]):
            return {"success": False, "error": "Signal missing required fields"}
        
        try:
            # Check if we already have an open position
            if symbol in self.open_positions:
                return {"success": False, "error": f"Position already open for {symbol}"}
            
            # Calculate position size if not provided
            if amount is None:
                position_size_usdc, asset_amount = self.calculate_position_size(signal)
            else:
                position_size_usdc = amount * entry_price
                asset_amount = amount
            
            # Check if we have enough balance
            if position_size_usdc > self.get_available_usdc():
                return {"success": False, "error": "Insufficient USDC balance"}
            
            # Create the order
            order = await self.exchange_manager.create_order(
                symbol=symbol,
                order_type="market",
                side=trade_type,
                amount=asset_amount,
                price=entry_price,
                exchange_id="binance"
            )
            
            if not order:
                return {"success": False, "error": "Failed to create order"}
            
            # Store position information
            self.open_positions[symbol] = {
                "symbol": symbol,
                "type": trade_type,
                "entry_price": entry_price,
                "amount": asset_amount,
                "cost": position_size_usdc,
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "entry_time": datetime.now(),
                "order_id": order.get("id"),
                "trade_id": order.get("id"),
                "strategy": signal.get("strategy"),
                "timeframe": signal.get("timeframe")
            }
            
            # Update balances
            await self.update_balances()
            
            # Store the trade in the database if available
            if self.db_manager:
                # This would be implemented to store the trade in the database
                pass
            
            logger.info(f"Opened {trade_type} position for {symbol}: {asset_amount} @ {entry_price}")
            
            return {
                "success": True,
                "position": self.open_positions[symbol],
                "order": order
            }
        
        except Exception as e:
            logger.exception(f"Error opening position: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def close_position(self, symbol: str, price: float = None, reason: str = "signal") -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            symbol: Symbol to close position for
            price: Optional override for exit price
            reason: Reason for closing (signal, stop_loss, take_profit)
        
        Returns:
            Dictionary with order details
        """
        if symbol not in self.open_positions:
            return {"success": False, "error": f"No open position for {symbol}"}
        
        position = self.open_positions[symbol]
        
        try:
            # Get the trade_id from the position
            trade_id = position.get("trade_id", f"unknown-{int(time.time())}-{symbol}")
            
            # Determine exit price - ALWAYS get current market price first
            real_market_price = await self._get_current_price(symbol)
            
            # Use provided price as override only if it's realistic (within 10% of market price)
            if price is not None and real_market_price > 0:
                # Check if the provided price is within a realistic range of the market price
                if 0.9 * real_market_price <= price <= 1.1 * real_market_price:
                    exit_price = price
                else:
                    logger.warning(f"Provided exit price ${price:.4f} is outside realistic range of market price ${real_market_price:.4f}. Using market price instead.")
                    exit_price = real_market_price
            else:
                exit_price = real_market_price
            
            # Final validation - exit price must be realistic
            if exit_price <= 0 or (real_market_price > 0 and (exit_price / real_market_price > 5 or real_market_price / exit_price > 5)):
                logger.error(f"Invalid exit price ${exit_price:.4f} compared to market price ${real_market_price:.4f}. Cannot close position.")
                return {"success": False, "error": "Invalid exit price (too far from market price)"}
            
            # Determine the opposing side for closing
            close_side = "sell" if position["type"] == "buy" else "buy"
            
            logger.info(f"Closing position for {symbol} with trade_id: {trade_id}, exit price: ${exit_price:.4f}")
            
            # Create the closing order
            order = await self.exchange_manager.create_order(
                symbol=symbol,
                order_type="market",
                side=close_side,
                amount=position["amount"],
                price=exit_price,  # Force using our validated exit price
                exchange_id="binance"
            )
            
            if not order:
                return {"success": False, "error": "Failed to create closing order"}
            
            # Override the order price with our validated exit price to ensure consistency
            if order and "price" in order:
                # Log any discrepancy
                if abs(order["price"] - exit_price) > 0.001 * exit_price:
                    logger.warning(f"Exchange returned price ${order['price']:.4f} differs from our exit price ${exit_price:.4f}. Using our validated price.")
                # Force our validated price
                order["price"] = exit_price
            
            # Calculate PnL
            if position["type"] == "buy":
                pnl = (exit_price - position["entry_price"]) * position["amount"]
                pnl_percentage = (exit_price / position["entry_price"] - 1) * 100
            else:
                pnl = (position["entry_price"] - exit_price) * position["amount"]
                pnl_percentage = (1 - exit_price / position["entry_price"]) * 100
            
            # Update statistics
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
                self.total_profit += pnl
            else:
                self.losing_trades += 1
                self.total_loss += abs(pnl)
            
            # Create result dictionary with trade details
            result = {
                "success": True,
                "symbol": symbol,
                "trade_id": trade_id,  # Include the trade_id in the result
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "amount": position["amount"],
                "side": position["type"],
                "pnl": pnl,
                "pnl_percentage": pnl_percentage,
                "entry_time": position["entry_time"],
                "exit_time": datetime.now(),
                "duration": (datetime.now() - position["entry_time"]).total_seconds() / 60,  # minutes
                "reason": reason,
                "strategy": position["strategy"],
                "order": order
            }
            
            # Remove the position from open positions
            del self.open_positions[symbol]
            
            # Update balances
            await self.update_balances()
            
            # Store the completed trade in the database if available
            if self.db_manager:
                # This would be implemented to store the completed trade
                pass
            
            logger.info(f"Closed {position['type']} position for {symbol} (trade_id: {trade_id}): {pnl:.2f} USDC ({pnl_percentage:.2f}%)")
            
            return result
        
        except Exception as e:
            logger.exception(f"Error closing position: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get the current price for a symbol."""
        try:
            # First try to get price directly from orchestrator's market data
            from app.core.orchestrator import BotOrchestrator
            orchestrator = BotOrchestrator._instance if hasattr(BotOrchestrator, '_instance') else None
            
            if orchestrator and hasattr(orchestrator, 'market_data') and orchestrator.market_data:
                if symbol in orchestrator.market_data:
                    market_info = orchestrator.market_data[symbol]
                    current_price = market_info.get('price')
                    if current_price is not None and current_price > 0:
                        logger.info(f"Using current ticker price for {symbol}: ${current_price:.4f}")
                        return current_price
            
            # Fallback to OHLCV data if market_data is not available
            ohlcv = await self.exchange_manager.fetch_ohlcv(
                symbol=symbol,
                timeframe="1m",
                limit=1,
                exchange_id="binance"
            )
            
            if not ohlcv.empty:
                price = ohlcv["close"].iloc[-1]
                logger.info(f"Using OHLCV fallback price for {symbol}: ${price:.4f}")
                return price
            else:
                logger.error(f"Failed to get current price for {symbol}")
                return 0.0
        
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return 0.0
    
    async def check_positions(self) -> List[Dict[str, Any]]:
        """
        Check all open positions against their stop loss and take profit levels.
        Return a list of actions that should be taken.
        """
        actions = []
        
        for symbol, position in list(self.open_positions.items()):
            try:
                current_price = await self._get_current_price(symbol)
                
                if current_price <= 0:
                    continue
                
                # Calculate unrealized PnL
                if position["type"] == "buy":
                    unrealized_pnl = (current_price - position["entry_price"]) * position["amount"]
                    
                    # Check stop loss
                    if position.get("stop_loss") and current_price <= position["stop_loss"]:
                        result = await self.close_position(symbol, current_price, "stop_loss")
                        if result["success"]:
                            actions.append(result)
                    
                    # Check take profit
                    elif position.get("take_profit") and current_price >= position["take_profit"]:
                        result = await self.close_position(symbol, current_price, "take_profit")
                        if result["success"]:
                            actions.append(result)
                
                else:  # sell position
                    unrealized_pnl = (position["entry_price"] - current_price) * position["amount"]
                    
                    # Check stop loss
                    if position.get("stop_loss") and current_price >= position["stop_loss"]:
                        result = await self.close_position(symbol, current_price, "stop_loss")
                        if result["success"]:
                            actions.append(result)
                    
                    # Check take profit
                    elif position.get("take_profit") and current_price <= position["take_profit"]:
                        result = await self.close_position(symbol, current_price, "take_profit")
                        if result["success"]:
                            actions.append(result)
                
                # Update unrealized PnL tracking
                position["unrealized_pnl"] = unrealized_pnl
                position["current_price"] = current_price
            
            except Exception as e:
                logger.error(f"Error checking position for {symbol}: {str(e)}")
        
        # Update total unrealized PnL
        self.unrealized_pnl = sum(position.get("unrealized_pnl", 0) for position in self.open_positions.values())
        
        return actions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio."""
        total_balance = self.balances.get("USDC", {}).get("total", 0.0)
        total_positions_value = sum(
            position.get("current_price", position["entry_price"]) * position["amount"] 
            for position in self.open_positions.values()
        )
        
        return {
            "total_balance_usdc": total_balance,
            "available_usdc": self.get_available_usdc(),
            "total_positions_value": total_positions_value,
            "unrealized_pnl": self.unrealized_pnl,
            "open_positions_count": len(self.open_positions),
            "open_positions": self.open_positions,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "net_profit": self.total_profit - self.total_loss,
            "profit_factor": self.total_profit / self.total_loss if self.total_loss > 0 else 0,
            "timestamp": datetime.now().isoformat()
        } 