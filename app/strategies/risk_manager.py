import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Advanced risk management functionality for trading strategies.
    Implements chandelier exits, trailing stops, and volatility-adjusted 
    position sizing for effective risk control.
    """
    
    def __init__(self, settings=None):
        self.settings = settings
        self.default_risk_per_trade = getattr(settings, 'RISK_PER_TRADE', 0.02)  # 2% default risk
        self.max_position_size = getattr(settings, 'MAX_POSITION_SIZE_USD', 1000)
        self.trailing_activation_pct = getattr(settings, 'TRAILING_ACTIVATION_PCT', 0.01)  # 1% profit to activate
        self.chandelier_multiplier = getattr(settings, 'CHANDELIER_MULTIPLIER', 3.0)
        logger.info("Advanced Risk Manager initialized")
    
    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_price: float, 
                              available_capital: float,
                              confidence: float = 0.5,
                              volatility: float = None) -> Tuple[float, float]:
        """
        Calculate optimal position size based on risk parameters and volatility.
        
        Args:
            entry_price: Entry price for the trade
            stop_price: Stop loss price
            available_capital: Available capital for trading
            confidence: Confidence level from strategy/ML model (0-1)
            volatility: Normalized volatility measure (optional, e.g., ATR/price)
            
        Returns:
            Tuple of (position size in USD, quantity of asset)
        """
        # Calculate risk amount in dollars
        capital_at_risk = available_capital * self.default_risk_per_trade
        
        # Calculate risk per unit (distance to stop)
        risk_per_unit = abs(entry_price - stop_price)
        
        # Avoid division by zero
        if risk_per_unit <= 0 or entry_price <= 0:
            logger.warning(f"Invalid risk parameters: entry={entry_price}, stop={stop_price}")
            return 0.0, 0.0
        
        # Base quantity calculation
        base_quantity = capital_at_risk / risk_per_unit
        base_position_size = base_quantity * entry_price
        
        # Adjust position size based on confidence
        confidence_factor = 0.5 + (confidence / 2)  # Map 0-1 to 0.5-1.0
        adjusted_position_size = base_position_size * confidence_factor
        
        # Apply volatility adjustment if available
        if volatility:
            # Inverse relationship with volatility - higher volatility, smaller position
            volatility_factor = 1.0 / max(0.5, min(2.0, volatility))
            adjusted_position_size *= volatility_factor
            logger.debug(f"Applied volatility adjustment factor: {volatility_factor:.2f}")
        
        # Ensure we don't exceed maximum position size
        final_position_size = min(adjusted_position_size, self.max_position_size)
        
        # Calculate final quantity
        final_quantity = final_position_size / entry_price
        
        logger.info(f"Position sizing: {final_position_size:.2f} USD ({final_quantity:.6f} units)")
        logger.debug(f"Position size factors: base={base_position_size:.2f}, confidence={confidence_factor:.2f}")
        
        return final_position_size, final_quantity
    
    def calculate_chandelier_exit(self, 
                                high_prices: np.ndarray, 
                                low_prices: np.ndarray,
                                close_prices: np.ndarray,
                                atr: float,
                                periods: int = 22,
                                trade_type: str = 'buy') -> float:
        """
        Calculate chandelier exit level based on ATR.
        For long positions: Highest high - ATR * multiplier
        For short positions: Lowest low + ATR * multiplier
        
        Args:
            high_prices: Array of high prices
            low_prices: Array of low prices
            close_prices: Array of close prices
            atr: Average True Range value
            periods: Number of periods to look back for highest high/lowest low
            trade_type: Type of trade ('buy' or 'sell')
            
        Returns:
            Chandelier exit price
        """
        if len(high_prices) < periods or len(low_prices) < periods:
            logger.warning(f"Not enough price data for chandelier exit calculation, need {periods} bars")
            if trade_type.lower() == 'buy':
                return close_prices[-1] * 0.95  # Default to 5% below current price
            else:
                return close_prices[-1] * 1.05  # Default to 5% above current price
        
        try:
            # Calculate highest high or lowest low over the lookback period
            if trade_type.lower() == 'buy':
                # For long position: Highest high - ATR * multiplier
                highest_high = np.max(high_prices[-periods:])
                chandelier_level = highest_high - (atr * self.chandelier_multiplier)
                
                # Don't set stop too far from current close
                max_distance = close_prices[-1] * 0.1  # Max 10% away
                if close_prices[-1] - chandelier_level > max_distance:
                    chandelier_level = close_prices[-1] - max_distance
                
            else:
                # For short position: Lowest low + ATR * multiplier
                lowest_low = np.min(low_prices[-periods:])
                chandelier_level = lowest_low + (atr * self.chandelier_multiplier)
                
                # Don't set stop too far from current close
                max_distance = close_prices[-1] * 0.1  # Max 10% away
                if chandelier_level - close_prices[-1] > max_distance:
                    chandelier_level = close_prices[-1] + max_distance
            
            return chandelier_level
        
        except Exception as e:
            logger.error(f"Error calculating chandelier exit: {str(e)}")
            return close_prices[-1] * (0.95 if trade_type.lower() == 'buy' else 1.05)
    
    def calculate_volatility_based_targets(self, 
                                         entry_price: float, 
                                         atr: float,
                                         close_prices: np.ndarray = None,
                                         trade_type: str = 'buy') -> Dict[str, float]:
        """
        Calculate volatility-adjusted profit targets and stop levels.
        
        Args:
            entry_price: Entry price for the trade
            atr: Average True Range value
            close_prices: Recent close prices (for additional context)
            trade_type: Type of trade ('buy' or 'sell')
            
        Returns:
            Dictionary with target levels
        """
        # Base ATR multiplier for initial stop loss
        initial_stop_multiplier = 1.5
        
        # Calculate stop loss based on ATR
        if trade_type.lower() == 'buy':
            stop_loss = entry_price - (atr * initial_stop_multiplier)
            
            # Calculate take profit levels with increasing multiples (risk:reward)
            take_profit_1 = entry_price + (atr * 2)  # 1:1.33 risk:reward
            take_profit_2 = entry_price + (atr * 3)  # 1:2 risk:reward
            take_profit_3 = entry_price + (atr * 5)  # 1:3.33 risk:reward
            
        else:  # sell
            stop_loss = entry_price + (atr * initial_stop_multiplier)
            
            take_profit_1 = entry_price - (atr * 2)
            take_profit_2 = entry_price - (atr * 3)
            take_profit_3 = entry_price - (atr * 5)
        
        # If we have close prices, adjust take profit based on recent volatility
        if close_prices is not None and len(close_prices) > 20:
            # Calculate recent volatility
            returns = np.diff(close_prices) / close_prices[:-1]
            recent_volatility = np.std(returns[-20:])
            historical_volatility = np.std(returns)
            
            # Adjust targets based on relative volatility
            volatility_ratio = recent_volatility / max(0.0001, historical_volatility)
            
            # If recent volatility is higher, extend targets
            if volatility_ratio > 1.2:
                multiplier = min(2.0, volatility_ratio)
                
                if trade_type.lower() == 'buy':
                    take_profit_2 = entry_price + (atr * 3 * multiplier)
                    take_profit_3 = entry_price + (atr * 5 * multiplier)
                else:
                    take_profit_2 = entry_price - (atr * 3 * multiplier)
                    take_profit_3 = entry_price - (atr * 5 * multiplier)
                
                logger.debug(f"Adjusted take profit for high volatility: multiplier={multiplier:.2f}")
        
        return {
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'take_profit_3': take_profit_3
        }
    
    def update_trailing_stop(self,
                           current_price: float,
                           entry_price: float,
                           highest_price: float,
                           current_stop: float,
                           atr: float,
                           trade_type: str = 'buy') -> Tuple[float, bool]:
        """
        Update trailing stop level based on price movement.
        
        Args:
            current_price: Current market price
            entry_price: Original entry price
            highest_price: Highest price since entry (for buys) or lowest price (for sells)
            current_stop: Current stop loss level
            atr: Current Average True Range value
            trade_type: Type of trade ('buy' or 'sell')
            
        Returns:
            Tuple of (new stop price, whether trailing is activated)
        """
        # Calculate profit percentage so far
        if trade_type.lower() == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
            
            # Check if trailing stop should be activated
            trailing_active = profit_pct >= self.trailing_activation_pct
            
            if trailing_active:
                # Calculate new trailing stop
                new_stop = highest_price - (atr * 2)
                
                # Only raise the stop, never lower it
                if new_stop > current_stop:
                    return new_stop, trailing_active
                else:
                    return current_stop, trailing_active
            else:
                return current_stop, trailing_active
            
        else:  # sell position
            profit_pct = (entry_price - current_price) / entry_price
            
            # Check if trailing stop should be activated
            trailing_active = profit_pct >= self.trailing_activation_pct
            
            if trailing_active:
                # Calculate new trailing stop
                new_stop = highest_price + (atr * 2)
                
                # Only lower the stop, never raise it
                if new_stop < current_stop:
                    return new_stop, trailing_active
                else:
                    return current_stop, trailing_active
            else:
                return current_stop, trailing_active
    
    def should_exit_trade(self,
                         position_data: Dict[str, Any],
                         current_price: float,
                         atr: float = None,
                         market_data: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Determine if a trade should be exited based on risk management rules.
        
        Args:
            position_data: Dictionary with position details
            current_price: Current market price
            atr: Current ATR value (optional)
            market_data: Recent market data for additional analysis (optional)
            
        Returns:
            Tuple of (should exit, reason)
        """
        # Extract position details
        entry_price = position_data.get('entry_price', 0)
        stop_loss = position_data.get('stop_loss', 0)
        take_profit = position_data.get('take_profit', 0)
        trade_type = position_data.get('type', 'buy')
        highest_price = position_data.get('highest_price', entry_price)
        lowest_price = position_data.get('lowest_price', entry_price)
        
        # Update highest/lowest seen
        if trade_type.lower() == 'buy':
            position_data['highest_price'] = max(highest_price, current_price)
        else:
            position_data['lowest_price'] = min(lowest_price, current_price)
        
        # Calculate current profit/loss
        if trade_type.lower() == 'buy':
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Check stop loss hit
            if current_price <= stop_loss:
                return True, "stop_loss"
            
            # Check take profit hit
            if take_profit > 0 and current_price >= take_profit:
                return True, "take_profit"
            
        else:  # sell position
            pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss hit
            if current_price >= stop_loss:
                return True, "stop_loss"
            
            # Check take profit hit
            if take_profit > 0 and current_price <= take_profit:
                return True, "take_profit"
        
        # Check trailing stop if we have ATR
        if atr is not None and 'trailing_active' in position_data:
            trailing_active = position_data.get('trailing_active', False)
            trailing_stop = position_data.get('trailing_stop', stop_loss)
            
            # Update trailing stop
            new_trailing_stop, trailing_active = self.update_trailing_stop(
                current_price=current_price,
                entry_price=entry_price,
                highest_price=position_data['highest_price'] if trade_type.lower() == 'buy' else position_data['lowest_price'],
                current_stop=trailing_stop,
                atr=atr,
                trade_type=trade_type
            )
            
            # Store updated values
            position_data['trailing_stop'] = new_trailing_stop
            position_data['trailing_active'] = trailing_active
            
            # Check if trailing stop has been hit
            if trailing_active:
                if (trade_type.lower() == 'buy' and current_price <= new_trailing_stop) or \
                   (trade_type.lower() == 'sell' and current_price >= new_trailing_stop):
                    return True, "trailing_stop"
        
        # Check for any other exit signals in market data
        if market_data is not None and not market_data.empty:
            latest = market_data.iloc[-1]
            
            # Exit on extreme RSI values in the opposite direction
            if trade_type.lower() == 'buy' and 'rsi_14' in latest and latest['rsi_14'] > 80 and pnl_pct > 0:
                return True, "rsi_overbought"
                
            if trade_type.lower() == 'sell' and 'rsi_14' in latest and latest['rsi_14'] < 20 and pnl_pct > 0:
                return True, "rsi_oversold"
            
            # Exit on divergence for profitable trades
            if pnl_pct > 0.02:  # 2% profit
                # Price making new high but RSI not confirming (for buys)
                if trade_type.lower() == 'buy' and 'rsi_divergence' in latest and latest['rsi_divergence'] < 0:
                    return True, "bearish_divergence"
                
                # Price making new low but RSI not confirming (for sells)
                if trade_type.lower() == 'sell' and 'rsi_divergence' in latest and latest['rsi_divergence'] > 0:
                    return True, "bullish_divergence"
        
        # No exit signal
        return False, "" 