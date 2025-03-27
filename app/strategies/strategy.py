import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from app.config.settings import Settings
from app.database.models import TradeType, Signal
from app.strategies.timeframe_selector import DynamicTimeframeSelector
from app.strategies.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class Strategy:
    """
    Base strategy class for defining trading rules.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.risk_manager = None
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals from a DataFrame of market data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def get_stop_loss(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """
        Base method for calculating stop loss levels.
        
        Args:
            price: Entry price
            trade_type: Trade type (buy/sell)
            market_data: Optional market data for advanced calculations
        """
        # If risk manager is available, use it for chandelier exit
        if self.risk_manager is not None and market_data is not None:
            try:
                # Extract needed price data
                high_prices = market_data['high'].values if 'high' in market_data.columns else None
                low_prices = market_data['low'].values if 'low' in market_data.columns else None
                close_prices = market_data['close'].values if 'close' in market_data.columns else None
                atr = market_data['atr'].iloc[-1] if 'atr' in market_data.columns else None
                
                if high_prices is not None and low_prices is not None and close_prices is not None and atr is not None:
                    # Calculate chandelier exit
                    return self.risk_manager.calculate_chandelier_exit(
                        high_prices=high_prices,
                        low_prices=low_prices,
                        close_prices=close_prices,
                        atr=atr,
                        trade_type=trade_type
                    )
            except Exception as e:
                logger.error(f"Error calculating chandelier exit: {str(e)}")
        
        # Default implementation with fixed percentage
        if trade_type == TradeType.BUY.value:
            return price * 0.97  # 3% stop loss
        else:
            return price * 1.03
    
    def get_take_profit(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """
        Base method for calculating take profit levels.
        
        Args:
            price: Entry price
            trade_type: Trade type (buy/sell)
            market_data: Optional market data for advanced calculations
        """
        # If risk manager is available, use it for volatility-based targets
        if self.risk_manager is not None and market_data is not None:
            try:
                atr = market_data['atr'].iloc[-1] if 'atr' in market_data.columns else None
                close_prices = market_data['close'].values if 'close' in market_data.columns else None
                
                if atr is not None:
                    # Get volatility-adjusted targets
                    targets = self.risk_manager.calculate_volatility_based_targets(
                        entry_price=price,
                        atr=atr,
                        close_prices=close_prices,
                        trade_type=trade_type
                    )
                    
                    # Return the first take profit level
                    return targets['take_profit_1']
            except Exception as e:
                logger.error(f"Error calculating volatility-based targets: {str(e)}")
        
        # Default implementation with fixed percentage
        if trade_type == TradeType.BUY.value:
            return price * 1.06  # 6% take profit
        else:
            return price * 0.94

    def analyze_market_structure(self, df):
        """
        Base implementation to analyze market structure.
        Returns a tuple of (trend, strength)
        """
        if df is None or len(df) < 5:
            return "neutral", 0
            
        # Check for required columns
        last = df.iloc[-1]
        if 'ema12' not in last or 'ema26' not in last:
            logger.warning("Missing required EMA columns in dataframe")
            return "neutral", 0
            
        # Determine trend based on EMA crossover
        trend = "neutral"
        trend_strength = 0
        
        if last['ema12'] > last['ema26']:
            trend = "bullish"
            trend_strength = (last['ema12'] / last['ema26'] - 1) * 100
        elif last['ema12'] < last['ema26']:
            trend = "bearish"
            trend_strength = (1 - last['ema12'] / last['ema26']) * 100
            
        # Adjust strength based on ADX if available
        if 'adx' in last and not pd.isna(last['adx']):
            if last['adx'] > 25:
                trend_strength *= 1.5
            elif last['adx'] < 20:
                trend_strength *= 0.8
                
        return trend, float(trend_strength)

class MACrossoverStrategy(Strategy):
    """
    Moving Average Crossover strategy implementation.
    """
    
    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__(name=f"MA_Crossover_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        logger.info(f"Initialized {self.name} strategy")
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: DataFrame with OHLCV data and indicators
        
        Returns:
            List of signal dictionaries
        """
        if data.empty or len(data) < self.slow_period:
            return []
        
        signals = []
        
        # Calculate MAs if not already in the dataframe
        if f'ema_{self.fast_period}' not in data.columns:
            data[f'ema_{self.fast_period}'] = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        
        if f'ema_{self.slow_period}' not in data.columns:
            data[f'ema_{self.slow_period}'] = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate crossover
        data['crossover'] = np.where(
            data[f'ema_{self.fast_period}'] > data[f'ema_{self.slow_period}'], 1,
            np.where(data[f'ema_{self.fast_period}'] < data[f'ema_{self.slow_period}'], -1, 0)
        )
        
        # Detect crossover points (when crossover value changes)
        data['crossover_change'] = data['crossover'].diff()
        
        # Get the last two rows for recent signal detection
        latest_rows = data.iloc[-2:].copy() if len(data) >= 2 else data.copy()
        
        for _, row in latest_rows.iterrows():
            if row['crossover_change'] == 2:  # Bullish crossover
                # Calculate additional confidence modifiers
                confidence = 0.6  # Base confidence
                
                # Adjust confidence based on trend strength if available
                if 'adx' in row:
                    trend_strength = row['adx'] / 100.0  # Normalize to 0-1
                    confidence = min(0.85, confidence + (trend_strength * 0.25))
                
                # Adjust confidence based on volume confirmation if available
                if 'volume' in data.columns and len(data) > 5:
                    avg_volume = data['volume'].iloc[-6:-1].mean()
                    if row['volume'] > avg_volume * 1.5:
                        confidence += 0.1
                
                signals.append({
                    'timestamp': row.name if isinstance(row.name, datetime) else datetime.now(),
                    'symbol': data.attrs.get('symbol', 'Unknown'),
                    'trade_type': TradeType.BUY.value,
                    'price': row['close'],
                    'confidence': confidence,
                    'strategy': self.name,
                    'timeframe': data.attrs.get('timeframe', 'Unknown')
                })
            elif row['crossover_change'] == -2:  # Bearish crossover
                # Calculate additional confidence modifiers
                confidence = 0.6  # Base confidence
                
                # Adjust confidence based on trend strength if available
                if 'adx' in row:
                    trend_strength = row['adx'] / 100.0  # Normalize to 0-1
                    confidence = min(0.85, confidence + (trend_strength * 0.25))
                
                # Adjust confidence based on volume confirmation if available
                if 'volume' in data.columns and len(data) > 5:
                    avg_volume = data['volume'].iloc[-6:-1].mean()
                    if row['volume'] > avg_volume * 1.5:
                        confidence += 0.1
                
                signals.append({
                    'timestamp': row.name if isinstance(row.name, datetime) else datetime.now(),
                    'symbol': data.attrs.get('symbol', 'Unknown'),
                    'trade_type': TradeType.SELL.value,
                    'price': row['close'],
                    'confidence': confidence,
                    'strategy': self.name,
                    'timeframe': data.attrs.get('timeframe', 'Unknown')
                })
        
        return signals
    
    def get_stop_loss(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """
        Calculate stop loss level using chandelier exit if available.
        
        Args:
            price: Entry price
            trade_type: Trade type (buy/sell)
            market_data: Market data for advanced calculations
        
        Returns:
            Stop loss price
        """
        return super().get_stop_loss(price, trade_type, market_data)
    
    def get_take_profit(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """
        Calculate take profit with volatility-based targets.
        
        Args:
            price: Entry price
            trade_type: Trade type (buy/sell)
            market_data: Market data for advanced calculations
        
        Returns:
            Take profit price
        """
        return super().get_take_profit(price, trade_type, market_data)

class RSIStrategy(Strategy):
    """
    RSI-based strategy implementation.
    """
    
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__(name=f"RSI_{period}")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        logger.info(f"Initialized {self.name} strategy")
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on RSI.
        
        Args:
            data: DataFrame with OHLCV data and indicators
        
        Returns:
            List of signal dictionaries
        """
        if data.empty or len(data) < self.period:
            return []
        
        signals = []
        
        # Calculate RSI if not already in the dataframe
        rsi_col = f'rsi_{self.period}'
        
        if rsi_col not in data.columns:
            return []  # Need RSI to be calculated by TA module
        
        # Get the last two rows for recent signal detection
        latest_rows = data.iloc[-2:].copy() if len(data) >= 2 else data.copy()
        
        # Generate signals based on RSI crossing thresholds
        for i, row in enumerate(latest_rows.iterrows()):
            index, current = row
            
            if i == 0 and len(latest_rows) < 2:
                continue  # Need at least two rows to detect crossings
            
            if i > 0:
                # Correctly assign the previous row Series to 'prev'
                prev = latest_rows.iloc[i-1]
                
                # Oversold to normal (buy signal)
                if prev[rsi_col] <= self.oversold and current[rsi_col] > self.oversold:
                    # Calculate confidence based on RSI and supporting indicators
                    confidence = 0.65  # Base confidence
                    
                    # Check for bullish divergence for higher confidence
                    if 'rsi_divergence' in current and current['rsi_divergence'] > 0:
                        confidence += 0.1
                    
                    # Check for support at moving average
                    if 'sma_50' in current and 'close' in current and current['close'] > current['sma_50']:
                        confidence += 0.05
                    
                    signals.append({
                        'timestamp': index if isinstance(index, datetime) else datetime.now(),
                        'symbol': data.attrs.get('symbol', 'Unknown'),
                        'trade_type': TradeType.BUY.value,
                        'price': current['close'],
                        'confidence': confidence,
                        'strategy': self.name,
                        'timeframe': data.attrs.get('timeframe', 'Unknown')
                    })
                
                # Overbought to normal (sell signal)
                elif prev[rsi_col] >= self.overbought and current[rsi_col] < self.overbought:
                    # Calculate confidence based on RSI and supporting indicators
                    confidence = 0.65  # Base confidence
                    
                    # Check for bearish divergence for higher confidence
                    if 'rsi_divergence' in current and current['rsi_divergence'] < 0:
                        confidence += 0.1
                    
                    # Check for resistance at moving average
                    if 'sma_50' in current and 'close' in current and current['close'] < current['sma_50']:
                        confidence += 0.05
                    
                    signals.append({
                        'timestamp': index if isinstance(index, datetime) else datetime.now(),
                        'symbol': data.attrs.get('symbol', 'Unknown'),
                        'trade_type': TradeType.SELL.value,
                        'price': current['close'],
                        'confidence': confidence,
                        'strategy': self.name,
                        'timeframe': data.attrs.get('timeframe', 'Unknown')
                    })
        
        return signals
    
    def get_stop_loss(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """Calculate advanced stop loss level using risk manager."""
        return super().get_stop_loss(price, trade_type, market_data)
    
    def get_take_profit(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """Calculate advanced take profit using risk manager."""
        return super().get_take_profit(price, trade_type, market_data)

class MLStrategy(Strategy):
    """
    Machine Learning based strategy implementation.
    Uses signals generated by ML models.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__(name="ML_Strategy")
        self.confidence_threshold = confidence_threshold
        logger.info(f"Initialized {self.name} strategy with confidence threshold {confidence_threshold}")
    
    async def generate_signals_from_predictions(self, predictions: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals from ML model predictions.
        
        Args:
            predictions: DataFrame with model predictions
        
        Returns:
            List of signal dictionaries
        """
        if predictions.empty:
            return []
        
        signals = []
        
        for _, prediction in predictions.iterrows():
            confidence = prediction['probability']
            signal_type = prediction['signal']
            
            # Generate signals based on confidence threshold
            if signal_type in ['strong_buy', 'buy'] and confidence >= self.confidence_threshold:
                signals.append({
                    'timestamp': prediction.get('timestamp', datetime.now()),
                    'symbol': prediction.get('symbol', 'Unknown'),
                    'trade_type': TradeType.BUY.value,
                    'price': prediction.get('price', 0.0),
                    'confidence': confidence,
                    'strategy': self.name,
                    'timeframe': prediction.get('timeframe', 'Unknown'),
                    'model': prediction.get('model', 'Unknown')
                })
            elif signal_type in ['strong_sell', 'sell'] and confidence >= self.confidence_threshold:
                signals.append({
                    'timestamp': prediction.get('timestamp', datetime.now()),
                    'symbol': prediction.get('symbol', 'Unknown'),
                    'trade_type': TradeType.SELL.value,
                    'price': prediction.get('price', 0.0),
                    'confidence': confidence,
                    'strategy': self.name,
                    'timeframe': prediction.get('timeframe', 'Unknown'),
                    'model': prediction.get('model', 'Unknown')
                })
        
        return signals
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on combined indicators.
        This is a fallback when ML predictions are not available.
        
        Args:
            data: DataFrame with OHLCV data and indicators
        
        Returns:
            List of signal dictionaries
        """
        if data.empty:
            return []
        
        signals = []
        latest = data.iloc[-1]
        
        # Use RSI and MACD for a basic signal
        rsi_ok = 'rsi_14' in latest and (latest['rsi_14'] < 30 or latest['rsi_14'] > 70)
        macd_ok = all(x in latest for x in ['macd', 'macd_signal']) and (
            abs(latest['macd'] - latest['macd_signal']) > 0.01 * latest['close']
        )
        
        if rsi_ok and macd_ok:
            trade_type = TradeType.BUY.value if latest.get('rsi_14', 50) < 30 else TradeType.SELL.value
            
            # Calculate confidence based on agreement of multiple indicators
            confidence = 0.55  # Base confidence
            
            # Check trend alignment
            if 'adx' in latest and 'di_plus' in latest and 'di_minus' in latest:
                trend_strength = latest['adx'] / 100.0
                
                if trade_type == TradeType.BUY.value and latest['di_plus'] > latest['di_minus']:
                    confidence += trend_strength * 0.1
                elif trade_type == TradeType.SELL.value and latest['di_minus'] > latest['di_plus']:
                    confidence += trend_strength * 0.1
            
            # Check for support/resistance at Bollinger Bands
            if 'bb_lower' in latest and 'bb_upper' in latest and 'close' in latest:
                if trade_type == TradeType.BUY.value and abs(latest['close'] - latest['bb_lower']) < 0.01 * latest['close']:
                    confidence += 0.05
                elif trade_type == TradeType.SELL.value and abs(latest['close'] - latest['bb_upper']) < 0.01 * latest['close']:
                    confidence += 0.05
            
            signals.append({
                'timestamp': latest.name if isinstance(latest.name, datetime) else datetime.now(),
                'symbol': data.attrs.get('symbol', 'Unknown'),
                'trade_type': trade_type,
                'price': latest['close'],
                'confidence': confidence,
                'strategy': f"{self.name}_Fallback",
                'timeframe': data.attrs.get('timeframe', 'Unknown')
            })
        
        return signals
    
    def get_stop_loss(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """Calculate dynamic stop loss level using risk manager."""
        return super().get_stop_loss(price, trade_type, market_data)
    
    def get_take_profit(self, price: float, trade_type: str, market_data: pd.DataFrame = None) -> float:
        """Calculate dynamic take profit level using risk manager."""
        return super().get_take_profit(price, trade_type, market_data)

class StrategyModule:
    """
    Module that manages multiple trading strategies and combines their signals.
    Integrates dynamic timeframe selection and advanced risk management.
    """
    
    def __init__(self, settings: Settings, ta_module=None, ml_module=None):
        self.settings = settings
        self.ta_module = ta_module
        self.ml_module = ml_module
        
        # Initialize risk manager
        self.risk_manager = RiskManager(settings)
        
        # Initialize timeframe selector
        self.timeframe_selector = DynamicTimeframeSelector(
            timeframes=settings.default_timeframes if hasattr(settings, 'default_timeframes') else ["1m", "5m", "15m", "1h"]
        )
        
        # Initialize strategies
        self.strategies = {
            "ma_crossover": MACrossoverStrategy(9, 21),
            "rsi": RSIStrategy(14, 70, 30),
            "ml": MLStrategy(0.7)
        }
        
        # Assign risk manager to strategies
        for strategy in self.strategies.values():
            strategy.risk_manager = self.risk_manager
        
        # Active signals
        self.active_signals = []
        
        # Performance tracking
        self.signals_performance = {}
        
        logger.info(f"Strategy module initialized with {len(self.strategies)} strategies")
    
    async def generate_signals(self, market_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals across all strategies for all provided market data.
        
        Args:
            market_data: Dictionary mapping symbol -> {timeframe -> DataFrame}
            
        Returns:
            List of signal dictionaries
        """
        all_signals = []
        
        # Check if market_data is valid and has the expected structure
        if not market_data or not isinstance(market_data, dict) or not market_data.get("success", False):
            logger.error("Invalid market data provided to generate_signals")
            return all_signals

        # Extract the actual market data from the response
        market_data_dict = market_data.get("data", {})
        if not isinstance(market_data_dict, dict):
            logger.error("Market data format is invalid")
            return all_signals
        
        # Prepare data for timeframe selector training
        try:
            # Make sure we have valid data for the timeframe selector
            performance_history = getattr(self.timeframe_selector, 'performance_history', None)
                
            # Only call prepare_training_features if we have valid market data
            if market_data_dict and any(isinstance(v, dict) for v in market_data_dict.values()):
                tf_training_data = self.timeframe_selector.prepare_training_features(
                    market_data_dict, 
                    performance_history
                )
                
                # Train timeframe selector if we have enough data
                if isinstance(tf_training_data, pd.DataFrame) and not tf_training_data.empty and len(tf_training_data) >= 50:
                    logger.info(f"Training timeframe selector with {len(tf_training_data)} samples")
                    self.timeframe_selector.train_model(tf_training_data)
            else:
                logger.warning("Insufficient market data for timeframe selector training")
        except Exception as e:
            logger.error(f"Error training timeframe selector: {str(e)}")
            import traceback
            logger.debug(f"Timeframe selector error details: {traceback.format_exc()}")
        
        # Process each symbol
        for symbol, timeframes_data in market_data_dict.items():
            # Skip if no data for symbol or invalid format
            if not isinstance(timeframes_data, dict):
                logger.warning(f"Invalid timeframes data format for symbol {symbol}")
                continue
            
            try:
                # Get ML confidence scores if ML module is available
                ml_confidence = {}
                if self.ml_module and hasattr(self.ml_module, 'make_predictions'):
                    # This would need to be implemented to get ML confidence by timeframe
                    pass
                
                # Select optimal timeframe for this symbol
                selected_timeframe = self.timeframe_selector.select_optimal_timeframe(
                    symbol, timeframes_data, ml_confidence
                )
                
                # Make sure timeframes_data is a dict before calling keys()
                if isinstance(timeframes_data, dict):
                    # Use all timeframes but prioritize the selected one
                    prioritized_timeframes = [selected_timeframe] + \
                        [tf for tf in timeframes_data.keys() if tf != selected_timeframe]
                else:
                    # Fallback if timeframes_data is not a dict
                    prioritized_timeframes = [selected_timeframe]
                
                logger.info(f"Selected optimal timeframe for {symbol}: {selected_timeframe}")
                
                # Generate signals for each timeframe, prioritizing the selected one
                for timeframe in prioritized_timeframes:
                    df = timeframes_data.get(timeframe)
                    # Check if df is DataFrame and not empty
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    
                    # Set metadata for the dataframe
                    df.attrs['symbol'] = symbol
                    df.attrs['timeframe'] = timeframe
                    
                    # Generate signals from each strategy
                    for strategy_name, strategy in self.strategies.items():
                        if strategy_name != "ml":  # Handle ML strategy separately
                            try: # Add inner try/except for more specific logging
                                logger.info(f"Calling {strategy_name}.generate_signals for {symbol} on {timeframe}") # Log before call
                                signals = await strategy.generate_signals(df)
                                logger.info(f"Finished {strategy_name}.generate_signals for {symbol} on {timeframe}, received {len(signals)} signals") # Log after call

                                # Add priority to signals from the selected timeframe
                                for signal in signals:
                                    signal['priority'] = 1.0 if timeframe == selected_timeframe else 0.5
                                    signal['optimal_timeframe'] = selected_timeframe == timeframe

                                all_signals.extend(signals)
                            except ValueError as ve: # Catch unpacking errors specifically
                                logger.error(f"ValueError during {strategy_name}.generate_signals for {symbol} on {timeframe}: {ve}", exc_info=True)
                                # Continue loop or re-raise if needed
                            except Exception as inner_e: # Catch other errors from strategy call
                                logger.error(f"Exception during {strategy_name}.generate_signals for {symbol} on {timeframe}: {inner_e}", exc_info=True)
                                # Continue loop or re-raise if needed
            
            except Exception as e: # Outer catch remains
                logger.error(f"Error generating signals for {symbol}: {str(e)}") # This is the one currently logging
        
        return all_signals
    
    async def update_signals(self, ml_predictions: Dict[str, Any]) -> None:
        """
        Update signals with ML model predictions.
        
        Args:
            ml_predictions: Dictionary with ML model predictions
        """
        if not ml_predictions or not isinstance(ml_predictions, dict):
            logger.warning("Invalid ML predictions format")
            return
            
        if not ml_predictions.get("success", False) or "predictions" not in ml_predictions:
            return
        
        predictions_df = ml_predictions["predictions"]
        
        # Generate signals from ML predictions
        ml_strategy = self.strategies.get("ml")
        # Check if predictions_df is a DataFrame and not empty
        if ml_strategy and isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
            signals = await ml_strategy.generate_signals_from_predictions(predictions_df)
            
            # Add priority to ML signals (higher priority)
            for signal in signals:
                signal['priority'] = 1.2  # Higher priority than regular signals
            
            self.active_signals.extend(signals)
            
            logger.info(f"Added {len(signals)} new ML-based signals")
    
    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """
        Get all active signals, prioritizing based on confidence and timeframe.
        
        Returns:
            List of active signal dictionaries
        """
        # Filter and prioritize signals
        filtered_signals = []
        
        # Group signals by symbol and trade_type to avoid conflicting signals
        signals_by_symbol = {}
        
        for signal in self.active_signals:
            symbol = signal.get('symbol', 'Unknown')
            trade_type = signal.get('trade_type', '')
            
            key = f"{symbol}_{trade_type}"
            
            if key not in signals_by_symbol:
                signals_by_symbol[key] = []
            
            signals_by_symbol[key].append(signal)
        
        # For each symbol and trade_type, pick the best signal
        for key, symbol_signals in signals_by_symbol.items():
            if not symbol_signals:
                continue
            
            # Calculate a score for each signal based on priority and confidence
            for signal in symbol_signals:
                priority = signal.get('priority', 1.0)
                confidence = signal.get('confidence', 0.5)
                optimal_timeframe = signal.get('optimal_timeframe', False)
                
                # Calculate a combined score
                score = (priority * 0.4) + (confidence * 0.6)
                
                # Boost score for optimal timeframe
                if optimal_timeframe:
                    score *= 1.1
                
                signal['score'] = score
            
            # Sort by score (descending)
            symbol_signals.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Add the best signal to filtered signals
            filtered_signals.append(symbol_signals[0])
        
        # Clear the active signals for the next round
        self.active_signals = []
        
        # Return the filtered signals
        return filtered_signals
    
    def get_signal_metadata(self, signal: Dict[str, Any], market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Enrich a signal with additional metadata like stop loss and take profit levels.
        Uses advanced risk management for stop loss and take profit calculations.
        
        Args:
            signal: Original signal dictionary
            market_data: DataFrame with market data for the symbol (optional)
        
        Returns:
            Enriched signal dictionary
        """
        strategy_name = signal.get('strategy', '')
        strategy = next((s for name, s in self.strategies.items() if strategy_name.startswith(name)), None)
        
        if not strategy:
            return signal
        
        # Create a copy to avoid modifying the original
        enriched_signal = signal.copy()
        
        # Get ATR if available in market data
        atr = None
        if market_data is not None and 'atr' in market_data.columns:
            atr = market_data['atr'].iloc[-1]
        
        # Calculate stop loss and take profit levels
        price = signal.get('price', 0)
        trade_type = signal.get('trade_type', '')
        
        if price > 0 and trade_type:
            # Calculate stop loss and take profit using the strategy's methods
            enriched_signal['stop_loss'] = strategy.get_stop_loss(price, trade_type, market_data)
            enriched_signal['take_profit'] = strategy.get_take_profit(price, trade_type, market_data)
            
            # If we have the risk manager and ATR, add more advanced risk data
            if self.risk_manager is not None and atr is not None:
                volatility_targets = self.risk_manager.calculate_volatility_based_targets(
                    entry_price=price,
                    atr=atr,
                    close_prices=market_data['close'].values if market_data is not None and 'close' in market_data.columns else None,
                    trade_type=trade_type
                )
                
                # Add multiple take profit levels
                enriched_signal['take_profit_levels'] = {
                    'tp1': volatility_targets['take_profit_1'],
                    'tp2': volatility_targets['take_profit_2'],
                    'tp3': volatility_targets['take_profit_3']
                }
                
                # Initialize trailing stop data
                enriched_signal['trailing_stop'] = enriched_signal['stop_loss']
                enriched_signal['trailing_active'] = False
                
                # Add position sizing based on risk management
                confidence = signal.get('confidence', 0.5)
                volatility = atr / price if price > 0 else None
                
                # Placeholder for available capital - in real implementation, get from balance manager
                available_capital = 1000.0
                
                position_size_usd, quantity = self.risk_manager.calculate_position_size(
                    entry_price=price,
                    stop_price=enriched_signal['stop_loss'],
                    available_capital=available_capital,
                    confidence=confidence,
                    volatility=volatility
                )
                
                enriched_signal['position_size_usd'] = position_size_usd
                enriched_signal['quantity'] = quantity
            
            # Calculate risk/reward ratio
            if trade_type == TradeType.BUY.value:
                risk = price - enriched_signal['stop_loss']
                reward = enriched_signal['take_profit'] - price
            else:
                risk = enriched_signal['stop_loss'] - price
                reward = price - enriched_signal['take_profit']
            
            if risk > 0:
                enriched_signal['risk_reward_ratio'] = reward / risk
            else:
                enriched_signal['risk_reward_ratio'] = 0
        
        return enriched_signal
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update performance metrics for timeframe selection and strategy improvement.
        
        Args:
            trade_result: Dictionary with trade result details
        """
        try:
            # Extract relevant trade data
            symbol = trade_result.get('symbol')
            timeframe = trade_result.get('timeframe')
            strategy = trade_result.get('strategy')
            pnl = trade_result.get('pnl', 0)
            pnl_pct = trade_result.get('pnl_percentage', 0)
            
            if not all([symbol, timeframe, strategy]):
                logger.warning("Missing required data for performance update")
                return
            
            # Update timeframe selector performance history
            if hasattr(self.timeframe_selector, 'update_performance'):
                self.timeframe_selector.update_performance(
                    symbol=symbol,
                    timeframe=timeframe,
                    performance_metric=pnl_pct  # Use percentage PnL as performance metric
                )
            
            # Store signal performance for future reference
            signal_key = f"{symbol}_{strategy}_{timeframe}"
            
            if signal_key not in self.signals_performance:
                self.signals_performance[signal_key] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0,
                    'avg_pnl_pct': 0
                }
            
            # Update performance metrics
            perf = self.signals_performance[signal_key]
            perf['total_trades'] += 1
            perf['total_pnl'] += pnl
            
            if pnl > 0:
                perf['winning_trades'] += 1
                
            perf['avg_pnl_pct'] = ((perf['avg_pnl_pct'] * (perf['total_trades'] - 1)) + pnl_pct) / perf['total_trades']
            
            logger.info(f"Updated performance for {signal_key}: win rate {perf['winning_trades']/perf['total_trades']:.2f}, avg PnL {perf['avg_pnl_pct']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def should_exit_trade(self, position_data: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """
        Check if a trade should be exited based on advanced risk management rules.
        
        Args:
            position_data: Dictionary with position details
            current_price: Current market price
            
        Returns:
            Tuple of (should exit, reason)
        """
        if self.risk_manager is None:
            # Default simple check
            if position_data.get('type') == 'buy':
                if current_price <= position_data.get('stop_loss', 0) or current_price >= position_data.get('take_profit', float('inf')):
                    return True, "default_exit"
            else:
                if current_price >= position_data.get('stop_loss', float('inf')) or current_price <= position_data.get('take_profit', 0):
                    return True, "default_exit"
            return False, ""
        
        # Get ATR if available in position data
        atr = position_data.get('atr', None)
        
        # Get market data if available
        market_data = position_data.get('market_data', None)
        
        # Use risk manager to check exit conditions
        return self.risk_manager.should_exit_trade(
            position_data=position_data,
            current_price=current_price,
            atr=atr,
            market_data=market_data
        ) 