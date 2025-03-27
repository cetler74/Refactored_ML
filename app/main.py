#!/usr/bin/env python3
import logging
import os
import sys
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Set up log directory
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app_debug.log")),
        logging.StreamHandler()
    ]
)

# Set specific loggers to different levels
logging.getLogger('apscheduler').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('ccxt').setLevel(logging.WARNING)
logging.getLogger('app.exchange.manager').setLevel(logging.DEBUG)  # Set to DEBUG for our token prefix debugging

# Set up file handler for trading bot log
trading_bot_handler = logging.FileHandler(os.path.join(log_dir, "trading_bot.log"))
trading_bot_handler.setLevel(logging.INFO)
trading_bot_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add the handler to the root logger
logging.getLogger().addHandler(trading_bot_handler)

logger = logging.getLogger(__name__)

# Override the StrategyModule implementation with our own
from app.strategies.strategy import StrategyModule as OriginalStrategyModule
from typing import Dict, Any, List, Optional, Tuple

# Define our custom TA-based StrategyModule before importing orchestrator
class CustomTAStrategyModule:
    """A replacement StrategyModule that generates signals based on technical analysis."""
    
    def __init__(self, settings=None, ta_module=None, ml_module=None):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.ta_module = ta_module
        self.ml_module = ml_module
        self.pd = pd
        self.np = np
        self.logger.info("Initialized CustomTAStrategyModule for market-based signal generation")
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for decision making."""
        try:
            # Convert list to DataFrame if needed
            if isinstance(df, list):
                # Check if it's list of lists (OHLCV format from exchange)
                if isinstance(df[0], list):
                    df = self.pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    # Convert timestamp if needed
                    if df['timestamp'].dtype != 'datetime64[ns]':
                        df['timestamp'] = self.pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
            
            # Ensure we have a DataFrame with at least 20 rows for meaningful calculations
            if not isinstance(df, self.pd.DataFrame):
                self.logger.warning(f"Input is not a DataFrame: {type(df)}")
                return None
                
            if len(df) < 20:
                self.logger.warning("Not enough data points for technical analysis (need at least 20)")
                return None
                
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                self.logger.warning(f"DataFrame missing required columns: {missing}")
                return None
            
            # Calculate SMA
            df['sma7'] = df['close'].rolling(window=7).mean()
            df['sma25'] = df['close'].rolling(window=25).mean()
            
            # Calculate EMA
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # First RSI calculation
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['middle_band'] = df['close'].rolling(window=20).mean()
            df['std_dev'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
            df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
            
            # ADX for trend strength
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
            minus_dm = minus_dm.abs().where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = self.pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            df['adx'] = dx.rolling(window=14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Fill NaN values with forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def analyze_market_structure(self, df):
        """Analyze market structure for trend direction and strength."""
        if df is None or len(df) < 5:
            # Return a tuple of default values instead of None, 0
            return "neutral", 0
        
        # Get the most recent data point
        last = df.iloc[-1]
        
        # Trend direction based on EMA crossover
        trend = "neutral"
        trend_strength = 0
        
        # Check for trend direction
        if 'ema12' not in last or 'ema26' not in last:
            self.logger.warning("Missing ema columns in dataframe")
            return "neutral", 0
            
        if last['ema12'] > last['ema26']:
            trend = "bullish"
            # Measure trend strength by distance between EMAs
            trend_strength = (last['ema12'] / last['ema26'] - 1) * 100
        elif last['ema12'] < last['ema26']:
            trend = "bearish"
            # Measure trend strength by distance between EMAs
            trend_strength = (1 - last['ema12'] / last['ema26']) * 100
        
        # Adjust trend strength based on ADX
        if 'adx' in last and not self.pd.isna(last['adx']):
            # ADX > 25 indicates strong trend
            if last['adx'] > 25:
                trend_strength *= 1.5
            elif last['adx'] < 20:
                trend_strength *= 0.8
        
        return trend, float(trend_strength)
        
    def calculate_signal_confidence(self, df, trend, trend_strength):
        """Calculate confidence level for the generated signal."""
        if df is None or len(df) < 5:
            return 0.5
        
        # Get the most recent data point
        last = df.iloc[-1]
        
        # Start with base confidence based on trend strength
        confidence = min(0.5 + (trend_strength / 100), 0.9)
        
        # RSI confirmation
        if 'rsi' in last:
            if trend == "bullish" and last['rsi'] > 50:
                confidence += 0.1
            elif trend == "bearish" and last['rsi'] < 50:
                confidence += 0.1
        
        # MACD confirmation
        if 'macd' in last and 'macd_signal' in last:
            if trend == "bullish" and last['macd'] > last['macd_signal']:
                confidence += 0.1
            elif trend == "bearish" and last['macd'] < last['macd_signal']:
                confidence += 0.1
        
        # Bollinger Band confirmation
        if 'close' in last and 'middle_band' in last:
            if trend == "bullish" and last['close'] > last['middle_band']:
                confidence += 0.05
            elif trend == "bearish" and last['close'] < last['middle_band']:
                confidence += 0.05
        
        # Volume confirmation
        if 'volume_ratio' in last:
            if last['volume_ratio'] > 1:
                confidence += 0.05
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        return confidence
        
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from the provided market data using technical analysis."""
        logger = logging.getLogger(__name__)
        logger.info("CustomTAStrategyModule generate_signals called with market data")
        
        if not market_data or not market_data.get("data"):
            logger.warning("No market data available, cannot generate signals")
            return []
        
        # Get the actual trading pairs from market data
        signals = []
        pair_data = market_data.get("data", {})
        
        for symbol, data in pair_data.items():
            logger.info(f"Analyzing pair: {symbol}")
            
            # Check if we have OHLCV data
            if not data or "ohlcv" not in data:
                logger.warning(f"No OHLCV data for {symbol}, skipping")
                continue
            
            try:
                # Get the OHLCV data for 1h timeframe (or fallback to available data)
                timeframe = '1h'
                ohlcv_data = None
                
                # Safely extract OHLCV data
                try:
                    if isinstance(data["ohlcv"], dict) and timeframe in data["ohlcv"]:
                        ohlcv_data = data["ohlcv"][timeframe]
                    else:
                        ohlcv_data = data["ohlcv"]
                except Exception as e:
                    logger.warning(f"Error extracting OHLCV data for {symbol}: {str(e)}")
                    continue
                
                # Safe DataFrame checking
                if ohlcv_data is None or (isinstance(ohlcv_data, pd.DataFrame) and ohlcv_data.empty) or len(ohlcv_data) < 30:
                    logger.warning(f"Not enough OHLCV data for {symbol}, need at least 30 candles for reliable analysis")
                    continue
                
                # Get current price - handle both list and DataFrame formats
                current_price = 0
                try:
                    if isinstance(ohlcv_data, pd.DataFrame):
                        current_price = ohlcv_data['close'].iloc[-1] if not ohlcv_data.empty else 0
                    else:
                        current_price = ohlcv_data[-1][4] if isinstance(ohlcv_data[0], list) else float(data.get("ticker", {}).get("last", 0))
                except Exception as e:
                    logger.warning(f"Error getting current price for {symbol}: {str(e)}")
                    # Try getting price from ticker as fallback
                    try:
                        current_price = float(data.get("ticker", {}).get("last", 0))
                    except:
                        logger.warning(f"Could not get price from ticker for {symbol}")
                        continue
                
                if current_price == 0:
                    logger.warning(f"Invalid price (0) for {symbol}, skipping")
                    continue
                    
                # Calculate technical indicators
                df = self.calculate_indicators(ohlcv_data)
                if df is None or (isinstance(df, pd.DataFrame) and df.empty) or len(df) < 5:
                    logger.warning(f"Failed to calculate indicators for {symbol}, skipping")
                    continue
                
                # Analyze market structure
                try:
                    trend, trend_strength = self.analyze_market_structure(df)
                except ValueError as e:
                    logger.error(f"Error unpacking values in analyze_market_structure for {symbol}: {str(e)}")
                    # Default values if unpacking fails
                    trend = "neutral"
                    trend_strength = 0.5
                
                # Get the last row of indicators
                last = df.iloc[-1]
                
                # Log the analysis results
                logger.info(f"Analysis for {symbol}: Trend={trend}, Strength={trend_strength:.2f}, RSI={last['rsi']:.2f}, MACD={last['macd']:.2f}")
                
                # For testing, force a buy signal when RSI < 70 or a sell signal when RSI > 30
                trade_type = None
                signal_confidence = 0.0
                
                if trend == "bullish" and last['rsi'] < 70:
                    trade_type = "buy"
                    try:
                        signal_confidence = self.calculate_signal_confidence(df, trend, trend_strength)
                    except Exception as e:
                        logger.error(f"Error calculating signal confidence for {symbol}: {str(e)}")
                        signal_confidence = 0.6  # Default confidence
                    logger.info(f"Bullish signal detected for {symbol} with RSI {last['rsi']:.2f} < 70, confidence: {signal_confidence:.2f}")
                elif trend == "bearish" and last['rsi'] > 30:
                    trade_type = "sell"
                    try:
                        signal_confidence = self.calculate_signal_confidence(df, trend, trend_strength)
                    except Exception as e:
                        logger.error(f"Error calculating signal confidence for {symbol}: {str(e)}")
                        signal_confidence = 0.6  # Default confidence
                    logger.info(f"Bearish signal detected for {symbol} with RSI {last['rsi']:.2f} > 30, confidence: {signal_confidence:.2f}")
                
                # For test purposes always create a sample signal for each pair
                # This ensures we always get a signal to test with
                if True:  # Always create a test signal
                    trade_type = "buy" if trend == "bullish" else "sell"
                    try:
                        signal_confidence = self.calculate_signal_confidence(df, trend, trend_strength)
                    except Exception as e:
                        logger.error(f"Error calculating signal confidence for {symbol}: {str(e)}")
                        signal_confidence = 0.6  # Default confidence
                    
                    # Create signal with consistent field structure
                    signal = {
                        "symbol": symbol,
                        "price": current_price,
                        "trade_type": trade_type,  # Always use trade_type for consistency
                        "confidence": signal_confidence,
                        "timestamp": datetime.now().isoformat(),
                        "strategy": "ta_strategy",
                        "timeframe": timeframe,
                        "reason": f"Test signal: {trend} trend with {trend_strength:.2f} strength, RSI: {last['rsi']:.2f}, MACD: {last['macd']:.2f}"
                    }
                    signals.append(signal)
                    logger.info(f"Generated TEST signal for {symbol}: {trade_type} at {current_price}")
                else:
                    logger.info(f"No clear signal for {symbol}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"CustomTAStrategyModule - generated {len(signals)} signals based on technical analysis")
        return signals
    
    async def update_signals(self, predictions):
        """Update signals with ML predictions (stub method for compatibility)"""
        logger = logging.getLogger(__name__)
        logger.info("CustomTAStrategyModule.update_signals called")
        return []
    
    def should_exit_trade(self, position, current_price):
        """Check if a trade should be exited based on current price and position data"""
        logger = logging.getLogger(__name__)
        logger.info(f"CustomTAStrategyModule.should_exit_trade called for position: {position['symbol'] if position else 'None'}")
        
        if not position or not current_price:
            return False, None
        
        # Simple exit strategy based on profit/loss
        entry_price = position.get('entry_price', 0)
        if entry_price == 0:
            return False, None
            
        position_type = position.get('trade_type', position.get('type', 'unknown'))
        
        # Calculate profit/loss percentage
        if position_type == 'buy':
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            # Exit if profit exceeds 3% or loss exceeds 2%
            if profit_pct >= 3:
                return True, f"profit_target_reached_{profit_pct:.2f}%"
            elif profit_pct <= -2:
                return True, f"stop_loss_triggered_{profit_pct:.2f}%"
                
        elif position_type == 'sell':
            profit_pct = (entry_price - current_price) / entry_price * 100
            
            # Exit if profit exceeds 3% or loss exceeds 2%
            if profit_pct >= 3:
                return True, f"profit_target_reached_{profit_pct:.2f}%"
            elif profit_pct <= -2:
                return True, f"stop_loss_triggered_{profit_pct:.2f}%"
        
        return False, None

# Replace the original StrategyModule with our custom one
import app.strategies.strategy
app.strategies.strategy.StrategyModule = CustomTAStrategyModule
logger.info("Replaced original StrategyModule with CustomTAStrategyModule for market-based signal generation")

# Import modules after environment and logging setup
from app.core.orchestrator import BotOrchestrator
from app.config.settings import get_settings

# Define the patched_generate_signals function before using it
async def patched_generate_signals(self):
    """Patched version of generate_signals that handles trade_type instead of side."""
    try:
        # Collect market data first
        market_data = await self.collect_market_data()
        if not market_data or not market_data.get("success", False):
            self.logger.warning("Could not generate signals: market data is unavailable")
            return {"success": False, "error": "Market data unavailable"}
        
        # Add debug logging
        self.logger.info("Calling strategy_module.generate_signals")
        
        # Make sure strategy_module is initialized
        if not self.strategy_module:
            await self.initialize_modules()

        # Check if strategy_module is a placeholder, and if so, reinitialize it
        if hasattr(self.strategy_module, '__class__') and self.strategy_module.__class__.__name__ == 'PlaceholderStrategyModule':
            self.logger.warning("Detected placeholder strategy module, replacing with real implementation")
            from app.strategies.strategy import StrategyModule
            self.strategy_module = StrategyModule(
                settings=self.settings,
                ta_module=self.ta_module,
                ml_module=self.ml_module
            )
            self.logger.info("Successfully replaced placeholder strategy module with real implementation")
            
        signals = await self.strategy_module.generate_signals(market_data)
        
        # Log signal details for debugging
        self.logger.info(f"Received {len(signals) if signals else 0} signals from strategy module")
        for i, signal in enumerate(signals):
            # Check if we got a signal with dict-like structure
            if not isinstance(signal, dict):
                self.logger.error(f"Signal {i} is not a dictionary: {signal}")
                continue
                
            # Log the keys in the signal
            self.logger.info(f"Signal {i} has keys: {list(signal.keys())}")
            
            # Log the actual signal content for debugging
            self.logger.info(f"Signal {i} content: {signal}")
        
        # Update signal history
        self.previous_signals = self.current_signals
        self.current_signals = signals
        
        # Log generated signals
        if signals:
            self.logger.info(f"Generated {len(signals)} trading signals")
            for i, signal in enumerate(signals):
                try:
                    # Always use get() with default values to avoid KeyError
                    symbol = signal.get('symbol', 'unknown')
                    price = signal.get('price', 0)
                    
                    # Handle any of the possible field names for trade direction
                    # but always store as trade_type for consistency
                    trade_type_value = (
                        signal.get('trade_type') or 
                        signal.get('side') or 
                        signal.get('direction') or 
                        'unknown'
                    )
                    
                    # Ensure all signals have consistent field names
                    if 'trade_type' not in signal:
                        signal['trade_type'] = trade_type_value
                        self.logger.info(f"Added missing trade_type field to signal {i}, value: {trade_type_value}")
                    
                    # Log with safely obtained values
                    self.logger.info(f"Signal {i}: {symbol} - {trade_type_value} at {price}")
                except Exception as e:
                    self.logger.error(f"Error processing signal {i}: {str(e)}")
                    self.logger.error(f"Signal content: {signal}")
        else:
            self.logger.info("No trading signals generated")
        
        # Execute signals if appropriate based on current mode
        if signals and getattr(self.settings, 'AUTO_EXECUTE_SIGNALS', False):
            await self.execute_signals(signals)
        
        # Update heartbeat to show this process is running
        self.update_heartbeat()
        
        return {"success": True, "signals": signals}
        
    except Exception as e:
        self.logger.exception(f"Error generating signals: {str(e)}")
        return {"success": False, "error": str(e)}

# Save the original method (after our function is defined)
original_generate_signals = BotOrchestrator.generate_signals

# Patch the methods
BotOrchestrator.patched_generate_signals = patched_generate_signals
BotOrchestrator.generate_signals = patched_generate_signals
logger.info("Applied patch to BotOrchestrator.generate_signals to use trade_type instead of side/direction")

# Also monkey patch the schedule_jobs method to use patched_generate_signals instead of generate_signals
original_schedule_jobs = BotOrchestrator.schedule_jobs

def patched_schedule_jobs(self):
    """Patched version of schedule_jobs to use patched_generate_signals."""
    if not self.scheduler:
        self.logger.error("Scheduler not initialized, cannot schedule jobs")
        return
    
    # Get existing job IDs to avoid duplication
    existing_jobs = [job.id for job in self.scheduler.get_jobs()]
    
    # Market data collection - every minute
    if 'collect_market_data' not in existing_jobs:
        self.scheduler.add_job(
            self.collect_market_data,
            'interval',
            minutes=1,
            id='collect_market_data'
        )
    
    # Signal generation - every 1 minute with patched version
    if 'patched_generate_signals' not in existing_jobs:
        self.scheduler.add_job(
            self.patched_generate_signals,  # Use the patched method
            'interval',
            minutes=1,
            id='patched_generate_signals'
        )
    
    # Other jobs remain the same
    # Update trading pairs - every 4 hours
    if 'update_trading_pairs' not in existing_jobs:
        self.scheduler.add_job(
            self.update_trading_pairs,
            'interval',
            hours=4,
            id='update_trading_pairs'
        )
    
    # ML training job - get training interval with a fallback
    if 'train_ml_models' not in existing_jobs:
        training_interval = getattr(self.config, 'TRAINING_INTERVAL_MINUTES', 60)
        self.scheduler.add_job(
            self.train_ml_models,
            'interval',
            minutes=training_interval,
            id='train_ml_models'
        )
    
    # ML prediction job - get prediction interval with a fallback
    if 'make_predictions' not in existing_jobs:
        prediction_interval = getattr(self.config, 'PREDICTION_INTERVAL_MINUTES', 5)
        self.scheduler.add_job(
            self.make_predictions,
            'interval',
            minutes=prediction_interval,
            id='make_predictions'
        )
    
    # Update position prices - every minute
    if 'update_position_prices' not in existing_jobs:
        self.scheduler.add_job(
            self.update_position_prices,
            'interval',
            minutes=1,
            id='update_position_prices'
        )
    
    # Heartbeat job - every 30 seconds
    if 'update_heartbeat' not in existing_jobs:
        self.scheduler.add_job(
            self.update_heartbeat,
            'interval',
            seconds=30,
            id='update_heartbeat'
        )
    
    self.logger.info("All jobs scheduled successfully")

# Apply the schedule_jobs patch
BotOrchestrator.schedule_jobs = patched_schedule_jobs
logger.info("Applied patch to BotOrchestrator.schedule_jobs to use patched_generate_signals")

async def async_main():
    """Async main entry point for the trading bot."""
    logger.info("Starting ML-Powered Trading Bot")
    
    # Get application settings
    settings = get_settings()
    logger.info(f"Application mode: {settings.APP_MODE}")
    
    # Initialize and start the bot orchestrator
    orchestrator = BotOrchestrator(settings)
    
    try:
        # Start the bot and keep it running
        await orchestrator.start()
        
        # Create a loop that keeps the program running
        while True:
            # Check every 60 seconds if we should continue running
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.exception(f"Error occurred: {str(e)}")
    finally:
        # Ensure proper shutdown
        await orchestrator.shutdown()
        logger.info("Trading bot shutdown complete")

def main():
    """Main entry point for the trading bot."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.exception(f"Error occurred in asyncio loop: {str(e)}")

if __name__ == "__main__":
    main() 