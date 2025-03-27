import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("test_strategy_fix")

async def test_strategy_fixes():
    """Test that our fixes to the strategy modules work correctly."""
    logger.info("Testing strategy module fixes...")
    
    # First, import the original StrategyModule class
    from app.strategies.strategy import StrategyModule, MACrossoverStrategy
    from app.config.settings import get_settings
    
    # Create a sample dataframe
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 105,
        'low': np.random.randn(100) + 95,
        'close': np.random.randn(100) + 102,
        'volume': np.random.randn(100) + 1000
    }, index=dates)
    
    # Add EMA columns needed for testing
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['adx'] = 30  # Placeholder ADX values
    df['rsi'] = 50  # Placeholder RSI values
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Set metadata for the dataframe
    df.attrs['symbol'] = 'ETH/USDC'
    df.attrs['timeframe'] = '1h'
    
    # Create an instance of the MACrossoverStrategy
    ma_strategy = MACrossoverStrategy(9, 21)
    
    # Test the analyze_market_structure method
    try:
        trend, strength = ma_strategy.analyze_market_structure(df)
        logger.info(f"MACrossoverStrategy.analyze_market_structure() result: {trend}, {strength}")
    except Exception as e:
        logger.error(f"Error in MACrossoverStrategy.analyze_market_structure(): {str(e)}")
    
    # Create a sample dataframe without EMA columns to test error handling
    df_no_ema = pd.DataFrame({
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 105,
        'low': np.random.randn(100) + 95,
        'close': np.random.randn(100) + 102,
        'volume': np.random.randn(100) + 1000
    }, index=dates)
    
    # Add some indicators but not EMAs
    df_no_ema['rsi'] = 50
    
    # Test analyze_market_structure with missing columns
    try:
        trend, strength = ma_strategy.analyze_market_structure(df_no_ema)
        logger.info(f"MACrossoverStrategy.analyze_market_structure() with missing columns: {trend}, {strength}")
    except Exception as e:
        logger.error(f"Error in MACrossoverStrategy.analyze_market_structure() with missing columns: {str(e)}")
    
    # Now test the CustomTAStrategyModule
    from app.main import CustomTAStrategyModule
    
    # Create a CustomTAStrategyModule instance
    custom_module = CustomTAStrategyModule()
    
    # Test the analyze_market_structure method
    try:
        trend, strength = custom_module.analyze_market_structure(df)
        logger.info(f"CustomTAStrategyModule.analyze_market_structure() result: {trend}, {strength}")
    except Exception as e:
        logger.error(f"Error in CustomTAStrategyModule.analyze_market_structure(): {str(e)}")
    
    # Test with missing columns
    try:
        trend, strength = custom_module.analyze_market_structure(df_no_ema)
        logger.info(f"CustomTAStrategyModule.analyze_market_structure() with missing columns: {trend}, {strength}")
    except Exception as e:
        logger.error(f"Error in CustomTAStrategyModule.analyze_market_structure() with missing columns: {str(e)}")
    
    # Test the generate_signals method of the MACrossoverStrategy
    try:
        signals = await ma_strategy.generate_signals(df)
        logger.info(f"MACrossoverStrategy.generate_signals() result: {len(signals)} signals")
    except Exception as e:
        logger.error(f"Error in MACrossoverStrategy.generate_signals(): {str(e)}")
    
    logger.info("Strategy fix tests completed.")

if __name__ == "__main__":
    asyncio.run(test_strategy_fixes()) 