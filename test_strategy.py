import logging
import asyncio
import traceback
import pandas as pd
from app.config.settings import get_settings
from app.ta.indicators import TechnicalAnalysisModule
from app.ml.model import MLModule
from app.database.manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("test_strategy")

# Import the CustomTAStrategyModule
from app.main import CustomTAStrategyModule

async def test_strategy_module():
    try:
        # Get settings
        logger.info("Getting settings...")
        settings = get_settings()
        
        # Create necessary modules
        logger.info("Creating modules...")
        db_manager = DatabaseManager(db_path="data/trading_bot.db", debug=False)
        await db_manager.initialize()
        ta_module = TechnicalAnalysisModule(settings)
        ml_module = MLModule(settings, db_manager)
        
        # Create strategy module
        logger.info("Creating strategy module...")
        strategy_module = CustomTAStrategyModule(
            settings=settings,
            ta_module=ta_module,
            ml_module=ml_module
        )
        
        # Create sample market data
        logger.info("Creating sample market data...")
        # Create sample OHLCV data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        sample_data = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [105 + i * 0.1 for i in range(100)],
            'low': [95 + i * 0.1 for i in range(100)],
            'close': [102 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Create market data structure that matches what the strategy module expects
        market_data = {
            'success': True,
            'data': {
                'BTC/USDC': {
                    'ohlcv': {
                        '1h': sample_data
                    },
                    'ticker': {
                        'last': 110.0
                    }
                },
                'ETH/USDC': {
                    'ohlcv': {
                        '1h': sample_data.copy()
                    },
                    'ticker': {
                        'last': 55.0
                    }
                }
            }
        }
        
        # Calculate indicators
        logger.info("Calculating indicators...")
        for symbol, data in market_data['data'].items():
            for timeframe, ohlcv in data['ohlcv'].items():
                # Add indicators to the data
                market_data['data'][symbol]['ohlcv'][timeframe] = await ta_module.calculate_indicators(ohlcv)
        
        # Call generate_signals
        logger.info("Calling generate_signals...")
        signals = await strategy_module.generate_signals(market_data)
        
        # Print results
        if signals:
            logger.info(f"Generated {len(signals)} signals:")
            for i, signal in enumerate(signals):
                logger.info(f"Signal {i+1}: {signal}")
        else:
            logger.info("No signals generated")
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_strategy_module()) 